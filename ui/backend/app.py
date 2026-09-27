import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
import urllib.error
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
from pathlib import Path

import boto3
import mlflow
import sagemaker
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sagemaker.inputs import TrainingInput

ROOT = Path(os.getenv("PROJECT_ROOT", "/workspace"))
TF_DIR = ROOT / "infra" / "terraform"
CACHE_DIR = ROOT / ".dataset_cache"
CODE_CACHE = ROOT / ".code_cache"
CACHE_DIR.mkdir(exist_ok=True)
CODE_CACHE.mkdir(exist_ok=True)

app = FastAPI(title="SageMaker Reproducible Research Platform", version="7.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}

platform_status_cache = {
    "checked_at": 0.0,
    "ready": False,
    "bucket": None,
    "role": None,
    "error": None,
}
PLATFORM_STATUS_TTL_SECONDS = 10.0
INSTANCE_TYPE = "ml.g4dn.xlarge"
ALLOWED_COUNTS = {1, 2, 4}
ALLOWED_FRAMEWORKS = {"pytorch", "tensorflow"}
SPOT_ESTIMATE = float(os.getenv("ESTIMATED_SPOT_USD_PER_INSTANCE_HOUR", "0.55"))

PUBLIC_DATASETS = {
    "cifar10": {
        "key": "cifar10",
        "name": "CIFAR-10",
        "description": "60,000 32x32 color images in 10 classes",
        "source_url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "filename": "cifar-10-python.tar.gz",
        "format": "cifar-python-tar",
        "recommended_version": "v3",
    }
}

def run_cmd(args, cwd=None):
    p = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError((p.stdout + "\n" + p.stderr).strip())
    return p.stdout.strip()

def terraform_output(name):
    return run_cmd(["terraform", "output", "-raw", name], cwd=TF_DIR)

def aws_region():
    return os.getenv("AWS_DEFAULT_REGION", "eu-central-1")

def s3_client():
    return boto3.client("s3", region_name=aws_region())

def load_json_from_s3(key, default):
    try:
        bucket = terraform_output("artifact_bucket")
        obj = s3_client().get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return default

def save_json_to_s3(key, value):
    bucket = terraform_output("artifact_bucket")
    s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(value, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

def load_dataset_registry():
    return load_json_from_s3("dataset-registry/manifest.json", {"datasets": []})

def save_dataset_registry(registry):
    save_json_to_s3("dataset-registry/manifest.json", registry)

def load_code_registry():
    return load_json_from_s3("code-registry/manifest.json", {"codes": []})

def save_code_registry(registry):
    save_json_to_s3("code-registry/manifest.json", registry)

def upsert_dataset(item):
    registry = load_dataset_registry()
    registry["datasets"] = [d for d in registry["datasets"] if d["id"] != item["id"]]
    registry["datasets"].append(item)
    save_dataset_registry(registry)
    return item

def upsert_code(item):
    registry = load_code_registry()
    registry["codes"] = [c for c in registry["codes"] if c["id"] != item["id"]]
    registry["codes"].append(item)
    save_code_registry(registry)
    return item

def log(job_id, message):
    jobs[job_id]["logs"].append(str(message))

def launch(job_id, fn, *args):
    def wrapper():
        try:
            jobs[job_id]["status"] = "running"
            fn(job_id, *args)
            if jobs[job_id]["status"] == "running":
                jobs[job_id]["status"] = "completed"
        except Exception as exc:
            log(job_id, f"ERROR: {exc}")
            jobs[job_id]["status"] = "failed"
    threading.Thread(target=wrapper, daemon=True).start()

def estimate_budget(count, minutes, total_budget, planned_runs):
    if count not in ALLOWED_COUNTS:
        raise HTTPException(400, "GPU workers must be 1, 2, or 4")
    if not 1 <= minutes <= 60:
        raise HTTPException(400, "max_minutes must be between 1 and 60")
    if total_budget <= 0 or planned_runs < 1:
        raise HTTPException(400, "Invalid budget configuration")
    per_run = total_budget / planned_runs
    estimated = count * (minutes / 60.0) * SPOT_ESTIMATE
    return {
        "estimated_run_cost_usd": round(estimated, 4),
        "per_run_budget_usd": round(per_run, 4),
        "within_target": estimated <= per_run,
    }

@app.get("/api/health")
def health():
    return {"ok": True, "project": "sagemaker-research-platform-full-ui"}


def detect_platform_resources(force=False):
    """
    Persistent readiness check.
    Does not depend on in-memory setup jobs, so browser refreshes and backend
    restarts still recover the real AWS/Terraform state.
    """
    now = time.time()
    if (
        not force
        and now - platform_status_cache["checked_at"] < PLATFORM_STATUS_TTL_SECONDS
    ):
        return dict(platform_status_cache)

    result = {
        "checked_at": now,
        "ready": False,
        "bucket": None,
        "role": None,
        "error": None,
    }

    try:
        bucket = terraform_output("artifact_bucket")
        role = terraform_output("sagemaker_role_arn")

        if not bucket or not role:
            raise RuntimeError("Terraform outputs are not available yet")

        # Verify that the resources still exist in AWS.
        s3_client().head_bucket(Bucket=bucket)

        role_name = role.rsplit("/", 1)[-1]
        boto3.client("iam").get_role(RoleName=role_name)

        result.update({
            "ready": True,
            "bucket": bucket,
            "role": role,
        })
    except Exception as exc:
        result["error"] = str(exc)

    platform_status_cache.update(result)
    return dict(platform_status_cache)


@app.get("/api/platform-status")
def platform_status():
    setup_jobs = [j for j in jobs.values() if j.get("type") == "setup"]
    latest = setup_jobs[-1] if setup_jobs else None

    persistent = detect_platform_resources(force=False)

    return {
        "ready": persistent["ready"],
        "latest_setup_job": latest,
        "bucket": persistent["bucket"],
        "role": persistent["role"],
        "detected_from_aws": True,
        "check_error": persistent["error"],
    }

@app.get("/api/catalog")
def public_catalog():
    return {"datasets": list(PUBLIC_DATASETS.values())}

@app.get("/api/budget-estimate")
def budget_estimate(
    instance_count: int = 1,
    max_minutes: int = 5,
    total_budget_usd: float = 1.0,
    planned_runs: int = 5,
):
    return estimate_budget(instance_count, max_minutes, total_budget_usd, planned_runs)

@app.get("/api/datasets")
def datasets():
    return load_dataset_registry()

@app.get("/api/codes")
def codes():
    return load_code_registry()

@app.get("/api/jobs")
def list_jobs():
    return list(jobs.values())[::-1]

@app.post("/api/setup")
def setup():
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "type": "setup",
        "status": "queued",
        "logs": [],
        "stage": "Queued",
        "progress_percent": 0,
    }
    launch(job_id, do_setup)
    return jobs[job_id]

def do_setup(job_id):
    jobs[job_id]["stage"] = "Checking AWS credentials"
    jobs[job_id]["progress_percent"] = 10
    log(job_id, run_cmd(["aws", "sts", "get-caller-identity"]))

    jobs[job_id]["stage"] = "Terraform init"
    jobs[job_id]["progress_percent"] = 30
    log(job_id, run_cmd(["terraform", "init", "-input=false"], cwd=TF_DIR))

    jobs[job_id]["stage"] = "Terraform validate"
    jobs[job_id]["progress_percent"] = 45
    log(job_id, run_cmd(["terraform", "validate"], cwd=TF_DIR))

    jobs[job_id]["stage"] = "Creating S3 + IAM"
    jobs[job_id]["progress_percent"] = 65
    log(job_id, run_cmd(["terraform", "apply", "-auto-approve", "-input=false"], cwd=TF_DIR))

    jobs[job_id]["stage"] = "Verifying resources"
    jobs[job_id]["progress_percent"] = 90
    bucket = terraform_output("artifact_bucket")
    role = terraform_output("sagemaker_role_arn")
    jobs[job_id]["bucket"] = bucket
    jobs[job_id]["role"] = role
    log(job_id, f"S3 bucket: {bucket}")
    log(job_id, f"SageMaker role: {role}")

    jobs[job_id]["stage"] = "Ready"
    jobs[job_id]["progress_percent"] = 100
    detect_platform_resources(force=True)


@app.post("/api/destroy")
def destroy():
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"id": job_id, "type": "destroy", "status": "queued", "logs": []}
    launch(job_id, do_destroy)
    return jobs[job_id]

def do_destroy(job_id):
    log(job_id, run_cmd(["terraform", "destroy", "-auto-approve", "-input=false"], cwd=TF_DIR))
    platform_status_cache.update({
        "checked_at": time.time(),
        "ready": False,
        "bucket": None,
        "role": None,
        "error": None,
    })


def registry_find_dataset(name, version):
    safe_name = name.strip().lower().replace(" ", "-")
    safe_version = version.strip().replace(" ", "-")
    target_id = f"{safe_name}:{safe_version}"
    return next((d for d in load_dataset_registry()["datasets"] if d["id"] == target_id), None)


def s3_object_exists(s3_uri):
    try:
        bucket = s3_uri.split("/", 3)[2]
        key = s3_uri.split("/", 3)[3]
        s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def file_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def probe_range_support(url):
    """
    Returns (supports_ranges, total_size).
    Tries HEAD first, then a 0-0 byte request if needed.
    """
    total = 0
    try:
        req = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": "sagemaker-research-platform/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            accepts = (resp.headers.get("Accept-Ranges") or "").lower()
            if "bytes" in accepts:
                return True, total
    except Exception:
        pass

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "sagemaker-research-platform/1.0",
                "Range": "bytes=0-0",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = getattr(resp, "status", None)
            content_range = resp.headers.get("Content-Range") or ""
            if status == 206 and "/" in content_range:
                total = int(content_range.split("/")[-1])
                return True, total
    except Exception:
        pass

    return False, total


def parallel_download(url, target_path, job_id, connections=4):
    supports_ranges, total = probe_range_support(url)
    jobs[job_id]["download_mode"] = (
        f"Parallel ({connections} connections)" if supports_ranges and total > 0 else "Standard"
    )
    jobs[job_id]["total_bytes"] = total

    if not supports_ranges or total <= 0:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "sagemaker-research-platform/1.0"},
        )
        started = time.time()
        downloaded = 0
        with urllib.request.urlopen(request, timeout=120) as response:
            if not total:
                total = int(response.headers.get("Content-Length") or 0)
                jobs[job_id]["total_bytes"] = total
            with open(target_path, "wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    jobs[job_id]["downloaded_bytes"] = downloaded
                    if total > 0:
                        jobs[job_id]["progress_percent"] = min(90, max(1, int(downloaded / total * 90)))
                    elapsed = max(time.time() - started, 0.001)
                    jobs[job_id]["speed_bps"] = downloaded / elapsed
                    if total > downloaded and jobs[job_id]["speed_bps"] > 0:
                        jobs[job_id]["eta_seconds"] = int((total - downloaded) / jobs[job_id]["speed_bps"])
        return

    # Parallel range download.
    part_dir = target_path.parent / f"{target_path.name}.parts"
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True)

    chunk = total // connections
    ranges = []
    for i in range(connections):
        start = i * chunk
        end = total - 1 if i == connections - 1 else ((i + 1) * chunk - 1)
        ranges.append((i, start, end))

    progress = {"bytes": 0}
    progress_lock = threading.Lock()
    started = time.time()

    def download_part(item):
        idx, start, end = item
        part_path = part_dir / f"part-{idx:02d}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "sagemaker-research-platform/1.0",
                "Range": f"bytes={start}-{end}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as response, open(part_path, "wb") as out:
            while True:
                data = response.read(1024 * 1024)
                if not data:
                    break
                out.write(data)
                with progress_lock:
                    progress["bytes"] += len(data)
                    downloaded = progress["bytes"]
                    jobs[job_id]["downloaded_bytes"] = downloaded
                    jobs[job_id]["progress_percent"] = min(90, max(1, int(downloaded / total * 90)))
                    elapsed = max(time.time() - started, 0.001)
                    speed = downloaded / elapsed
                    jobs[job_id]["speed_bps"] = speed
                    jobs[job_id]["eta_seconds"] = int((total - downloaded) / speed) if speed > 0 else None
        return idx, part_path

    parts = {}
    try:
        with ThreadPoolExecutor(max_workers=connections) as executor:
            futures = [executor.submit(download_part, r) for r in ranges]
            for future in as_completed(futures):
                idx, path = future.result()
                parts[idx] = path

        with open(target_path, "wb") as merged:
            for idx in range(connections):
                with open(parts[idx], "rb") as part:
                    shutil.copyfileobj(part, merged)
    finally:
        shutil.rmtree(part_dir, ignore_errors=True)


@app.post("/api/datasets/register-public")
def register_public_dataset(catalog_key: str = Form(...), version: str = Form(...)):
    if catalog_key not in PUBLIC_DATASETS:
        raise HTTPException(404, "Public dataset preset not found")
    preset = PUBLIC_DATASETS[catalog_key]
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "type": "dataset-download",
        "dataset_name": preset["name"],
        "version": version,
        "status": "queued",
        "logs": [],
        "stage": "Queued",
        "progress_percent": 0,
        "downloaded_bytes": 0,
        "total_bytes": 0,
        "download_mode": "Checking...",
        "speed_bps": 0,
        "eta_seconds": None,
        "cache_hit": None,
    }
    launch(job_id, do_register_public_dataset, preset, version)
    return jobs[job_id]

def do_register_public_dataset(job_id, preset, version):
    bucket = terraform_output("artifact_bucket")
    safe_name = preset["name"].lower().replace(" ", "-")
    safe_version = version.strip().replace(" ", "-")
    local_path = CACHE_DIR / preset["filename"]

    # 1) Check S3/registry first.
    jobs[job_id]["stage"] = "Checking S3"
    jobs[job_id]["progress_percent"] = 2
    existing = registry_find_dataset(preset["name"], version)
    if existing and s3_object_exists(existing["s3_uri"]):
        jobs[job_id]["cache_hit"] = "s3"
        jobs[job_id]["download_mode"] = "Skipped — already in S3"
        jobs[job_id]["dataset"] = existing
        jobs[job_id]["progress_percent"] = 100
        jobs[job_id]["stage"] = "Already registered"
        jobs[job_id]["downloaded_bytes"] = 0
        jobs[job_id]["total_bytes"] = 0
        log(job_id, f"{preset['name']}:{version} already exists in S3. Download skipped.")
        return

    # 2) Check local cache.
    if local_path.exists() and local_path.stat().st_size > 0:
        jobs[job_id]["cache_hit"] = "local"
        jobs[job_id]["download_mode"] = "Local cache"
        jobs[job_id]["stage"] = "Using local cache"
        jobs[job_id]["progress_percent"] = 90
        jobs[job_id]["downloaded_bytes"] = local_path.stat().st_size
        jobs[job_id]["total_bytes"] = local_path.stat().st_size
        log(job_id, f"Using cached file {local_path.name}")
    else:
        # 3) Faster HTTP downloader with automatic Range support detection.
        jobs[job_id]["cache_hit"] = None
        jobs[job_id]["stage"] = "Downloading"
        jobs[job_id]["progress_percent"] = 1
        log(job_id, f"Downloading {preset['name']} from official public source")
        parallel_download(preset["source_url"], local_path, job_id, connections=4)

    jobs[job_id]["stage"] = "Hashing"
    jobs[job_id]["progress_percent"] = 92
    sha256 = file_sha256(local_path)

    key = f"datasets/{safe_name}/{safe_version}/{preset['filename']}"
    jobs[job_id]["stage"] = "Uploading to S3"
    jobs[job_id]["progress_percent"] = 95
    s3_client().upload_file(str(local_path), bucket, key)

    jobs[job_id]["stage"] = "Registering"
    jobs[job_id]["progress_percent"] = 98
    item = {
        "id": f"{safe_name}:{safe_version}",
        "name": preset["name"],
        "version": version,
        "filename": preset["filename"],
        "s3_uri": f"s3://{bucket}/{key}",
        "sha256": sha256,
        "source_type": "public-catalog",
        "source_url": preset["source_url"],
        "format": preset["format"],
        "registered_at": int(time.time()),
    }
    upsert_dataset(item)

    jobs[job_id]["dataset"] = item
    jobs[job_id]["stage"] = "Registered"
    jobs[job_id]["progress_percent"] = 100
    if not jobs[job_id]["total_bytes"]:
        jobs[job_id]["total_bytes"] = local_path.stat().st_size
    jobs[job_id]["downloaded_bytes"] = jobs[job_id]["total_bytes"]
    jobs[job_id]["eta_seconds"] = 0


@app.post("/api/datasets/upload")
async def upload_dataset(name: str = Form(...), version: str = Form(...), dataset: UploadFile = File(...)):
    payload = await dataset.read()
    sha256 = hashlib.sha256(payload).hexdigest()
    bucket = terraform_output("artifact_bucket")
    safe_name = name.strip().replace(" ", "-")
    safe_version = version.strip().replace(" ", "-")
    key = f"datasets/{safe_name}/{safe_version}/{dataset.filename}"
    s3_client().put_object(Bucket=bucket, Key=key, Body=payload)
    item = {
        "id": f"{safe_name}:{safe_version}",
        "name": name,
        "version": version,
        "filename": dataset.filename,
        "s3_uri": f"s3://{bucket}/{key}",
        "sha256": sha256,
        "source_type": "manual-upload",
        "source_url": None,
        "format": "user-file",
        "registered_at": int(time.time()),
    }
    return upsert_dataset(item)

@app.post("/api/codes/upload")
async def upload_code(
    name: str = Form(...),
    version: str = Form(...),
    framework: str = Form(...),
    code_zip: UploadFile = File(...),
):
    if framework not in ALLOWED_FRAMEWORKS:
        raise HTTPException(400, "Framework must be pytorch or tensorflow")
    if not code_zip.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Training code must be a ZIP file")

    payload = await code_zip.read()
    sha256 = hashlib.sha256(payload).hexdigest()

    # Validate ZIP contains train.py, either root or one top-level folder.
    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "code.zip"
        zpath.write_bytes(payload)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(Path(td) / "src")
        src = Path(td) / "src"
        if not (src / "train.py").exists():
            dirs = [p for p in src.iterdir() if p.is_dir()]
            if len(dirs) == 1 and (dirs[0] / "train.py").exists():
                src = dirs[0]
            else:
                raise HTTPException(400, "ZIP must contain train.py")

    bucket = terraform_output("artifact_bucket")
    safe_name = name.strip().replace(" ", "-")
    safe_version = version.strip().replace(" ", "-")
    key = f"training-code/{safe_name}/{safe_version}/{code_zip.filename}"
    s3_client().put_object(Bucket=bucket, Key=key, Body=payload)

    item = {
        "id": f"{safe_name}:{safe_version}",
        "name": name,
        "version": version,
        "framework": framework,
        "filename": code_zip.filename,
        "s3_uri": f"s3://{bucket}/{key}",
        "sha256": sha256,
        "registered_at": int(time.time()),
    }
    return upsert_code(item)

def materialize_code(code_item, job_id):
    target = CODE_CACHE / job_id
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    bucket = code_item["s3_uri"].split("/", 3)[2]
    key = code_item["s3_uri"].split("/", 3)[3]
    zpath = target / "code.zip"
    s3_client().download_file(bucket, key, str(zpath))

    with zipfile.ZipFile(zpath) as z:
        z.extractall(target / "src")
    src = target / "src"
    if (src / "train.py").exists():
        return src
    dirs = [p for p in src.iterdir() if p.is_dir()]
    if len(dirs) == 1 and (dirs[0] / "train.py").exists():
        return dirs[0]
    raise RuntimeError("Registered training code no longer contains train.py")

def build_estimator(
    framework, source_dir, role, count, output, max_minutes, epochs, code_hash,
    use_subset, train_max_per_class, test_max_per_class, subset_seed
):
    from sagemaker.pytorch import PyTorch
    from sagemaker.tensorflow import TensorFlow

    common = dict(
        entry_point="train.py",
        source_dir=str(source_dir),
        role=role,
        instance_type=INSTANCE_TYPE,
        instance_count=count,
        output_path=output,
        hyperparameters={
            "epochs": epochs,
            "batch-size": 64,
            "code-hash": code_hash,
            "use-subset": str(use_subset).lower(),
            "train-max-per-class": train_max_per_class,
            "test-max-per-class": test_max_per_class,
            "subset-seed": subset_seed,
        },
        use_spot_instances=True,
        max_run=max_minutes * 60,
        max_wait=max_minutes * 60 + 300,
        disable_profiler=True,
    )
    if framework == "pytorch":
        return PyTorch(framework_version="2.2.0", py_version="py310", **common)
    return TensorFlow(framework_version="2.15.0", py_version="py310", **common)

@app.post("/api/run-sagemaker")
def run_sagemaker(
    framework: str = Form(...),
    dataset_id: str = Form(...),
    code_id: str = Form(...),
    epochs: int = Form(1),
    instance_count: int = Form(1),
    max_minutes: int = Form(5),
    total_budget_usd: float = Form(1.0),
    planned_runs: int = Form(5),
    use_subset: bool = Form(True),
    train_max_per_class: int = Form(200),
    test_max_per_class: int = Form(50),
    subset_seed: int = Form(42),
):
    budget = estimate_budget(instance_count, max_minutes, total_budget_usd, planned_runs)
    if not budget["within_target"]:
        raise HTTPException(400, "Current configuration exceeds the per-run budget target")

    if train_max_per_class < 1 or test_max_per_class < 1:
        raise HTTPException(400, "Subset samples per class must be >= 1")
    if train_max_per_class > 5000 or test_max_per_class > 1000:
        raise HTTPException(400, "Subset samples per class exceed CIFAR-10 class limits")

    dataset = next((d for d in load_dataset_registry()["datasets"] if d["id"] == dataset_id), None)
    code = next((c for c in load_code_registry()["codes"] if c["id"] == code_id), None)
    if not dataset:
        raise HTTPException(404, "Dataset version not found")
    if not code:
        raise HTTPException(404, "Training code version not found")
    if code["framework"] != framework:
        raise HTTPException(400, "Selected training code framework does not match selected framework")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "type": "sagemaker",
        "framework": framework,
        "dataset": dataset,
        "code": code,
        "epochs": epochs,
        "instance_count": instance_count,
        "instance_type": INSTANCE_TYPE,
        "max_minutes": max_minutes,
        "budget": budget,
        "use_subset": use_subset,
        "train_max_per_class": train_max_per_class,
        "test_max_per_class": test_max_per_class,
        "subset_seed": subset_seed,
        "status": "queued",
        "logs": [],
        "stage": "Queued",
        "mlflow_run_id": None,
    }
    launch(job_id, do_sagemaker)
    return jobs[job_id]

def do_sagemaker(job_id):
    job = jobs[job_id]
    bucket = terraform_output("artifact_bucket")
    role = terraform_output("sagemaker_role_arn")
    session = boto3.Session(region_name=aws_region())
    sm_session = sagemaker.Session(boto_session=session, default_bucket=bucket)

    job["stage"] = "Preparing training code"
    source_dir = materialize_code(job["code"], job_id)
    output = f"s3://{bucket}/outputs/{job_id}"

    estimator = build_estimator(
        job["framework"], source_dir, role, job["instance_count"], output,
        job["max_minutes"], job["epochs"], job["code"]["sha256"],
        job["use_subset"], job["train_max_per_class"],
        job["test_max_per_class"], job["subset_seed"]
    )
    estimator.sagemaker_session = sm_session

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("sagemaker-research-platform")

    with mlflow.start_run(run_name=f"{job['code']['name']}-{job_id[:8]}") as active:
        job["mlflow_run_id"] = active.info.run_id
        mlflow.log_params({
            "dataset_id": job["dataset"]["id"],
            "dataset_sha256": job["dataset"]["sha256"],
            "dataset_s3_uri": job["dataset"]["s3_uri"],
            "training_code_id": job["code"]["id"],
            "training_code_sha256": job["code"]["sha256"],
            "training_code_s3_uri": job["code"]["s3_uri"],
            "framework": job["framework"],
            "epochs": job["epochs"],
            "gpu_workers": job["instance_count"],
            "instance_type": job["instance_type"],
            "managed_spot": True,
            "max_minutes": job["max_minutes"],
            "use_subset": job["use_subset"],
            "train_max_per_class": job["train_max_per_class"],
            "test_max_per_class": job["test_max_per_class"],
            "subset_seed": job["subset_seed"],
        })
        mlflow.log_metric("estimated_run_cost_usd", job["budget"]["estimated_run_cost_usd"])
        mlflow.log_metric("per_run_budget_usd", job["budget"]["per_run_budget_usd"])

        job["stage"] = "SageMaker training"
        start = time.time()
        estimator.fit(
            inputs={"training": TrainingInput(job["dataset"]["s3_uri"])},
            wait=True,
            logs=True,
        )
        elapsed = time.time() - start

        job["elapsed_seconds"] = round(elapsed, 2)
        job["output_path"] = output
        mlflow.log_metric("elapsed_seconds", job["elapsed_seconds"])
        mlflow.set_tag("sagemaker_output_path", output)
        mlflow.set_tag("reproducible", "true")

    job["stage"] = "Completed"
    log(job_id, f"Completed in {job['elapsed_seconds']} sec")
    log(job_id, f"MLflow run: {job['mlflow_run_id']}")
    log(job_id, f"Artifacts: {output}")
