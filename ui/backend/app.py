import hashlib
import json
import os
import subprocess
import tarfile
import threading
import time
import urllib.request
import uuid
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
CACHE_DIR.mkdir(exist_ok=True)

app = FastAPI(title="SageMaker Reproducible Research Platform", version="6.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}
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
        "recommended_version": "v1",
    }
}


def run_cmd(args, cwd=None):
    p = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError((p.stdout + "\n" + p.stderr).strip())
    return p.stdout.strip()


def terraform_output(name):
    return run_cmd(["terraform", "output", "-raw", name], cwd=TF_DIR)


def current_git_commit():
    try:
        return run_cmd(["git", "rev-parse", "HEAD"], cwd=ROOT)
    except Exception:
        return os.getenv("CODE_VERSION", "unknown")


def git_dirty():
    try:
        return bool(run_cmd(["git", "status", "--porcelain"], cwd=ROOT))
    except Exception:
        return False


def aws_region():
    return os.getenv("AWS_DEFAULT_REGION", "eu-central-1")


def s3_client():
    return boto3.client("s3", region_name=aws_region())


def registry_key():
    return "dataset-registry/manifest.json"


def load_registry():
    try:
        bucket = terraform_output("artifact_bucket")
        obj = s3_client().get_object(Bucket=bucket, Key=registry_key())
        return json.loads(obj["Body"].read())
    except Exception:
        return {"datasets": []}


def save_registry(registry):
    bucket = terraform_output("artifact_bucket")
    s3_client().put_object(
        Bucket=bucket,
        Key=registry_key(),
        Body=json.dumps(registry, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def upsert_dataset(item):
    registry = load_registry()
    registry["datasets"] = [d for d in registry["datasets"] if d["id"] != item["id"]]
    registry["datasets"].append(item)
    save_registry(registry)
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
    return {"ok": True, "project": "sagemaker-research-platform-reproducible-catalog"}


@app.get("/api/catalog")
def public_catalog():
    return {"datasets": list(PUBLIC_DATASETS.values())}


@app.get("/api/code-version")
def code_version():
    commit = current_git_commit()
    dirty = git_dirty()
    return {
        "git_commit": commit,
        "dirty": dirty,
        "reproducible": commit != "unknown" and not dirty,
    }


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
    return load_registry()


@app.post("/api/datasets/register-public")
def register_public_dataset(
    catalog_key: str = Form(...),
    version: str = Form(...),
):
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
    }
    launch(job_id, do_register_public_dataset, preset, version)
    return jobs[job_id]


def do_register_public_dataset(job_id, preset, version):
    bucket = terraform_output("artifact_bucket")
    safe_name = preset["name"].lower().replace(" ", "-")
    safe_version = version.strip().replace(" ", "-")
    local_path = CACHE_DIR / preset["filename"]

    log(job_id, f"Downloading {preset['name']} from official public source")
    log(job_id, preset["source_url"])
    urllib.request.urlretrieve(preset["source_url"], local_path)

    sha256 = hashlib.sha256(local_path.read_bytes()).hexdigest()
    key = f"datasets/{safe_name}/{safe_version}/{preset['filename']}"

    log(job_id, f"Uploading immutable source archive to s3://{bucket}/{key}")
    s3_client().upload_file(str(local_path), bucket, key)

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
    log(job_id, f"Registered {item['name']}:{item['version']}")
    log(job_id, f"SHA256: {sha256}")


@app.post("/api/datasets/upload")
async def upload_dataset(
    name: str = Form(...),
    version: str = Form(...),
    dataset: UploadFile = File(...),
):
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


@app.get("/api/jobs")
def list_jobs():
    return list(jobs.values())[::-1]


@app.post("/api/setup")
def setup():
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"id": job_id, "type": "setup", "status": "queued", "logs": []}
    launch(job_id, do_setup)
    return jobs[job_id]


def do_setup(job_id):
    log(job_id, run_cmd(["aws", "sts", "get-caller-identity"]))
    log(job_id, run_cmd(["terraform", "init", "-input=false"], cwd=TF_DIR))
    log(job_id, run_cmd(["terraform", "validate"], cwd=TF_DIR))
    log(job_id, run_cmd(["terraform", "apply", "-auto-approve", "-input=false"], cwd=TF_DIR))
    log(job_id, f"S3 bucket: {terraform_output('artifact_bucket')}")
    log(job_id, f"SageMaker role: {terraform_output('sagemaker_role_arn')}")


@app.post("/api/destroy")
def destroy():
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"id": job_id, "type": "destroy", "status": "queued", "logs": []}
    launch(job_id, do_destroy)
    return jobs[job_id]


def do_destroy(job_id):
    log(job_id, run_cmd(["terraform", "destroy", "-auto-approve", "-input=false"], cwd=TF_DIR))


def build_estimator(framework, source_dir, role, count, output, max_minutes, epochs, commit):
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
            "git-commit": commit,
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
    epochs: int = Form(1),
    instance_count: int = Form(1),
    max_minutes: int = Form(5),
    total_budget_usd: float = Form(1.0),
    planned_runs: int = Form(5),
):
    if framework not in ALLOWED_FRAMEWORKS:
        raise HTTPException(400, "Framework must be PyTorch or TensorFlow")
    if not 1 <= epochs <= 50:
        raise HTTPException(400, "epochs must be between 1 and 50")
    if git_dirty():
        raise HTTPException(400, "Commit current code changes before starting a reproducible run")

    budget = estimate_budget(instance_count, max_minutes, total_budget_usd, planned_runs)
    if not budget["within_target"]:
        raise HTTPException(400, "Current configuration exceeds the per-run budget target")

    dataset = next((d for d in load_registry()["datasets"] if d["id"] == dataset_id), None)
    if not dataset:
        raise HTTPException(404, "Dataset version not found")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "type": "sagemaker",
        "framework": framework,
        "dataset": dataset,
        "epochs": epochs,
        "instance_count": instance_count,
        "instance_type": INSTANCE_TYPE,
        "max_minutes": max_minutes,
        "git_commit": current_git_commit(),
        "git_dirty": False,
        "budget": budget,
        "status": "queued",
        "logs": [],
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

    source_dir = ROOT / "examples" / (
        "pytorch_vision" if job["framework"] == "pytorch" else "tensorflow_vision"
    )
    output = f"s3://{bucket}/outputs/{job_id}"

    estimator = build_estimator(
        job["framework"], source_dir, role, job["instance_count"], output,
        job["max_minutes"], job["epochs"], job["git_commit"]
    )
    estimator.sagemaker_session = sm_session

    log(job_id, f"Dataset: {job['dataset']['name']}:{job['dataset']['version']}")
    log(job_id, f"Dataset SHA256: {job['dataset']['sha256']}")
    log(job_id, f"Dataset source: {job['dataset'].get('source_url') or 'manual upload'}")
    log(job_id, f"Code commit: {job['git_commit']}")
    log(job_id, f"GPU: {job['instance_count']} x {INSTANCE_TYPE}")
    log(job_id, f"Epochs: {job['epochs']}")
    log(job_id, "Managed Spot: true")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("sagemaker-research-platform")

    with mlflow.start_run(run_name=f"{job['framework']}-{job_id[:8]}") as active:
        job["mlflow_run_id"] = active.info.run_id
        mlflow.log_params({
            "dataset_id": job["dataset"]["id"],
            "dataset_sha256": job["dataset"]["sha256"],
            "dataset_s3_uri": job["dataset"]["s3_uri"],
            "dataset_source_type": job["dataset"].get("source_type", "unknown"),
            "dataset_source_url": job["dataset"].get("source_url") or "",
            "git_commit": job["git_commit"],
            "framework": job["framework"],
            "epochs": job["epochs"],
            "gpu_workers": job["instance_count"],
            "instance_type": job["instance_type"],
            "managed_spot": True,
            "max_minutes": job["max_minutes"],
        })
        mlflow.log_metric("estimated_run_cost_usd", job["budget"]["estimated_run_cost_usd"])
        mlflow.log_metric("per_run_budget_usd", job["budget"]["per_run_budget_usd"])

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

    log(job_id, f"Completed in {job['elapsed_seconds']} sec")
    log(job_id, f"MLflow run: {job['mlflow_run_id']}")
    log(job_id, f"Artifacts: {output}")
