import os, subprocess, threading, time, uuid, zipfile, shutil
from pathlib import Path
from typing import Optional

import boto3
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(os.getenv("PROJECT_ROOT", "/workspace"))
TF_DIR = ROOT / "infra" / "terraform"
UPLOADS = ROOT / ".local_runs"
RESULTS = ROOT / "results"
UPLOADS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

app = FastAPI(title="SageMaker Research Platform API", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}

ALLOWED_INSTANCE_COUNTS = {1, 2, 4}
ALLOWED_INSTANCE_TYPE = "ml.g4dn.xlarge"

# This is only a configurable planning estimate for the UI/backend budget guard.
# It is NOT the AWS bill and it is intentionally conservative.
DEFAULT_ESTIMATED_SPOT_USD_PER_INSTANCE_HOUR = float(
    os.getenv("ESTIMATED_SPOT_USD_PER_INSTANCE_HOUR", "0.55")
)

def run_cmd(cmd, cwd=None, env=None):
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError((p.stdout + "\n" + p.stderr).strip())
    return p.stdout.strip()

def log(job_id, msg):
    jobs[job_id]["logs"].append(str(msg))

def launch(job_id, fn, *args):
    def wrapper():
        try:
            jobs[job_id]["status"] = "running"
            fn(job_id, *args)
            if jobs[job_id]["status"] == "running":
                jobs[job_id]["status"] = "completed"
        except Exception as e:
            log(job_id, f"ERROR: {e}")
            jobs[job_id]["status"] = "failed"
    threading.Thread(target=wrapper, daemon=True).start()

def tf_output(name):
    return run_cmd(["terraform", "output", "-raw", name], cwd=TF_DIR)

def estimate_cost(instance_count: int, max_minutes: int):
    hours = max_minutes / 60.0
    return round(instance_count * hours * DEFAULT_ESTIMATED_SPOT_USD_PER_INSTANCE_HOUR, 4)

def validate_budget(instance_count: int, max_minutes: int, total_budget_usd: float, planned_runs: int):
    if instance_count not in ALLOWED_INSTANCE_COUNTS:
        raise HTTPException(400, "instance_count must be 1, 2 or 4")
    if max_minutes < 1 or max_minutes > 60:
        raise HTTPException(400, "max_minutes must be between 1 and 60")
    if total_budget_usd <= 0:
        raise HTTPException(400, "total_budget_usd must be > 0")
    if planned_runs < 1 or planned_runs > 100:
        raise HTTPException(400, "planned_runs must be between 1 and 100")

    per_run_budget = total_budget_usd / planned_runs
    estimated = estimate_cost(instance_count, max_minutes)
    return {
        "estimated_run_cost_usd": estimated,
        "per_run_budget_usd": round(per_run_budget, 4),
        "within_target": estimated <= per_run_budget,
    }

@app.get("/api/health")
def health():
    return {"ok": True, "project": "sagemaker-research-platform-budget"}

@app.get("/api/budget-estimate")
def budget_estimate(
    instance_count: int = 1,
    max_minutes: int = 5,
    total_budget_usd: float = 1.0,
    planned_runs: int = 5,
):
    return validate_budget(instance_count, max_minutes, total_budget_usd, planned_runs)

@app.get("/api/jobs")
def list_jobs():
    return list(jobs.values())[::-1]

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]

@app.post("/api/setup")
def setup():
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"id":job_id,"type":"setup","status":"queued","logs":[],"created_at":time.time()}
    launch(job_id, do_setup)
    return jobs[job_id]

def do_setup(job_id):
    log(job_id, "AWS identity:")
    log(job_id, run_cmd(["aws","sts","get-caller-identity"]))
    log(job_id, "Terraform init")
    log(job_id, run_cmd(["terraform","init","-input=false"], cwd=TF_DIR))
    log(job_id, "Terraform validate")
    log(job_id, run_cmd(["terraform","validate"], cwd=TF_DIR))
    log(job_id, "Creating S3 + IAM only (no GPU compute)")
    log(job_id, run_cmd(["terraform","apply","-auto-approve","-input=false"], cwd=TF_DIR))
    log(job_id, f"Bucket: {tf_output('artifact_bucket')}")
    log(job_id, f"Role: {tf_output('sagemaker_role_arn')}")

@app.post("/api/destroy")
def destroy():
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"id":job_id,"type":"destroy","status":"queued","logs":[],"created_at":time.time()}
    launch(job_id, do_destroy)
    return jobs[job_id]

def do_destroy(job_id):
    log(job_id, "Destroying S3/IAM resources...")
    log(job_id, run_cmd(["terraform","destroy","-auto-approve","-input=false"], cwd=TF_DIR))
    log(job_id, "Destroy complete.")

def local_image(framework):
    return {
        "pytorch": "pytorch/pytorch:2.2.2-cpu",
        "sklearn": "python:3.11-slim",
        "tensorflow": "tensorflow/tensorflow:2.15.0"
    }[framework]

def do_local(job_id, framework, source_dir, dataset_path, max_minutes):
    image = local_image(framework)
    run_dir = source_dir.parent
    model_dir = run_dir / "model"
    model_dir.mkdir(exist_ok=True)

    install = ""
    if (source_dir/"requirements.txt").exists():
        install = "pip install --no-cache-dir -r /job/requirements.txt && "

    cmd = [
        "docker","run","--rm",
        "-v",f"{source_dir}:/job",
        "-v",f"{model_dir}:/model",
        "-e","LOCAL_MODEL_DIR=/model",
        "-e","LOCAL_DATA_DIR=/data",
    ]
    if dataset_path:
        data_dir = run_dir/"data"
        data_dir.mkdir(exist_ok=True)
        shutil.copy2(dataset_path, data_dir/dataset_path.name)
        cmd += ["-v",f"{data_dir}:/data"]

    # Keep local and cloud entrypoint behavior aligned.
    cmd += [
        image, "bash", "-lc",
        f"timeout {max_minutes*60}s bash -lc '{install}python /job/train.py --epochs 1 --batch-size 64'"
    ]

    log(job_id, f"Local image: {image}")
    log(job_id, f"Local time limit: {max_minutes} minute(s)")
    start=time.time()
    out=run_cmd(cmd)
    elapsed=time.time()-start
    log(job_id, out)
    log(job_id, f"Local run completed in {elapsed:.1f}s")
    jobs[job_id]["elapsed_seconds"]=round(elapsed,1)

def framework_estimator(framework, source_dir, role, instance_count, output_path, max_minutes):
    from sagemaker.pytorch import PyTorch
    from sagemaker.tensorflow import TensorFlow
    from sagemaker.sklearn.estimator import SKLearn

    max_run_seconds = int(max_minutes * 60)
    # Allow extra wait time for Managed Spot capacity, but not unlimited waiting.
    max_wait_seconds = max_run_seconds + 300

    common = dict(
        entry_point="train.py",
        source_dir=str(source_dir),
        role=role,
        instance_type=ALLOWED_INSTANCE_TYPE,
        instance_count=instance_count,
        output_path=output_path,
        hyperparameters={
            "epochs": 1,
            "batch-size": 64,
        },
        use_spot_instances=True,
        max_run=max_run_seconds,
        max_wait=max_wait_seconds,
        disable_profiler=True,
    )

    if framework=="pytorch":
        return PyTorch(framework_version="2.2.0",py_version="py310",**common)
    if framework=="tensorflow":
        return TensorFlow(framework_version="2.15.0",py_version="py310",**common)
    if framework=="sklearn":
        return SKLearn(framework_version="1.2-1",py_version="py3",**common)
    raise ValueError("Unsupported framework")

def do_sagemaker(
    job_id, framework, source_dir, dataset_path,
    instance_count, max_minutes, total_budget_usd, planned_runs
):
    import sagemaker
    from sagemaker.inputs import TrainingInput

    budget = validate_budget(instance_count, max_minutes, total_budget_usd, planned_runs)
    jobs[job_id]["budget"] = budget

    bucket=tf_output("artifact_bucket")
    role=tf_output("sagemaker_role_arn")
    region=os.getenv("AWS_DEFAULT_REGION","eu-central-1")
    session=boto3.Session(region_name=region)
    sm_session=sagemaker.Session(boto_session=session, default_bucket=bucket)

    inputs=None
    if dataset_path:
        key=f"datasets/{job_id}/{dataset_path.name}"
        boto3.client("s3",region_name=region).upload_file(str(dataset_path),bucket,key)
        uri=f"s3://{bucket}/{key}"
        inputs={"training":TrainingInput(uri)}
        log(job_id,f"Dataset uploaded to {uri}")

    output=f"s3://{bucket}/outputs/{job_id}"
    est=framework_estimator(
        framework, source_dir, role, instance_count, output, max_minutes
    )
    est.sagemaker_session=sm_session

    log(job_id, f"Budget target: ${total_budget_usd:.2f} total / {planned_runs} runs")
    log(job_id, f"Per-run budget target: ${budget['per_run_budget_usd']:.2f}")
    log(job_id, f"Planning estimate for this run: ${budget['estimated_run_cost_usd']:.2f}")
    log(job_id, f"Managed Spot only; max training runtime: {max_minutes} minute(s)")
    log(job_id, f"Starting SageMaker: {instance_count} x {ALLOWED_INSTANCE_TYPE} ({framework})")

    start=time.time()
    est.fit(inputs=inputs,wait=True,logs=True)
    elapsed=time.time()-start

    jobs[job_id]["elapsed_seconds"]=round(elapsed,1)
    jobs[job_id]["output_path"]=output
    log(job_id,f"SageMaker completed in {elapsed:.1f}s")
    log(job_id,f"Artifacts: {output}")

async def save_uploads(training_zip, dataset, job_id):
    job_dir=UPLOADS/job_id
    source=job_dir/"source"
    source.mkdir(parents=True)
    zpath=job_dir/"training.zip"
    zpath.write_bytes(await training_zip.read())

    with zipfile.ZipFile(zpath) as z:
        z.extractall(source)

    if not (source/"train.py").exists():
        dirs=[p for p in source.iterdir() if p.is_dir()]
        if len(dirs)==1 and (dirs[0]/"train.py").exists():
            source=dirs[0]

    if not (source/"train.py").exists():
        raise HTTPException(400,"ZIP must contain train.py")

    dataset_path=None
    if dataset:
        dataset_path=job_dir/dataset.filename
        dataset_path.write_bytes(await dataset.read())

    return source,dataset_path

@app.post("/api/run-local")
async def run_local(
    framework:str=Form(...),
    max_minutes:int=Form(5),
    training_zip:UploadFile=File(...),
    dataset:Optional[UploadFile]=File(None),
):
    if framework not in ["pytorch","sklearn","tensorflow"]:
        raise HTTPException(400,"Unsupported framework")
    if max_minutes < 1 or max_minutes > 60:
        raise HTTPException(400,"max_minutes must be between 1 and 60")

    job_id=str(uuid.uuid4())
    source,dataset_path=await save_uploads(training_zip,dataset,job_id)
    jobs[job_id]={
        "id":job_id,"type":"local","name":training_zip.filename,
        "status":"queued","logs":[],"created_at":time.time()
    }
    launch(job_id,do_local,framework,source,dataset_path,max_minutes)
    return jobs[job_id]

@app.post("/api/run-sagemaker")
async def run_sagemaker(
    framework:str=Form(...),
    instance_count:int=Form(1),
    max_minutes:int=Form(5),
    total_budget_usd:float=Form(1.0),
    planned_runs:int=Form(5),
    training_zip:UploadFile=File(...),
    dataset:Optional[UploadFile]=File(None),
):
    if framework not in ["pytorch","sklearn","tensorflow"]:
        raise HTTPException(400,"Unsupported framework")

    budget = validate_budget(instance_count, max_minutes, total_budget_usd, planned_runs)
    if not budget["within_target"]:
        raise HTTPException(
            400,
            f"Estimated run cost ${budget['estimated_run_cost_usd']:.2f} "
            f"exceeds per-run target ${budget['per_run_budget_usd']:.2f}. "
            "Increase total budget, reduce planned runs, reduce runtime, or use fewer GPUs."
        )

    job_id=str(uuid.uuid4())
    source,dataset_path=await save_uploads(training_zip,dataset,job_id)
    jobs[job_id]={
        "id":job_id,"type":"sagemaker","name":training_zip.filename,
        "status":"queued","logs":[],"created_at":time.time(),
        "budget":budget
    }
    launch(
        job_id,do_sagemaker,framework,source,dataset_path,
        instance_count,max_minutes,total_budget_usd,planned_runs
    )
    return jobs[job_id]

@app.post("/api/stop-sagemaker/{training_job_name}")
def stop_sagemaker(training_job_name: str):
    region=os.getenv("AWS_DEFAULT_REGION","eu-central-1")
    boto3.client("sagemaker",region_name=region).stop_training_job(
        TrainingJobName=training_job_name
    )
    return {"ok": True, "training_job_name": training_job_name}
