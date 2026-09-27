# SageMaker Research Platform — Full UI Reproducible Workflow

The complete researcher workflow is now handled in the React UI.

## UI flow

```text
1. Setup AWS Resources
2. Dataset Registry
   - Download CIFAR-10 from trusted public source
   - Live progress beside "Download & Register to S3"
   - Store exact archive + SHA256 in S3
3. Training Code Registry
   - Researcher uploads training-code.zip
   - Must contain train.py
   - Store exact ZIP + SHA256 in S3
4. Training Configuration
   - Select dataset version
   - Select training code version
   - Framework
   - Epochs
   - 1 / 2 / 4 GPU workers
5. Budget Guardrail
6. Run Training
   - SageMaker Managed Spot
   - MLflow automatic
```

There is no Run History section in this version.

## Local services

```text
Frontend  http://localhost:7475
Backend   http://localhost:7474
MLflow    http://localhost:5050
```

## Reproducibility

Each training run is tied to:

- dataset ID/version
- dataset SHA256
- dataset S3 URI
- training code ID/version
- training code SHA256
- training code S3 URI
- framework
- epochs
- GPU worker count
- fixed instance type
- runtime guardrail
- budget estimate
- SageMaker output path
- MLflow run ID

## Training code ZIP

```text
training-code.zip
├── train.py             required
├── requirements.txt     optional
├── config.yaml          optional
└── README.md            optional
```

The backend validates `train.py` before registering the code.

## Start

```cmd
docker compose up --build
```

Open:

```text
http://localhost:7475
```


## Balanced subset controls

The Training Configuration UI now supports an optional deterministic balanced subset:

```text
Use balanced subset: ON
Train max / class: 200
Test max / class: 50
Subset seed: 42
```

For CIFAR-10 this means approximately:

```text
2,000 training images
500 test images
```

The subset parameters are sent to SageMaker and recorded automatically in MLflow, so the run remains reproducible.

Researcher training code should accept these optional CLI parameters:

```text
--use-subset
--train-max-per-class
--test-max-per-class
--subset-seed
```


## Faster dataset registration

Public dataset registration now uses this order:

```text
1. Check S3 registry/object
   -> already exists: skip download and upload

2. Check local .dataset_cache
   -> cache hit: skip internet download

3. HTTP capability probe
   -> Range requests supported: 4 parallel byte-range downloads
   -> otherwise: standard single-stream download

4. SHA256
5. Upload once to S3
6. Register immutable version
```

The UI shows:

- download mode
- progress %
- downloaded MB / total MB
- MB/s
- ETA
- S3 cache hit
- local cache hit


## Platform readiness in the UI

`Setup AWS Resources` now reports explicit stages and progress:

```text
Checking AWS credentials
Terraform init
Terraform validate
Creating S3 + IAM
Verifying resources
Ready
```

On success the UI shows:

```text
AWS Resources Ready ✓
S3 bucket: ...
SageMaker role: ...
```

On failure the UI shows the latest setup error.

AWS-dependent actions are disabled until setup is ready:

- Download & Register to S3
- Upload & Register Training Code
- Run Training


## Persistent AWS setup state

The Platform status no longer depends on an in-memory setup job.

On each refresh the backend recovers readiness from the Terraform outputs and
verifies the S3 bucket and SageMaker IAM role still exist in AWS. The result is
cached briefly to avoid repeatedly invoking Terraform on every UI poll.

Therefore `AWS Resources Ready` survives:

- browser refresh
- reopening the UI
- backend restart
- `docker compose down` / `docker compose up`

As long as the Terraform state and AWS resources still exist.
