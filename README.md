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


## Live UI console

The UI now exposes backend activity directly in the browser:

- Setup Console streams AWS CLI and Terraform output line-by-line.
- Dataset Console shows S3 lookup, cache/download, SHA256, upload and registration stages.
- Activity Console shows the latest job and automatically refreshes with the existing polling loop.
- Dataset registration shows `Starting...` immediately after the button is clicked.

This avoids having to open Docker logs just to understand what the platform is doing.


## Existing AWS resources mode

This version does not provision or destroy AWS infrastructure from the UI.

Existing S3 bucket:
`sagemaker-research-platform-2fa2b53f`

Existing SageMaker role name:
`sagemaker-research-platform-2fa2b53f-execution`

Dataset Registry and Training Code Registry upload directly to the existing S3 bucket.
The UI only verifies access. Training uses the existing SageMaker IAM role.


## Repair: Dataset catalog + MLflow visibility

This build restores the read-only `/api/catalog` and `/api/jobs` endpoints used by
the React UI. CIFAR-10 therefore appears in the Dataset dropdown again.

MLflow now has a Docker health check and `restart: unless-stopped`. The Platform
card also shows whether MLflow is reachable at `http://localhost:5050`.


## MLflow / SQLAlchemy compatibility fix

MLflow is pinned to `2.16.2` and SQLAlchemy is pinned to `2.0.36`.
This avoids the SQLAlchemy 2.1 removal of `FallbackAsyncAdaptedQueuePool`,
which prevents MLflow 2.16.2 from starting with its SQLite backend.


## Dataset Registry helper fix

Added the missing `registry_find_dataset()` and `s3_object_exists()` helpers used by
`Download & Register to S3`.

The flow now correctly:

1. checks the registry manifest,
2. verifies whether the exact S3 object already exists,
3. skips re-download/re-upload when the same dataset version is already present,
4. otherwise continues with cache/download -> SHA256 -> S3 upload -> registration.
