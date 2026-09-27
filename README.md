# SageMaker Research Platform — Reproducible ML + MLflow

This version is structured like an internal ML platform rather than a file-upload demo.

## Researcher flow

```text
Dataset:      CIFAR-10-v3
Code version: git commit 8ac72f
GPU:          g4dn.xlarge
Epochs:       5

[ Run Training ]
```

Flow:

```text
React UI
   ↓
Dataset Registry in S3
   ↓
Select dataset version + training parameters
   ↓
Validate committed Git version + budget
   ↓
Start SageMaker Managed Spot Training Job
   ↓
Automatically track the run in MLflow
```

## Local services

```text
Frontend  http://localhost:7475
Backend   http://localhost:7474
MLflow    http://localhost:5050
```

Prometheus and Grafana are removed. Loki is also removed in this focused version.

## Reproducibility

Every cloud run records:

- dataset logical name and version
- dataset SHA256 content hash
- exact S3 URI
- exact Git commit
- framework
- epochs
- 1 / 2 / 4 GPU workers
- fixed instance type `ml.g4dn.xlarge`
- Managed Spot
- maximum runtime
- budget estimate
- SageMaker output path
- MLflow run ID

The UI blocks training when the Git working tree has uncommitted changes.

## Dataset Registry

Researchers upload a dataset once with:

```text
Name: CIFAR-10
Version: v3
File: ...
```

The backend stores it in S3 and adds it to a versioned registry manifest.
Subsequent experiments select that dataset version instead of choosing arbitrary files.

## MLflow

MLflow is automatic for SageMaker runs. A run is created when cloud training starts, so the configuration is captured even before training completes.

Tracked values include dataset ID/hash, Git commit, framework, epochs, GPU workers, instance type, runtime, budget estimate, and S3 output path.

## Budget defaults

```text
Total target:    $1
Planned runs:    5
Max runtime/run: 5 minutes
Per-run target:  $0.20
```

These are editable in the UI.

## Run

```cmd
docker compose up --build
```

Then open:

```text
http://localhost:7475
```


## Public Dataset Catalog

The Dataset Registry now has a preferred no-file-picker flow:

```text
Public Dataset Catalog
Dataset: CIFAR-10
Registry version: v3

[ Download & Register to S3 ]
```

The backend downloads the official CIFAR-10 source archive once, computes SHA256, uploads that exact archive to the project S3 bucket, and registers its metadata.

Each later training run selects the registered S3 version and records its SHA256 and source URL in MLflow.

Manual file upload is still available under **Advanced** for private datasets.
