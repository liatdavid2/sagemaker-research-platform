# SageMaker Research Platform — Budget Edition

A local-first research training platform with optional SageMaker Managed Spot GPU execution and GitHub Actions CI/CD.

## Default budget target

The UI starts with:

```text
Total budget target: $1
Planned runs:        5
Max runtime/run:     5 minutes
Per-run target:      $0.20
```

All three parameters are editable.

The platform uses a conservative **planning estimate** to decide whether the selected GPU count fits the configured per-run target. This estimate is not an AWS billing guarantee because Managed Spot pricing, availability and startup behavior can vary.

## Cost guardrails

- Managed Spot training only.
- Instance type fixed to `ml.g4dn.xlarge`.
- Allowed instance counts: 1, 2, 4.
- Default maximum training time: 5 minutes.
- Maximum runtime is configurable in the UI.
- SageMaker `max_run` is set from the chosen runtime.
- `max_wait` is bounded to `max_run + 5 minutes`.
- Backend rejects a SageMaker request when its planning estimate exceeds:
  `total_budget / planned_runs`.
- Local run is always available first and creates no AWS training compute.

## Architecture

```text
Researcher ZIP + optional dataset
              |
              v
          React UI
              |
              v
          FastAPI
          /     \
         /       \
 Local Docker   SageMaker Managed Spot
 (free test)    (temporary GPU only)
                    |
                    v
                S3 artifacts
```

## Real datasets only

Included examples use real public datasets:

- PyTorch Vision: CIFAR-10, automatically downloaded; only a small real subset is used for the short demo.
- scikit-learn: Wisconsin Diagnostic Breast Cancer.
- TensorFlow: Wisconsin Diagnostic Breast Cancer.
- `sample_data/wisconsin_breast_cancer.csv` is a real example CSV exported from scikit-learn.

No synthetic-data fallback is used.

## Researcher ZIP format

```text
research_job.zip
├── train.py             required
├── requirements.txt     optional
├── config.yaml          optional
└── README.md            optional
```

Local variables:
- `LOCAL_DATA_DIR`
- `LOCAL_MODEL_DIR`

SageMaker variables:
- `SM_CHANNEL_TRAINING`
- `SM_MODEL_DIR`

## Ready-made researcher ZIPs

```text
researcher_job_zips/
├── pytorch_vision_job.zip
├── sklearn_tabular_job.zip
└── tensorflow_tabular_job.zip
```

## Local start

Create `.env` from `.env.example`, then:

```cmd
docker compose up --build
```

Open:

```text
http://localhost:7475
```

Backend Swagger:

```text
http://localhost:7474/docs
```

## Recommended workflow

1. Choose budget / planned runs / max runtime.
2. Upload a researcher ZIP.
3. Click **Run Locally First**.
4. Verify the code and dataset.
5. Click **Setup AWS Resources** once.
6. Choose 1 / 2 / 4 GPUs.
7. Run on SageMaker only if the UI says the selection is within the budget target.
8. Click **Destroy AWS Resources** when finished.

Setup creates only S3 + IAM. No GPU is kept alive.

## GitHub Actions

- `ci.yml`: Python syntax, Terraform fmt/validate, React build, Docker Compose config.
- `docker-build.yml`: Docker image build.
- `terraform-plan.yml`: safe Terraform static checks; production organizations can add GitHub OIDC.
- `manual-sagemaker-note.yml`: template for a manually triggered cloud training workflow without repository AWS keys.

## Security / production note

For GitHub-to-AWS automation, prefer GitHub OIDC + an AWS IAM role instead of long-lived AWS access keys in GitHub secrets.

## Quota note

SageMaker quotas are separate from EC2 Spot quotas. Before cloud execution, verify that the account has non-zero quota for `ml.g4dn.xlarge` training or Spot training in the selected region.
