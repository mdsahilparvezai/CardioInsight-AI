# MLOps Integration Guide — CardioInsight AI

> **Status**: Optional integrations — the core app runs without these.

---

## MLflow Experiment Tracking

### Enable MLflow

Set environment variables:
```bash
export CARDIOINSIGHT_MLFLOW_ENABLED=true
export CARDIOINSIGHT_MLFLOW_TRACKING_URI=file://./mlruns
export CARDIOINSIGHT_MLFLOW_EXPERIMENT=cardioinsight-heart-disease
```

Or in `.env`:
```
CARDIOINSIGHT_MLFLOW_ENABLED=true
CARDIOINSIGHT_MLFLOW_TRACKING_URI=file://./mlruns
CARDIOINSIGHT_MLFLOW_EXPERIMENT=cardioinsight-heart-disease
```

### What Gets Logged

When enabled, `train.py` automatically logs to MLflow:

| Category | Artifacts |
|----------|-----------|
| **Parameters** | All config values, model hyperparameters, engine, seed |
| **Metrics** | Train/val/test: accuracy, precision, recall, specificity, F1, ROC-AUC |
| **Artifacts** | Model pickle, preprocessor, metadata JSON, comparison.csv |
| **Tags** | `model_type`, `engine`, `git_commit` (if available) |

### View Experiments

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file://./mlruns
# Open http://localhost:5000
```

### Remote Tracking Server

For team collaboration, use a remote MLflow server:
```bash
export CARDIOINSIGHT_MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
# Requires MLFLOW_TRACKING_USERNAME/PASSWORD or token auth
```

---

## DVC Data & Pipeline Versioning

### Initialize DVC

```bash
# Install DVC
pip install dvc

# Initialize in project root
dvc init

# Configure remote storage (e.g., S3, GCS, Azure, SSH)
dvc remote add -d myremote s3://my-bucket/cardioinsight-dvc
```

### Track Dataset

```bash
# If using real UCI data
dvc add data/heart_disease.csv
git add data/heart_disease.csv.dvc .gitignore
git commit -m "Track heart disease dataset with DVC"
```

### Define Pipeline (`dvc.yaml`)

```yaml
stages:
  prepare_data:
    cmd: python -c "from src.data import generate_synthetic_dataset; generate_synthetic_dataset().to_csv('data/synthetic.csv', index=False)"
    outs:
      - data/synthetic.csv

  train:
    cmd: python train.py --engine sklearn
    deps:
      - src/
      - data/synthetic.csv
      - configs/config.yaml
    params:
      - model.engine
      - model.cv_folds
      - data.test_size
    outs:
      - models/comparison.csv
      - models/model_*.pkl

  evaluate:
    cmd: python -c "
import pandas as pd
from src.models.io import load_artifacts, latest_model_path
path = latest_model_path()
artifact = load_artifacts(path)
print('Model:', artifact['metadata']['model_type'])
print('Test F1:', artifact['metadata']['test_metrics']['f1'])
print('Test AUC:', artifact['metadata']['test_metrics']['roc_auc'])
"
    deps:
      - models/
    metrics:
      - models/comparison.csv:
          xpath: "model_name,split,f1,roc_auc"
```

### Run Pipeline

```bash
# Run full pipeline
dvc repro

# Run specific stage
dvc repro train

# View metrics
dvc metrics show
dvc metrics diff
```

### Push/Pull Data

```bash
# Push data and models to remote
dvc push

# Pull on another machine
dvc pull
```

---

## GitHub Actions Integration

The CI workflow (`.github/workflows/ci.yml`) includes:

1. **Test matrix** on Python 3.9–3.12
2. **Linting** with ruff + black
3. **Training smoke test** (NumPy engine, quick mode)
4. **Docker build** on main branch pushes
5. **Security scans** (pip-audit, bandit)

### Enable MLflow in CI

Add repository secrets:
- `MLFLOW_TRACKING_URI` — your tracking server URL
- `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` — if auth required

Then uncomment the `mlops` job in `ci.yml`.

### Enable DVC in CI

```yaml
- name: Setup DVC
  uses: iterative/setup-dvc@v2

- name: Pull DVC data
  run: dvc pull
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

---

## Model Registry (Advanced)

For production-grade model management, consider:

1. **MLflow Model Registry** — register best model, stage transitions (Staging → Production)
2. **DVC Model Registry** — `dvc model` commands for versioning
3. **Custom registry** — simple JSON index in `models/registry.json`

Example registration in `train.py` (after saving artifact):
```python
if cfg.get("mlflow", {}).get("enabled"):
    import mlflow
    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model",
        "CardioInsightHeartDisease"
    )
```

---

## Monitoring & Drift Detection (Future)

For a production system, add:

| Concern | Tool | Integration Point |
|---------|------|-------------------|
| **Data drift** | Evidently AI / WhyLogs | Compare live input distributions to training |
| **Concept drift** | Monitor target distribution shift | Scheduled job on prediction logs |
| **Performance monitoring** | MLflow / Prometheus | Log predictions + outcomes (when available) |
| **Alerting** | PagerDuty / Slack | Threshold on AUC drop, data quality failures |

---

## Summary: Minimal vs Full Stack

| Feature | Minimal (Default) | Full MLOps |
|---------|-------------------|------------|
| Training | `python train.py` | `dvc repro` + MLflow |
| Tracking | Local CSV/JSON | MLflow server |
| Data versioning | Git (small) / none | DVC + remote storage |
| Model registry | File-based (`latest_model_path`) | MLflow/DVC registry |
| CI/CD | GitHub Actions (test + build) | + MLflow/DVC steps |
| Monitoring | None | Evidently + alerts |

**Start minimal.** Add MLflow/DVC when you need experiment comparison across runs, team collaboration, or audit trails.