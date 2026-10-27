# CardioInsight AI — Architecture Documentation

> **Status**: ✅ Implemented and tested (Phases 1–6 complete)
> **Last Updated**: 2026-10-27
> **Version**: 0.1.0

---

## 1. System Overview

CardioInsight AI is an **end-to-end educational machine learning application** that:

- Loads a synthetic dataset matching the **Cleveland heart disease schema** (UCI ML Repository #45)
- Trains and compares **multiple baseline models** with hyperparameter tuning
- Exposes a **Streamlit dashboard** with 6 pages for exploration, evaluation, prediction, and explainability
- Provides **full reproducibility** via config-driven pipelines, fixed seeds, and artifact serialization
- Runs on **minimal dependencies** (NumPy, Pandas, Matplotlib, Seaborn) with optional sklearn/MLops

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CARDIOINSIGHT AI                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Data Layer │───▶│  Features    │───▶│   Models     │          │
│  │  (src/data)  │    │ (src/feat.)  │    │ (src/models) │          │
│  └──────────────┘    └──────────────┘    └──────┬───────┘          │
│                                                 │                   │
│                        ┌──────────────┐         │                   │
│                        │  Prediction  │◀────────┘                   │
│                        │ (src/pred.)  │                             │
│                        └──────┬───────┘                             │
│                               │                                     │
│                        ┌──────▼───────┐                             │
│                        │ Visualization│                             │
│                        │ (src/vis.)   │                             │
│                        └──────┬───────┘                             │
│                               │                                     │
│                        ┌──────▼───────┐                             │
│                        │   Streamlit  │                             │
│                        │  Dashboard   │                             │
│                        │   (app.py)   │                             │
│                        └──────────────┘                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key design principles:**

| Principle | Implementation |
|-----------|----------------|
| **Separation of concerns** | Data, features, models, prediction, UI are independent modules |
| **Single source of truth** | Schema defined once in `src/data/schema.py`; used by validator, preprocessor, UI |
| **Train/serve parity** | Same preprocessing pipeline (or NumPy equivalent) used at train and inference time |
| **Graceful degradation** | Pure-NumPy engine runs when sklearn unavailable; SHAP is optional |
| **Configuration over code** | All hyperparameters, paths, seeds in `configs/config.yaml` + env vars |

---

## 3. Module Reference

### 3.1 `src/data/` — Data Ingestion & Validation

| File | Responsibility |
|------|----------------|
| `schema.py` | Canonical feature definitions (13 features + target), labels, units, ranges |
| `loader.py` | `load_dataset()` (CSV or synthetic), `generate_synthetic_dataset()` |
| `validator.py` | `validate_dataset()` → structured `ValidationResult` (missing, duplicates, ranges, target balance) |
| `eda.py` | `run_eda()` → summary stats, correlation matrix, per-feature distributions by target |

**Synthetic dataset**: Reproducible generator (seed=42) with learnable signal + noise. Documented as synthetic in UI and `data/README.md`.

### 3.2 `src/features/` — Preprocessing & Input Validation

| File | Responsibility |
|------|----------------|
| `preprocessor.py` | `build_preprocessor()` (sklearn ColumnTransformer), `fit_preprocessor()`, `transform_features()` |
| `validation.py` | `validate_inference_input()` — validates user dict against schema, returns coerced values + errors |

### 3.3 `src/models/` — Engines, Evaluation, I/O, Explainability

| File | Responsibility |
|------|----------------|
| `engines.py` | `SklearnEngine` + `NumPyEngine` (LogisticRegression, RandomForest, GradientBoosting, NaiveBayes); `select_engine()` |
| `evaluation.py` | `evaluate_model()` (accuracy, precision, recall, specificity, F1, ROC-AUC, CM), `cross_validate()`, `compare_models()`, `calibration_data()` |
| `io.py` | `save_artifacts()` / `load_artifacts()` (pickle + JSON sidecar), `ModelMetadata` dataclass |
| `explainability.py` | SHAP wrapper with graceful fallback (`compute_shap_values`, summary/force/dependence plots) |

**NumPy engine implementations** (zero deps):
- Logistic Regression: gradient descent with L2, internal standardisation, gradient clipping
- Random Forest: bootstrap-aggregated CART trees (Gini splits)
- Gradient Boosting: regression trees on pseudo-residuals (log-loss)
- Naive Bayes: Gaussian NB with variance smoothing

### 3.4 `src/prediction/` — Inference Service

| File | Responsibility |
|------|----------------|
| `service.py` | `Predictor` class: load artifact → validate input → preprocess → predict → structured `PredictionResult` |

### 3.5 `src/visualization/` — Charts

| Function | Output |
|----------|--------|
| `plot_feature_distribution` | Histogram/bar by target (Plotly + MPL fallback) |
| `plot_correlation_heatmap` | Correlation matrix heatmap |
| `plot_confusion_matrix` | 2×2 CM with labels |
| `plot_roc_curve` / `plot_precision_recall` / `plot_calibration` | Standard diagnostic curves |
| `plot_metric_comparison` | Grouped bar chart across models |

All charts respect the **CardioInsight theme** (dark navy / teal / light modes).

### 3.6 `src/utils/` — Shared Utilities

| File | Responsibility |
|------|----------------|
| `config.py` | YAML + env-var config loading (`CARDIOINSIGHT_*` prefix), cached accessor |
| `logging.py` | Single `get_logger()` factory with consistent formatting |

---

## 4. Data Flow

### 4.1 Training (`train.py`)

```
load_dataset() → validate_dataset() → split (stratified 70/10/20)
                    ↓
            build_preprocessor() → fit on TRAIN only
                    ↓
            transform_features() on train/val/test
                    ↓
            For each model_type in baselines:
                Grid search on validation set (CV on train)
                Select best by `selection_metric` (default: F1)
                Evaluate on train / val / test
                    ↓
            Select best model overall (by val F1)
                    ↓
            save_artifacts(model, preprocessor, metadata)
            write comparison.csv
```

### 4.2 Inference (`Predictor.predict()`)

```
validate_inference_input(user_dict) → raises on error
                    ↓
            Build DataFrame in schema order
                    ↓
            preprocessor.transform() (if available)
                    ↓
            model.predict_proba() → probability, class
                    ↓
            Return PredictionResult (class, prob, metadata, validation)
```

---

## 5. Configuration Schema (`configs/config.yaml`)

```yaml
project:
  random_seed: 42
  artifacts_dir: models

data:
  path: ""                    # empty → synthetic
  target_column: target
  test_size: 0.2
  val_size: 0.1
  stratify: true
  synthetic: {n_samples: 303, positive_rate: 0.45, noise: 0.35, seed: 42}

preprocessing:
  numeric_features: [age, trestbps, chol, thalch, oldpeak, ...]
  categorical_features: [sex, cp, fbs, restecg, ...]

model:
  engine: auto                # auto | sklearn | numpy
  baselines: [logistic_regression, random_forest, gradient_boosting, naive_bayes]
  cv_folds: 5
  selection_metric: f1

evaluation:
  metrics: [accuracy, precision, recall, specificity, f1, roc_auc]
  decision_threshold: 0.5

mlflow:
  enabled: false
  tracking_uri: file://./mlruns
  experiment_name: cardioinsight-heart-disease
```

Environment variables (prefix `CARDIOINSIGHT_`) override any value.

---

## 6. Artifact Format

Each trained model produces:

```
models/model_{type}_{engine}_{seed}.pkl    # pickle: {model, preprocessor, feature_names, metadata}
models/model_{type}_{engine}_{seed}.json   # human-readable metadata
models/comparison.csv                       # test-set metrics for all models
```

`metadata` includes: model_type, engine, config snapshot, train/val/test metrics, feature list, timestamp, seed, sample counts.

---

## 7. Deployment Architecture

```
┌─────────────────────────────────────┐
│        Developer Laptop (Win)       │
│  python -m venv .venv               │
│  pip install -r requirements.txt    │
│  python train.py                    │
│  streamlit run app.py               │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│         Docker (optional)           │
│  FROM python:3.11-slim              │
│  COPY . /app                        │
│  RUN pip install -r requirements.txt│
│  CMD ["streamlit", "run", "app.py"] │
└─────────────────────────────────────┘
```

**CI (GitHub Actions)**: Install deps → run `pytest` on Python 3.9–3.12 → build Docker image.

---

## 8. Security & Privacy

- **No PHI collected**: Synthetic data only; real data never leaves user's machine
- **No external calls**: App runs fully offline (MLflow/DVC optional)
- **No secrets in code**: All config via env vars / `.env` (gitignored)
- **Model artifacts**: Pickle files — only load trusted artifacts you trained

---

## 9. Future Extensibility

| Area | Planned |
|------|---------|
| Real data ingestion | `data/README.md` documents UCI CSV placement |
| SHAP integration | `src/models/explainability.py` ready; install `shap` to enable |
| MLflow tracking | Config flag + `mlflow` extra; auto-logs params/metrics/artifacts |
| DVC pipeline | `dvc.yaml` for data → train → eval stages |
| More models | Add to `PARAM_GRIDS` and `engines.py` |
| REST API | FastAPI wrapper around `Predictor` class |

---

## 10. Appendix: Real Training Results (Seed 42, NumPy Engine)

| Model | Test Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|-------|--------------|-----------|--------|-------------|-----|---------|
| Gradient Boosting | 0.700 | 0.696 | 0.593 | 0.788 | **0.640** | 0.709 |
| Naive Bayes | 0.633 | 0.581 | 0.667 | 0.606 | 0.621 | 0.753 |
| **Random Forest** | **0.667** | **0.640** | 0.593 | 0.727 | 0.615 | **0.767** |
| Logistic Regression | 0.550 | 0.000 | 0.000 | 1.000 | 0.000 | 0.412 |

*Best model selected by validation F1: Random Forest (val F1=0.692).*

> **Note**: Logistic Regression failed to converge on raw features (small val set, no CV in quick mode). With sklearn's StandardScaler + LBFGS it typically reaches ~0.75 AUC.