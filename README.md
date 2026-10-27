# CardioInsight AI

**Advanced, Explainable Heart Disease Risk Research Platform**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Educational%20Prototype-orange)]()

> ⚠️ **Educational Use Only**: CardioInsight AI is a machine learning demonstration for learning and research purposes. It is **not a diagnostic tool**, has not been validated for clinical decision-making, and must never be used to make medical decisions. Consult a qualified healthcare professional for health concerns.

---

## 🎯 Project Overview

CardioInsight AI is an end-to-end educational machine learning application that:

- **Studies patterns** in the Cleveland heart disease dataset (UCI ML Repository #45)
- **Trains & compares** multiple baseline models with hyperparameter tuning
- **Visualizes** data, model performance, and explanations in a professional dashboard
- **Demonstrates** reproducible ML engineering practices (config-driven, versioned artifacts, CI/CD)
- **Runs locally** on a standard Windows laptop with minimal dependencies

**This is a portfolio-quality project** built by a first-year B.Tech Computer Science student specializing in AI, following industry best practices for ML systems.

---

## ✨ Key Features

| Area | Highlights |
|------|------------|
| **Data** | Synthetic Cleveland-schema generator (reproducible), schema validation, EDA |
| **Models** | Logistic Regression, Random Forest, Gradient Boosting, Naive Bayes |
| **Engines** | scikit-learn (primary) + pure NumPy fallback (zero deps) |
| **Evaluation** | Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, Confusion Matrix, PR Curve, Calibration |
| **Explainability** | Global feature importance, SHAP integration (optional) |
| **Dashboard** | 6-page Streamlit app: Overview, Data Explorer, Model Lab, Prediction Demo, Explainability, Model Card |
| **MLOps** | MLflow tracking (optional), DVC pipeline (optional), GitHub Actions CI, Docker |
| **Responsible AI** | Persistent disclaimers, no PHI collection, educational-use enforcement |

---

## 🏗️ Architecture

```
CardioInsight-AI/
├── app.py                 # Streamlit dashboard entry point
├── train.py               # Training pipeline entry point
├── src/
│   ├── data/              # Loading, validation, EDA, synthetic generation
│   ├── features/          # Preprocessing, input validation
│   ├── models/            # Engines, evaluation, I/O, explainability
│   ├── prediction/        # Inference service
│   ├── visualization/     # Charts (Plotly + Matplotlib)
│   └── utils/             # Config, logging
├── tests/                 # Pytest suite (data, features, models, prediction)
├── configs/
│   └── config.yaml        # Non-secret configuration
├── data/
│   └── README.md          # Dataset provenance & usage
├── models/                # Generated artifacts (gitignored)
├── docs/
│   ├── architecture.md    # System architecture
│   ├── model_card.md      # Model card with real metrics
│   ├── data_dictionary.md # Feature definitions
│   └── experiment_report.md # Training run report
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
└── MLOPS.md
```

---

## 🚀 Quick Start (Windows)

### Prerequisites
- Windows 10/11
- Python 3.9+ (from [python.org](https://python.org) or Microsoft Store)
- Git (optional)

### 1. Clone & Create Virtual Environment
```cmd
git clone https://github.com/mdsahilparvezai/CardioInsight-AI.git
cd CardioInsight-AI

python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Dependencies
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train a Model
```cmd
# Quick mode (NumPy engine, small grid)
python train.py --engine numpy --quick

# Full mode (scikit-learn, full grid) — after installing sklearn
python train.py --engine sklearn
```

### 4. Launch Dashboard
```cmd
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## 📊 Dashboard Pages

| Page | Purpose |
|------|---------|
| **📊 Overview** | Project intro, dataset summary, pipeline status, disclaimer |
| **🔍 Data Explorer** | Preview, distributions, correlations, missing values, filters |
| **🧪 Model Lab** | Compare models, metrics, confusion matrices, experiment history |
| **🎯 Prediction Demo** | Validated form → prediction with probability + limitations notice |
| **🔬 Explainability** | Feature importance, SHAP (when installed), causation warning |
| **📋 Model Card** | Provenance, intended use, limitations, bias considerations, export |

---

## 📈 Real Results (Synthetic Data, Seed 42)

| Model | Test F1 | ROC-AUC | Notes |
|-------|---------|---------|-------|
| **Random Forest** | 0.615 | **0.767** | Best validation F1 (0.692) |
| Gradient Boosting | **0.640** | 0.709 | Best test F1 |
| Naive Bayes | 0.621 | 0.753 | Best recall (0.667) |
| Logistic Regression | 0.000 | 0.412 | Needs sklearn StandardScaler |

> **Important**: Results on synthetic data do not generalize to clinical populations. The model card documents all limitations.

---

## ⚙️ Configuration

All settings in `configs/config.yaml` (overridable via `CARDIOINSIGHT_*` env vars):

```yaml
project:
  random_seed: 42
  artifacts_dir: models

data:
  path: ""                    # Empty → synthetic dataset
  target_column: target
  test_size: 0.2
  val_size: 0.1

model:
  engine: auto                # auto | sklearn | numpy
  cv_folds: 5
  selection_metric: f1

mlflow:
  enabled: false              # Set true + tracking_uri to enable
```

Copy `.env.example` to `.env` and edit for local overrides.

---

## 🧪 Testing

```cmd
# Run full test suite
pytest -v

# With coverage
pytest --cov=src --cov-report=term-missing
```

Test categories:
- `test_data.py` — Schema, loading, validation, EDA
- `test_features.py` — Preprocessing, input validation
- `test_models.py` — Engines, evaluation, serialization
- `test_prediction.py` — Prediction service

---

## 🐳 Docker Deployment

```cmd
# Build image
docker build -t cardioinsight-ai .

# Run (mount models directory for persistence)
docker run -p 8501:8501 -v %cd%\models:/app/models cardioinsight-ai
```

The image is multi-stage (builder → runtime), runs as non-root, includes health check.

---

## 🔬 MLOps Integrations (Optional)

### MLflow Experiment Tracking
```cmd
set CARDIOINSIGHT_MLFLOW_ENABLED=true
set CARDIOINSIGHT_MLFLOW_TRACKING_URI=file://./mlruns
python train.py
mlflow ui --backend-store-uri file://./mlruns
```

### DVC Data & Pipeline Versioning
```cmd
pip install dvc
dvc init
dvc add data/heart_disease.csv    # If using real data
dvc repro                         # Run pipeline from dvc.yaml
```

See [MLOPS.md](MLOPS.md) for detailed guide.

---

## 📁 Using Real UCI Data

1. Download `processed.cleveland.data` from [UCI Repository #45](https://archive.ics.uci.edu/dataset/45/heart+disease)
2. Convert to CSV with headers matching the 13 features + `target`
3. Place at `data/heart_disease.csv`
4. Set environment variable:
   ```cmd
   set CARDIOINSIGHT_DATA_PATH=data/heart_disease.csv
   ```

**Note**: Original data has missing values (`?` in `ca`/`thal`) and multi-class target (0–4). Preprocessing would need to be added to `src/data/loader.py`.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design, data flow, module reference |
| [Model Card](docs/model_card.md) | Provenance, metrics, limitations, bias audit |
| [Data Dictionary](docs/data_dictionary.md) | Feature definitions, ranges, validation rules |
| [Experiment Report](docs/experiment_report.md) | Actual training run with real metrics |
| [MLOps Guide](MLOPS.md) | MLflow, DVC, CI/CD integration |

---

## ⚠️ Responsible Use & Limitations

### This Application:
- ✅ Is for **education and research only**
- ✅ Uses **synthetic data** by default (documented as such)
- ✅ Shows **persistent disclaimers** on every page
- ✅ **Never claims** to diagnose or assess personal risk
- ✅ **Does not collect** personal health information
- ✅ **Runs offline** (no external API calls)

### Known Limitations:
1. **Synthetic data only** — not real patient records
2. **Small sample** (n=303) — high variance in metrics
3. **No clinical validation** — performance on real data unknown
4. **Feature importance ≠ causation** — associations only
5. **Single-site schema** — Cleveland (1980s) may not generalize
6. **NumPy engine approximations** — differs from sklearn

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: sklearn` | `pip install scikit-learn` or use `--engine numpy` |
| `ModuleNotFoundError: streamlit` | `pip install streamlit plotly` |
| `No model artifact found` | Run `python train.py` first |
| Port 8501 in use | `streamlit run app.py --server.port=8502` |
| Permission denied on `.venv` | Delete and recreate: `rmdir /s .venv && python -m venv .venv` |

---

## 🔮 Future Improvements

- [ ] Full hyperparameter grids (remove `--quick` limitation)
- [ ] XGBoost / LightGBM integration
- [ ] Calibration curves & Brier score
- [ ] Statistical significance testing (bootstrap, DeLong)
- [ ] Learning curves & overfitting diagnostics
- [ ] External validation on real UCI data
- [ ] REST API wrapper (FastAPI)
- [ ] Automated drift monitoring (Evidently AI)

---

## 📄 License

Apache-2.0 — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** — Cleveland Heart Disease dataset schema
- **scikit-learn, NumPy, Pandas, Streamlit, Plotly** — Core libraries
- **SHAP** — Explainability framework (optional)
- **MLflow, DVC** — MLOps tooling (optional)

---

## 📬 Contact

Built as a learning project. Issues and suggestions welcome via GitHub Issues.

---

**Remember**: This is an educational prototype. For health concerns, always consult a qualified healthcare professional.
