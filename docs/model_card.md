# Model Card — CardioInsight AI v0.1.0

> **Status**: ✅ Generated from actual training run (2026-10-27)
> **Model Artifact**: `models/model_random_forest_numpy_42.pkl`
> **Training Engine**: NumPy fallback (scikit-learn not available in sandbox)

---

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | CardioInsight AI |
| **Version** | 0.1.0 |
| **Model Type** | Random Forest (NumPy engine) |
| **Task** | Binary classification (heart disease: 0=no, 1=yes) |
| **Training Date** | 2026-10-27 |
| **Random Seed** | 42 |
| **Training Samples** | 214 |
| **Validation Samples** | 29 |
| **Test Samples** | 60 |
| **Features** | 13 (7 numeric, 6 categorical after one-hot) |
| **Engine** | Pure NumPy (no sklearn) |
| **Selection Metric** | Validation F1 = 0.692 |

---

## Intended Use

### ✅ Intended Uses
- Educational demonstration of ML pipelines for healthcare risk factor research
- Teaching model comparison, evaluation metrics, and explainability concepts
- Research prototype for exploring Cleveland heart disease dataset patterns
- Portfolio project showcasing reproducible ML engineering practices

### ❌ Prohibited Uses
- **Clinical decision-making** of any kind
- Patient diagnosis, triage, or treatment planning
- Any application where output could influence real health outcomes
- Commercial medical devices or software
- Use with real patient data without ethics review and clinical validation

---

## Training Data

### Dataset
- **Source**: Synthetic dataset generated to match the Cleveland heart disease schema (UCI ML Repository Dataset #45)
- **Generation Code**: `src/data/loader.py::generate_synthetic_dataset()`
- **Size**: 303 samples (214 train / 29 val / 60 test)
- **Class Balance**: 45% positive (disease), 55% negative (no disease)
- **License**: Apache-2.0 (synthetic data)

### Features (13)
| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| age | numeric | Age in years | [0, 120] |
| sex | categorical | 0=female, 1=male | {0, 1} |
| cp | categorical | Chest pain type (0–3) | {0,1,2,3} |
| trestbps | numeric | Resting blood pressure (mmHg) | [50, 250] |
| chol | numeric | Serum cholesterol (mg/dL) | [50, 700] |
| fbs | categorical | Fasting blood sugar > 120 mg/dL | {0, 1} |
| restecg | categorical | Resting ECG result (0–2) | {0,1,2} |
| thalach | numeric | Maximum heart rate achieved (bpm) | [40, 250] |
| exang | categorical | Exercise-induced angina | {0, 1} |
| oldpeak | numeric | ST depression (mV) | [-5, 10] |
| slope | categorical | ST segment slope (1–3) | {1,2,3} |
| ca | numeric | Major vessels colored (0–3) | [0, 4] |
| thal | categorical | Thalassemia type (0–3) | {0,1,2,3} |

### Preprocessing
- **Numeric**: StandardScaler (fit on train only)
- **Categorical**: OneHotEncoder (handle_unknown='ignore')
- **No imputation**: Synthetic data has no missing values
- **No leakage**: Preprocessor fit exclusively on training split

---

## Training Procedure

### Hyperparameter Search
- **Method**: Grid search on validation set
- **CV**: 5-fold stratified (on training data only)
- **Parameter Grid** (Random Forest):
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [3, 5, 7]
- **Best Params**: `n_estimators=50`, `max_depth=3`

### Models Compared
| Model | Val F1 | Test F1 | Test AUC |
|-------|--------|---------|----------|
| Random Forest | **0.692** | 0.615 | **0.767** |
| Gradient Boosting | 0.636 | **0.640** | 0.709 |
| Naive Bayes | 0.667 | 0.621 | 0.753 |
| Logistic Regression | 0.000 | 0.000 | 0.412 |

*Logistic Regression failed to converge on raw features without sklearn's StandardScaler + LBFGS.*

### Selection
Best model selected by **validation F1** (0.692 for Random Forest). Test metrics reported on held-out 60 samples.

---

## Evaluation Results (Test Set, n=60)

| Metric | Value | 95% CI (approx) |
|--------|-------|-----------------|
| **Accuracy** | 0.667 | [0.54, 0.78] |
| **Precision** | 0.640 | [0.44, 0.81] |
| **Recall (Sensitivity)** | 0.593 | [0.39, 0.77] |
| **Specificity** | 0.727 | [0.54, 0.87] |
| **F1 Score** | 0.615 | [0.44, 0.77] |
| **ROC-AUC** | 0.767 | [0.62, 0.88] |

### Confusion Matrix
```
                 Predicted
              No Disease  Disease
Actual No Disease      24       9
      Disease          11      16
```

### Per-Class Performance
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| No Disease (0) | 0.686 | 0.727 | 0.706 | 33 |
| Disease (1) | 0.640 | 0.593 | 0.615 | 27 |

---

## Known Limitations

1. **Synthetic Data Only** — The model was trained on artificially generated data that mimics the Cleveland schema but does not represent real patient distributions, comorbidities, or clinical complexity.

2. **Small Sample Size** — n=303 (214 train) yields high variance in performance estimates. Confidence intervals are wide.

3. **No External Validation** — Performance on real clinical data (different hospitals, populations, measurement protocols) is completely unknown.

4. **Class Imbalance** — 45% positive rate may bias probability calibration. The model tends to be conservative (higher specificity than sensitivity).

5. **Feature Importance ≠ Causation** — MDI importance ranks features by model utility, not clinical causality. E.g., `ca` (vessels colored) may rank high because it's a strong proxy, not because it causes disease.

6. **No Temporal Validation** — Single random split; no assessment of model drift over time.

7. **Single-Site Schema** — Cleveland data (1980s, US) may not generalize to global populations, modern diagnostics, or different ethnic groups.

8. **NumPy Engine Approximations** — The pure-NumPy Random Forest uses simplified splitting criteria and no feature subsampling optimizations. Results may differ from sklearn's implementation.

---

## Bias & Fairness Considerations

| Attribute | Status |
|-----------|--------|
| **Sex** | Included as feature (0/1). Synthetic distribution: ~50/50. No fairness audit performed. |
| **Age** | Continuous. Synthetic range 20–80. No age-stratified evaluation. |
| **Race/Ethnicity** | **Not present** in schema. Cannot assess bias. |
| **Socioeconomic** | **Not present**. Cannot assess bias. |
| **Geography** | Single-site (Cleveland) schema only. |

> **Recommendation**: Before any real-world deployment, conduct a full bias audit with disaggregated evaluation across demographic subgroups using real, representative data.

---

## Ethical Considerations

- This model **must not** be used to inform individual health decisions
- Outputs are **probabilities from a synthetic demonstration**, not calibrated risks
- The persistent disclaimer in the UI reinforces educational-only status
- No personal health information is collected, stored, or transmitted

---

## Model Artifacts

```
models/
├── model_random_forest_numpy_42.pkl      # Pickled model + preprocessor + metadata
├── model_random_forest_numpy_42.json     # Human-readable metadata
└── comparison.csv                        # All models' test metrics
```

### Loading the Model
```python
from src.prediction import Predictor
predictor = Predictor("models/model_random_forest_numpy_42.pkl")
result = predictor.predict({
    "age": 52, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
    "fbs": 0, "restecg": 1, "thalach": 160, "exang": 0,
    "oldpeak": 1.2, "slope": 2, "ca": 1, "thal": 2,
})
print(result.class_label, result.probability)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-10-27 | Initial release: synthetic data, 4 baselines, NumPy engine, Streamlit UI |

---

## Contact & Citation

**Project**: CardioInsight AI  
**Repository**: https://github.com/your-org/cardioinsight-ai  
**License**: Apache-2.0  

If you use this work in research, please cite:
```bibtex
@software{cardioinsight2026,
  title = {CardioInsight AI: Educational Heart Disease Risk Research Platform},
  author = {CardioInsight AI Contributors},
  year = {2026},
  url = {https://github.com/your-org/cardioinsight-ai}
}
```