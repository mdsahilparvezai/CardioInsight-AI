# Data Dictionary — CardioInsight AI

> **Version**: 0.1.0  
> **Schema Source**: Cleveland Heart Disease Dataset (UCI ML Repository #45)  
> **Status**: ✅ Implemented in `src/data/schema.py`

---

## Overview

This document defines every column in the dataset used by CardioInsight AI. The schema is **enforced at load time** by `validate_schema()` and used for input validation in the prediction form.

All ranges below are **structural bounds** derived from the dataset's documented feature types (e.g., age is a count in years, blood pressure is mmHg). They are used to reject physically impossible inputs and flag out-of-range values for review. **They are NOT clinical normal ranges**, and the application never presents them as guidance for personal health decisions.

---

## Target Variable

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `target` | integer | 0, 1 | **0 = No heart disease**, **1 = Heart disease present** (angiographic stenosis ≥ 50% in at least one major vessel) |

---

## Feature Columns (13)

### 1. `age` — Age
| Property | Value |
|----------|-------|
| **Type** | numeric (integer) |
| **Unit** | years |
| **Structural Range** | [0, 120] |
| **Description** | Age of the patient in years at the time of examination. |

### 2. `sex` — Sex
| Property | Value |
|----------|-------|
| **Type** | categorical |
| **Unit** | code |
| **Allowed Values** | {0, 1} |
| **Description** | **0 = Female**, **1 = Male**. |

### 3. `cp` — Chest Pain Type
| Property | Value |
|----------|-------|
| **Type** | categorical |
| **Unit** | code |
| **Allowed Values** | {0, 1, 2, 3} |
| **Description** | Chest pain classification:<br>• **0** = Typical angina<br>• **1** = Atypical angina<br>• **2** = Non-anginal pain<br>• **3** = Asymptomatic |

### 4. `trestbps` — Resting Blood Pressure
| Property | Value |
|----------|-------|
| **Type** | numeric (integer) |
| **Unit** | mmHg |
| **Structural Range** | [50, 250] |
| **Description** | Resting systolic blood pressure in mmHg on admission. |

### 5. `chol` — Serum Cholesterol
| Property | Value |
|----------|-------|
| **Type** | numeric (integer) |
| **Unit** | mg/dL |
| **Structural Range** | [50, 700] |
| **Description** | Serum cholesterol in mg/dL. |

### 6. `fbs` — Fasting Blood Sugar
| Property | Value |
|----------|-------|
| **Type** | categorical |
| **Unit** | code |
| **Allowed Values** | {0, 1} |
| **Description** | **1** if fasting blood sugar > 120 mg/dL; **0** otherwise. |

### 7. `restecg` — Resting Electrocardiographic Results
| Property | Value |
|----------|-------|
| **Type** | categorical |
| **Unit** | code |
| **Allowed Values** | {0, 1, 2} |
| **Description** | Resting ECG classification:<br>• **0** = Normal<br>• **1** = ST-T wave abnormality (T wave inversions and/or ST elevation/depression > 0.05 mV)<br>• **2** = Left ventricular hypertrophy (Estes' criteria) |

### 8. `thalach` — Maximum Heart Rate Achieved
| Property | Value |
|----------|-------|
| **Type** | numeric (integer) |
| **Unit** | bpm (beats per minute) |
| **Structural Range** | [40, 250] |
| **Description** | Maximum heart rate achieved during exercise stress test. |

### 9. `exang` — Exercise Induced Angina
| Property | Value |
|----------|-------|
| **Type** | categorical |
| **Unit** | code |
| **Allowed Values** | {0, 1} |
| **Description** | **1** if exercise induced angina; **0** otherwise. |

### 10. `oldpeak` — ST Depression
| Property | Value |
|----------|-------|
| **Type** | numeric (float) |
| **Unit** | mV (millivolts) |
| **Structural Range** | [-5, 10] |
| **Description** | ST depression induced by exercise relative to rest. Negative values indicate ST elevation. |

### 11. `slope` — ST Segment Slope
| Property | Value |
|----------|-------|
| **Type** | categorical |
| **Unit** | code |
| **Allowed Values** | {1, 2, 3} |
| **Description** | Slope of the peak exercise ST segment:<br>• **1** = Upsloping<br>• **2** = Flat<br>• **3** = Downsloping |

### 12. `ca` — Major Vessels Colored by Fluoroscopy
| Property | Value |
|----------|-------|
| **Type** | numeric (integer) |
| **Unit** | count |
| **Structural Range** | [0, 4] |
| **Description** | Number of major vessels (0–3) colored by fluoroscopy. In the original Cleveland data, this has missing values coded as `?`; our synthetic generator produces complete data. |

### 13. `thal` — Thalassemia
| Property | Value |
|----------|-------|
| **Type** | categorical |
| **Unit** | code |
| **Allowed Values** | {0, 1, 2, 3} |
| **Description** | Thalassemia blood disorder indicator (from thalium stress test):<br>• **0** = Normal (or missing in original data)<br>• **1** = Fixed defect<br>• **2** = Reversible defect<br>• **3** = Other/Unspecified |

---

## Schema Enforcement

The schema is defined programmatically in `src/data/schema.py`:

```python
FEATURE_COLUMNS = (
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
)
TARGET_COLUMN = "target"
```

### Validation Rules (enforced by `validate_dataset()`)

1. **All 13 features + target must be present** — missing columns raise `ValueError`
2. **No extra columns allowed** — unexpected columns raise `ValueError` (forces intentional schema updates)
3. **Categorical values must be in allowed set** — out-of-domain values flagged
4. **Numeric values must be within structural range** — out-of-bounds values flagged
5. **No missing values** — `NaN` counts reported per column
6. **No duplicate rows** — exact duplicates counted
7. **Target distribution reported** — class counts and imbalance ratio

### Example Validation Output (Synthetic Data, Seed 42)

```json
{
  "schema_ok": true,
  "missing_values": {},
  "invalid_values": {},
  "duplicate_rows": 0,
  "target_distribution": {0: 167, 1: 136},
  "target_imbalance_ratio": 0.449,
  "n_rows": 303,
  "n_columns": 14,
  "errors": []
}
```

---

## Using Real UCI Data

To use the actual Cleveland dataset:

1. Download `processed.cleveland.data` from UCI ML Repository #45
2. Convert to CSV with headers matching `FEATURE_COLUMNS + [TARGET_COLUMN]`
3. Place at `data/heart_disease.csv`
4. Set environment variable: `CARDIOINSIGHT_DATA_PATH=data/heart_disease.csv`

**Note**: The original Cleveland data has:
- 303 rows (after removing 6 rows with missing `ca`/`thal`)
- Missing values coded as `?` in `ca` and `thal`
- Target values 0–4 (0=healthy, 1–4=disease severity); commonly binarized to 0/1

Our loader expects the binary target. Preprocessing (imputation, binarization) would need to be added to `loader.py` for raw UCI data.

---

## Synthetic Data Generation

When no path is provided, `generate_synthetic_dataset()` creates a reproducible dataset:

```python
generate_synthetic_dataset(
    n_samples=303,
    seed=42,
    positive_rate=0.45,   # ~45% disease
    noise=0.35,           # measurement noise on continuous features
)
```

**Properties:**
- Deterministic: same seed → identical dataframe
- Learnable signal: features correlate with target
- Realistic noise: not perfectly separable
- Complete: no missing values
- Documented as synthetic in UI and metadata

---

## Feature Metadata for UI

The prediction form uses `FEATURE_METADATA` to render human-readable labels, units, and constraints:

```python
FEATURE_METADATA["age"] = FeatureMeta(
    name="age",
    label="Age",
    unit="years",
    kind="numeric",
    min_value=0,
    max_value=120,
    description="Age in years."
)
```

This powers the Streamlit form's `st.number_input` / `st.selectbox` configuration and validation messages.