"""Smoke test for Phase 2: data loading, validation, EDA.

Run: python _smoke_phase2.py
"""

import sys

sys.path.insert(0, ".")

from src.data import (
    load_dataset, validate_dataset, run_eda,
    validate_schema, generate_synthetic_dataset,
)
from src.utils import get_config, reset_config

reset_config()
cfg = get_config()

print("=== 1. Load dataset ===")
df = load_dataset(path=cfg["data"]["path"], random_seed=cfg["project"]["random_seed"])
print(f"shape: {df.shape}")
print(f"columns: {list(df.columns)}")

print("\n=== 2. Schema validation ===")
try:
    validate_schema(df.columns)
    print("schema OK")
except ValueError as exc:
    print(f"schema FAILED: {exc}")
    sys.exit(1)

print("\n=== 3. Dataset validation ===")
result = validate_dataset(df)
print(f"valid: {result.is_valid}")
print(f"missing: {result.missing_values}")
print(f"duplicates: {result.duplicate_rows}")
print(f"invalid values: {result.invalid_values}")
print(f"target distribution: {result.target_distribution}")
print(f"imbalance ratio: {result.target_imbalance_ratio:.3f}")
if result.errors:
    for e in result.errors:
        print(f"  error: {e}")

print("\n=== 4. EDA ===")
eda = run_eda(df)
print(f"target dist: {eda['target_distribution']}")
print(f"correlation matrix shape: {eda['correlation_matrix'].shape}")
print("feature distributions computed for:", list(eda["feature_distributions"].keys()))

print("\n=== 5. Reproducibility check ===")
df2 = generate_synthetic_dataset(seed=cfg["project"]["random_seed"])
same = df.equals(df2)
print(f"deterministic regeneration: {same}")

print("\n=== 6. Inference input validation ===")
from src.features.validation import validate_inference_input
sample = {
    "age": 52, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
    "fbs": 0, "restecg": 1, "thalach": 160, "exang": 0,
    "oldpeak": 1.2, "slope": 2, "ca": 1, "thal": 2,
}
v = validate_inference_input(sample)
print(f"valid: {v.valid}, errors: {v.errors}")
print(f"coerced: {v.coerced}")

print("\n=== 7. Bad input handling ===")
bad = {"age": 9999, "sex": 5, "cp": 2, "trestbps": 30, "chol": 240,
       "fbs": 0, "restecg": 1, "thalach": 160, "exang": 0,
       "oldpeak": 1.2, "slope": 2, "ca": 1, "thal": 2}
v2 = validate_inference_input(bad)
print(f"valid: {v2.valid}")
for e in v2.errors:
    print(f"  error: {e}")

print("\n=== Phase 2 smoke test PASSED ===")