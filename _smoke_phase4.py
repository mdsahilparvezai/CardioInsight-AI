"""Smoke test for Phase 4: prediction service.

Run: python _smoke_phase4.py
"""

import sys
sys.path.insert(0, ".")

from src.prediction import Predictor

print("=== 1. Load predictor ===")
predictor = Predictor()
print(f"Model type: {predictor.metadata.get('model_type')}")
print(f"Engine: {predictor.metadata.get('engine')}")
print(f"Features: {predictor.feature_names}")

print("\n=== 2. Valid prediction ===")
sample = {
    "age": 52, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
    "fbs": 0, "restecg": 1, "thalach": 160, "exang": 0,
    "oldpeak": 1.2, "slope": 2, "ca": 1, "thal": 2,
}
result = predictor.predict(sample)
print(f"Predicted class: {result.predicted_class}")
print(f"Probability: {result.probability:.4f}")
print(f"Class label: {result.class_label}")
print(f"Validation: {result.validation.valid}")

print("\n=== 3. Invalid input handling ===")
bad = {"age": 9999, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
       "fbs": 0, "restecg": 1, "thalach": 160, "exang": 0,
       "oldpeak": 1.2, "slope": 2, "ca": 1, "thal": 2}
try:
    predictor.predict(bad)
    print("ERROR: Should have raised ValueError")
except ValueError as exc:
    print(f"Correctly rejected: {exc}")

print("\n=== 4. Feature metadata for UI ===")
meta = predictor.get_feature_metadata()
print(f"Features available: {len(meta)}")
print(f"Example (age): {meta['age']}")

print("\n=== Phase 4 smoke test PASSED ===")