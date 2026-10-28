"""Tests for the features layer: preprocessing and input validation."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from src.features import (
    build_preprocessor,
    fit_preprocessor,
    transform_features,
    validate_inference_input,
)
from src.data import generate_synthetic_dataset
from src.data.schema import FEATURE_COLUMNS, FEATURE_METADATA


class TestPreprocessor:
    """Preprocessing pipeline tests."""

    def test_build_preprocessor_returns_none_without_sklearn(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sklearn", None)
        preprocessor = build_preprocessor()
        assert preprocessor is None

    def test_fit_preprocessor_raises_on_none(self):
        df = generate_synthetic_dataset(seed=42)
        X = df[list(FEATURE_COLUMNS)]
        with pytest.raises(RuntimeError, match="No preprocessor available"):
            fit_preprocessor(None, X)

    def test_transform_features_raises_on_none(self):
        df = generate_synthetic_dataset(seed=42)
        X = df[list(FEATURE_COLUMNS)]
        with pytest.raises(RuntimeError, match="No preprocessor available"):
            transform_features(None, X)


class TestInputValidation:
    """Inference input validation tests."""

    def _valid_input(self) -> dict:
        return {
            "age": 52, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
            "fbs": 0, "restecg": 1, "thalach": 160, "exang": 0,
            "oldpeak": 1.2, "slope": 2, "ca": 1, "thal": 2,
        }

    def test_valid_input_passes(self):
        result = validate_inference_input(self._valid_input())
        assert result.valid
        assert result.errors == []
        assert set(result.coerced.keys()) == set(FEATURE_COLUMNS)

    def test_missing_feature_rejected(self):
        inp = self._valid_input()
        del inp["age"]
        result = validate_inference_input(inp)
        assert not result.valid
        assert any("Age" in e and "required" in e for e in result.errors)

    def test_invalid_categorical_rejected(self):
        inp = self._valid_input()
        inp["sex"] = 5
        result = validate_inference_input(inp)
        assert not result.valid
        assert any("Sex" in e and "must be one of" in e for e in result.errors)

    def test_numeric_below_min_rejected(self):
        inp = self._valid_input()
        inp["age"] = -5
        result = validate_inference_input(inp)
        assert not result.valid
        assert any("Age" in e and "must be >=" in e for e in result.errors)

    def test_numeric_above_max_rejected(self):
        inp = self._valid_input()
        inp["age"] = 999
        result = validate_inference_input(inp)
        assert not result.valid
        assert any("Age" in e and "must be <=" in e for e in result.errors)

    def test_unknown_feature_rejected(self):
        inp = self._valid_input()
        inp["unknown_feature"] = 123
        result = validate_inference_input(inp)
        assert not result.valid
        assert any("Unknown feature" in e for e in result.errors)

    def test_coerced_values_correct_types(self):
        inp = {k: str(v) for k, v in self._valid_input().items()}
        result = validate_inference_input(inp)
        assert result.valid
        assert isinstance(result.coerced["age"], float)
        assert isinstance(result.coerced["sex"], int)
        assert result.coerced["sex"] == 1

    def test_lookup_by_label_works(self):
        # Use labels instead of names
        inp = {
            "Age": 52, "Sex": 1, "Chest Pain Type": 2,
            "Resting Blood Pressure": 130, "Serum Cholesterol": 240,
            "Fasting Blood Sugar": 0, "Resting ECG": 1,
            "Maximum Heart Rate": 160, "Exercise Induced Angina": 0,
            "ST Depression": 1.2, "ST Slope": 2,
            "Major Vessels Colored": 1, "Thalassemia": 2,
        }
        result = validate_inference_input(inp)
        assert result.valid
        assert set(result.coerced.keys()) == set(FEATURE_COLUMNS)

    def test_multiple_errors_reported(self):
        inp = {
            "age": 999, "sex": 5, "cp": 2, "trestbps": 30, "chol": 240,
            "fbs": 0, "restecg": 1, "thalach": 160, "exang": 0,
            "oldpeak": 1.2, "slope": 2, "ca": 1, "thal": 2,
        }
        result = validate_inference_input(inp)
        assert not result.valid
        assert len(result.errors) >= 2  # age and sex should both error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
