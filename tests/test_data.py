"""Tests for the data layer: schema, loading, validation, EDA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import (
    load_dataset,
    validate_dataset,
    run_eda,
    validate_schema,
    generate_synthetic_dataset,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_METADATA,
)


class TestSchema:
    """Schema definition tests."""

    def test_feature_columns_complete(self):
        assert len(FEATURE_COLUMNS) == 13
        expected = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ]
        assert list(FEATURE_COLUMNS) == expected

    def test_target_column(self):
        assert TARGET_COLUMN == "target"

    def test_numeric_categorical_partition(self):
        all_features = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
        assert all_features == set(FEATURE_COLUMNS)
        assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)

    def test_feature_metadata_coverage(self):
        for name in FEATURE_COLUMNS:
            assert name in FEATURE_METADATA
            meta = FEATURE_METADATA[name]
            assert meta.name == name
            assert meta.label
            assert meta.unit
            assert meta.kind in ("numeric", "categorical")

    def test_validate_schema_passes_on_correct_columns(self):
        cols = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
        validate_schema(cols)  # should not raise

    def test_validate_schema_fails_on_missing_feature(self):
        cols = list(FEATURE_COLUMNS)[:-1] + [TARGET_COLUMN]
        with pytest.raises(ValueError, match="missing required columns"):
            validate_schema(cols)

    def test_validate_schema_fails_on_missing_target(self):
        cols = list(FEATURE_COLUMNS)
        with pytest.raises(ValueError, match="missing required columns"):
            validate_schema(cols)

    def test_validate_schema_fails_on_extra_columns(self):
        cols = list(FEATURE_COLUMNS) + [TARGET_COLUMN, "extra_col"]
        with pytest.raises(ValueError, match="unexpected columns"):
            validate_schema(cols)


class TestDataLoading:
    """Dataset loading and synthetic generation tests."""

    def test_generate_synthetic_dataset_shape(self):
        df = generate_synthetic_dataset(n_samples=100, seed=123)
        assert df.shape == (100, 14)  # 13 features + target
        assert list(df.columns[-1:]) == [TARGET_COLUMN]

    def test_generate_synthetic_deterministic(self):
        df1 = generate_synthetic_dataset(n_samples=50, seed=42)
        df2 = generate_synthetic_dataset(n_samples=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_generate_synthetic_different_seeds(self):
        df1 = generate_synthetic_dataset(n_samples=50, seed=1)
        df2 = generate_synthetic_dataset(n_samples=50, seed=2)
        assert not df1.equals(df2)

    def test_generate_synthetic_target_balance(self):
        df = generate_synthetic_dataset(n_samples=1000, seed=42, positive_rate=0.3)
        rate = df[TARGET_COLUMN].mean()
        assert 0.25 < rate < 0.35  # approximate

    def test_load_dataset_uses_synthetic_when_path_none(self):
        df = load_dataset(path=None, random_seed=42)
        assert df.shape[0] > 0
        assert TARGET_COLUMN in df.columns

    def test_synthetic_has_no_missing_values(self):
        df = generate_synthetic_dataset(seed=42)
        assert df.isna().sum().sum() == 0

    def test_synthetic_categorical_values_in_domain(self):
        df = generate_synthetic_dataset(seed=42)
        for name, meta in FEATURE_METADATA.items():
            if meta.kind == "categorical":
                invalid = set(df[name].unique()) - set(meta.allowed_values)
                assert not invalid, f"{name} has invalid values: {invalid}"

    def test_synthetic_numeric_values_in_range(self):
        df = generate_synthetic_dataset(seed=42)
        for name, meta in FEATURE_METADATA.items():
            if meta.kind == "numeric":
                lo, hi = meta.min_value, meta.max_value
                if lo is not None:
                    assert (df[name] >= lo).all(), f"{name} below min"
                if hi is not None:
                    assert (df[name] <= hi).all(), f"{name} above max"


class TestValidation:
    """Dataset validation tests."""

    def test_validate_clean_dataset(self):
        df = generate_synthetic_dataset(seed=42)
        result = validate_dataset(df)
        assert result.is_valid
        assert result.schema_ok
        assert result.missing_values == {}
        assert result.invalid_values == {}
        assert result.duplicate_rows == 0
        assert result.target_distribution == {0: 167, 1: 136}

    def test_validate_detects_missing(self):
        df = generate_synthetic_dataset(seed=42)
        df.loc[0, "age"] = np.nan
        result = validate_dataset(df)
        assert not result.is_valid
        assert "age" in result.missing_values
        assert result.missing_values["age"] == 1

    def test_validate_detects_duplicates(self):
        df = generate_synthetic_dataset(seed=42)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        result = validate_dataset(df)
        assert result.duplicate_rows == 1
        assert not result.is_valid

    def test_validate_detects_invalid_categorical(self):
        df = generate_synthetic_dataset(seed=42)
        df.loc[0, "sex"] = 99
        result = validate_dataset(df)
        assert "sex" in result.invalid_values
        assert 99 in result.invalid_values["sex"]

    def test_validate_detects_out_of_range_numeric(self):
        df = generate_synthetic_dataset(seed=42)
        df.loc[0, "age"] = 999
        result = validate_dataset(df)
        assert "age" in result.invalid_values
        assert 999 in result.invalid_values["age"]


class TestEDA:
    """Exploratory data analysis tests."""

    def test_run_eda_returns_expected_keys(self):
        df = generate_synthetic_dataset(seed=42)
        eda = run_eda(df)
        assert "shape" in eda
        assert "summary_statistics" in eda
        assert "missing_values" in eda
        assert "correlation_matrix" in eda
        assert "target_distribution" in eda
        assert "feature_distributions" in eda

    def test_eda_shape_correct(self):
        df = generate_synthetic_dataset(seed=42)
        eda = run_eda(df)
        assert eda["shape"]["rows"] == 303
        assert eda["shape"]["columns"] == 14

    def test_eda_feature_distributions_complete(self):
        df = generate_synthetic_dataset(seed=42)
        eda = run_eda(df)
        assert set(eda["feature_distributions"].keys()) == set(FEATURE_COLUMNS)

    def test_eda_numeric_feature_stats(self):
        df = generate_synthetic_dataset(seed=42)
        eda = run_eda(df)
        age_dist = eda["feature_distributions"]["age"]
        assert age_dist["kind"] == "numeric"
        assert "mean" in age_dist
        assert "std" in age_dist
        assert "by_target" in age_dist

    def test_eda_categorical_feature_stats(self):
        df = generate_synthetic_dataset(seed=42)
        eda = run_eda(df)
        sex_dist = eda["feature_distributions"]["sex"]
        assert sex_dist["kind"] == "categorical"
        assert "value_counts" in sex_dist
        assert "by_target" in sex_dist

    def test_correlation_matrix_symmetric(self):
        df = generate_synthetic_dataset(seed=42)
        eda = run_eda(df)
        corr = eda["correlation_matrix"]
        # Check symmetry (within numerical precision)
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)

    def test_correlation_diagonal_is_one(self):
        df = generate_synthetic_dataset(seed=42)
        eda = run_eda(df)
        corr = eda["correlation_matrix"]
        np.testing.assert_allclose(np.diag(corr.values), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])