"""Tests for the models layer: engines, evaluation, I/O."""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import (
    select_engine,
    available_engines,
    evaluate_model,
    cross_validate,
    compare_models,
    save_artifacts,
    load_artifacts,
    ModelMetadata,
    PARAM_GRIDS,
)
from src.data import generate_synthetic_dataset
from src.data.schema import FEATURE_COLUMNS


class TestEngines:
    """Model engine tests."""

    def test_available_engines_includes_numpy(self):
        engines = available_engines()
        assert "numpy" in engines

    def test_select_engine_numpy(self):
        for mtype in ["logistic_regression", "random_forest",
                      "gradient_boosting", "naive_bayes"]:
            engine = select_engine(mtype, engine="numpy", random_state=42)
            assert engine is not None
            assert hasattr(engine, "fit")
            assert hasattr(engine, "predict")
            assert hasattr(engine, "predict_proba")

    def test_select_engine_invalid_model_type(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            select_engine("invalid_model", engine="numpy")

    def test_select_engine_invalid_engine(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            select_engine("logistic_regression", engine="invalid")

    def test_engine_fit_predict_roundtrip(self):
        df = generate_synthetic_dataset(n_samples=100, seed=42)
        X = df[list(FEATURE_COLUMNS)].values
        y = df["target"].values

        for mtype in ["random_forest", "naive_bayes"]:
            engine = select_engine(mtype, engine="numpy", random_state=42)
            engine.fit(X, y)
            preds = engine.predict(X)
            probas = engine.predict_proba(X)
            assert preds.shape == (100,)
            assert probas.shape == (100, 2)
            assert set(preds).issubset({0, 1})
            assert np.allclose(probas.sum(axis=1), 1.0)

    def test_engine_get_set_params(self):
        engine = select_engine("random_forest", engine="numpy",
                               n_estimators=50, max_depth=3)
        params = engine.get_params()
        assert params["model_type"] == "random_forest"
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 3

        engine.set_params(n_estimators=100)
        params = engine.get_params()
        assert params["n_estimators"] == 100


class TestEvaluation:
    """Evaluation metric tests."""

    def test_evaluate_model_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])

        res = evaluate_model(y_true, y_pred, y_proba, "test_model", "test")
        assert res.accuracy == 1.0
        assert res.precision == 1.0
        assert res.recall == 1.0
        assert res.specificity == 1.0
        assert res.f1 == 1.0
        assert res.roc_auc == 1.0

    def test_evaluate_model_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_proba = np.array([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3], [0.8, 0.2]])

        res = evaluate_model(y_true, y_pred, y_proba, "test_model", "test")
        assert res.accuracy == 0.0
        assert res.f1 == 0.0
        assert res.roc_auc == 0.0

    def test_evaluate_model_confusion_matrix_shape(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.6, 0.4],
                            [0.9, 0.1], [0.2, 0.8]])

        res = evaluate_model(y_true, y_pred, y_proba, "test_model", "test")
        assert res.confusion_matrix.shape == (2, 2)
        # TN, FP, FN, TP
        tn, fp, fn, tp = res.confusion_matrix.ravel()
        assert tn + fp + fn + tp == 6

    def test_compare_models_returns_dataframe(self):
        df = generate_synthetic_dataset(n_samples=100, seed=42)
        X = df[list(FEATURE_COLUMNS)].values
        y = df["target"].values

        models = {}
        for mtype in ["random_forest", "naive_bayes"]:
            engine = select_engine(mtype, engine="numpy", random_state=42)
            engine.fit(X, y)
            models[mtype] = engine

        comp = compare_models(models, X, y, "test")
        assert isinstance(comp, pd.DataFrame)
        assert len(comp) == 2
        assert "model_name" in comp.columns
        assert "f1" in comp.columns
        # Should be sorted by F1 descending
        assert comp.iloc[0]["f1"] >= comp.iloc[1]["f1"]

    def test_calibration_data(self):
        from src.models.evaluation import calibration_data
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6],
                            [0.7, 0.3], [0.2, 0.8], [0.1, 0.9], [0.6, 0.4]])

        cal = calibration_data(y_true, y_proba, n_bins=5)
        assert isinstance(cal, pd.DataFrame)
        assert "bin" in cal.columns
        assert "mean_predicted" in cal.columns
        assert "fraction_positive" in cal.columns


class TestIO:
    """Artifact serialization tests."""

    def test_save_load_artifacts_roundtrip(self):
        df = generate_synthetic_dataset(n_samples=50, seed=42)
        X = df[list(FEATURE_COLUMNS)].values
        y = df["target"].values

        engine = select_engine("random_forest", engine="numpy",
                               n_estimators=10, max_depth=3, random_state=42)
        engine.fit(X, y)

        metadata = ModelMetadata(
            model_type="random_forest",
            engine="numpy",
            config={"test": True},
            train_metrics={"f1": 0.8},
            val_metrics={"f1": 0.75},
            test_metrics={"f1": 0.7},
            feature_names=list(FEATURE_COLUMNS),
            trained_at="2026-01-01T00:00:00",
            random_seed=42,
            n_train_samples=50,
            n_features=X.shape[1],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.pkl"
            save_artifacts(engine, None, metadata, path)

            assert path.exists()
            assert path.with_suffix(".json").exists()

            loaded = load_artifacts(path)
            assert "model" in loaded
            assert "metadata" in loaded
            assert loaded["metadata"]["model_type"] == "random_forest"
            assert loaded["feature_names"] == list(FEATURE_COLUMNS)

            # Test that loaded model works
            loaded_model = loaded["model"]
            preds = loaded_model.predict(X)
            assert preds.shape == (50,)

    def test_metadata_json_readable(self):
        df = generate_synthetic_dataset(n_samples=50, seed=42)
        X = df[list(FEATURE_COLUMNS)].values
        y = df["target"].values

        engine = select_engine("naive_bayes", engine="numpy", random_state=42)
        engine.fit(X, y)

        metadata = ModelMetadata(
            model_type="naive_bayes",
            engine="numpy",
            config={},
            train_metrics={},
            val_metrics={},
            test_metrics={},
            feature_names=list(FEATURE_COLUMNS),
            trained_at="2026-01-01T00:00:00",
            random_seed=42,
            n_train_samples=50,
            n_features=X.shape[1],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.pkl"
            save_artifacts(engine, None, metadata, path)

            json_path = path.with_suffix(".json")
            with json_path.open() as fh:
                data = json.load(fh)
            assert data["model_type"] == "naive_bayes"
            assert data["engine"] == "numpy"
            assert data["feature_names"] == list(FEATURE_COLUMNS)

    def test_latest_model_path_none_when_empty(self):
        from src.models.io import latest_model_path
        with tempfile.TemporaryDirectory() as tmpdir:
            path = latest_model_path(tmpdir)
            assert path is None


class TestParamGrids:
    """Parameter grid tests."""

    def test_param_grids_cover_all_models(self):
        expected_models = ["logistic_regression", "random_forest",
                           "gradient_boosting", "naive_bayes"]
        for model in expected_models:
            assert model in PARAM_GRIDS

    def test_param_grids_have_numpy_engine(self):
        for model in PARAM_GRIDS:
            assert "numpy" in PARAM_GRIDS[model]

    def test_param_grid_values_are_lists(self):
        for model in PARAM_GRIDS:
            for engine in PARAM_GRIDS[model]:
                for param, values in PARAM_GRIDS[model][engine].items():
                    assert isinstance(values, list), f"{model}.{engine}.{param} not a list"
                    assert len(values) > 0, f"{model}.{engine}.{param} empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
