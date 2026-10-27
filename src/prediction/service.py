"""Prediction service: load artifacts and produce validated predictions.

The :class:`Predictor` class encapsulates the entire inference pipeline:

1. Load model + preprocessor + metadata from a single artifact file.
2. Validate user input against the feature schema.
3. Apply the *exact same* preprocessing used during training.
4. Return class label, probability, and a human-readable explanation.

No external dependencies beyond numpy/pandas.  Works with both the sklearn
and NumPy engines because the serialized artifact contains the fitted model
object directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.schema import FEATURE_COLUMNS, FEATURE_METADATA
from ..features.validation import validate_inference_input, ValidationResult
from ..models.io import load_artifacts, latest_model_path
from ..utils.logging import get_logger

log = get_logger("prediction.service")


@dataclass
class PredictionResult:
    """Structured prediction output for the UI."""

    predicted_class: int  # 0 or 1
    probability: float    # P(class=1)
    class_label: str      # "No Disease" / "Disease"
    input_features: dict[str, Any]
    validation: ValidationResult
    model_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": int(self.predicted_class),
            "probability": float(self.probability),
            "class_label": self.class_label,
            "input_features": self.input_features,
            "validation": self.validation.to_dict(),
            "model_metadata": self.model_metadata,
        }


class Predictor:
    """High-level prediction API."""

    def __init__(self, artifact_path: str | Path | None = None):
        """Load a trained model artifact.

        Parameters
        ----------
        artifact_path:
            Path to a ``.pkl`` file produced by :func:`src.models.io.save_artifacts`.
            If ``None``, the most recent file in the ``models/`` directory is used.
        """
        if artifact_path is None:
            artifact_path = latest_model_path()
        if artifact_path is None:
            raise FileNotFoundError(
                "No model artifact found. Train a model first with `python train.py`."
            )
        self.artifact_path = Path(artifact_path)
        artifact = load_artifacts(self.artifact_path)
        self.model = artifact["model"]
        self.preprocessor = artifact["preprocessor"]
        self.feature_names = artifact["feature_names"]
        self.metadata = artifact["metadata"]
        log.info("Loaded predictor from %s (model_type=%s, engine=%s)",
                 self.artifact_path,
                 self.metadata.get("model_type"),
                 self.metadata.get("engine"))

    def _preprocess(self, features: dict[str, Any]) -> np.ndarray:
        """Convert a feature dict to the model's input array."""
        # Build DataFrame in the correct column order
        row = pd.DataFrame([features], columns=self.feature_names)
        if self.preprocessor is not None:
            from src.features.preprocessor import transform_features
            return transform_features(self.preprocessor, row)
        return row.values

    def predict(self, user_input: dict[str, Any]) -> PredictionResult:
        """Run validation, preprocessing, and inference on a single input.

        Parameters
        ----------
        user_input:
            Dictionary mapping feature names (or labels) to user-supplied values.

        Returns
        -------
        PredictionResult
            Contains the class label, probability, and validation details.
        """
        # 1. Validate
        validation = validate_inference_input(user_input)
        if not validation.valid:
            raise ValueError("Input validation failed: " + "; ".join(validation.errors))

        # 2. Preprocess
        X = self._preprocess(validation.coerced)

        # 3. Predict
        proba = self.model.predict_proba(X)
        prob_pos = float(proba[0, 1])
        pred_class = int(prob_pos >= 0.5)
        class_label = "Disease" if pred_class == 1 else "No Disease"

        return PredictionResult(
            predicted_class=pred_class,
            probability=prob_pos,
            class_label=class_label,
            input_features=validation.coerced,
            validation=validation,
            model_metadata={
                "model_type": self.metadata.get("model_type"),
                "engine": self.metadata.get("engine"),
                "trained_at": self.metadata.get("trained_at"),
                "random_seed": self.metadata.get("random_seed"),
                "n_train_samples": self.metadata.get("n_train_samples"),
                "test_f1": self.metadata.get("test_metrics", {}).get("f1"),
                "test_auc": self.metadata.get("test_metrics", {}).get("roc_auc"),
            },
        )

    def predict_batch(self, inputs: list[dict[str, Any]]) -> list[PredictionResult]:
        """Predict multiple inputs (for testing)."""
        return [self.predict(inp) for inp in inputs]

    def get_feature_metadata(self) -> dict[str, Any]:
        """Return feature metadata for building the UI form."""
        return {
            name: {
                "label": meta.label,
                "unit": meta.unit,
                "kind": meta.kind,
                "allowed_values": list(meta.allowed_values) if meta.allowed_values else None,
                "min_value": meta.min_value,
                "max_value": meta.max_value,
                "description": meta.description,
            }
            for name, meta in FEATURE_METADATA.items()
        }