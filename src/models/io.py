"""Artifact serialization: save/load models, preprocessors, and metadata.

Artifacts are stored as a single pickle file containing:

* ``model`` - the fitted estimator (sklearn or NumPy engine)
* ``preprocessor`` - the fitted ColumnTransformer (or None)
* ``feature_names`` - list of input feature names in model order
* ``metadata`` - dict with training info (engine, model_type, config, metrics, date)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..data.schema import FEATURE_COLUMNS
from ..utils.logging import get_logger

log = get_logger("models.io")


@dataclass
class ModelMetadata:
    """Everything needed to reproduce or inspect a trained model."""

    model_type: str
    engine: str
    config: dict
    train_metrics: dict
    val_metrics: dict
    test_metrics: dict | None
    feature_names: list[str]
    trained_at: str
    random_seed: int
    n_train_samples: int
    n_features: int

    def to_dict(self) -> dict:
        return asdict(self)


def _default_serializer(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_artifacts(
    model: Any,
    preprocessor: Any | None,
    metadata: ModelMetadata,
    path: str | Path,
) -> Path:
    """Serialize model + preprocessor + metadata to a single file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": metadata.feature_names,
        "metadata": metadata.to_dict(),
    }
    with path.open("wb") as fh:
        pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
    # Also write a human-readable metadata JSON sidecar
    meta_path = path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata.to_dict(), fh, indent=2, default=_default_serializer)
    log.info("Saved model artifact to %s (%d bytes)", path, path.stat().st_size)
    return path


def load_artifacts(path: str | Path) -> dict:
    """Load artifact and return the raw dict.

    Caller is responsible for extracting ``model``, ``preprocessor``,
    ``feature_names``, and ``metadata``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    with path.open("rb") as fh:
        artifact = pickle.load(fh)
    log.info("Loaded model artifact from %s", path)
    return artifact


def latest_model_path(artifacts_dir: str | Path = "models") -> Path | None:
    """Return the most recently modified .pkl file in the artifacts dir."""
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        return None
    pkls = list(artifacts_dir.glob("*.pkl"))
    if not pkls:
        return None
    return max(pkls, key=lambda p: p.stat().st_mtime)