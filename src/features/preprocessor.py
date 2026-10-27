"""Feature preprocessing pipelines.

Training and inference MUST use the exact same preprocessing.  To guarantee
that, the pipeline is fit once on training data, serialized, and reloaded for
inference.  This module exposes the build/fit/transform API; the serialization
logic lives in ``src/models/io.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from ..utils.logging import get_logger

log = get_logger("features.preprocessor")


def build_preprocessor():
    """Build a scikit-learn preprocessing pipeline.

    Returns ``None`` when scikit-learn is unavailable; the caller must then
    use the pure-NumPy fallback in ``src/models/engines.py``.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
    except ImportError:
        log.warning("scikit-learn not available; preprocessing pipeline disabled.")
        return None

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(NUMERIC_FEATURES)),
            ("cat", categorical_transformer, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
    return preprocessor


def fit_preprocessor(preprocessor, X: pd.DataFrame):
    """Fit a preprocessing pipeline on training data only."""
    if preprocessor is None:
        raise RuntimeError("No preprocessor available; install scikit-learn.")
    preprocessor.fit(X)
    return preprocessor


def transform_features(preprocessor, X: pd.DataFrame) -> np.ndarray:
    """Apply a fitted preprocessor to a feature frame."""
    if preprocessor is None:
        raise RuntimeError("No preprocessor available; install scikit-learn.")
    return preprocessor.transform(X)