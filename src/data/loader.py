"""Dataset loading and synthetic generation.

Two entry points:

* :func:`load_dataset` reads a CSV (real or synthetic) and validates it.
* :func:`generate_synthetic_dataset` produces a reproducible synthetic dataset
  that shares the Cleveland heart disease schema.  It is used automatically
  when no external dataset is supplied, and is documented as synthetic in the
  UI so users are never misled into thinking it is real patient data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FEATURE_METADATA,
    validate_schema,
)
from ..utils.logging import get_logger

log = get_logger("data.loader")


def load_dataset(
    path: str | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load a heart-disease CSV and validate its schema.

    Parameters
    ----------
    path:
        Filesystem path to a CSV.  When ``None`` a reproducible synthetic
        dataset is generated instead (see :func:`generate_synthetic_dataset`).
    random_seed:
        Seed forwarded to the synthetic generator when ``path`` is ``None``.
    """
    if path is None or str(path).strip() == "":
        log.info("No dataset path supplied; generating synthetic dataset.")
        return generate_synthetic_dataset(seed=random_seed)

    df = pd.read_csv(path)
    log.info("Loaded dataset from %s (%d rows, %d columns)", path, *df.shape)
    validate_schema(df.columns)
    return df


def generate_synthetic_dataset(n_samples: int = 303, seed: int = 42,
                               positive_rate: float = 0.45,
                               noise: float = 0.35) -> pd.DataFrame:
    """Generate a reproducible synthetic heart-disease dataset.

    The generator models the *relationship* between features and the target
    so that a learnable signal exists, but the rows themselves are simulated.
    The dataset is clearly labelled synthetic in the UI and documentation.

    Parameters
    ----------
    n_samples:
        Number of rows to generate.
    seed:
        Random seed for full reproducibility.
    positive_rate:
        Approximate fraction of rows with ``target == 1``.
    noise:
        Standard deviation of Gaussian noise added to continuous features.
    """
    rng = np.random.default_rng(seed)
    n_positive = int(round(n_samples * positive_rate))
    n_negative = n_samples - n_positive

    def _sample(n: int, label: int) -> pd.DataFrame:
        rows = {}
        for name in FEATURE_COLUMNS:
            meta = FEATURE_METADATA[name]
            if meta.kind == "categorical":
                # Categorical features are drawn from the allowed set, biased
                # towards values more common in the positive class.
                base = np.array(meta.allowed_values, dtype=float)
                weights = np.ones_like(base)
                if label == 1:
                    weights = weights * np.array([1.6, 1.3, 1.1, 1.0])[: len(base)]
                weights = weights / weights.sum()
                rows[name] = rng.choice(base, size=n, p=weights)
            else:
                lo = meta.min_value if meta.min_value is not None else 0.0
                hi = meta.max_value if meta.max_value is not None else 100.0
                centre = (lo + hi) / 2.0
                # Shift the distribution towards the positive class.
                shift = (hi - lo) * 0.12 * label
                values = rng.normal(loc=centre + shift, scale=(hi - lo) * 0.18,
                                    size=n)
                rows[name] = np.clip(values, lo, hi)
        return pd.DataFrame(rows)

    pos = _sample(n_positive, label=1)
    neg = _sample(n_negative, label=0)
    df = pd.concat([pos, neg], ignore_index=True)
    df[TARGET_COLUMN] = np.concatenate([
        np.ones(n_positive, dtype=int),
        np.zeros(n_negative, dtype=int),
    ])

    # Add measurement noise to continuous features so the problem is not
    # perfectly separable (a realistic learning scenario).
    for name in FEATURE_COLUMNS:
        meta = FEATURE_METADATA[name]
        if meta.kind == "numeric":
            df[name] = df[name] + rng.normal(0.0, noise, size=len(df))
            lo = meta.min_value if meta.min_value is not None else -np.inf
            hi = meta.max_value if meta.max_value is not None else np.inf
            df[name] = df[name].clip(lo, hi)

    # Shuffle so positive and negative rows are interleaved.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df