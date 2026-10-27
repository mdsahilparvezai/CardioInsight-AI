"""Exploratory data analysis helpers.

Produces the descriptive statistics and summary figures used by the Data
Explorer page.  All functions are pure: they take a DataFrame and return
data/figures; they never mutate their input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FEATURE_METADATA,
    NUMERIC_FEATURES,
)
from ..utils.logging import get_logger

log = get_logger("data.eda")


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``describe()`` plus the per-column missing-value count."""
    stats = df.describe(include="all").T
    stats["missing"] = df.isna().sum()
    stats["unique"] = df.nunique()
    return stats


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts, sorted descending, with a total row."""
    counts = df.isna().sum()
    out = counts[counts > 0].rename("missing_count").to_frame()
    out["column"] = out.index
    out = out[["column", "missing_count"]].sort_values(
        "missing_count", ascending=False
    )
    total = int(df.isna().sum().sum())
    if total:
        out.loc["TOTAL"] = {"column": "TOTAL", "missing_count": total}
    return out


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between numeric features and the target."""
    numeric_cols = list(NUMERIC_FEATURES)
    if TARGET_COLUMN in df.columns:
        numeric_cols = numeric_cols + [TARGET_COLUMN]
    available = [c for c in numeric_cols if c in df.columns]
    return df[available].corr(method="pearson")


def target_distribution(df: pd.DataFrame) -> dict[str, Any]:
    """Counts and percentages of the target classes."""
    if TARGET_COLUMN not in df.columns:
        return {"counts": {}, "percentages": {}, "n": 0}
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    total = int(counts.sum()) or 1
    return {
        "counts": {int(k): int(v) for k, v in counts.items()},
        "percentages": {int(k): round(100.0 * v / total, 2)
                        for k, v in counts.items()},
        "n": total,
    }


def feature_distribution(df: pd.DataFrame, feature: str) -> dict[str, Any]:
    """Summary statistics for one feature, split by target class when present."""
    if feature not in df.columns:
        raise KeyError(f"Unknown feature: {feature}")
    series = df[feature]
    out: dict[str, Any] = {
        "feature": feature,
        "label": FEATURE_METADATA[feature].label,
        "unit": FEATURE_METADATA[feature].unit,
        "kind": FEATURE_METADATA[feature].kind,
        "n": int(series.notna().sum()),
        "missing": int(series.isna().sum()),
        "unique": int(series.nunique()),
    }
    if FEATURE_METADATA[feature].kind == "numeric":
        out.update({
            "mean": float(series.mean()) if series.notna().any() else None,
            "std": float(series.std()) if series.notna().any() else None,
            "min": float(series.min()) if series.notna().any() else None,
            "max": float(series.max()) if series.notna().any() else None,
        })
    else:
        out["value_counts"] = {
            int(k): int(v) for k, v in series.value_counts().sort_index().items()
        }
    if TARGET_COLUMN in df.columns:
        grouped = df.groupby(TARGET_COLUMN)[feature].agg(
            ["mean", "std", "count"]
        )
        out["by_target"] = {
            int(k): {"mean": float(v["mean"]), "std": float(v["std"]),
                     "count": int(v["count"])}
            for k, v in grouped.iterrows()
        }
    return out


def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    """Run the full EDA and return a dictionary of results.

    The returned dict is JSON-serialisable and is what the Data Explorer page
    consumes.  Figures are produced separately by the visualization module.
    """
    log.info("Running EDA on %d rows, %d columns.", *df.shape)
    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "summary_statistics": summary_statistics(df),
        "missing_values": missing_value_summary(df),
        "correlation_matrix": correlation_matrix(df),
        "target_distribution": target_distribution(df),
        "feature_distributions": {
            f: feature_distribution(df, f) for f in FEATURE_COLUMNS
        },
    }