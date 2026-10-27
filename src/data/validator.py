"""Dataset validation: schema, missing values, duplicates, and value ranges.

Every check returns structured, machine-readable diagnostics so the UI can
surface them without re-implementing the logic.  Validation never modifies
the input frame; it only inspects it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FEATURE_METADATA,
    validate_schema,
)
from ..utils.logging import get_logger

log = get_logger("data.validator")


@dataclass
class ValidationResult:
    """Structured output of :func:`validate_dataset`."""

    schema_ok: bool
    missing_values: dict[str, int] = field(default_factory=dict)
    invalid_values: dict[str, list[Any]] = field(default_factory=dict)
    duplicate_rows: int = 0
    target_distribution: dict[int, int] = field(default_factory=dict)
    target_imbalance_ratio: float = 1.0
    n_rows: int = 0
    n_columns: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.schema_ok and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_ok": self.schema_ok,
            "missing_values": dict(self.missing_values),
            "invalid_values": {str(k): v for k, v in self.invalid_values.items()},
            "duplicate_rows": self.duplicate_rows,
            "target_distribution": dict(self.target_distribution),
            "target_imbalance_ratio": self.target_imbalance_ratio,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "errors": list(self.errors),
        }


def _check_missing(df: pd.DataFrame) -> dict[str, int]:
    missing = df.isna().sum()
    return {col: int(count) for col, count in missing.items() if count > 0}


def _check_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def _check_target(df: pd.DataFrame) -> tuple[dict[int, int], float]:
    counts = df[TARGET_COLUMN].value_counts().to_dict()
    counts = {int(k): int(v) for k, v in counts.items()}
    total = sum(counts.values()) or 1
    minority = min(counts.values()) if counts else 0
    return counts, minority / total


def _check_value_ranges(df: pd.DataFrame) -> dict[str, list[Any]]:
    """Check that categorical values are allowed and numerics are in range."""
    problems: dict[str, list[Any]] = {}
    for name, meta in FEATURE_METADATA.items():
        if name not in df.columns:
            continue
        series = df[name]
        bad: list[Any] = []
        if meta.kind == "categorical":
            allowed = set(meta.allowed_values)
            bad = [v for v in series.unique() if v not in allowed]
        else:
            lo = meta.min_value
            hi = meta.max_value
            if lo is not None:
                bad += [v for v in series.unique() if pd.notna(v) and v < lo]
            if hi is not None:
                bad += [v for v in series.unique() if pd.notna(v) and v > hi]
        if bad:
            problems[name] = sorted(set(bad))
    return problems


def validate_dataset(df: pd.DataFrame) -> ValidationResult:
    """Run all dataset checks and return a :class:`ValidationResult`.

    The schema check raises ``ValueError`` on failure; that exception is
    caught and recorded so callers always receive a structured result.
    """
    result = ValidationResult(
        schema_ok=False,
        n_rows=int(len(df)),
        n_columns=int(df.shape[1]),
    )

    try:
        validate_schema(df.columns)
        result.schema_ok = True
    except ValueError as exc:
        result.errors.append(f"Schema validation failed: {exc}")

    result.missing_values = _check_missing(df)
    if result.missing_values:
        result.errors.append(
            f"Missing values detected: {result.missing_values}"
        )

    result.invalid_values = _check_value_ranges(df)
    if result.invalid_values:
        result.errors.append(
            f"Invalid/out-of-range values detected: {result.invalid_values}"
        )

    result.duplicate_rows = _check_duplicates(df)
    if result.duplicate_rows:
        result.errors.append(
            f"{result.duplicate_rows} duplicate row(s) detected."
        )

    try:
        dist, ratio = _check_target(df)
        result.target_distribution = dist
        result.target_imbalance_ratio = ratio
        if TARGET_COLUMN not in df.columns:
            result.errors.append(f"Target column '{TARGET_COLUMN}' missing.")
    except Exception as exc:  # pragma: no cover - defensive
        result.errors.append(f"Target analysis failed: {exc}")

    if not result.is_valid:
        log.warning("Dataset validation produced %d issue(s).", len(result.errors))
    else:
        log.info("Dataset validation passed (%d rows, %d columns).",
                 result.n_rows, result.n_columns)
    return result