"""Data layer: schema, loading, validation, synthetic generation, EDA."""

from .schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_METADATA,
    validate_schema,
)
from .loader import load_dataset, generate_synthetic_dataset
from .validator import validate_dataset
from .eda import run_eda

__all__ = [
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "FEATURE_METADATA",
    "validate_schema",
    "load_dataset",
    "generate_synthetic_dataset",
    "validate_dataset",
    "run_eda",
]