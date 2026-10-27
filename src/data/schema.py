"""Dataset schema definitions for CardioInsight AI.

The schema mirrors the Cleveland heart disease dataset (UCI Machine Learning
Repository, "Heart Disease" dataset #45) so that a real copy of that CSV can be
dropped in without any code changes.  Every feature has a human-readable label,
a unit (where verified), and an allowed value range used for input validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


TARGET_COLUMN = "target"


@dataclass(frozen=True)
class FeatureMeta:
    """Metadata for a single model input feature."""

    name: str
    label: str
    unit: str
    kind: str  # "numeric" | "categorical"
    allowed_values: tuple = ()
    min_value: float | None = None
    max_value: float | None = None
    description: str = ""


# ---------------------------------------------------------------------------
# Feature definitions
#
# Ranges below are *structural* bounds derived from the dataset's documented
# feature types (e.g. age is a count in years, blood pressure is mmHg).  They
# are used to reject physically impossible inputs and to flag out-of-range
# values for review.  They are NOT clinical normal ranges, and the application
# never presents them as guidance for a personal health decision.
# ---------------------------------------------------------------------------
_FEATURES = [
    FeatureMeta("age", "Age", "years", "numeric", min_value=0, max_value=120,
                description="Age in years."),
    FeatureMeta("sex", "Sex", "code", "categorical", (0, 1),
                description="0 = female, 1 = male."),
    FeatureMeta("cp", "Chest Pain Type", "code", "categorical", (0, 1, 2, 3),
                description="Typical angina (0) to asymptomatic (3)."),
    FeatureMeta("trestbps", "Resting Blood Pressure", "mmHg", "numeric",
                min_value=50, max_value=250,
                description="Resting blood pressure (mmHg)."),
    FeatureMeta("chol", "Serum Cholesterol", "mg/dL", "numeric",
                min_value=50, max_value=700,
                description="Serum cholesterol in mg/dL."),
    FeatureMeta("fbs", "Fasting Blood Sugar", "code", "categorical", (0, 1),
                description="1 if fasting blood sugar > 120 mg/dL."),
    FeatureMeta("restecg", "Resting ECG", "code", "categorical", (0, 1, 2),
                description="Resting electrocardiographic results."),
    FeatureMeta("thalach", "Maximum Heart Rate", "bpm", "numeric",
                min_value=40, max_value=250,
                description="Maximum heart rate achieved."),
    FeatureMeta("exang", "Exercise Induced Angina", "code", "categorical", (0, 1),
                description="1 if exercise induced angina."),
    FeatureMeta("oldpeak", "ST Depression", "mV", "numeric",
                min_value=-5, max_value=10,
                description="ST depression induced by exercise relative to rest."),
    FeatureMeta("slope", "ST Slope", "code", "categorical", (1, 2, 3),
                description="Slope of the peak exercise ST segment."),
    FeatureMeta("ca", "Major Vessels Colored", "count", "numeric",
                min_value=0, max_value=4,
                description="Number of major vessels colored by fluoroscopy."),
    FeatureMeta("thal", "Thalassemia", "code", "categorical", (0, 1, 2, 3),
                description="Thalassemia blood disorder indicator."),
]

FEATURE_METADATA = {f.name: f for f in _FEATURES}
NUMERIC_FEATURES = tuple(f.name for f in _FEATURES if f.kind == "numeric")
CATEGORICAL_FEATURES = tuple(f.name for f in _FEATURES if f.kind == "categorical")
FEATURE_COLUMNS = tuple(f.name for f in _FEATURES)


def feature_names() -> tuple[str, ...]:
    """Return the ordered tuple of model input feature names."""
    return FEATURE_COLUMNS


def numeric_feature_names() -> tuple[str, ...]:
    """Return the ordered tuple of numeric feature names."""
    return NUMERIC_FEATURES


def categorical_feature_names() -> tuple[str, ...]:
    """Return the ordered tuple of categorical feature names."""
    return CATEGORICAL_FEATURES


def validate_schema(columns: Iterable[str]) -> None:
    """Raise ``ValueError`` if ``columns`` does not match the expected schema.

    The check is strict: every expected feature must be present, and the target
    column must be present.  Extra columns are reported but tolerated so that
    downstream code can drop them explicitly.
    """
    present = set(columns)
    missing = [f for f in FEATURE_COLUMNS if f not in present]
    if TARGET_COLUMN not in present:
        missing.append(TARGET_COLUMN)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Expected schema: {list(FEATURE_COLUMNS) + [TARGET_COLUMN]}."
        )
    extra = sorted(present - set(FEATURE_COLUMNS) - {TARGET_COLUMN})
    if extra:
        raise ValueError(
            f"Dataset contains unexpected columns not in the schema: {extra}. "
            f"Either drop them or add them to the schema intentionally."
        )