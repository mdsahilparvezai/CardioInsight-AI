"""Input validation for the prediction form.

Validates that a user-supplied feature dictionary matches the expected schema
before it reaches the model.  Returns structured diagnostics so the UI can
show precise, actionable error messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..data.schema import FEATURE_METADATA, TARGET_COLUMN
from ..utils.logging import get_logger

log = get_logger("features.validation")


@dataclass
class ValidationResult:
    """Result of validating a single prediction input."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    coerced: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": list(self.errors),
                "coerced": dict(self.coerced)}


def _coerce_value(meta, raw: Any) -> tuple[Any, str | None]:
    """Coerce a raw input to the feature's expected type.

    Returns ``(value, error_message)`` where ``error_message`` is ``None`` on
    success.  Never raises.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None, f"{meta.label} is required."
    try:
        if meta.kind == "categorical":
            value = int(float(raw))
            if value not in meta.allowed_values:
                return None, (
                    f"{meta.label} must be one of {list(meta.allowed_values)}, "
                    f"got {raw!r}."
                )
            return value, None
        value = float(raw)
        if meta.min_value is not None and value < meta.min_value:
            return None, (
                f"{meta.label} must be >= {meta.min_value}, got {value}."
            )
        if meta.max_value is not None and value > meta.max_value:
            return None, (
                f"{meta.label} must be <= {meta.max_value}, got {value}."
            )
        return value, None
    except (TypeError, ValueError):
        return None, f"{meta.label} must be a valid {meta.kind} value."


def validate_inference_input(values: dict[str, Any]) -> ValidationResult:
    """Validate a user-supplied feature dictionary.

    ``values`` may use either feature names or human-readable labels as keys.
    Missing keys, out-of-range values, and type errors are all reported.
    """
    result = ValidationResult(valid=True)

    # Allow lookups by label as well as by name.
    label_to_name = {m.label: name for name, m in FEATURE_METADATA.items()}
    normalised: dict[str, Any] = {}
    provided_keys: set[str] = set()
    for key, raw in values.items():
        name = key
        if key not in FEATURE_METADATA and key in label_to_name:
            name = label_to_name[key]
        if name not in FEATURE_METADATA:
            result.errors.append(f"Unknown feature: {key!r}.")
            continue
        provided_keys.add(name)
        coerced, err = _coerce_value(FEATURE_METADATA[name], raw)
        if err:
            result.errors.append(err)
        else:
            normalised[name] = coerced

    # Check for features the user never supplied at all.
    for name in FEATURE_METADATA:
        if name not in provided_keys:
            result.errors.append(
                f"{FEATURE_METADATA[name].label} ({name}) is required."
            )

    result.coerced = normalised
    result.valid = not result.errors
    return result