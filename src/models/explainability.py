"""Explainability utilities: SHAP integration with graceful fallback.

SHAP is an optional heavy dependency.  This module provides a unified API that
works whether `shap` is installed or not.  When unavailable, the UI displays
an informative message instead of crashing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..utils.logging import get_logger

log = get_logger("models.explainability")


def _check_shap() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


HAS_SHAP = _check_shap()


def compute_shap_values(model: Any, X: np.ndarray,
                        feature_names: list[str] | None = None) -> dict:
    """Compute SHAP values for a model.

    Returns a dict with keys:
    - "values": SHAP values array (n_samples, n_features)
    - "base_value": expected model output
    - "feature_names": list of feature names
    - "available": bool (False if shap not installed)
    """
    if not HAS_SHAP:
        log.warning("SHAP not installed; explainability unavailable.")
        return {
            "values": None,
            "base_value": None,
            "feature_names": feature_names or [],
            "available": False,
            "message": "Install 'shap' package to enable SHAP explanations.",
        }

    import shap

    # Try tree explainer first (fastest for tree models)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # shap_values can be list [class0, class1] or array (n_samples, n_features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
        return {
            "values": shap_values,
            "base_value": explainer.expected_value,
            "feature_names": feature_names or [f"f{i}" for i in range(X.shape[1])],
            "available": True,
        }
    except Exception:
        # Fall back to kernel explainer (model-agnostic but slow)
        try:
            explainer = shap.KernelExplainer(model.predict_proba, X[:50])  # background sample
            shap_values = explainer.shap_values(X[:20])  # limit for speed
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            return {
                "values": shap_values,
                "base_value": explainer.expected_value,
                "feature_names": feature_names or [f"f{i}" for i in range(X.shape[1])],
                "available": True,
            }
        except Exception as exc:
            log.error("SHAP computation failed: %s", exc)
            return {
                "values": None,
                "base_value": None,
                "feature_names": feature_names or [],
                "available": False,
                "message": f"SHAP computation failed: {exc}",
            }


def shap_summary_plot(shap_result: dict, max_display: int = 10) -> Any:
    """Generate a SHAP summary (beeswarm) plot. Returns a matplotlib figure or None."""
    if not HAS_SHAP or not shap_result["available"]:
        return None

    import shap
    import matplotlib.pyplot as plt

    values = shap_result["values"]
    feature_names = shap_result["feature_names"]

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        values,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.tight_layout()
    return fig


def shap_force_plot(shap_result: dict, sample_idx: int = 0) -> Any:
    """Generate a SHAP force plot for one sample. Returns HTML string or None."""
    if not HAS_SHAP or not shap_result["available"]:
        return None

    import shap

    values = shap_result["values"]
    base_value = shap_result["base_value"]
    feature_names = shap_result["feature_names"]

    if values is None or len(values) <= sample_idx:
        return None

    return shap.force_plot(
        base_value,
        values[sample_idx],
        feature_names=feature_names,
        matplotlib=False,
    )


def shap_dependence_plot(shap_result: dict, feature_idx: int,
                          interaction_idx: int | None = None) -> Any:
    """Generate a SHAP dependence plot. Returns a matplotlib figure or None."""
    if not HAS_SHAP or not shap_result["available"]:
        return None

    import shap
    import matplotlib.pyplot as plt

    values = shap_result["values"]
    feature_names = shap_result["feature_names"]

    fig, ax = plt.subplots(figsize=(6, 4))
    shap.dependence_plot(
        feature_idx,
        values,
        features=feature_names,
        interaction_index=interaction_idx,
        show=False,
        ax=ax,
    )
    plt.tight_layout()
    return fig


def format_shap_explanation(shap_result: dict, sample_idx: int = 0,
                            top_k: int = 5) -> str:
    """Generate a human-readable text explanation from SHAP values."""
    if not shap_result["available"] or shap_result["values"] is None:
        return "SHAP explanation not available (install 'shap' package)."

    values = shap_result["values"]
    feature_names = shap_result["feature_names"]

    if len(values) <= sample_idx:
        return "Sample index out of range."

    sample_shap = values[sample_idx]
    # Get top-k features by absolute SHAP value
    top_indices = np.argsort(np.abs(sample_shap))[-top_k:][::-1]

    lines = [f"SHAP explanation for sample {sample_idx}:"]
    for idx in top_indices:
        fname = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
        sval = sample_shap[idx]
        direction = "increases" if sval > 0 else "decreases"
        lines.append(f"  • {fname}: {sval:+.4f} (SHAP) → {direction} disease probability")

    base = shap_result["base_value"]
    if isinstance(base, (list, np.ndarray)):
        base = base[1] if len(base) > 1 else base[0]
    lines.append(f"  Base probability (E[model]): {base:.4f}")
    lines.append(f"  Model output: {base + sample_shap.sum():.4f}")

    return "\n".join(lines)