"""Model engines, evaluation, and artifact serialization."""

from .engines import select_engine, available_engines, PARAM_GRIDS
from .evaluation import evaluate_model, compare_models, cross_validate
from .io import save_artifacts, load_artifacts, ModelMetadata
from .explainability import (
    compute_shap_values, shap_summary_plot, shap_force_plot,
    shap_dependence_plot, format_shap_explanation, HAS_SHAP,
)

__all__ = ["select_engine", "available_engines", "PARAM_GRIDS",
           "evaluate_model", "compare_models", "cross_validate",
           "save_artifacts", "load_artifacts", "ModelMetadata",
           "compute_shap_values", "shap_summary_plot", "shap_force_plot",
           "shap_dependence_plot", "format_shap_explanation", "HAS_SHAP"]