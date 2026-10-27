"""Chart helpers for the dashboard.

Built on matplotlib (static) and plotly (interactive).  Every function returns
a figure object; the Streamlit pages decide how to render them.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plotly is optional; the app gracefully degrades to matplotlib if unavailable.
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# Style constants (CardioInsight AI visual identity)
# ---------------------------------------------------------------------------
DARK_NAVY = "#0B1D3A"
TEAL = "#00BFA6"
TEAL_LIGHT = "#4FE0D1"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F7FA"
MED_GRAY = "#D0D5DD"
DARK_GRAY = "#344054"
RED = "#E03E3E"
GREEN = "#10B981"

DARK_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=DARK_NAVY,
        plot_bgcolor="#112648",
        font=dict(color=WHITE, family="Inter, system-ui, sans-serif"),
        colorway=[TEAL, TEAL_LIGHT, "#60A5FA", "#F59E0B", RED, GREEN],
        margin=dict(l=50, r=30, t=50, b=50),
        xaxis=dict(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F"),
        yaxis=dict(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F"),
    )
)

LIGHT_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=LIGHT_GRAY,
        plot_bgcolor=WHITE,
        font=dict(color=DARK_GRAY, family="Inter, system-ui, sans-serif"),
        colorway=[TEAL, "#0891B2", "#2563EB", "#D97706", RED, GREEN],
        margin=dict(l=50, r=30, t=50, b=50),
        xaxis=dict(gridcolor=MED_GRAY, zerolinecolor=MED_GRAY),
        yaxis=dict(gridcolor=MED_GRAY, zerolinecolor=MED_GRAY),
    )
)


def _apply_theme(fig: Any, dark: bool = True) -> Any:
    """Apply the CardioInsight theme to a plotly figure."""
    if _HAS_PLOTLY and hasattr(fig, "update_layout"):
        tmpl = DARK_TEMPLATE if dark else LIGHT_TEMPLATE
        fig.update_layout(tmpl["layout"])
    return fig


def plot_feature_distribution(
    df: pd.DataFrame,
    feature: str,
    target_col: str = "target",
    dark: bool = True,
    interactive: bool = True,
) -> Any:
    """Histogram / bar chart of one feature, split by target."""
    from ..data.schema import FEATURE_METADATA
    meta = FEATURE_METADATA.get(feature)
    title = f"{meta.label if meta else feature} Distribution"
    x_label = f"{meta.label if meta else feature} ({meta.unit})" if meta else feature

    if _HAS_PLOTLY and interactive:
        if meta and meta.kind == "numeric":
            fig = px.histogram(
                df, x=feature, color=target_col, nbins=30,
                color_discrete_map={0: GREEN, 1: RED},
                labels={feature: x_label, target_col: "Target"},
                title=title,
                opacity=0.75,
                barmode="overlay",
            )
        else:
            fig = px.histogram(
                df, x=feature, color=target_col,
                color_discrete_map={0: GREEN, 1: RED},
                labels={feature: x_label, target_col: "Target"},
                title=title,
                barmode="group",
            )
        return _apply_theme(fig, dark)

    # Matplotlib fallback
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {0: GREEN, 1: RED}
    if meta and meta.kind == "numeric":
        for cls in sorted(df[target_col].unique()):
            subset = df[df[target_col] == cls][feature].dropna()
            ax.hist(subset, bins=20, alpha=0.6, label=f"Target={cls}",
                    color=colors.get(cls, "gray"), density=True)
        ax.set_ylabel("Density")
    else:
        counts = df.groupby([feature, target_col]).size().unstack(fill_value=0)
        counts.plot(kind="bar", ax=ax, color=[colors.get(c, "gray") for c in counts.columns])
        ax.set_ylabel("Count")
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    dark: bool = True,
    interactive: bool = True,
) -> Any:
    """Correlation heatmap."""
    if _HAS_PLOTLY and interactive:
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Feature Correlation Matrix",
        )
        return _apply_theme(fig, dark)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    dark: bool = True,
    interactive: bool = True,
) -> Any:
    """2x2 confusion matrix heatmap."""
    labels = ["No Disease (0)", "Disease (1)"]
    if _HAS_PLOTLY and interactive:
        fig = px.imshow(
            cm, text_auto=True, aspect="auto",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels, y=labels,
            color_continuous_scale="Blues",
            title="Confusion Matrix",
        )
        return _apply_theme(fig, dark)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    dark: bool = True,
    interactive: bool = True,
) -> Any:
    """ROC curve."""
    if _HAS_PLOTLY and interactive:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})",
            line=dict(color=TEAL, width=3),
            fill="tozeroy", fillcolor=f"rgba(0,191,166,0.1)",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(color="gray", width=2, dash="dash"),
        ))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        return _apply_theme(fig, dark)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=TEAL, lw=3, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=2, ls="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color=TEAL)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_precision_recall(
    precision: np.ndarray,
    recall: np.ndarray,
    ap: float,
    dark: bool = True,
    interactive: bool = True,
) -> Any:
    """Precision-Recall curve."""
    if _HAS_PLOTLY and interactive:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall, y=precision, mode="lines",
            name=f"PR (AP={ap:.3f})",
            line=dict(color=TEAL, width=3),
            fill="tozerox", fillcolor=f"rgba(0,191,166,0.1)",
        ))
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
        )
        return _apply_theme(fig, dark)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=TEAL, lw=3, label=f"PR (AP={ap:.3f})")
    ax.fill_between(recall, precision, alpha=0.1, color=TEAL)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_calibration(
    cal_df: pd.DataFrame,
    dark: bool = True,
    interactive: bool = True,
) -> Any:
    """Calibration curve (reliability diagram)."""
    if _HAS_PLOTLY and interactive:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cal_df["mean_predicted"], y=cal_df["fraction_positive"],
            mode="lines+markers", name="Model",
            line=dict(color=TEAL, width=3), marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfectly Calibrated",
            line=dict(color="gray", width=2, dash="dash"),
        ))
        fig.update_layout(
            title="Calibration Curve (Reliability Diagram)",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
        )
        return _apply_theme(fig, dark)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(cal_df["mean_predicted"], cal_df["fraction_positive"],
            "o-", color=TEAL, lw=3, label="Model")
    ax.plot([0, 1], [0, 1], color="gray", lw=2, ls="--", label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_metric_comparison(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    dark: bool = True,
    interactive: bool = True,
) -> Any:
    """Grouped bar chart comparing models across metrics."""
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]

    plot_df = df.melt(id_vars=["model_name"], value_vars=metrics,
                      var_name="metric", value_name="score")

    if _HAS_PLOTLY and interactive:
        fig = px.bar(
            plot_df, x="model_name", y="score", color="metric",
            barmode="group", text_auto=".3f",
            title="Model Comparison on Test Set",
            labels={"model_name": "Model", "score": "Score", "metric": "Metric"},
            height=500,
        )
        fig.update_layout(yaxis=dict(range=[0, 1.05]))
        return _apply_theme(fig, dark)

    # Matplotlib grouped bar
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.8 / len(metrics)
    for i, m in enumerate(metrics):
        vals = df[m].values
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=m, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_name"], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison on Test Set")
    ax.legend()
    fig.tight_layout()
    return fig