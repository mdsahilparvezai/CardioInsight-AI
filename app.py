#!/usr/bin/env python
"""CardioInsight AI - Streamlit dashboard.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_dataset, validate_dataset, run_eda
from src.features.validation import validate_inference_input
from src.models.io import load_artifacts, latest_model_path
from src.prediction import Predictor
from src.utils import get_config, reset_config, get_logger
from src.visualization import (
    plot_feature_distribution,
    plot_correlation_heatmap,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    plot_calibration,
    plot_metric_comparison,
)

log = get_logger("app")


# ---------------------------------------------------------------------------
# Page configuration & global style
# ---------------------------------------------------------------------------
reset_config()
cfg = get_config()

st.set_page_config(
    page_title=cfg["app"].get("page_title", "CardioInsight AI"),
    page_icon=cfg["app"].get("page_icon", "❤️"),
    layout=cfg["app"].get("layout", "wide"),
    initial_sidebar_state="expanded",
)

DISCLAIMER = cfg["app"].get("disclaimer", "")

# Custom CSS for CardioInsight visual identity
st.markdown("""
<style>
:root {
    --navy: #0B1D3A;
    --teal: #00BFA6;
    --teal-light: #4FE0D1;
    --white: #FFFFFF;
    --light-gray: #F5F7FA;
    --med-gray: #D0D5DD;
    --dark-gray: #344054;
    --red: #E03E3E;
    --green: #10B981;
}
[data-testid="stAppViewContainer"] { background-color: var(--light-gray); }
[data-testid="stSidebar"] { background-color: var(--navy); color: var(--white); }
[data-testid="stSidebar"] * { color: var(--white) !important; }
.stButton>button {
    background-color: var(--teal);
    color: var(--white);
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: background-color 0.15s;
}
.stButton>button:hover { background-color: var(--teal-light); color: var(--navy); }
.stMetric { background: var(--white); border: 1px solid var(--med-gray); border-radius: 8px; padding: 1rem; }
.stAlert { border-radius: 8px; }
.disclaimer-banner {
    background: #FFF3CD;
    border: 1px solid #FFD43B;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    color: #856404;
}
.section-card {
    background: var(--white);
    border: 1px solid var(--med-gray);
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }
.metric-card { background: var(--white); border: 1px solid var(--med-gray); border-radius: 8px; padding: 1rem; text-align: center; }
.metric-value { font-size: 2rem; font-weight: 700; color: var(--navy); }
.metric-label { font-size: 0.85rem; color: var(--dark-gray); margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)


def render_disclaimer():
    """Render the persistent educational disclaimer."""
    st.markdown(
        f'<div class="disclaimer-banner">⚠️ <strong>Educational Use Only:</strong> {DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


def load_cached_data():
    """Load and cache the dataset + EDA results."""
    if "dataset" not in st.session_state:
        with st.spinner("Loading dataset..."):
            df = load_dataset(path=cfg["data"]["path"],
                              random_seed=cfg["project"]["random_seed"])
            st.session_state.dataset = df
            st.session_state.eda = run_eda(df)
            val_res = validate_dataset(df)
            st.session_state.validation = val_res
    return st.session_state.dataset, st.session_state.eda, st.session_state.validation


def load_cached_predictor():
    """Load and cache the predictor."""
    if "predictor" not in st.session_state:
        try:
            st.session_state.predictor = Predictor()
            st.session_state.predictor_loaded = True
        except FileNotFoundError:
            st.session_state.predictor = None
            st.session_state.predictor_loaded = False
    return st.session_state.predictor


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ❤️ CardioInsight AI")
    st.caption("Educational ML Research Platform")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔍 Data Explorer", "🧪 Model Lab",
         "🎯 Prediction Demo", "🔬 Explainability", "📋 Model Card"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"Version: 0.1.0")
    st.caption(f"Engine: {cfg['model']['engine']}")
    st.caption(f"Seed: {cfg['project']['random_seed']}")

    # Model status
    predictor = load_cached_predictor()
    if predictor:
        st.success(f"Model loaded: {predictor.metadata.get('model_type')}")
    else:
        st.warning("No trained model found")


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "📊 Overview":
    render_disclaimer()
    st.title("📊 Project Overview")
    st.markdown("""
CardioInsight AI is an **educational machine learning platform** that explores
patterns in the Cleveland heart disease dataset and demonstrates how predictive
models can classify records for research purposes.

**This is not a medical device.** It has not been validated for clinical use
and must never be used to make health decisions.
""")

    df, eda, val = load_cached_data()

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{eda['shape']['rows']}</div>
            <div class="metric-label">Records</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{eda['shape']['columns'] - 1}</div>
            <div class="metric-label">Features</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        pos = eda['target_distribution']['percentages'].get(1, 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{pos:.1f}%</div>
            <div class="metric-label">Positive Class</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        predictor = load_cached_predictor()
        model_name = predictor.metadata.get("model_type", "—") if predictor else "—"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{model_name}</div>
            <div class="metric-label">Best Model</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Dataset summary
    st.subheader("Dataset Summary")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            eda["summary_statistics"].round(3),
            use_container_width=True,
            height=350,
        )
    with col2:
        st.markdown("**Target Distribution**")
        tgt = eda["target_distribution"]
        st.bar_chart(pd.DataFrame({
            "Class": ["No Disease (0)", "Disease (1)"],
            "Count": [tgt["counts"].get(0, 0), tgt["counts"].get(1, 0)],
        }).set_index("Class"))

    # Pipeline status
    st.subheader("Pipeline Status")
    steps = [
        ("Data Ingestion", "✅ Complete", "Synthetic dataset (Cleveland schema)"),
        ("Schema Validation", "✅ Passed" if val.schema_ok else "❌ Failed", val.errors[0] if val.errors else "All columns present"),
        ("Missing Values", "✅ None" if not val.missing_values else f"⚠️ {val.missing_values}", ""),
        ("Duplicates", "✅ None" if val.duplicate_rows == 0 else f"⚠️ {val.duplicate_rows}", ""),
        ("Value Ranges", "✅ Valid" if not val.invalid_values else "⚠️ Issues", ""),
        ("Train/Val/Test Split", "✅ Complete", f"{cfg['data']['test_size']:.0%} test, {cfg['data']['val_size']:.0%} val"),
        ("Model Training", "✅ Complete" if predictor else "⏳ Not trained", f"Best: {model_name}"),
        ("Model Evaluation", "✅ Complete" if predictor else "⏳ Pending", "Test F1, AUC computed"),
    ]
    for name, status, detail in steps:
        c1, c2, c3 = st.columns([2, 2, 4])
        c1.markdown(f"**{name}**")
        c2.markdown(status)
        c3.caption(detail)


# ---------------------------------------------------------------------------
# Page: Data Explorer
# ---------------------------------------------------------------------------
elif page == "🔍 Data Explorer":
    render_disclaimer()
    st.title("🔍 Data Explorer")

    df, eda, val = load_cached_data()

    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Feature distributions
    st.subheader("Feature Distributions")
    feature = st.selectbox(
        "Select feature",
        options=list(eda["feature_distributions"].keys()),
        format_func=lambda f: eda["feature_distributions"][f]["label"],
    )
    fig = plot_feature_distribution(df, feature, dark=True)
    st.plotly_chart(fig, use_container_width=True)

    # Target distribution
    st.subheader("Target Class Distribution")
    tgt_df = pd.DataFrame({
        "Class": ["No Disease (0)", "Disease (1)"],
        "Count": [eda["target_distribution"]["counts"].get(0, 0),
                  eda["target_distribution"]["counts"].get(1, 0)],
        "Percentage": [eda["target_distribution"]["percentages"].get(0, 0),
                       eda["target_distribution"]["percentages"].get(1, 0)],
    })
    st.bar_chart(tgt_df.set_index("Class")["Count"])

    # Missing value analysis
    st.subheader("Missing Value Analysis")
    missing_df = eda["missing_values"]
    if missing_df.empty:
        st.success("No missing values detected.")
    else:
        st.dataframe(missing_df, use_container_width=True)

    # Correlation matrix
    st.subheader("Feature Correlation Matrix")
    corr = eda["correlation_matrix"]
    fig = plot_correlation_heatmap(corr, dark=True)
    st.plotly_chart(fig, use_container_width=True)

    # Interactive filters
    st.subheader("Interactive Filter")
    col1, col2 = st.columns(2)
    with col1:
        age_range = st.slider("Age range", int(df["age"].min()), int(df["age"].max()),
                              (int(df["age"].min()), int(df["age"].max())))
    with col2:
        target_filter = st.multiselect("Target class", [0, 1], default=[0, 1],
                                       format_func=lambda x: "No Disease" if x == 0 else "Disease")
    filtered = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])
                  & (df["target"].isin(target_filter))]
    st.caption(f"Showing {len(filtered)} of {len(df)} records")
    st.dataframe(filtered, use_container_width=True, height=300)


# ---------------------------------------------------------------------------
# Page: Model Lab
# ---------------------------------------------------------------------------
elif page == "🧪 Model Lab":
    render_disclaimer()
    st.title("🧪 Model Lab")

    predictor = load_cached_predictor()
    if not predictor:
        st.warning("No trained model found. Run `python train.py` first.")
        st.stop()

    # Load comparison table
    comp_path = Path("models/comparison.csv")
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
    else:
        st.warning("Comparison table not found. Re-run training.")
        st.stop()

    st.subheader("Model Comparison (Test Set)")
    fig = plot_metric_comparison(comp_df, dark=True)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed metrics table
    st.subheader("Detailed Metrics")
    display_cols = ["model_name", "split", "accuracy", "precision", "recall",
                    "specificity", "f1", "roc_auc"]
    st.dataframe(comp_df[display_cols].round(4), use_container_width=True)

    # Per-model drill-down
    st.subheader("Per-Model Details")
    model_name = st.selectbox("Select model", comp_df["model_name"].unique())
    row = comp_df[comp_df["model_name"] == model_name].iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{row['accuracy']:.3f}")
        st.metric("Precision", f"{row['precision']:.3f}")
    with col2:
        st.metric("Recall (Sensitivity)", f"{row['recall']:.3f}")
        st.metric("Specificity", f"{row['specificity']:.3f}")
    with col3:
        st.metric("F1 Score", f"{row['f1']:.3f}")
        st.metric("ROC-AUC", f"{row['roc_auc']:.3f}")

    # Confusion matrix
    cm = np.array(row["confusion_matrix"])
    fig = plot_confusion_matrix(cm, dark=True)
    st.plotly_chart(fig, use_container_width=True)

    # ROC & PR curves (only if we have test probas cached)
    st.caption("ROC/PR curves require stored probability scores; not available in quick mode.")

    # Experiment history (MLflow placeholder)
    st.subheader("Experiment Tracking")
    st.info("MLflow integration is optional. Enable `CARDIOINSIGHT_MLFLOW_ENABLED=true` to track experiments.")


# ---------------------------------------------------------------------------
# Page: Prediction Demo
# ---------------------------------------------------------------------------
elif page == "🎯 Prediction Demo":
    render_disclaimer()
    st.title("🎯 Prediction Demo")

    predictor = load_cached_predictor()
    if not predictor:
        st.warning("No trained model found. Run `python train.py` first.")
        st.stop()

    st.markdown("""
Enter patient features below to see a model prediction. **This is a demonstration
only — the output is not a diagnosis and must not be used for medical decisions.**
""")

    meta = predictor.get_feature_metadata()

    with st.form("prediction_form"):
        cols = st.columns(3)
        user_input = {}

        for i, (fname, fmeta) in enumerate(meta.items()):
            with cols[i % 3]:
                if fmeta["kind"] == "categorical":
                    allowed = fmeta["allowed_values"]
                    user_input[fname] = st.selectbox(
                        fmeta["label"],
                        options=allowed,
                        format_func=lambda x, m=fmeta: f"{x} ({m['description']})",
                        help=f"Unit: {fmeta['unit']}. {fmeta['description']}",
                    )
                else:
                    user_input[fname] = st.number_input(
                        fmeta["label"],
                        min_value=float(fmeta["min_value"]) if fmeta["min_value"] is not None else None,
                        max_value=float(fmeta["max_value"]) if fmeta["max_value"] is not None else None,
                        value=float((fmeta["min_value"] + fmeta["max_value"]) / 2) if fmeta["min_value"] is not None and fmeta["max_value"] is not None else 0.0,
                        step=1.0 if fmeta["unit"] in ("years", "mmHg", "mg/dL", "bpm", "count") else 0.1,
                        help=f"Unit: {fmeta['unit']}. {fmeta['description']}",
                    )

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        try:
            with st.spinner("Running inference..."):
                result = predictor.predict(user_input)

            st.divider()
            st.subheader("Prediction Result")

            # Result cards
            col1, col2, col3 = st.columns(3)
            with col1:
                color = "red" if result.predicted_class == 1 else "green"
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {color};">
                    <div class="metric-label">Predicted Class</div>
                    <div class="metric-value" style="color: {color};">{result.class_label}</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Probability (Disease)</div>
                    <div class="metric-value">{result.probability:.1%}</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model</div>
                    <div class="metric-value" style="font-size: 1.2rem;">{result.model_metadata['model_type']}</div>
                </div>""", unsafe_allow_html=True)

            # Limitations notice
            st.error("""
**⚠️ Important Limitations**
- This prediction is based on a synthetic educational dataset, not real patient data.
- The model has not been clinically validated and may produce inaccurate results.
- Feature importance does not imply causation.
- Consult a qualified healthcare professional for any health concerns.
""")

            # Input summary
            with st.expander("Input Features Used"):
                input_df = pd.DataFrame([
                    {"Feature": meta[f]["label"], "Value": v, "Unit": meta[f]["unit"]}
                    for f, v in result.input_features.items()
                ])
                st.dataframe(input_df, use_container_width=True)

        except ValueError as exc:
            st.error(f"Input validation failed: {exc}")


# ---------------------------------------------------------------------------
# Page: Explainability
# ---------------------------------------------------------------------------
elif page == "🔬 Explainability":
    render_disclaimer()
    st.title("🔬 Explainability")

    predictor = load_cached_predictor()
    if not predictor:
        st.warning("No trained model found. Run `python train.py` first.")
        st.stop()

    st.markdown("""
This page shows **global feature importance** for the trained model.
For tree-based models, importance is computed as the mean decrease in impurity
(MDI). For linear models, it uses absolute coefficient weights.

⚠️ **Important:** Feature importance and SHAP explanations describe *associations
learned by the model*, not causal relationships. They do not establish causation.
""")

    model = predictor.model
    feature_names = predictor.feature_names

    # Try to extract feature importance
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "model_") and hasattr(model.model_, "feature_importances_"):
        importance = model.model_.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).ravel()
    elif hasattr(model, "model_") and hasattr(model.model_, "coef_"):
        importance = np.abs(model.model_.coef_).ravel()

    if importance is not None and len(importance) == len(feature_names):
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance,
        }).sort_values("Importance", ascending=True)

        # Map to human-readable labels
        from src.data.schema import FEATURE_METADATA
        imp_df["Label"] = imp_df["Feature"].map(lambda f: FEATURE_METADATA[f].label)

        st.subheader("Global Feature Importance")
        fig = plot_metric_comparison(
            imp_df.rename(columns={"Label": "model_name", "Importance": "f1"}),
            metrics=["f1"],  # hack to reuse the bar chart
            dark=True,
        )
        fig.update_layout(title="Feature Importance (MDI / |coef|)",
                          xaxis_title="Importance", yaxis_title="Feature")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(imp_df[["Label", "Importance"]].round(4), use_container_width=True)
    else:
        st.warning("Feature importance not available for this model type.")

    # SHAP placeholder
    st.subheader("SHAP Explanations")
    st.info("""
SHAP (SHapley Additive exPlanations) provides per-sample feature attributions.
Install the `shap` package (`pip install shap`) to enable this feature.

The SHAP library is heavy (~200 MB) and optional.  When available, this page
will show:
- Summary plot (beeswarm) of SHAP values across the test set
- Force plot for individual predictions
- Dependence plots for top features
""")


# ---------------------------------------------------------------------------
# Page: Model Card
# ---------------------------------------------------------------------------
elif page == "📋 Model Card":
    render_disclaimer()
    st.title("📋 Model Card")

    predictor = load_cached_predictor()
    if not predictor:
        st.warning("No trained model found. Run `python train.py` first.")
        st.stop()

    meta = predictor.metadata

    st.subheader("Model Details")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Model Type:** {meta.get('model_type', 'Unknown')}")
        st.markdown(f"**Engine:** {meta.get('engine', 'Unknown')}")
        st.markdown(f"**Trained At:** {meta.get('trained_at', 'Unknown')}")
        st.markdown(f"**Random Seed:** {meta.get('random_seed', 'Unknown')}")
    with col2:
        st.markdown(f"**Training Samples:** {meta.get('n_train_samples', 'Unknown')}")
        st.markdown(f"**Features:** {meta.get('n_features', 'Unknown')}")
        test_f1 = meta.get("test_metrics", {}).get("f1")
        test_auc = meta.get("test_metrics", {}).get("roc_auc")
        if test_f1 is not None:
            st.markdown(f"**Test F1:** {test_f1:.3f}")
        if test_auc is not None:
            st.markdown(f"**Test ROC-AUC:** {test_auc:.3f}")

    st.divider()

    st.subheader("Dataset Provenance")
    st.markdown("""
- **Source:** Synthetic dataset generated to match the Cleveland heart disease schema
  (UCI Machine Learning Repository, Dataset #45).
- **Generation:** Documented in `src/data/loader.py::generate_synthetic_dataset`.
- **License:** Apache-2.0 (synthetic data).
- **Real data alternative:** Place a UCI Cleveland CSV at `data/heart_disease.csv`
  and set `CARDIOINSIGHT_DATA_PATH=data/heart_disease.csv`.
""")

    st.subheader("Intended Use")
    st.markdown("""
- **Intended:** Educational demonstration of ML pipelines, model comparison,
  and explainability for heart disease risk factor research.
- **Prohibited:** Clinical decision-making, patient diagnosis, treatment planning,
  or any use that could impact real health outcomes.
""")

    st.subheader("Training Methodology")
    st.markdown(f"""
- **Preprocessing:** StandardScaler (numeric) + OneHotEncoder (categorical)
- **Split:** Stratified train/val/test ({cfg['data']['test_size']:.0%}/{cfg['data']['val_size']:.0%}/{1-cfg['data']['test_size']-cfg['data']['val_size']:.0%})
- **Hyperparameter Tuning:** Grid search on validation set ({cfg['model']['cv_folds']}-fold CV)
- **Selection Metric:** {cfg['model']['selection_metric'].upper()}
- **Models Compared:** Logistic Regression, Random Forest, Gradient Boosting, Naive Bayes
""")

    st.subheader("Evaluation Results")
    if meta.get("test_metrics"):
        tm = meta["test_metrics"]
        st.json(tm)

    st.subheader("Known Limitations")
    st.markdown("""
1. **Small dataset** (n=303 synthetic) — high variance in estimates.
2. **Synthetic data** — does not reflect real patient distributions or clinical complexity.
3. **Class imbalance** (~45% positive) — may bias probability calibration.
4. **No external validation** — performance on real clinical data is unknown.
5. **Feature importance ≠ causation** — associations only.
6. **No temporal validation** — model may not generalize to future populations.
7. **Single-site schema** — Cleveland data may not represent global populations.
""")

    st.subheader("Bias & Generalizability Considerations")
    st.markdown("""
- The synthetic generator uses fixed random seeds; results are reproducible but
  do not capture real-world variability.
- Demographic features (age, sex) are present but the synthetic distribution
  may not match any real population.
- No sensitive attributes (race, ethnicity, socioeconomic status) are included,
  so bias along those axes cannot be assessed.
""")

    # Export model card
    if st.button("📥 Download Model Card (JSON)"):
        card = {
            "model_details": {
                "name": "CardioInsight AI",
                "version": "0.1.0",
                "type": meta.get("model_type"),
                "engine": meta.get("engine"),
                "trained_at": meta.get("trained_at"),
            },
            "dataset": {
                "name": "Cleveland Heart Disease (synthetic)",
                "source": "UCI ML Repository #45 (schema only)",
                "samples": meta.get("n_train_samples"),
                "features": meta.get("n_features"),
            },
            "evaluation": meta.get("test_metrics"),
            "limitations": [
                "Synthetic data only",
                "No clinical validation",
                "Small sample size",
                "No external validation",
            ],
            "ethical_considerations": [
                "Not for clinical use",
                "Educational demonstration only",
                "Feature importance ≠ causation",
            ],
        }
        st.download_button(
            "Download JSON",
            data=json.dumps(card, indent=2),
            file_name="model_card.json",
            mime="application/json",
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    """Streamlit entry point."""
    pass


if __name__ == "__main__":
    main()