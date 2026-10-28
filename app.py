#!/usr/bin/env python
"""CardioInsight AI - Premium Streamlit Dashboard.

Advanced, Explainable Heart Disease Risk Research Platform.
Run with: streamlit run app.py
"""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path
from typing import Any

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
    plot_metric_comparison,
)

log = get_logger("app")

# ---------------------------------------------------------------------------
# Page configuration & premium theme
# ---------------------------------------------------------------------------
reset_config()
cfg = get_config()

st.set_page_config(
    page_title="CardioInsight AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "CardioInsight AI v0.1.0 — Educational ML Research Platform",
    },
)

DISCLAIMER = cfg["app"].get("disclaimer", "")

# ---------------------------------------------------------------------------
# Premium CSS — Healthcare Research Aesthetic
# ---------------------------------------------------------------------------
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    /* Light theme (default) */
    --bg-primary: #F8FAFC;
    --bg-secondary: #FFFFFF;
    --bg-tertiary: #F1F5F9;
    --bg-card: rgba(255, 255, 255, 0.85);
    --bg-card-hover: rgba(255, 255, 255, 0.95);
    --border-primary: #E2E8F0;
    --border-accent: #00BFA6;
    --text-primary: #0F172A;
    --text-secondary: #475569;
    --text-muted: #94A3B8;
    --accent-primary: #00BFA6;
    --accent-light: #4FE0D1;
    --accent-dark: #009982;
    --accent-glow: rgba(0, 191, 166, 0.15);
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
    --info: #3B82F6;
    --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.05);
    --shadow-md: 0 4px 12px rgba(15, 23, 42, 0.08);
    --shadow-lg: 0 12px 32px rgba(15, 23, 42, 0.12);
    --shadow-glow: 0 0 24px rgba(0, 191, 166, 0.2);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 24px;
    --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-normal: 250ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-slow: 400ms cubic-bezier(0.4, 0, 0.2, 1);
}

/* Dark theme */
[data-theme="dark"] {
    --bg-primary: #0B1D3A;
    --bg-secondary: #112648;
    --bg-tertiary: #153058;
    --bg-card: rgba(17, 38, 72, 0.85);
    --bg-card-hover: rgba(21, 48, 88, 0.95);
    --border-primary: #1E3A5F;
    --border-accent: #00BFA6;
    --text-primary: #F8FAFC;
    --text-secondary: #CBD5E1;
    --text-muted: #64748B;
    --accent-glow: rgba(0, 191, 166, 0.25);
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.2);
    --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.3);
    --shadow-lg: 0 12px 32px rgba(0, 0, 0, 0.4);
    --shadow-glow: 0 0 32px rgba(0, 191, 166, 0.3);
}

/* Global reset */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Smooth transitions for theme switching */
*, *::before, *::after {
    transition: background-color var(--transition-normal),
                border-color var(--transition-normal),
                color var(--transition-normal),
                box-shadow var(--transition-normal) !important;
}

/* Main container */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* Sidebar premium styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
    border-right: 1px solid var(--border-primary) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Glassmorphism card */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    transition: all var(--transition-normal);
}
.glass-card:hover {
    background: var(--bg-card-hover);
    box-shadow: var(--shadow-lg);
    border-color: var(--border-accent);
    transform: translateY(-2px);
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    text-align: center;
    transition: all var(--transition-normal);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-primary), var(--accent-light));
    opacity: 0;
    transition: opacity var(--transition-fast);
}
.metric-card:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow-lg), var(--shadow-glow);
    transform: translateY(-4px);
}
.metric-card:hover::before { opacity: 1; }
.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-dark)) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: var(--shadow-md), var(--shadow-glow) !important;
    transition: all var(--transition-fast) !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left var(--transition-normal);
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-lg), var(--shadow-glow) !important;
}
.stButton > button:hover::before { left: 100%; }
.stButton > button:active { transform: translateY(0) !important; }

/* Primary button variant */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-dark)) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-primary) !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--bg-card) !important;
    border-color: var(--border-accent) !important;
}

/* Form inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div,
.stSlider > div > div > div {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-primary) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    transition: all var(--transition-fast) !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stSelectbox > div > div > div:focus-within {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-primary) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: var(--bg-tertiary);
    padding: 0.5rem;
    border-radius: var(--radius-md);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    padding: 0.75rem 1.5rem !important;
    transition: all var(--transition-fast) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-primary) !important;
    color: white !important;
    box-shadow: var(--shadow-md) !important;
}

/* Alerts */
.stAlert {
    border-radius: var(--radius-md) !important;
    border: none !important;
    background: var(--bg-card) !important;
    backdrop-filter: blur(20px) !important;
}

/* Progress spinner */
.stSpinner > div {
    border-top-color: var(--accent-primary) !important;
}

/* Chatbot styles */
.chatbot-container {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    z-index: 1000;
}
.chatbot-toggle {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-dark));
    border: none;
    color: white;
    font-size: 1.5rem;
    cursor: pointer;
    box-shadow: var(--shadow-lg), var(--shadow-glow);
    transition: all var(--transition-normal);
    display: flex;
    align-items: center;
    justify-content: center;
}
.chatbot-toggle:hover {
    transform: scale(1.1) rotate(15deg);
    box-shadow: var(--shadow-lg), 0 0 40px rgba(0, 191, 166, 0.4);
}
.chatbot-panel {
    position: absolute;
    bottom: 80px;
    right: 0;
    width: 380px;
    max-height: 500px;
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-lg);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    animation: slideUp 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.chatbot-header {
    padding: 1rem 1.25rem;
    border-bottom: 1px solid var(--border-primary);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-tertiary);
}
.chatbot-header h4 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
}
.chatbot-close {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    padding: 0.25rem;
    border-radius: var(--radius-sm);
    transition: all var(--transition-fast);
}
.chatbot-close:hover {
    background: var(--bg-primary);
    color: var(--text-primary);
}
.chatbot-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}
.chatbot-message {
    max-width: 85%;
    padding: 0.75rem 1rem;
    border-radius: var(--radius-md);
    font-size: 0.9rem;
    line-height: 1.5;
    animation: fadeIn 0.2s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.chatbot-message.user {
    align-self: flex-end;
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-dark));
    color: white;
    border-bottom-right-radius: 4px;
}
.chatbot-message.assistant {
    align-self: flex-start;
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border-primary);
    border-bottom-left-radius: 4px;
}
.chatbot-message.typing {
    color: var(--text-muted);
    font-style: italic;
}
.chatbot-input-area {
    padding: 1rem;
    border-top: 1px solid var(--border-primary);
    background: var(--bg-tertiary);
}
.chatbot-input {
    width: 100%;
    padding: 0.75rem 1rem;
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-md);
    background: var(--bg-secondary);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    resize: none;
    min-height: 48px;
    max-height: 120px;
    transition: all var(--transition-fast);
}
.chatbot-input:focus {
    outline: none;
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 3px var(--accent-glow);
}
.chatbot-send {
    margin-top: 0.5rem;
    width: 100%;
}

/* Section cards */
.section-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: all var(--transition-normal);
}
.section-card:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow-md);
}

/* Disclaimer banner */
.disclaimer-banner {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05));
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    font-size: 0.9rem;
    color: var(--text-secondary);
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    animation: slideDown 0.4s ease-out;
}
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}
.disclaimer-banner svg { flex-shrink: 0; margin-top: 0.125rem; }

/* Page header */
.page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border-primary);
}
.page-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    background: linear-gradient(135deg, var(--text-primary), var(--accent-primary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.page-subtitle {
    color: var(--text-muted);
    font-size: 1rem;
    margin: 0.25rem 0 0 0;
}

/* Badge */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-success { background: rgba(16, 185, 129, 0.15); color: var(--success); }
.badge-warning { background: rgba(245, 158, 11, 0.15); color: var(--warning); }
.badge-danger { background: rgba(239, 68, 68, 0.15); color: var(--danger); }
.badge-info { background: rgba(59, 130, 246, 0.15); color: var(--info); }
.badge-primary { background: rgba(0, 191, 166, 0.15); color: var(--accent-primary); }

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-primary), transparent);
    margin: 1.5rem 0;
}

/* Animation utilities */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
.animate-pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
.animate-spin { animation: spin 1s linear infinite; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-tertiary); }
::-webkit-scrollbar-thumb { background: var(--border-primary); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }

/* Tooltip */
.tooltip { position: relative; }
.tooltip::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    padding: 0.5rem 0.75rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
    color: var(--text-secondary);
    white-space: nowrap;
    opacity: 0;
    visibility: hidden;
    transition: all var(--transition-fast);
    z-index: 100;
    box-shadow: var(--shadow-lg);
}
.tooltip:hover::after { opacity: 1; visibility: visible; transform: translateX(-50%) translateY(-4px); }

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Theme toggle state
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ---------------------------------------------------------------------------
# Chatbot Assistant
# ---------------------------------------------------------------------------
class ChatbotAssistant:
    """AI-powered chatbot for CardioInsight AI assistance."""

    def __init__(self):
        self.knowledge_base = {
            "project": "CardioInsight AI is an educational ML platform for heart disease risk research using the Cleveland dataset schema.",
            "models": "We compare Logistic Regression, Random Forest, Gradient Boosting, and Naive Bayes with hyperparameter tuning.",
            "data": "Uses synthetic data matching Cleveland heart disease schema (UCI ML Repository #45). Real data can be added via CARDIOINSIGHT_DATA_PATH.",
            "prediction": "The Prediction Demo page lets you enter patient features and see model predictions with probabilities.",
            "explainability": "Feature importance (MDI) and SHAP explanations show which features influence predictions.",
            "limitations": "This is educational only — not for clinical use. Synthetic data, small sample, no external validation.",
            "metrics": "We evaluate Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, Confusion Matrix, and Calibration.",
            "training": "Run 'python train.py --engine numpy --quick' for fast training, or 'python train.py --engine sklearn' for full sklearn training.",
            "deployment": "Docker support: 'docker build -t cardioinsight-ai .' then 'docker run -p 8501:8501 cardioinsight-ai'",
        }

    def get_response(self, user_message: str) -> str:
        """Generate response based on keywords in user message."""
        msg = user_message.lower()

        # Greeting
        if any(w in msg for w in ["hi", "hello", "hey", "start"]):
            return "👋 Hello! I'm your CardioInsight AI assistant. I can help you navigate the platform, explain features, or answer questions about the models and data. What would you like to know?"

        # Project overview
        if any(w in msg for w in ["what is", "about", "project", "overview"]):
            return self.knowledge_base["project"]

        # Models
        if any(w in msg for w in ["model", "algorithm", "random forest", "logistic", "gradient", "naive bayes"]):
            return self.knowledge_base["models"]

        # Data
        if any(w in msg for w in ["data", "dataset", "cleveland", "uci", "synthetic"]):
            return self.knowledge_base["data"]

        # Prediction
        if any(w in msg for w in ["predict", "prediction", "demo", "patient", "input"]):
            return self.knowledge_base["prediction"]

        # Explainability
        if any(w in msg for w in ["explain", "shap", "feature importance", "importance", "why"]):
            return self.knowledge_base["explainability"]

        # Limitations
        if any(w in msg for w in ["limitation", "caution", "warning", "clinical", "medical", "diagnos"]):
            return self.knowledge_base["limitations"]

        # Metrics
        if any(w in msg for w in ["metric", "accuracy", "precision", "recall", "f1", "auc", "roc"]):
            return self.knowledge_base["metrics"]

        # Training
        if any(w in msg for w in ["train", "training", "run", "engine", "sklearn", "numpy"]):
            return self.knowledge_base["training"]

        # Deployment
        if any(w in msg for w in ["deploy", "docker", "container", "production"]):
            return self.knowledge_base["deployment"]

        # Default helpful response
        return (
            "I'm here to help with CardioInsight AI! You can ask me about:\n"
            "• **Project overview** — What this platform does\n"
            "• **Models** — Random Forest, Gradient Boosting, Logistic Regression, Naive Bayes\n"
            "• **Data** — Synthetic Cleveland schema, real data options\n"
            "• **Predictions** — How to use the Prediction Demo\n"
            "• **Explainability** — Feature importance & SHAP\n"
            "• **Metrics** — Accuracy, F1, AUC, etc.\n"
            "• **Training** — How to run train.py\n"
            "• **Deployment** — Docker & CI/CD\n\n"
            "What would you like to explore?"
        )

# Initialize chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ChatbotAssistant()
if "chatbot_open" not in st.session_state:
    st.session_state.chatbot_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset():
    df = load_dataset(path=cfg["data"]["path"], random_seed=cfg["project"]["random_seed"])
    eda = run_eda(df)
    val = validate_dataset(df)
    return df, eda, val

@st.cache_resource(show_spinner=False)
def get_predictor():
    try:
        return Predictor()
    except FileNotFoundError:
        return None

# ---------------------------------------------------------------------------
# Header / Navigation
# ---------------------------------------------------------------------------
def render_header():
    """Render the premium header with theme toggle and chatbot."""
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="
                width: 48px; height: 48px;
                border-radius: 12px;
                background: linear-gradient(135deg, var(--accent-primary), var(--accent-dark));
                display: flex; align-items: center; justify-content: center;
                box-shadow: var(--shadow-md), var(--shadow-glow);
            ">
                <span style="font-size: 1.5rem;">❤️</span>
            </div>
            <div>
                <h1 style="margin: 0; font-size: 1.5rem; font-weight: 700; color: var(--text-primary);">
                    CardioInsight AI
                </h1>
                <p style="margin: 0; font-size: 0.8rem; color: var(--text-muted);">
                    Educational ML Research Platform
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Theme toggle
        theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
        if st.button(theme_icon, key="theme_toggle", help="Toggle theme"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()

    with col3:
        # Chatbot toggle
        chat_icon = "💬" if not st.session_state.chatbot_open else "✕"
        if st.button(chat_icon, key="chatbot_toggle", help="AI Assistant"):
            st.session_state.chatbot_open = not st.session_state.chatbot_open
            st.rerun()

def render_chatbot():
    """Render the AI chatbot panel."""
    if not st.session_state.chatbot_open:
        return

    st.markdown("""
    <div class="chatbot-container">
        <div class="chatbot-panel">
            <div class="chatbot-header">
                <h4>🤖 AI Assistant</h4>
                <button class="chatbot-close" onclick="document.querySelector('[data-testid=chatbot_toggle]').click()">✕</button>
            </div>
            <div class="chatbot-messages">
    """, unsafe_allow_html=True)

    # Render chat history
    for msg in st.session_state.chat_history:
        role_class = "user" if msg["role"] == "user" else "assistant"
        st.markdown(f'<div class="chatbot-message {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

    # Show typing indicator
    if st.session_state.get("chatbot_typing", False):
        st.markdown('<div class="chatbot-message assistant typing">🤖 Thinking...</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Input area
    st.markdown('<div class="chatbot-input-area">', unsafe_allow_html=True)

    with st.form("chatbot_form", clear_on_submit=True):
        user_input = st.text_area(
            "Message",
            placeholder="Ask about models, data, predictions...",
            key="chatbot_input",
            label_visibility="collapsed",
            height=60,
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            send = st.form_submit_button("Send", use_container_width=True)
        with col2:
            clear = st.form_submit_button("Clear", use_container_width=True, type="secondary")

    st.markdown("</div></div></div>", unsafe_allow_html=True)

    if clear:
        st.session_state.chat_history = []
        st.rerun()

    if send and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        st.session_state.chatbot_typing = True
        st.rerun()

def process_chatbot():
    """Process chatbot response after user message."""
    if st.session_state.get("chatbot_typing", False):
        # Get the last user message
        user_msg = st.session_state.chat_history[-1]["content"]
        # Simulate thinking time
        time.sleep(0.5)
        # Get response
        response = st.session_state.chatbot.get_response(user_msg)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.chatbot_typing = False
        st.rerun()

def render_disclaimer():
    """Render the persistent educational disclaimer."""
    st.markdown(f"""
    <div class="disclaimer-banner">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
            <line x1="12" y1="9" x2="12" y2="13"></line>
            <line x1="12" y1="17" x2="12.01" y2="17"></line>
        </svg>
        <div>
            <strong>⚠️ Educational Use Only:</strong> {DISCLAIMER}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview():
    df, eda, val = get_dataset()
    predictor = get_predictor()
    model_name = predictor.metadata.get("model_type", "—") if predictor else "—"

    # Hero section
    st.markdown("""
    <div class="glass-card" style="padding: 2.5rem; margin-bottom: 2rem; background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-tertiary) 100%);">
        <div style="display: flex; align-items: flex-start; gap: 1.5rem; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 300px;">
                <h2 style="margin: 0 0 0.5rem; font-size: 1.75rem; font-weight: 700; color: var(--text-primary);">
                    Advanced Heart Disease Risk Research Platform
                </h2>
                <p style="margin: 0; font-size: 1rem; color: var(--text-secondary); line-height: 1.6;">
                    CardioInsight AI explores patterns in the Cleveland heart disease dataset using
                    reproducible ML pipelines. Compare models, visualize explanations, and understand
                    risk factors — all in an interactive research environment.
                </p>
            </div>
            <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                <span class="badge badge-primary">Educational Prototype</span>
                <span class="badge badge-info">v0.1.0</span>
                <span class="badge badge-success">NumPy Engine</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_disclaimer()

    # Key metrics
    st.markdown("### 📊 Key Metrics")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{eda['shape']['rows']}</div>
            <div class="metric-label">Records</div>
        </div>""", unsafe_allow_html=True)
    with mcol2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{eda['shape']['columns'] - 1}</div>
            <div class="metric-label">Features</div>
        </div>""", unsafe_allow_html=True)
    with mcol3:
        pos = eda['target_distribution']['percentages'].get(1, 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{pos:.1f}%</div>
            <div class="metric-label">Positive Class</div>
        </div>""", unsafe_allow_html=True)
    with mcol4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{model_name}</div>
            <div class="metric-label">Best Model</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Dataset summary + Pipeline status
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="section-card">
            <h3 style="margin: 0 0 1rem; font-size: 1.1rem; font-weight: 600;">Dataset Summary</h3>
        """, unsafe_allow_html=True)
        st.dataframe(eda["summary_statistics"].round(3), use_container_width=True, height=300)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-card">
            <h3 style="margin: 0 0 1rem; font-size: 1.1rem; font-weight: 600;">Pipeline Status</h3>
        """, unsafe_allow_html=True)

        steps = [
            ("Data Ingestion", True, "Synthetic Cleveland schema"),
            ("Schema Validation", val.schema_ok, "All columns present" if val.schema_ok else val.errors[0]),
            ("Missing Values", not val.missing_values, "None" if not val.missing_values else str(val.missing_values)),
            ("Duplicates", val.duplicate_rows == 0, f"{val.duplicate_rows} found" if val.duplicate_rows else "None"),
            ("Value Ranges", not val.invalid_values, "Valid" if not val.invalid_values else "Issues detected"),
            ("Train/Val/Test Split", True, f"{cfg['data']['test_size']:.0%} test, {cfg['data']['val_size']:.0%} val"),
            ("Model Training", predictor is not None, f"Best: {model_name}" if predictor else "Not trained"),
            ("Model Evaluation", predictor is not None, "Test F1, AUC computed" if predictor else "Pending"),
        ]

        for name, status, detail in steps:
            icon = "✅" if status else ("⚠️" if "warning" in str(detail).lower() else "⏳")
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem; background: var(--bg-tertiary); border-radius: var(--radius-sm); margin-bottom: 0.5rem;">
                <span style="font-size: 1.25rem;">{icon}</span>
                <div style="flex: 1;">
                    <div style="font-weight: 500; color: var(--text-primary);">{name}</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">{detail}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def page_data_explorer():
    df, eda, val = get_dataset()

    st.markdown("""
    <div class="page-header">
        <div>
            <h1 class="page-title">🔍 Data Explorer</h1>
            <p class="page-subtitle">Interactive exploration of the Cleveland heart disease dataset</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    render_disclaimer()

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Preview", "📈 Distributions", "🎯 Target", "🔗 Correlations", "🔍 Filters"])

    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True, height=400)
        st.caption(f"Showing 20 of {len(df)} records • {df.shape[1]} columns")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        feature = st.selectbox(
            "Select Feature",
            options=list(eda["feature_distributions"].keys()),
            format_func=lambda f: eda["feature_distributions"][f]["label"],
            key="feature_dist_select"
        )
        fig = plot_feature_distribution(df, feature, dark=(st.session_state.theme == "dark"), interactive=True)
        st.plotly_chart(fig, use_container_width=True)

        # Feature stats
        fdist = eda["feature_distributions"][feature]
        scol1, scol2, scol3, scol4 = st.columns(4)
        if fdist["kind"] == "numeric":
            with scol1:
                st.metric("Mean", f"{fdist.get('mean', 0):.2f} {fdist.get('unit', '')}")
            with scol2:
                st.metric("Std Dev", f"{fdist.get('std', 0):.2f}")
            with scol3:
                st.metric("Min", f"{fdist.get('min', 0):.2f}")
            with scol4:
                st.metric("Max", f"{fdist.get('max', 0):.2f}")
        else:
            st.write("**Value Counts:**")
            for val, cnt in fdist.get("value_counts", {}).items():
                st.write(f"• {val}: {cnt}")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        tgt = eda["target_distribution"]
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.metric("No Disease (0)", tgt["counts"].get(0, 0), f"{tgt['percentages'].get(0, 0):.1f}%")
        with tcol2:
            st.metric("Disease (1)", tgt["counts"].get(1, 0), f"{tgt['percentages'].get(1, 0):.1f}%")
        st.write("**Imbalance Ratio:** {:.3f}".format(val.target_imbalance_ratio))
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        corr = eda["correlation_matrix"]
        fig = plot_correlation_heatmap(corr, dark=(st.session_state.theme == "dark"), interactive=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            age_range = st.slider("Age Range", int(df["age"].min()), int(df["age"].max()),
                                  (int(df["age"].min()), int(df["age"].max())))
        with fcol2:
            target_filter = st.multiselect("Target Class", [0, 1], default=[0, 1],
                                           format_func=lambda x: "No Disease" if x == 0 else "Disease")
        filtered = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])
                      & (df["target"].isin(target_filter))]
        st.caption(f"Showing {len(filtered)} of {len(df)} records")
        st.dataframe(filtered, use_container_width=True, height=350)
        st.markdown('</div>', unsafe_allow_html=True)

def page_model_lab():
    predictor = get_predictor()
    if not predictor:
        st.warning("No trained model found. Run `python train.py` first.")
        return

    comp_path = Path("models/comparison.csv")
    if not comp_path.exists():
        st.warning("Comparison table not found. Re-run training.")
        return

    comp_df = pd.read_csv(comp_path)

    st.markdown("""
    <div class="page-header">
        <div>
            <h1 class="page-title">🧪 Model Lab</h1>
            <p class="page-subtitle">Compare trained models, analyze metrics, and understand performance trade-offs</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    render_disclaimer()

    # Model comparison chart
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Model Comparison (Test Set)")
    fig = plot_metric_comparison(comp_df, dark=(st.session_state.theme == "dark"), interactive=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Detailed metrics
    st.markdown("#### Detailed Metrics")
    display_cols = ["model_name", "split", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    st.dataframe(comp_df[display_cols].round(4), use_container_width=True)

    # Per-model drill-down
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    model_name = st.selectbox("Select Model for Deep Dive", comp_df["model_name"].unique(), key="model_drilldown")
    row = comp_df[comp_df["model_name"] == model_name].iloc[0]

    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{row['accuracy']:.3f}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric-card" style="margin-top: 1rem;">
            <div class="metric-value">{row['precision']:.3f}</div>
            <div class="metric-label">Precision</div>
        </div>""", unsafe_allow_html=True)
    with mcol2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{row['recall']:.3f}</div>
            <div class="metric-label">Recall (Sensitivity)</div>
        </div>
        <div class="metric-card" style="margin-top: 1rem;">
            <div class="metric-value">{row['specificity']:.3f}</div>
            <div class="metric-label">Specificity</div>
        </div>""", unsafe_allow_html=True)
    with mcol3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{row['f1']:.3f}</div>
            <div class="metric-label">F1 Score</div>
        </div>
        <div class="metric-card" style="margin-top: 1rem;">
            <div class="metric-value">{row['roc_auc']:.3f}</div>
            <div class="metric-label">ROC-AUC</div>
        </div>""", unsafe_allow_html=True)

    # Confusion matrix
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Confusion Matrix")
    cm_raw = row["confusion_matrix"]
    if isinstance(cm_raw, str):
        cm = np.array(ast.literal_eval(cm_raw), dtype=float)
    else:
        cm = np.array(cm_raw, dtype=float)
    fig = plot_confusion_matrix(cm, dark=(st.session_state.theme == "dark"), interactive=True)
    st.plotly_chart(fig, use_container_width=True)

    tn, fp, fn, tp = cm.ravel()
    ccol1, ccol2, ccol3, ccol4 = st.columns(4)
    with ccol1:
        st.metric("True Negatives", tn)
    with ccol2:
        st.metric("False Positives", fp)
    with ccol3:
        st.metric("False Negatives", fn)
    with ccol4:
        st.metric("True Positives", tp)
    st.markdown('</div>', unsafe_allow_html=True)

def page_prediction_demo():
    predictor = get_predictor()
    if not predictor:
        st.warning("No trained model found. Run `python train.py` first.")
        return

    st.markdown("""
    <div class="page-header">
        <div>
            <h1 class="page-title">🎯 Prediction Demo</h1>
            <p class="page-subtitle">Enter patient features to see model predictions with probability scores</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    render_disclaimer()

    meta = predictor.get_feature_metadata()

    with st.form("prediction_form"):
        st.markdown("### Patient Features")
        cols = st.columns(3)
        user_input = {}

        for i, (fname, fmeta) in enumerate(meta.items()):
            with cols[i % 3]:
                if fmeta["kind"] == "categorical":
                    allowed = fmeta["allowed_values"]
                    user_input[fname] = st.selectbox(
                        fmeta["label"],
                        options=allowed,
                        format_func=lambda x, m=fmeta: f"{x} — {m['description']}",
                        help=f"Unit: {fmeta['unit']}. {fmeta['description']}",
                        key=f"pred_{fname}"
                    )
                else:
                    user_input[fname] = st.number_input(
                        fmeta["label"],
                        min_value=float(fmeta["min_value"]) if fmeta["min_value"] is not None else None,
                        max_value=float(fmeta["max_value"]) if fmeta["max_value"] is not None else None,
                        value=float((fmeta["min_value"] + fmeta["max_value"]) / 2) if fmeta["min_value"] is not None and fmeta["max_value"] is not None else 0.0,
                        step=1.0 if fmeta["unit"] in ("years", "mmHg", "mg/dL", "bpm", "count") else 0.1,
                        help=f"Unit: {fmeta['unit']}. {fmeta['description']}",
                        key=f"pred_{fname}"
                    )

        submitted = st.form_submit_button("🔍 Analyze Risk", use_container_width=True, type="primary")

    if submitted:
        try:
            with st.spinner("Running inference..."):
                result = predictor.predict(user_input)

            # Result display
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📋 Analysis Result")

            color = "var(--danger)" if result.predicted_class == 1 else "var(--success)"
            icon = "⚠️" if result.predicted_class == 1 else "✅"
            label = "Elevated Risk Detected" if result.predicted_class == 1 else "Low Risk Indicated"

            rcol1, rcol2, rcol3 = st.columns(3)
            with rcol1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {color};">
                    <div class="metric-label">{icon} Prediction</div>
                    <div class="metric-value" style="color: {color}; font-size: 1.5rem;">{label}</div>
                </div>""", unsafe_allow_html=True)
            with rcol2:
                prob_color = "var(--danger)" if result.probability > 0.7 else ("var(--warning)" if result.probability > 0.4 else "var(--success)")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Disease Probability</div>
                    <div class="metric-value" style="color: {prob_color};">{result.probability:.1%}</div>
                </div>""", unsafe_allow_html=True)
            with rcol3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model Used</div>
                    <div class="metric-value" style="font-size: 1.2rem;">{result.model_metadata['model_type']}</div>
                </div>""", unsafe_allow_html=True)

            # Limitations
            st.error("""
            **⚠️ Important Limitations**
            - This prediction is based on a synthetic educational dataset, not real patient data.
            - The model has not been clinically validated and may produce inaccurate results.
            - Feature importance does not imply causation.
            - **Consult a qualified healthcare professional for any health concerns.**
            """)

            # Input summary
            with st.expander("📝 Input Features Used", expanded=True):
                input_df = pd.DataFrame([
                    {"Feature": meta[f]["label"], "Value": v, "Unit": meta[f]["unit"]}
                    for f, v in result.input_features.items()
                ])
                st.dataframe(input_df, use_container_width=True)

            # Quick actions
            st.markdown("#### Quick Actions")
            a1, a2, a3 = st.columns(3)
            with a1:
                if st.button("🔬 View Explainability", use_container_width=True):
                    st.info("Navigate to the **Explainability** page to see feature importance and SHAP explanations.")
            with a2:
                if st.button("📊 Compare Models", use_container_width=True):
                    st.info("Navigate to the **Model Lab** to compare all trained models.")
            with a3:
                if st.button("📋 Model Card", use_container_width=True):
                    st.info("Navigate to the **Model Card** for full documentation.")

        except ValueError as exc:
            st.error(f"Input validation failed: {exc}")

def page_explainability():
    predictor = get_predictor()
    if not predictor:
        st.warning("No trained model found. Run `python train.py` first.")
        return

    st.markdown("""
    <div class="page-header">
        <div>
            <h1 class="page-title">🔬 Explainability</h1>
            <p class="page-subtitle">Understand which features drive model predictions</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    render_disclaimer()

    st.markdown("""
    <div class="section-card">
        <h4 style="margin: 0 0 1rem;">How to Interpret</h4>
        <p style="margin: 0; color: var(--text-secondary);">
            <strong>Feature Importance (MDI)</strong> shows which features the model relies on most for predictions.
            Higher values mean the feature contributes more to reducing impurity in the trees.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card" style="border-left: 4px solid var(--warning);">
        <h4 style="margin: 0 0 0.5rem;">⚠️ Important Caveat</h4>
        <p style="margin: 0; color: var(--text-secondary);">
            Feature importance and SHAP explanations describe <strong>associations learned by the model</strong>,
            not causal relationships. They do not establish causation. Clinical interpretation requires
            domain expertise and validation on real patient data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Global feature importance
    model = predictor.model
    feature_names = predictor.feature_names

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
        from src.data.schema import FEATURE_METADATA

        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance,
        }).sort_values("Importance", ascending=True)

        imp_df["Label"] = imp_df["Feature"].map(lambda f: FEATURE_METADATA[f].label)

        st.markdown("#### Global Feature Importance")
        fig = plot_metric_comparison(
            imp_df.rename(columns={"Label": "model_name", "Importance": "f1"}),
            metrics=["f1"],
            dark=(st.session_state.theme == "dark"),
            interactive=True,
        )
        fig.update_layout(
            title="Feature Importance (Mean Decrease Impurity)",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.dataframe(imp_df[["Label", "Importance"]].round(4), use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")

    # SHAP section
    st.markdown("#### SHAP Explanations")
    if st.button("🔧 Install SHAP to Enable", use_container_width=True, type="secondary"):
        st.code("pip install shap", language="bash")

    st.info("""
    SHAP (SHapley Additive exPlanations) provides per-sample feature attributions. When installed, this page shows:
    • **Summary Plot** — Beeswarm of SHAP values across the test set
    • **Force Plot** — Individual prediction breakdown
    • **Dependence Plots** — Feature interactions for top features

    The SHAP library is heavy (~200 MB) and optional.
    """)

def page_model_card():
    predictor = get_predictor()
    if not predictor:
        st.warning("No trained model found. Run `python train.py` first.")
        return

    meta = predictor.metadata

    st.markdown("""
    <div class="page-header">
        <div>
            <h1 class="page-title">📋 Model Card</h1>
            <p class="page-subtitle">Comprehensive documentation of the trained model</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    render_disclaimer()

    # Model Details
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Model Details")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        for label, value in [
            ("Model Type", meta.get('model_type', 'Unknown')),
            ("Engine", meta.get('engine', 'Unknown')),
            ("Trained At", meta.get('trained_at', 'Unknown')[:19].replace('T', ' ')),
            ("Random Seed", str(meta.get('random_seed', 'Unknown'))),
        ]:
            st.markdown(f"**{label}:** {value}")
    with dcol2:
        for label, value in [
            ("Training Samples", str(meta.get('n_train_samples', 'Unknown'))),
            ("Features", str(meta.get('n_features', 'Unknown'))),
            ("Test F1", f"{meta.get('test_metrics', {}).get('f1', 0):.3f}" if meta.get('test_metrics') else "N/A"),
            ("Test ROC-AUC", f"{meta.get('test_metrics', {}).get('roc_auc', 0):.3f}" if meta.get('test_metrics') else "N/A"),
        ]:
            st.markdown(f"**{label}:** {value}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Intended Use
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
    ### Intended Use
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
        <div>
            <h4 style="color: var(--success); margin: 0 0 0.5rem;">✅ Intended Uses</h4>
            <ul style="margin: 0; padding-left: 1.25rem; color: var(--text-secondary);">
                <li>Educational demonstration of ML pipelines</li>
                <li>Model comparison and evaluation research</li>
                <li>Explainability concept exploration</li>
                <li>Heart disease risk factor pattern analysis</li>
            </ul>
        </div>
        <div>
            <h4 style="color: var(--danger); margin: 0 0 0.5rem;">❌ Prohibited Uses</h4>
            <ul style="margin: 0; padding-left: 1.25rem; color: var(--text-secondary);">
                <li>Clinical decision-making</li>
                <li>Patient diagnosis or triage</li>
                <li>Treatment planning</li>
                <li>Any real health outcome impact</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Limitations
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
    ### Known Limitations
    <ol style="margin: 0; padding-left: 1.25rem; color: var(--text-secondary); line-height: 2;">
        <li><strong>Synthetic Data Only</strong> — Does not reflect real patient distributions</li>
        <li><strong>Small Sample</strong> (n=303) — High variance in performance estimates</li>
        <li><strong>No External Validation</strong> — Performance on real clinical data unknown</li>
        <li><strong>Class Imbalance</strong> (~45% positive) — May bias probability calibration</li>
        <li><strong>Feature Importance ≠ Causation</strong> — Associations only</li>
        <li><strong>No Temporal Validation</strong> — Model drift not assessed</li>
        <li><strong>Single-Site Schema</strong> — Cleveland (1980s) may not generalize</li>
    </ol>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Bias considerations
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
    ### Bias & Generalizability Considerations
    <div style="color: var(--text-secondary);">
        <ul style="margin: 0; padding-left: 1.25rem; line-height: 2;">
            <li><strong>Sex</strong> — Included as feature (0/1), ~50/50 synthetic distribution</li>
            <li><strong>Age</strong> — Continuous, range 20–80, no age-stratified evaluation</li>
            <li><strong>Race/Ethnicity</strong> — <em>Not present</em> in schema, cannot assess bias</li>
            <li><strong>Socioeconomic</strong> — <em>Not present</em>, cannot assess bias</li>
            <li><strong>Geography</strong> — Single-site (Cleveland) schema only</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Export
    if st.button("📥 Download Model Card (JSON)", use_container_width=True):
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
                "Synthetic data only", "No clinical validation",
                "Small sample size", "No external validation"
            ],
            "ethical_considerations": [
                "Not for clinical use", "Educational demonstration only",
                "Feature importance ≠ causation"
            ],
        }
        st.download_button(
            "Download JSON",
            data=json.dumps(card, indent=2),
            file_name="model_card.json",
            mime="application/json",
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    # Process chatbot if needed
    process_chatbot()

    # Render header
    render_header()

    # Render chatbot panel
    render_chatbot()

    # Sidebar navigation
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            [
                "📊 Overview",
                "🔍 Data Explorer",
                "🧪 Model Lab",
                "🎯 Prediction Demo",
                "🔬 Explainability",
                "📋 Model Card",
            ],
            label_visibility="collapsed",
            key="nav_radio",
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Quick stats in sidebar
        predictor = get_predictor()
        if predictor:
            st.markdown(f"""
            <div class="glass-card" style="padding: 1rem; margin-bottom: 1rem;">
                <strong style="color: var(--text-primary);">Model Status</strong><br>
                <span style="color: var(--success);">● Loaded</span> {predictor.metadata.get('model_type', 'Unknown')}<br>
                <small style="color: var(--text-muted);">Test F1: {predictor.metadata.get('test_metrics', {}).get('f1', 0):.3f}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="padding: 1rem; margin-bottom: 1rem;">
                <strong style="color: var(--text-primary);">Model Status</strong><br>
                <span style="color: var(--warning);">● Not Loaded</span><br>
                <small style="color: var(--text-muted);">Run train.py first</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Quick actions
        st.markdown("**Quick Actions**")
        if st.button("🔄 Reload Data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.caption(f"Version 0.1.0 • Seed: {cfg['project']['random_seed']}")
        st.caption(f"Engine: {cfg['model']['engine']} • Theme: {st.session_state.theme.title()}")

    # Route to page
    if page == "📊 Overview":
        page_overview()
    elif page == "🔍 Data Explorer":
        page_data_explorer()
    elif page == "🧪 Model Lab":
        page_model_lab()
    elif page == "🎯 Prediction Demo":
        page_prediction_demo()
    elif page == "🔬 Explainability":
        page_explainability()
    elif page == "📋 Model Card":
        page_model_card()

if __name__ == "__main__":
    main()