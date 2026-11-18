"""Loan Approval Intelligence Dashboard.

Run:  streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Approval Dashboard", page_icon="🏦", layout="wide",
                   initial_sidebar_state="expanded")

from dashboard import views  # noqa: E402  (must come after set_page_config)
from dashboard.style import inject_css  # noqa: E402
from src.config import METRICS_PATH, MODEL_PATH, PREDS_PATH  # noqa: E402
from src.data import load_raw, prepare_for_analysis  # noqa: E402
from src.predict import load_model  # noqa: E402


@st.cache_data(show_spinner=False)
def get_data():
    return prepare_for_analysis(load_raw())


@st.cache_data(show_spinner=False)
def get_metrics():
    return json.loads(METRICS_PATH.read_text())


@st.cache_data(show_spinner=False)
def get_preds():
    return pd.read_csv(PREDS_PATH)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


inject_css()

if not (MODEL_PATH.exists() and METRICS_PATH.exists()):
    st.error("Trained model not found. Run `python -m src.train` once, then restart the app.")
    st.stop()

df, metrics = get_data(), get_metrics()

PAGES = {
    "📊 Overview": lambda: views.render_overview(df, metrics),
    "🔎 Data Explorer": lambda: views.render_explorer(df),
    "🤖 Model Performance": lambda: views.render_models(metrics, get_preds()),
    "🧮 Decision Simulator": lambda: views.render_simulator(get_model(), df, metrics),
    "ℹ️ About": lambda: views.render_about(metrics),
}

with st.sidebar:
    st.markdown("## 🏦 Loan Approval")
    st.caption("ML-powered screening dashboard")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    fm = metrics["models"][metrics["final_model"]]
    st.markdown(f'<span class="badge">{metrics["final_model"]}</span>'
                f'<span class="badge">{fm["accuracy"]:.1%} acc</span>'
                f'<span class="badge">AUC {fm["roc_auc"]:.3f}</span>', unsafe_allow_html=True)
    st.caption(f"Trained {metrics['meta']['trained_at']}")
    st.markdown('<div class="footer">For learning and demonstration only. Not for real lending decisions.</div>',
                unsafe_allow_html=True)

PAGES[page]()
