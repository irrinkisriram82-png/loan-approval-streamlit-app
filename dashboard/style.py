"""Look and feel: CSS and small HTML components (works in light and dark themes)."""
import inspect

import streamlit as st

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1250px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.hero {padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 1.1rem; color: #fff;
       background: linear-gradient(120deg, #1f4e79 0%, #2f6f9f 60%, #3b8ea5 100%);}
.hero h1 {color:#fff; margin:0; font-size:1.7rem;}
.hero p {margin:.25rem 0 0 0; opacity:.9; font-size:.95rem;}
.kpi {padding: .9rem 1rem; border-radius: 12px; border: 1px solid rgba(128,128,128,.28);
      background: rgba(128,128,128,.07); border-left: 5px solid #1f4e79; height: 100%;}
.kpi.green {border-left-color:#2e7d5b;} .kpi.red {border-left-color:#c0504d;} .kpi.amber {border-left-color:#e0a030;}
.kpi-label {font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; opacity:.7;}
.kpi-value {font-size:1.75rem; font-weight:700; line-height:1.25;}
.kpi-sub {font-size:.8rem; opacity:.65;}
.verdict {padding: 1.1rem 1.3rem; border-radius: 14px; color:#fff; margin-bottom:.6rem;}
.verdict h2 {margin:0; color:#fff; font-size:1.6rem;}
.verdict p {margin:.2rem 0 0 0; opacity:.92;}
.note {padding:.6rem .9rem; border-radius:10px; margin:.35rem 0; font-size:.92rem;
       border:1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.06);}
.note.good {border-left:4px solid #2e7d5b;} .note.ok {border-left:4px solid #3b8ea5;}
.note.warn {border-left:4px solid #e0a030;} .note.bad {border-left:4px solid #c0504d;}
.badge {display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.75rem;
        background: rgba(31,78,121,.15); border:1px solid rgba(31,78,121,.35);}
.footer {opacity:.6; font-size:.8rem; padding-top:1rem;}
[data-testid="stSidebar"] .badge {margin-right:.25rem;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi_html(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return (f'<div class="kpi {tone}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


def kpi_row(items):
    """items: list of (label, value, sub, tone)."""
    cols = st.columns(len(items))
    for col, (label, value, sub, tone) in zip(cols, items):
        col.markdown(kpi_html(label, value, sub, tone), unsafe_allow_html=True)


def note(text: str, level: str = "ok"):
    st.markdown(f'<div class="note {level}">{text}</div>', unsafe_allow_html=True)


def show(fig):
    """Render a Plotly figure full-width (compatible with old and new Streamlit versions)."""
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, width="stretch")          # newer Streamlit
    else:
        st.plotly_chart(fig, use_container_width=True)  # older Streamlit
