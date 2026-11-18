"""Plotly figure builders. Each function takes data and returns a figure; no Streamlit here."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import (C_ACCENT, C_APPROVED, C_GREY, C_PRIMARY, C_REJECTED,
                        CIBIL_LABELS, LABELS, NUMERIC)

STATUS_COLORS = {"Approved": C_APPROVED, "Rejected": C_REJECTED}


def _layout(fig, title=None, height=360, legend=True):
    fig.update_layout(
        title={"text": title, "x": 0.0, "font": {"size": 15}} if title else None,
        height=height,
        margin={"l": 10, "r": 10, "t": 50 if title else 20, "b": 10},
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
    )
    return fig


# ---------------------------------------------------------------- data exploration
def status_donut(df):
    counts = df["loan_status"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.62, sort=False,
        marker={"colors": [STATUS_COLORS[s] for s in counts.index]},
        textinfo="percent", hovertemplate="%{label}: %{value:,} applications<extra></extra>"))
    fig.add_annotation(text=f"<b>{len(df):,}</b><br>applications", showarrow=False, font={"size": 14})
    return _layout(fig, "Loan decisions", legend=True)


def approval_by_band(df):
    g = df.groupby("cibil_band", observed=False).agg(rate=("approved", "mean"), n=("approved", "size"))
    g = g.reindex(CIBIL_LABELS)
    colors = [C_REJECTED if r < 0.5 else C_APPROVED for r in g["rate"].fillna(0)]
    fig = go.Figure(go.Bar(
        x=g.index, y=(g["rate"] * 100).round(1), marker_color=colors,
        text=[f"{r*100:.1f}%" if n else "" for r, n in zip(g["rate"].fillna(0), g["n"])],
        textposition="outside", customdata=g["n"],
        hovertemplate="%{x}<br>Approval rate: %{y}%<br>Applications: %{customdata:,}<extra></extra>"))
    fig.update_yaxes(range=[0, 112], title="Approval rate (%)")
    return _layout(fig, "Approval rate by CIBIL score band", legend=False)


def hist_by_status(df, col, scale=1.0, unit=""):
    fig = go.Figure()
    for status in ["Approved", "Rejected"]:
        fig.add_trace(go.Histogram(x=df.loc[df["loan_status"] == status, col] / scale, name=status,
                                   marker_color=STATUS_COLORS[status], opacity=0.75, nbinsx=30))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=f"{LABELS.get(col, col)}{unit}")
    fig.update_yaxes(title="Applications")
    return _layout(fig, f"Distribution of {LABELS.get(col, col).lower()}")


def box_cibil(df):
    fig = go.Figure()
    for status in ["Approved", "Rejected"]:
        fig.add_trace(go.Box(y=df.loc[df["loan_status"] == status, "cibil_score"], name=status,
                             marker_color=STATUS_COLORS[status], boxmean=True))
    fig.update_yaxes(title="CIBIL score")
    return _layout(fig, "CIBIL score by decision", legend=False)


def scatter_income_loan(df, max_points=2500):
    d = df.sample(min(len(df), max_points), random_state=1)
    fig = go.Figure()
    for status in ["Rejected", "Approved"]:
        s = d[d["loan_status"] == status]
        fig.add_trace(go.Scattergl(x=s["income_annum"] / 1e5, y=s["loan_amount"] / 1e5, mode="markers",
                                   name=status, marker={"size": 5, "color": STATUS_COLORS[status], "opacity": 0.5}))
    fig.update_xaxes(title="Annual income (lakh)")
    fig.update_yaxes(title="Loan amount (lakh)")
    return _layout(fig, "Income vs loan amount")


def term_rate(df):
    g = df.groupby("loan_term").agg(rate=("approved", "mean"), n=("approved", "size")).reset_index()
    g = g[g["n"] >= 20]  # hide terms with too few applications to be meaningful
    fig = go.Figure(go.Bar(x=g["loan_term"].astype(str), y=(g["rate"] * 100).round(1), marker_color=C_PRIMARY,
                           customdata=g["n"],
                           hovertemplate="%{x} years<br>Approval rate: %{y}%<br>Applications: %{customdata:,}<extra></extra>"))
    fig.add_hline(y=df["approved"].mean() * 100, line_dash="dash", line_color=C_GREY,
                  annotation_text="overall", annotation_position="top left")
    fig.update_xaxes(title="Loan term (years)")
    fig.update_yaxes(title="Approval rate (%)", range=[0, 100])
    return _layout(fig, "Approval rate by loan term", legend=False)


def corr_heatmap(df):
    cols = NUMERIC + ["approved"]
    corr = df[cols].corr().round(2)
    names = [LABELS.get(c, "Approved") for c in cols]
    fig = go.Figure(go.Heatmap(z=corr.values, x=names, y=names, zmin=-1, zmax=1, zmid=0,
                               colorscale="RdBu", reversescale=True, text=corr.values,
                               texttemplate="%{text}", textfont={"size": 10}))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-40)
    return _layout(fig, "Correlation matrix", height=520, legend=False)


# ---------------------------------------------------------------- model performance
def model_comparison(models: dict, final: str):
    names = list(models)
    metric_defs = [("accuracy", "Test accuracy"), ("roc_auc", "ROC-AUC"), ("f1", "F1-score")]
    colors = [C_PRIMARY, C_ACCENT, C_GREY]
    fig = go.Figure()
    for (key, label), colr in zip(metric_defs, colors):
        fig.add_trace(go.Bar(x=names, y=[models[n][key] for n in names], name=label, marker_color=colr,
                             text=[f"{models[n][key]:.3f}" for n in names], textposition="outside",
                             textfont={"size": 10}))
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0.8, 1.02], title="Score")
    return _layout(fig, "Model comparison on the held-out test set", height=400)


def roc_fig(roc: dict, models: dict, final: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": C_GREY},
                             name="Random guess", hoverinfo="skip"))
    for name, curve in roc.items():
        is_final = name == final
        fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines",
                                 name=f"{name} (AUC {models[name]['roc_auc']:.3f})",
                                 line={"width": 4 if is_final else 1.6}))
    fig.update_xaxes(title="False positive rate")
    fig.update_yaxes(title="True positive rate")
    fig.update_layout(legend={"orientation": "v", "x": 0.55, "y": 0.05, "yanchor": "bottom"})
    return _layout(fig, "ROC curves", height=420, legend=True)


def confusion_fig(cm, title="Confusion matrix"):
    cm = np.array(cm)
    labels = ["Rejected", "Approved"]
    total = cm.sum()
    text = [[f"<b>{cm[i][j]:,}</b><br>{cm[i][j]/total:.1%}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=cm, x=[f"Predicted {l}" for l in labels], y=[f"Actual {l}" for l in labels],
                               colorscale=[[0, "#eef3f9"], [1, C_PRIMARY]], showscale=False,
                               text=text, texttemplate="%{text}", textfont={"size": 15}))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, height=330, legend=False)


def importance_fig(importance: dict):
    items = sorted(importance.items(), key=lambda kv: kv[1]["mean"])
    fig = go.Figure(go.Bar(
        x=[v["mean"] * 100 for _, v in items], y=[LABELS[k] for k, _ in items], orientation="h",
        error_x={"type": "data", "array": [v["std"] * 100 for _, v in items]},
        marker_color=[C_PRIMARY if v["mean"] > 0.01 else C_GREY for _, v in items]))
    fig.update_xaxes(title="Drop in test accuracy when the feature is shuffled (percentage points)")
    return _layout(fig, "Permutation feature importance", height=420, legend=False)


def threshold_fig(preds: pd.DataFrame, threshold: float):
    ts = np.linspace(0.05, 0.95, 91)
    y, p = preds["y_true"].values, preds["proba"].values
    prec, rec, acc = [], [], []
    for t in ts:
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        prec.append(tp / pred.sum() if pred.sum() else np.nan)
        rec.append(tp / (y == 1).sum())
        acc.append((pred == (y == 1)).mean())
    fig = go.Figure()
    for name, vals, colr in [("Accuracy", acc, C_PRIMARY), ("Precision", prec, C_APPROVED), ("Recall", rec, C_ACCENT)]:
        fig.add_trace(go.Scatter(x=ts, y=vals, mode="lines", name=name, line={"color": colr, "width": 2.5}))
    fig.add_vline(x=threshold, line_dash="dash", line_color=C_REJECTED)
    fig.update_xaxes(title="Decision threshold")
    fig.update_yaxes(title="Score", range=[0.5, 1.02])
    return _layout(fig, "Trade-off as the approval threshold changes", height=340)


# ---------------------------------------------------------------- simulator
def gauge(prob: float, colour: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Probability of approval", "font": {"size": 14}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 1},
               "bar": {"color": colour, "thickness": 0.3},
               "steps": [{"range": [0, 50], "color": "rgba(192,80,77,0.18)"},
                         {"range": [50, 100], "color": "rgba(46,125,91,0.18)"}],
               "threshold": {"line": {"color": "#444", "width": 3}, "thickness": 0.8, "value": 50}}))
    fig.update_layout(height=270, margin={"l": 20, "r": 20, "t": 50, "b": 10})
    return fig


def cibil_sweep_fig(scores, probs, current_score, current_prob):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scores, y=probs * 100, mode="lines", line={"color": C_PRIMARY, "width": 3},
                             name="Approval probability", fill="tozeroy", fillcolor="rgba(31,78,121,0.10)"))
    fig.add_trace(go.Scatter(x=[current_score], y=[current_prob * 100], mode="markers",
                             marker={"size": 13, "color": C_REJECTED, "line": {"width": 2, "color": "white"}},
                             name="This applicant"))
    fig.add_hline(y=50, line_dash="dot", line_color=C_GREY)
    fig.update_xaxes(title="CIBIL score")
    fig.update_yaxes(title="Approval probability (%)", range=[0, 102])
    return _layout(fig, "What-if: effect of the CIBIL score on this application", height=320)
