"""The five dashboard pages. Each ``render_*`` function draws one page."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import predict as P
from src.config import LABELS
from dashboard import charts as C
from dashboard.style import hero, kpi_row, note, show


def inr(x: float) -> str:
    """Format a rupee amount in lakh / crore."""
    if x >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if x >= 1e5:
        return f"₹{x / 1e5:.1f} L"
    return f"₹{x:,.0f}"


# ===================================================================== 1. overview
def render_overview(df: pd.DataFrame, metrics: dict):
    hero("Loan Approval Intelligence Dashboard",
         "Portfolio overview, model performance and a live decision simulator for retail loan screening.")
    final = metrics["final_model"]
    fm = metrics["models"][final]
    kpi_row([
        ("Applications", f"{len(df):,}", "records analysed", ""),
        ("Approval rate", f"{df['approved'].mean():.1%}", f"{int(df['approved'].sum()):,} approved", "green"),
        ("Rejection rate", f"{1 - df['approved'].mean():.1%}", f"{int((1 - df['approved']).sum()):,} rejected", "red"),
        ("Average CIBIL", f"{df['cibil_score'].mean():.0f}", "scale 300 - 900", "amber"),
        ("Model accuracy", f"{fm['accuracy']:.1%}", f"{final}, held-out test set", ""),
    ])
    st.write("")
    c1, c2 = st.columns([1, 1.6])
    with c1:
        show(C.status_donut(df))
    with c2:
        show(C.approval_by_band(df))

    st.subheader("Key findings")
    low = df[df["cibil_score"] < 550]["approved"].mean()
    high = df[df["cibil_score"] >= 550]["approved"].mean()
    left, right = st.columns(2)
    with left:
        note(f"<b>CIBIL score is the deciding factor.</b> Applicants below 550 were approved only "
             f"{low:.1%} of the time; applicants at 550 or above were approved {high:.1%} of the time.", "bad")
        note("<b>Income, loan size, education, employment type and assets</b> show almost no direct "
             "relationship with the decision in this dataset.", "ok")
    with right:
        note(f"<b>{final} performs best.</b> It reaches {fm['accuracy']:.1%} accuracy and "
             f"{fm['roc_auc']:.3f} ROC-AUC, against a {metrics['meta']['majority_baseline']:.1%} "
             f"majority-class baseline.", "good")
        note("<b>Shorter terms were approved slightly more often</b> (2 to 4 years) than long terms, "
             "but the effect is small compared with CIBIL.", "warn")

    st.subheader("Data quality")
    dq = metrics["meta"]["data_quality"]
    kpi_row([
        ("Rows / columns", f"{dq['rows']:,} / {dq['columns']}", "raw file", ""),
        ("Missing cells", f"{dq['missing_cells']}", f"in {dq['columns_with_missing']} columns, imputed", "amber"),
        ("Duplicate rows", f"{dq['duplicates']}", "none found", "green"),
        ("Train / test", f"{metrics['meta']['train_rows']:,} / {metrics['meta']['test_rows']:,}", "stratified 80 / 20", ""),
    ])


# ===================================================================== 2. explorer
def render_explorer(df: pd.DataFrame):
    hero("Data Explorer", "Slice the applicant data and see how every group behaves.")
    with st.expander("Filters", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        edu = f1.multiselect("Education", sorted(df["education_label"].unique()),
                             default=sorted(df["education_label"].unique()))
        se = f2.multiselect("Self-employed", sorted(df["self_employed"].unique()),
                            default=sorted(df["self_employed"].unique()))
        cibil = f3.slider("CIBIL score", 300, 900, (300, 900))
        term = f4.slider("Loan term (years)", int(df["loan_term"].min()), int(df["loan_term"].max()),
                         (int(df["loan_term"].min()), int(df["loan_term"].max())))
    d = df[df["education_label"].isin(edu) & df["self_employed"].isin(se)
           & df["cibil_score"].between(*cibil) & df["loan_term"].between(*term)]
    if d.empty:
        st.warning("No applications match these filters.")
        return
    kpi_row([
        ("Applications", f"{len(d):,}", f"{len(d) / len(df):.0%} of all", ""),
        ("Approval rate", f"{d['approved'].mean():.1%}", f"overall {df['approved'].mean():.1%}", "green"),
        ("Avg income", inr(d["income_annum"].mean()), "per year", ""),
        ("Avg loan", inr(d["loan_amount"].mean()), f"{d['loan_to_income'].mean():.1f}x income", "amber"),
        ("Avg CIBIL", f"{d['cibil_score'].mean():.0f}", "", ""),
    ])
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Distributions", "Relationships", "Correlations", "Data table"])
    with t1:
        a, b = st.columns(2)
        with a:
            show(C.hist_by_status(d, "cibil_score"))
        with b:
            show(C.box_cibil(d))
        a, b = st.columns(2)
        with a:
            show(C.hist_by_status(d, "income_annum", 1e5, " (lakh)"))
        with b:
            show(C.hist_by_status(d, "loan_amount", 1e5, " (lakh)"))
    with t2:
        a, b = st.columns(2)
        with a:
            show(C.scatter_income_loan(d))
        with b:
            show(C.term_rate(d))
        show(C.approval_by_band(d))
    with t3:
        show(C.corr_heatmap(d))
        note("CIBIL score is the only feature strongly related to approval. Income, loan amount and the "
             "asset values are highly correlated with <i>each other</i> but not with the decision.", "ok")
    with t4:
        cols = ["loan_id", "no_of_dependents", "education_label", "self_employed", "income_annum", "loan_amount",
                "loan_term", "cibil_score", "total_assets", "loan_status"]
        cols = [c for c in cols if c in d.columns]
        st.dataframe(d[cols], hide_index=True, height=420)
        st.download_button("Download filtered data (CSV)", d.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_applications.csv", mime="text/csv")


# ===================================================================== 3. models
def render_models(metrics: dict, preds: pd.DataFrame):
    hero("Model Performance", "Five algorithms, tuned with 5-fold cross-validation and compared on unseen data.")
    models, final = metrics["models"], metrics["final_model"]
    fm = models[final]
    kpi_row([
        ("Selected model", final, "best cross-validation accuracy", ""),
        ("Test accuracy", f"{fm['accuracy']:.2%}", f"baseline {metrics['meta']['majority_baseline']:.1%}", "green"),
        ("ROC-AUC", f"{fm['roc_auc']:.3f}", "1.0 = perfect separation", ""),
        ("Precision / recall", f"{fm['precision']:.2f} / {fm['recall']:.2f}", "Approved class", "amber"),
        ("CV accuracy", f"{fm['cv_accuracy']:.2%}", f"± {fm['cv_std']:.2%} across folds", ""),
    ])
    st.write("")
    board = pd.DataFrame([{
        "Model": n, "Selected": "✔" if n == final else "",
        "CV accuracy": m["cv_accuracy"], "Test accuracy": m["accuracy"], "Precision": m["precision"],
        "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"], "Train accuracy": m["train_accuracy"],
    } for n, m in models.items()]).sort_values("CV accuracy", ascending=False)
    st.subheader("Leaderboard")
    st.dataframe(board.style.format({c: "{:.4f}" for c in board.columns[2:]}), hide_index=True)
    show(C.model_comparison(models, final))

    a, b = st.columns(2)
    with a:
        show(C.roc_fig(metrics["roc"], models, final))
    with b:
        show(C.confusion_fig(metrics["confusion_matrix"], f"Confusion matrix - {final}"))
        tn, fp = metrics["confusion_matrix"][0]
        fn, tp = metrics["confusion_matrix"][1]
        note(f"Of {tn + fp + fn + tp:,} test applications, <b>{fp}</b> rejected ones were wrongly approved "
             f"(false positives, the costlier error for a lender) and <b>{fn}</b> approved ones were wrongly rejected.", "ok")

    st.subheader("What drives the model?")
    show(C.importance_fig(metrics["permutation_importance"]))
    note("<b>CIBIL score dominates.</b> Shuffling it alone drops accuracy by far more than any other feature. "
         "Features near zero contribute almost nothing to the decision.", "ok")

    st.subheader("Threshold analysis")
    th = st.slider("Approval threshold (approve when probability is at least)", 0.10, 0.90, 0.50, 0.05)
    y, p = preds["y_true"].values, preds["proba"].values
    pred = (p >= th).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    kpi_row([
        ("Accuracy", f"{(tp + tn) / len(y):.2%}", "", ""),
        ("Precision", f"{tp / max(tp + fp, 1):.2%}", "approved calls that were right", "green"),
        ("Recall", f"{tp / max(tp + fn, 1):.2%}", "good applicants found", "amber"),
        ("Wrong approvals", f"{fp}", "false positives", "red"),
    ])
    a, b = st.columns([1.6, 1])
    with a:
        show(C.threshold_fig(preds, th))
    with b:
        show(C.confusion_fig([[tn, fp], [fn, tp]], f"At threshold {th:.2f}"))
    st.caption("Raising the threshold makes the model more cautious: fewer wrong approvals, more good applicants turned away.")

    with st.expander("Hyper-parameters searched and selected"):
        rows = [{"Model": n, "Search space": "; ".join(f"{k}: {', '.join(v)}" for k, v in m["search_space"].items()),
                 "Best setting": ", ".join(f"{k}={v}" for k, v in m["best_params"].items())} for n, m in models.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)


# ===================================================================== 4. simulator
PRESETS = {
    "Strong applicant": dict(no_of_dependents=2, education="Graduate", self_employed="No", income_annum=9_600_000,
                             loan_amount=29_900_000, loan_term=12, cibil_score=778, residential_assets_value=2_400_000,
                             commercial_assets_value=17_600_000, luxury_assets_value=22_700_000, bank_asset_value=8_000_000),
    "Just above the cut-off (CIBIL 552)": dict(no_of_dependents=3, education="Graduate", self_employed="No", income_annum=7_000_000,
                                 loan_amount=20_000_000, loan_term=10, cibil_score=552, residential_assets_value=5_000_000,
                                 commercial_assets_value=6_000_000, luxury_assets_value=14_000_000, bank_asset_value=6_000_000),
    "Just below the cut-off (CIBIL 545)": dict(no_of_dependents=3, education="Graduate", self_employed="No", income_annum=7_000_000,
                                 loan_amount=20_000_000, loan_term=10, cibil_score=545, residential_assets_value=5_000_000,
                                 commercial_assets_value=6_000_000, luxury_assets_value=14_000_000, bank_asset_value=6_000_000),
    "Weak applicant": dict(no_of_dependents=0, education="NotGraduate", self_employed="Yes", income_annum=4_100_000,
                           loan_amount=12_200_000, loan_term=8, cibil_score=417, residential_assets_value=2_700_000,
                           commercial_assets_value=2_200_000, luxury_assets_value=8_800_000, bank_asset_value=3_300_000),
}
LEVEL_ICON = {"good": "✅", "ok": "ℹ️", "warn": "⚠️", "bad": "⛔"}


def _money_input(col, label, key, preset, df, step=100_000):
    lo, hi = int(df[key].min()), int(df[key].max())
    value = int(min(max(preset[key], lo), hi))
    return col.number_input(label, min_value=lo, max_value=hi, value=value, step=int(step), help=f"Range in data: {inr(lo)} to {inr(hi)}")


def render_simulator(model, df: pd.DataFrame, metrics: dict):
    hero("Loan Decision Simulator", "Enter an applicant's details; the model returns a decision, risk level and explanation instantly.")
    choice = st.selectbox("Load an example applicant", list(PRESETS), index=0)
    pre = PRESETS[choice]
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Applicant details")
        a, b = st.columns(2)
        edu = a.selectbox("Education", ["Graduate", "NotGraduate"], index=["Graduate", "NotGraduate"].index(pre["education"]),
                          format_func=lambda v: "Graduate" if v == "Graduate" else "Not graduate")
        se = b.selectbox("Self-employed", ["No", "Yes"], index=["No", "Yes"].index(pre["self_employed"]))
        dep = a.number_input("Dependents", 0, 5, int(pre["no_of_dependents"]))
        term = b.number_input("Loan term (years)", 2, 20, int(pre["loan_term"]))
        cibil = st.slider("CIBIL score", 300, 900, int(pre["cibil_score"]))
        income = _money_input(st, "Annual income (₹)", "income_annum", pre, df)
        loan = _money_input(st, "Loan amount requested (₹)", "loan_amount", pre, df)
        st.markdown("**Assets (₹)**")
        a, b = st.columns(2)
        resid = _money_input(a, "Residential", "residential_assets_value", pre, df)
        commer = _money_input(b, "Commercial", "commercial_assets_value", pre, df)
        lux = _money_input(a, "Luxury", "luxury_assets_value", pre, df)
        bank = _money_input(b, "Bank", "bank_asset_value", pre, df)

    inputs = dict(no_of_dependents=dep, education=edu, self_employed=se, income_annum=income, loan_amount=loan,
                  loan_term=term, cibil_score=cibil, residential_assets_value=resid, commercial_assets_value=commer,
                  luxury_assets_value=lux, bank_asset_value=bank)
    prob = P.predict_proba(model, inputs)
    verdict = P.decision(prob)
    tier, colour = P.risk_tier(prob)
    d = P.derived_metrics(inputs)

    with right:
        st.subheader("Assessment")
        bg = "#2e7d5b" if verdict == "Approved" else "#c0504d"
        st.markdown(f'<div class="verdict" style="background:{bg}"><h2>Likely {verdict.upper()}</h2>'
                    f'<p>Risk level: <b>{tier}</b> &nbsp;·&nbsp; Model: {metrics["final_model"]}</p></div>',
                    unsafe_allow_html=True)
        show(C.gauge(prob, colour))
        kpi_row([
            ("Loan / income", f"{d['loan_to_income']:.1f}x", inr(loan), ""),
            ("Asset cover", f"{d['asset_cover']:.1f}x", f"assets {inr(d['total_assets'])}", "green" if d["asset_cover"] >= 1 else "amber"),
            ("Repayment burden", f"{d['annual_repayment_burden']:.0%}", "of income per year, principal only", "amber"),
        ])
        st.write("")
        st.markdown("**Why?**")
        for level, text in P.insights(inputs):
            note(f"{LEVEL_ICON[level]} {text}", level)

    scores, probs = P.cibil_sweep(model, inputs)
    show(C.cibil_sweep_fig(scores, probs, cibil, prob))

    report = pd.DataFrame([{**inputs, "approval_probability": round(prob, 4), "decision": verdict, "risk_level": tier}])
    st.download_button("Download this assessment (CSV)", report.to_csv(index=False).encode("utf-8"),
                       file_name="loan_assessment.csv", mime="text/csv")
    st.caption("Decision-support demonstration trained on a public learning dataset. It must not be used for real lending decisions.")


# ===================================================================== 5. about
def render_about(metrics: dict):
    hero("About this project", "Methodology, design decisions and limitations.")
    meta = metrics["meta"]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Methodology")
        st.markdown(
            f"""
1. **Data**: {meta['rows']:,} loan applications, 11 applicant features, binary target (Approved / Rejected). Public Kaggle dataset by Archit Sharma.
2. **Cleaning**: {meta['data_quality']['missing_cells']} missing cells imputed *inside the pipeline* (median / most frequent), so values are learned from training data only.
3. **Split**: stratified 80 / 20 ({meta['train_rows']:,} train, {meta['test_rows']:,} test), fixed random state.
4. **Models**: Logistic Regression (baseline), KNN, SVM, Decision Tree, Random Forest, each tuned by grid search with 5-fold stratified cross-validation.
5. **Selection**: highest cross-validation accuracy on the training data. The test set is used only once, for reporting.
6. **Explainability**: permutation importance and a what-if CIBIL curve for each application.
7. **Serving**: one scikit-learn `Pipeline` is saved with Joblib; the app loads the same object, so training and prediction can never drift apart.
"""
        )
    with c2:
        st.subheader("Limitations (stated honestly)")
        st.markdown(
            """
- **One feature decides almost everything.** In this dataset the decision follows a CIBIL cut-off near 550, so the very high accuracy may not carry over to real lending data where many factors interact.
- **Labels are past decisions**, not repayment outcomes. The model copies earlier decisions, including any bias in them.
- **Single train / test split** of 856 records; differences of one or two applications between models are within noise.
- **Random Forest memorises the training data** (100% train accuracy); cross-validation and test results show the gap is small.
- **Not a lending tool.** Built for learning and demonstration only.
"""
        )
        st.subheader("Technology")
        st.markdown(
            '<span class="badge">Python</span><span class="badge">pandas</span><span class="badge">scikit-learn</span>'
            '<span class="badge">Plotly</span><span class="badge">Streamlit</span><span class="badge">Joblib</span>'
            '<span class="badge">pytest</span><span class="badge">Docker</span>', unsafe_allow_html=True)
    st.caption(f"Model trained {meta['trained_at']} with scikit-learn {meta['sklearn_version']} in {meta['train_seconds']} s.")
