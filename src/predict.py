"""Prediction helpers used by the dashboard and by the tests."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .config import FEATURES, MODEL_PATH


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def to_frame(inputs: dict) -> pd.DataFrame:
    missing = [f for f in FEATURES if f not in inputs]
    if missing:
        raise ValueError(f"Missing input fields: {missing}")
    return pd.DataFrame([inputs])[FEATURES]


def predict_proba(model, inputs: dict) -> float:
    """Probability that the loan is approved."""
    return float(model.predict_proba(to_frame(inputs))[0, 1])


def decision(prob: float, threshold: float = 0.5) -> str:
    return "Approved" if prob >= threshold else "Rejected"


def risk_tier(prob: float) -> tuple[str, str]:
    """Map approval probability to a (label, colour) pair for display."""
    if prob >= 0.85:
        return "Low risk", "#2e7d5b"
    if prob >= 0.50:
        return "Moderate risk", "#e0a030"
    if prob >= 0.15:
        return "High risk", "#d9782d"
    return "Very high risk", "#c0504d"


def derived_metrics(inputs: dict) -> dict:
    income = max(inputs["income_annum"], 1)
    loan = max(inputs["loan_amount"], 1)
    assets = (inputs["residential_assets_value"] + inputs["commercial_assets_value"]
              + inputs["luxury_assets_value"] + inputs["bank_asset_value"])
    return {
        "loan_to_income": loan / income,
        "total_assets": assets,
        "asset_cover": assets / loan,
        "annual_repayment_burden": loan / max(inputs["loan_term"], 1) / income,
    }


def insights(inputs: dict) -> list[tuple[str, str]]:
    """Plain-language observations about the application: (level, text)."""
    out = []
    c = inputs["cibil_score"]
    d = derived_metrics(inputs)
    if c >= 750:
        out.append(("good", f"CIBIL score {c} is in the very good / excellent range."))
    elif c >= 550:
        out.append(("ok", f"CIBIL score {c} is above the 550 level that separated most approvals from rejections in the historical data."))
    else:
        out.append(("bad", f"CIBIL score {c} is below 550. In the historical data only about 10% of such applications were approved."))
    if d["loan_to_income"] > 3.5:
        out.append(("warn", f"Loan is {d['loan_to_income']:.1f}x annual income, at the high end of the range seen in the data."))
    else:
        out.append(("good", f"Loan is {d['loan_to_income']:.1f}x annual income."))
    if d["asset_cover"] >= 2:
        out.append(("good", f"Declared assets cover the loan {d['asset_cover']:.1f} times."))
    elif d["asset_cover"] < 1:
        out.append(("warn", f"Declared assets cover only {d['asset_cover']:.1f}x of the loan amount."))
    if inputs["loan_term"] >= 16:
        out.append(("warn", "Long repayment term (16+ years); longer terms had slightly lower approval rates."))
    return out


def cibil_sweep(model, inputs: dict, lo: int = 300, hi: int = 900, step: int = 10):
    """Approval probability as the CIBIL score changes, everything else fixed (what-if curve)."""
    scores = np.arange(lo, hi + 1, step)
    frame = pd.concat([to_frame(inputs)] * len(scores), ignore_index=True)
    frame["cibil_score"] = scores
    return scores, model.predict_proba(frame)[:, 1]
