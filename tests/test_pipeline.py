"""Automated tests. Run with:  pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest

from src import predict as P
from src.config import FEATURES, METRICS_PATH, MODEL_PATH, PREDS_PATH
from src.data import load_raw, prepare_for_analysis, split_xy

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m src.train` first")

STRONG = dict(no_of_dependents=2, education="Graduate", self_employed="No", income_annum=9_600_000,
              loan_amount=29_900_000, loan_term=12, cibil_score=778, residential_assets_value=2_400_000,
              commercial_assets_value=17_600_000, luxury_assets_value=22_700_000, bank_asset_value=8_000_000)
WEAK = dict(STRONG, cibil_score=417, education="NotGraduate", self_employed="Yes")


@pytest.fixture(scope="module")
def model():
    return P.load_model()


def test_data_loads_and_is_clean():
    df = load_raw()
    assert df.shape == (4278, 13)
    assert df.duplicated().sum() == 0
    assert set(df["education"].dropna().unique()) == {"Graduate", "NotGraduate"}
    X, y = split_xy(df)
    assert list(X.columns) == FEATURES and set(y.unique()) == {0, 1}


def test_analysis_frame_has_no_missing_values():
    d = prepare_for_analysis(load_raw())
    assert d.isnull().sum().sum() == 0
    assert {"approved", "cibil_band", "loan_to_income", "total_assets"} <= set(d.columns)


def test_probability_in_range_and_decisions(model):
    assert 0 <= P.predict_proba(model, STRONG) <= 1
    assert P.decision(P.predict_proba(model, STRONG)) == "Approved"
    assert P.decision(P.predict_proba(model, WEAK)) == "Rejected"


def test_cibil_cutoff_behaviour(model):
    """The historical data has a cut-off at CIBIL 550; the model should reflect it."""
    low = P.predict_proba(model, dict(STRONG, cibil_score=500))
    high = P.predict_proba(model, dict(STRONG, cibil_score=700))
    assert low < 0.2 and high > 0.8


def test_sweep_shape(model):
    scores, probs = P.cibil_sweep(model, STRONG)
    assert len(scores) == len(probs) == 61 and probs.min() >= 0 and probs.max() <= 1


def test_missing_field_raises():
    with pytest.raises(ValueError):
        P.to_frame({"cibil_score": 700})


def test_unknown_category_does_not_crash(model):
    assert 0 <= P.predict_proba(model, dict(STRONG, education="Other")) <= 1


def test_saved_metrics_consistent():
    m = json.loads(METRICS_PATH.read_text())
    preds = pd.read_csv(PREDS_PATH)
    acc = (preds["y_true"] == preds["y_pred"]).mean()
    assert np.isclose(acc, m["models"][m["final_model"]]["accuracy"], atol=1e-4)
    assert m["models"][m["final_model"]]["accuracy"] > m["meta"]["majority_baseline"]


def test_risk_tiers():
    assert P.risk_tier(0.95)[0] == "Low risk" and P.risk_tier(0.02)[0] == "Very high risk"
