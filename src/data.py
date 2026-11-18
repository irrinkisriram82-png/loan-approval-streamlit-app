"""Data loading, cleaning and feature preparation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (CATEGORICAL, CIBIL_BINS, CIBIL_LABELS, DATA_PATH, FEATURES,
                     NUMERIC, POSITIVE, TARGET)


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Read the CSV and strip stray whitespace from column names and text values."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype("string").str.strip()
            df[col] = s.astype(object).where(s.notna(), np.nan)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def split_xy(df: pd.DataFrame):
    """Return the feature table and the 0/1 target (1 = Approved).

    Missing values are deliberately left in X: imputation happens inside the
    model pipeline, so it is learned from the training data only.
    """
    df = df.dropna(subset=[TARGET])
    X = df[FEATURES].copy()
    y = (df[TARGET] == POSITIVE).astype(int)
    return X, y


def prepare_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Imputed copy of the data with extra analytic columns, used by the dashboard."""
    d = df.copy()
    for col in NUMERIC:
        d[col] = d[col].fillna(d[col].median())
    for col in CATEGORICAL + [TARGET]:
        d[col] = d[col].fillna(d[col].mode()[0])
    d["loan_term"] = d["loan_term"].round().astype(int)
    d["approved"] = (d[TARGET] == POSITIVE).astype(int)
    d["cibil_band"] = pd.cut(d["cibil_score"], CIBIL_BINS, labels=CIBIL_LABELS)
    d["total_assets"] = (d["residential_assets_value"] + d["commercial_assets_value"]
                         + d["luxury_assets_value"] + d["bank_asset_value"])
    d["loan_to_income"] = d["loan_amount"] / d["income_annum"].replace(0, np.nan)
    d["education_label"] = d["education"].replace({"NotGraduate": "Not Graduate"})
    return d


def data_quality_report(df: pd.DataFrame) -> dict:
    """Small summary used on the dashboard (rows, missing cells, duplicates)."""
    miss = df.isnull().sum()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_cells": int(miss.sum()),
        "columns_with_missing": int((miss > 0).sum()),
        "duplicates": int(df.duplicated().sum()),
    }
