"""Train, tune, compare and save the loan-approval models.

Run from the project root:   python -m src.train

Every model is a scikit-learn Pipeline (imputation -> encoding -> scaling -> classifier),
so the *same* object is used for training, evaluation and the web app. This removes the
risk of the app transforming inputs differently from training.
"""
from __future__ import annotations

import json
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # SVC(probability=True) notice in sklearn 1.9
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import (CATEGORICAL, CV_FOLDS, FEATURES, METRICS_PATH, MODEL_DIR, MODEL_PATH,
                     NUMERIC, PREDS_PATH, RANDOM_STATE, TEST_SIZE)
from .data import data_quality_report, load_raw, split_xy


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def model_specs():
    """(estimator, hyper-parameter grid) for each algorithm compared in the project."""
    return {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                                {"clf__C": [1.0]}),
        "KNN": (KNeighborsClassifier(),
                {"clf__n_neighbors": [3, 5, 7, 9], "clf__weights": ["uniform", "distance"]}),
        "SVM": (SVC(probability=True, random_state=RANDOM_STATE),
                {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                          {"clf__criterion": ["gini", "entropy"],
                           "clf__max_depth": [3, 5, 7, None],
                           "clf__min_samples_leaf": [1, 2, 4]}),
        "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE),
                          {"clf__n_estimators": [64, 128, 256],
                           "clf__max_features": [2, 3, 4],
                           "clf__bootstrap": [True, False]}),
    }


def evaluate(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def _downsample_roc(y_true, proba, points=150):
    fpr, tpr, _ = roc_curve(y_true, proba)
    idx = np.unique(np.linspace(0, len(fpr) - 1, points).astype(int))
    return {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}


def main():
    t0 = time.time()
    df = load_raw()
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results, roc, fitted, probas = {}, {}, {}, {}
    for name, (estimator, grid) in model_specs().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1).fit(X_tr, y_tr)
        best = gs.best_estimator_
        proba = best.predict_proba(X_te)[:, 1]
        m = evaluate(y_te, proba)
        m.update({
            "cv_accuracy": float(gs.best_score_),
            "cv_std": float(gs.cv_results_["std_test_score"][gs.best_index_]),
            "train_accuracy": float(accuracy_score(y_tr, best.predict(X_tr))),
            "best_params": {k.replace("clf__", ""): (v if v is None or isinstance(v, (bool, int, float, str)) else str(v))
                            for k, v in gs.best_params_.items()},
            "search_space": {k.replace("clf__", ""): [str(v) for v in vs] for k, vs in grid.items()},
        })
        results[name], fitted[name], probas[name] = m, best, proba
        roc[name] = _downsample_roc(y_te, proba)
        print(f"{name:20s} test acc {m['accuracy']:.4f}  auc {m['roc_auc']:.4f}  cv {m['cv_accuracy']:.4f}")

    # Model selection uses cross-validation on the TRAINING data only (never the test set).
    final_name = max(results, key=lambda n: (results[n]["cv_accuracy"], results[n]["roc_auc"]))
    final = fitted[final_name]
    final_proba = probas[final_name]
    pred = (final_proba >= 0.5).astype(int)

    perm = permutation_importance(final, X_te, y_te, n_repeats=10, scoring="accuracy",
                                  random_state=RANDOM_STATE, n_jobs=-1)
    importance = {f: {"mean": float(m), "std": float(s)}
                  for f, m, s in zip(FEATURES, perm.importances_mean, perm.importances_std)}

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    pd.DataFrame({"y_true": y_te.values, "proba": final_proba, "y_pred": pred}).to_csv(PREDS_PATH, index=False)

    metrics = {
        "meta": {
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "sklearn_version": sklearn.__version__,
            "random_state": RANDOM_STATE,
            "train_seconds": round(time.time() - t0, 1),
            "rows": int(len(X)),
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
            "test_approved": int(y_te.sum()),
            "test_rejected": int((1 - y_te).sum()),
            "majority_baseline": float(max(y_te.mean(), 1 - y_te.mean())),
            "data_quality": data_quality_report(df),
        },
        "final_model": final_name,
        "models": results,
        "roc": roc,
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "permutation_importance": importance,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSelected model: {final_name}  ->  saved to {MODEL_PATH.name}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
