"""Central configuration: paths, feature lists and constants shared by the whole project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "loan_approval_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "loan_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
PREDS_PATH = MODEL_DIR / "test_predictions.csv"

NUMERIC = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
CATEGORICAL = ["education", "self_employed"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "loan_status"
POSITIVE = "Approved"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

CIBIL_BINS = [299, 549, 649, 749, 799, 900]
CIBIL_LABELS = ["Poor (300-549)", "Fair (550-649)", "Good (650-749)",
                "Very good (750-799)", "Excellent (800-900)"]

LABELS = {
    "no_of_dependents": "Dependents",
    "income_annum": "Annual income",
    "loan_amount": "Loan amount",
    "loan_term": "Loan term (years)",
    "cibil_score": "CIBIL score",
    "residential_assets_value": "Residential assets",
    "commercial_assets_value": "Commercial assets",
    "luxury_assets_value": "Luxury assets",
    "bank_asset_value": "Bank assets",
    "education": "Education",
    "self_employed": "Self-employed",
}

# Colour palette used by every chart so the dashboard looks consistent.
C_APPROVED = "#2e7d5b"
C_REJECTED = "#c0504d"
C_PRIMARY = "#1f4e79"
C_ACCENT = "#e0a030"
C_GREY = "#8a94a6"
