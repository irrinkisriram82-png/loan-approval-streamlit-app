# Loan Approval Intelligence Dashboard

An end-to-end machine-learning project: data cleaning, model comparison, explainability and an
interactive Streamlit dashboard that predicts whether a loan application will be **Approved** or **Rejected**.

| | |
|---|---|
| **Final model** | Random Forest (selected by 5-fold cross-validation on the training data) |
| **Test accuracy** | ~98.3% (majority-class baseline 62.1%), ROC-AUC 0.998 |
| **Data** | 4,278 applications, 11 features ([Kaggle: Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)) |
| **Stack** | Python, pandas, scikit-learn, Plotly, Streamlit, Joblib, pytest, Docker |

## Dashboard pages
1. **Overview**: KPIs, decision split, approval rate by CIBIL band, key findings, data quality.
2. **Data Explorer**: filter by education, employment, CIBIL and term; distributions, scatter, correlation matrix, downloadable table.
3. **Model Performance**: leaderboard of 5 algorithms, ROC curves, confusion matrix, permutation importance, interactive threshold analysis.
4. **Decision Simulator**: enter an applicant, get a decision, probability gauge, risk level, plain-language reasons, and a what-if CIBIL curve.
5. **About**: methodology and honest limitations.

## Quick start
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py             # opens http://localhost:8501
```
The trained model is already included in `models/`. To retrain (about 30 seconds) and test:
```bash
python -m src.train
pip install -r requirements-dev.txt && pytest -q
```
Docker: `docker build -t loan-dashboard . && docker run -p 8501:8501 loan-dashboard`

## Project structure
```
app.py                 Streamlit entry point (navigation, caching)
dashboard/             charts.py (Plotly), views.py (5 pages), style.py (CSS)
src/                   config.py, data.py, train.py, predict.py
data/                  loan_approval_dataset.csv
models/                loan_model.joblib, metrics.json, test_predictions.csv
tests/                 pytest suite
```

## Design decisions
- **One Pipeline for everything.** Imputation, one-hot encoding, scaling and the classifier live in a single scikit-learn `Pipeline`, so the app applies exactly the transformations used in training and nothing is learned from test data.
- **Model selection without peeking.** The winner is chosen by cross-validation on training data; the test set is used once for reporting.
- **Explainability.** Permutation importance on raw features, plus a per-applicant what-if curve.
- **Reproducible.** Fixed random state, pinned scikit-learn version, automated tests and CI.

## Key finding and limitation
The historical decisions follow a **CIBIL score cut-off near 550** (about 10% approved below, about 99% above).
Income, loan size, education and assets add almost nothing, so the very high accuracy may not transfer to real
lending data. The labels are past decisions, not repayment outcomes. This project is for learning and
demonstration only and must not be used for real lending decisions.
