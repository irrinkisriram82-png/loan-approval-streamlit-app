# Viva / external examination guide

## 60-second pitch
"I built a loan-approval prediction system. I cleaned a 4,278-record dataset, compared five algorithms
(Logistic Regression, KNN, SVM, Decision Tree, Random Forest) with grid search and 5-fold cross-validation,
and selected the Random Forest, which reaches about 98% accuracy and 0.998 ROC-AUC on unseen data.
The model is served through a Streamlit dashboard with data exploration, model diagnostics and a live decision simulator."

## Suggested demo order (5 minutes)
1. **Overview**: show the KPIs and the approval-by-CIBIL chart: "CIBIL below 550 means about 10% approval, above 550 about 99%."
2. **Data Explorer**: drag the CIBIL slider; show that the approval rate collapses below 550.
3. **Model Performance**: leaderboard, ROC curves, confusion matrix, then move the threshold slider.
4. **Decision Simulator**: load "Just above the cut-off (CIBIL 552)" then "Just below (CIBIL 545)". The decision flips; show the what-if curve.
5. **About**: say the limitations yourself before they ask.

## Likely questions and answers
- **Why Random Forest, when the Decision Tree scored slightly higher on the test set?** Selection used cross-validation on training data (RF 98.3% vs 97.8%) and RF has the best ROC-AUC. The test-set gap is one or two applications, which is noise on 856 records.
- **Is 98% accuracy too good?** Yes, and I say so. The dataset's decision is almost a single CIBIL cut-off, which trees capture with one split. Real lending data would be harder.
- **How did you avoid data leakage?** Imputation and scaling are inside the Pipeline and fitted on training folds only; the test set is used once; `loan_id` is excluded.
- **Why does Logistic Regression score lower?** It draws a smooth linear boundary, while the real rule is a sharp step.
- **Which error matters more?** A wrong approval (false positive). The threshold slider shows how raising the threshold reduces them at the cost of recall.
- **What is permutation importance?** Shuffle one feature on the test set and measure the accuracy drop; a big drop means the model relies on it.
- **Why is precision/recall reported, not only accuracy?** Classes are 62/38, so accuracy alone can mislead.
- **What would you improve?** Real repayment-outcome data, gradient boosting with SHAP, fairness checks across groups, threshold chosen by cost.
