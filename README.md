# employee-attrition-prediction
End-to-end ML pipeline predicting employee attrition using Random Forest, XGBoost, and Gradient Boosting and finding out which outperforms the other two models
# Employee Attrition Prediction
> Binary Classification · Ensemble ML · HR Analytics · 2025

## Problem
Predict which employees are likely to leave, enabling proactive HR intervention
before attrition happens — saving cost and institutional knowledge.

## Dataset
1,500 employees · 19 features · 10.5% minority class (attrition = yes)

## My Approach
- Handled class imbalance using `class_weight='balanced'` and sample weighting
- Engineered 6 domain features: Work Pressure Index, Salary Per Level,
  Commute Stress, Perf-Sat Gap, Low Pay High Hours flag, Experience Category
- Feature selection: averaged ranks across Pearson, RF Importance,
  Mutual Information, and RFE — selected final 10 features
- Trained and tuned Random Forest (300 trees), XGBoost, Gradient Boosting
  on stratified 80/20 split with StandardScaler preprocessing

## Results
| Model | Recall | ROC-AUC | F1 |
|---|---|---|---|
| Random Forest ✅ | 21.88% | 0.654 | 25% |
| XGBoost | — | — | — |
| Gradient Boosting | — | — | — |

**Why Recall?** Missing an at-risk employee costs more than a false alarm —
the same reasoning used in model risk frameworks.

## Business Recommendation
- Flag employees with model score ≥ 0.6 for proactive HR intervention
- Enforce overtime caps in high Work Pressure Index segments
- Conduct salary benchmarking in low pay bands

## Tech Stack
`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Matplotlib` `Seaborn`

## What I'd do differently
- Collect more attrition-positive samples to improve Recall
- Try SHAP values for better explainability of individual predictions
- Deploy as a lightweight Flask API for HR teams to query in real time

## Run it yourself
```bash
pip install pandas scikit-learn xgboost matplotlib seaborn
```
Open `notebook.ipynb` in Jupyter or Google Colab.
