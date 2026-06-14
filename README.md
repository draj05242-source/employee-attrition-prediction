# 🧑‍💼 Employee Attrition Prediction

> Predicting employee attrition using feature engineering, feature selection, and ensemble classification models — built for a Hackathon (Group 5, Mentor 1).

---

## 📌 Overview

Employee attrition is one of the costliest HR challenges for organizations. This project builds an end-to-end machine learning pipeline to **predict which employees are at risk of leaving**, enabling proactive HR intervention before talent is lost.

The pipeline covers the full data science lifecycle: data understanding → cleaning → feature engineering → feature selection → model training → evaluation → business interpretation.

---

## 🗂️ Project Structure

```
├── Hackathon_Mentor1_Group_5.ipynb   # Main notebook (all parts)
├── Mentor 1 data set.csv             # Input dataset (place in /content/)
└── README.md
```

---

## 🔄 Pipeline Walkthrough

### Part 1 — Data Understanding
- Load and inspect the dataset (shape, dtypes, summary stats)
- Identify categorical vs. numerical columns
- Analyze and visualize the **target variable** (`Attrition`) distribution
- Detect class imbalance between "Stayed" and "Left" employees

### Part 2 — Data Cleaning
- Detect and impute missing values:
  - **Numerical** (`Distance_From_Home_KM`, `Job_Satisfaction`, `Training_Hours`) → median imputation
  - **Categorical** (`Education_Level`) → mode imputation
- Remove duplicate rows
- Verify data types and final dataset shape

### Part 3 — Feature Engineering
Six domain-driven features are created to enhance predictive power:

| Feature | Description |
|---|---|
| `Work_Pressure_Index` | Weekly hours ÷ job satisfaction — captures overwork-dissatisfaction interaction |
| `Salary_Per_Level` | Monthly salary ÷ job level — flags underpaid employees relative to seniority |
| `Commute_Stress` | Distance × weekly hours — quantifies daily commute burden |
| `Experience_Category` | Tenure buckets: New / Junior / Mid / Senior |
| `Perf_Sat_Gap` | Performance rating − job satisfaction — flags disengaged high performers |
| `Low_Pay_High_Hours` | Binary flag: bottom 25% salary AND top 25% work hours |

### Part 4 — Feature Selection
Four methods are used and aggregated via **consensus ranking**:
- **Pearson Correlation** with the target
- **Random Forest Feature Importance**
- **Mutual Information**
- **Recursive Feature Elimination (RFE)** → selects top 10 features for model training

### Part 5 — Model Building
Three ensemble classifiers trained on RFE-selected features:

| Model | Key Hyperparameters |
|---|---|
| **Random Forest** | 300 trees, `class_weight='balanced'`, max depth 8 |
| **XGBoost** | 300 estimators, `scale_pos_weight` for imbalance, LR 0.05 |
| **Gradient Boosting** | 300 estimators, sample weights for balance, LR 0.05 |

All models use `StandardScaler` and a stratified 80/20 train-test split.

### Part 6 — Model Evaluation
Models are compared across five metrics:
- Accuracy, Precision, **Recall** (primary), F1 Score, **ROC AUC**

Visualizations include: metrics bar chart, confusion matrices, and ROC curves.

> 🏆 **Selection criterion:** Recall is prioritized — missing a future leaver is more costly to the business than a false alarm.

### Part 7 — Business Interpretation
- **Top attrition drivers** identified from feature importance
- **Overtime analysis**: employees on overtime leave at significantly higher rates
- **Salary band analysis**: lower-paid employees show higher attrition
- **High-risk employee identification**: individual risk scores and risk tiers (Low / Medium / High)
- **Department-level attrition rates** visualized

---

## 📊 Key Business Insights

1. **Job satisfaction, monthly salary, and work hours** are the strongest predictors of attrition.
2. **Overtime employees leave at significantly higher rates** — auditing mandatory overtime policies could reduce turnover.
3. **Lower salary bands drive attrition** — targeted retention bonuses for at-risk groups are a high-ROI intervention.
4. **Highest-risk profile**: younger, lower-paid employees with low job satisfaction, long commutes, and overtime.
5. **Actionable HR lever**: Focus retention efforts on employees flagged as **High Risk**, especially those with high `Work_Pressure_Index` and low `Job_Satisfaction`.

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` |
| Boosting | `xgboost` |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/employee-attrition-prediction.git
cd employee-attrition-prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 3. Add the dataset
Place `YourData.csv` in the `/content/` directory (or update the path in the notebook).

### 4. Run the notebook
Open `Hackathon_Mentor1_Group_5.ipynb` in Jupyter or Google Colab and run all cells.

---

## 📈 Expected Dataset Columns

The dataset should include (at minimum):

`Employee_ID`, `Age`, `Department`, `Attrition`, `Monthly_Salary`, `Job_Level`, `Job_Satisfaction`, `Performance_Rating`, `Work_Hours_Per_Week`, `Distance_From_Home_KM`, `Years_At_Company`, `Overtime`, `Training_Hours`, `Education_Level`

---
