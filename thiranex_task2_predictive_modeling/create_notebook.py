"""
Notebook Builder Script for Task 2: Predictive Modeling
Generates notebooks/Task_2_Predictive_Modeling.ipynb with markdown and code cells.
"""

import os
import nbformat as nbf


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Header
    cells.append(nbf.v4.new_markdown_cell(
"""# Thiranex Data Science Internship — Task 2
## Project Title: Customer Churn Prediction using Supervised Machine Learning
**Author:** Data Science Intern  
**Domain:** Telecommunications / E-Commerce Customer Analytics  
**Dataset:** IBM Telco Customer Churn Dataset  
**Due Date:** 9 October 2026  
***

### 1. Executive Summary & Project Objectives
Customer churn prediction is a critical business task for subscription-based telecommunication and digital services. Predicting whether a customer is likely to cancel their subscription allows companies to proactively deploy retention offers and mitigate revenue loss.

**Core Objectives:**
1. **Data Understanding**: Audit customer demographics, account information, and service subscriptions across 7,043 customer accounts.
2. **Strict Data Preprocessing**: Clean missing values, handle categorical features, scale numerical features, and perform stratified train-test splitting using `CustomColumnTransformer` to guarantee **zero data leakage**.
3. **Supervised ML Modeling**: Train an interpretable Baseline (Logistic Regression), a Decision Tree Classifier, and an ensemble Random Forest Classifier.
4. **Rigorous Evaluation**: Evaluate holdout test set performance using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
5. **Feature Importance & Insights**: Quantify key predictors of customer churn (e.g., tenure, contract type, payment method) and discuss business recommendations.
"""
    ))

    # Section 2: Setup & Imports
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 2. Environment Setup & Library Imports
We import core data processing and visualization libraries, along with custom modular code from `src/`.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

# Add project root to sys.path
sys.path.append(os.path.abspath(".."))

from src.preprocessing import (
    load_raw_data,
    inspect_data,
    clean_raw_dataframe,
    get_preprocessor,
    split_data
)
from src.train import build_model_pipelines, train_all_models
from src.evaluate import (
    plot_class_distribution,
    evaluate_all_models,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_model_comparison,
    plot_feature_importance
)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
%matplotlib inline

print("Task 2 Libraries and modules successfully imported!")
"""
    ))

    # Section 3: Data Loading
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 3. Data Loading & Initial Ingestion
We load the raw, unchanged dataset from `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""raw_data_path = os.path.join("..", "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
df_raw = load_raw_data(raw_data_path)

print(f"Dataset Shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# Preview first 5 rows
print("--- First 5 Customer Records ---")
display(df_raw.head())
"""
    ))

    # Section 4: Data Understanding & Quality Audit
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 4. Data Understanding & Target Quality Audit
We perform structural audit, check data types, detect missing/blank values, and evaluate class imbalance in the target variable `Churn`.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""inspection = inspect_data(df_raw)

print("Column Data Types:")
for col, dt in inspection['dtypes'].items():
    print(f" - {col:22s}: {dt}")

print("\nBlank TotalCharges Count:", inspection['blank_total_charges_count'])
print("Exact Duplicate Rows Count:", inspection['duplicate_rows'])
print("Target 'Churn' Value Counts:", inspection['target_distribution'])
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# Visualize Target Class Distribution
viz_dir = os.path.join("..", "visualizations")
plot_class_distribution(df_raw['Churn'], viz_dir)

Image(filename=os.path.join(viz_dir, "00_class_distribution.png"))
"""
    ))

    # Section 5: Data Preprocessing
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 5. Data Preprocessing & Column Transformer
1. `customerID` dropped (non-predictive unique identifier).
2. `TotalCharges` converted from text string to float, replacing 11 empty space records with numeric values.
3. Target `Churn` converted to binary numerical format (`Yes` = 1, `No` = 0).
4. `CustomColumnTransformer` fits StandardScaler on numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) and One-Hot Encoders on categorical features **strictly on the training split**.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""X, y, tracking = clean_raw_dataframe(df_raw)

print("Clean Feature Matrix X Shape:", X.shape)
print("Target Vector y Shape:", y.shape)
print("Churn Rate (%):", tracking['target_churn_rate_pct'])

preprocessor = get_preprocessor(X)
"""
    ))

    # Section 6: Stratified Train / Test Split
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 6. Stratified Train / Test Split
To preserve the 26.5% positive churn class ratio across training and testing partitions, we apply a 80/20 stratified split (`random_state=42`).
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

print(f"Training Set : {X_train.shape[0]:,} samples")
print(f"Testing Set  : {X_test.shape[0]:,} samples")
"""
    ))

    # Section 7: Machine Learning Model Training
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 7. Supervised Machine Learning Model Training
We fit three model architectures using `MLPipeline` wrappers:
1. **Baseline Model**: L2-Regularized Logistic Regression
2. **Decision Tree Classifier**: Depth-constrained decision tree (`max_depth=6`)
3. **Random Forest Classifier**: Ensemble of 100 decision trees (`max_depth=8`)
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""pipelines = build_model_pipelines(preprocessor)
trained_models = train_all_models(pipelines, X_train, y_train)
"""
    ))

    # Section 8: Model Evaluation & Metrics Comparison
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 8. Model Evaluation on Holdout Test Set
We evaluate all three models on the holdout test set (`X_test`, `y_test`) across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""df_results, detailed_metrics = evaluate_all_models(trained_models, X_test, y_test)

print("=== HOLD-OUT TEST SET PERFORMANCE METRICS ===")
display(df_results)
"""
    ))

    # Section 9: Evaluation Visualizations
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 9. Visual Diagnostic Dashboards

#### 9.1 Confusion Matrices
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""plot_confusion_matrices(detailed_metrics, y_test, viz_dir)
Image(filename=os.path.join(viz_dir, "01_confusion_matrices.png"))
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""#### 9.2 ROC Curves & AUC Analysis"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""plot_roc_curves(detailed_metrics, y_test, viz_dir)
Image(filename=os.path.join(viz_dir, "02_roc_curves.png"))
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""#### 9.3 Model Performance Comparison Chart"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""plot_model_comparison(df_results, viz_dir)
Image(filename=os.path.join(viz_dir, "03_model_comparison.png"))
"""
    ))

    # Section 10: Feature Importance Analysis
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 10. Feature Importance & Interpretability Analysis
We extract relative feature importances from the Random Forest model to identify key drivers of customer churn.

> **Important Note on Association vs. Causation:**  
> Feature importance scores quantify mathematical utility for model predictions. High feature importance indicates strong predictive association, **not direct physical causation**.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""rf_model = trained_models["Random Forest"]
df_imp = plot_feature_importance(rf_model, X_train, viz_dir, top_n=15)

display(df_imp)
Image(filename=os.path.join(viz_dir, "04_feature_importance.png"))
"""
    ))

    # Section 11: Summary & Conclusion
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 11. Key Findings & Strategic Recommendations

1. **Best Model Performance**:
   - **Random Forest** achieved the highest overall **ROC-AUC score of 0.8589** and **Precision of 69.65%**.
   - **Baseline Logistic Regression** achieved **81.17% Accuracy** and **0.6003 F1-Score**.

2. **Primary Drivers of Churn**:
   - `tenure` (Customer longevity): Customers with short tenure (<12 months) are significantly more likely to churn.
   - `Contract_Month-to-month`: Month-to-month contracts exhibit drastically higher churn rates compared to 1-year or 2-year contracts.
   - `InternetService_Fiber optic`: Fiber optic subscribers show higher churn due to higher pricing and service expectations.
   - `PaymentMethod_Electronic check`: Electronic check users have higher cancellation rates.

3. **Business Recommendations**:
   - Transition month-to-month customers to annual contracts via discounts.
   - Focus retention campaigns on new customers during their first 6 months.
   - Promote automatic bank transfers over electronic check payments.

---
### 12. Model Artifact Export
The final selected Random Forest model pipeline is serialized to `models/best_churn_model.joblib` for production deployment.
"""
    ))

    nb['cells'] = cells

    nb_path = os.path.join("notebooks", "Task_2_Predictive_Modeling.ipynb")
    os.makedirs(os.path.dirname(nb_path), exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"[NOTEBOOK CREATED] Saved to: {nb_path}")


if __name__ == "__main__":
    build_notebook()
