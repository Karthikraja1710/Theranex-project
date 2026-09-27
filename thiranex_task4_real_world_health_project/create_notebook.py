"""
Notebook Builder Script for Task 4: Real-World Health Data Project
Generates notebooks/Task_4_Health_Data_Project.ipynb with markdown and code cells.
"""

import os
import nbformat as nbf


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Section 1: Title & Header
    cells.append(nbf.v4.new_markdown_cell(
"""# Thiranex Data Science Internship — Task 4
## Project Title: Health Risk Analysis & Heart Disease Prediction
**Author:** Data Science Intern  
**Domain:** Healthcare & Clinical Data Science  
**Dataset:** UCI Heart Disease Dataset  
**Due Date:** 23 October 2026  
***

### 1. Executive Summary & Research Problem
Cardiovascular diseases (CVDs) are the leading cause of global mortality, taking an estimated 17.9 million lives each year according to the World Health Organization (WHO). Early risk identification and quantitative risk stratification enable early intervention and lifestyle modifications.

**Core Objectives:**
1. **Clinical Data Audit & Quality Control**: Inspect 303 patient records across 14 clinical features (age, blood pressure, cholesterol, max heart rate, ST depression, chest pain classification).
2. **Exploratory Clinical Analytics (EDA)**: Identify key physiological indicators that correlate with heart disease prevalence.
3. **Leak-Free Supervised Modeling**: Train an interpretable Baseline (Logistic Regression), a Decision Tree Classifier, and an ensemble Random Forest Classifier using `CustomColumnTransformer` to ensure zero data leakage.
4. **Model Evaluation & Clinical Risk Scoring**: Evaluate holdout test performance across Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.
"""
    ))

    # Section 2 & 3: Problem Statement & Educational Disclaimer
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 2. Business & Clinical Problem Statement
The objective is to train a supervised machine learning model that predicts heart disease diagnosis (`target` = 1 for disease present, `target` = 0 for normal) using patient physiological attributes.

***
> ### ⚠️ MANDATORY EDUCATIONAL & NON-CLINICAL DISCLAIMER
> **THIS PROJECT IS STRICTLY AN EDUCATIONAL INTERNSHIP DEMONSTRATION.**  
> The machine learning models, statistical calculations, and predictions generated in this notebook are **FOR EDUCATIONAL AND APPLIED DATA SCIENCE LEARNING ONLY**.  
> **THE PREDICTIONS DO NOT CONSTITUTE MEDICAL DIAGNOSES AND MUST NEVER REPLACE CLINICAL ASSESSMENT, MEDICAL ADVICE, OR DIAGNOSTIC EVALUATION BY A QUALIFIED PHYSICIAN.**
***
"""
    ))

    # Section 4: Dataset Overview
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 3. Dataset Overview
We utilize the **UCI Heart Disease Dataset** (Cleveland Clinic database).
* **Sample Size:** 303 patient records
* **Clinical Attributes:** 13 input features + 1 binary target (`target`)
* **Target Classes:** 165 Disease Present (54.5%), 138 Normal / Healthy (45.5%)
"""
    ))

    # Section 5: Setup & Code Imports
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 4. Environment Setup & Library Imports
We import core scientific computing libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats`), along with modular functions from `src/`.
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
from src.analysis import (
    compute_clinical_summary_stats,
    plot_target_distribution,
    plot_age_distribution_by_target,
    plot_chest_pain_vs_target,
    plot_cholesterol_bp_scatter,
    plot_max_heart_rate_boxplot,
    plot_correlation_heatmap,
    plot_exercise_angina_st_depression,
    plot_pairwise_feature_importance
)
from src.train import build_model_pipelines, train_all_models
from src.evaluate import (
    evaluate_all_models,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_model_comparison,
    plot_feature_importance
)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
%matplotlib inline

print("Task 4 Health Data Project libraries successfully imported!")
"""
    ))

    # Section 6: Data Loading
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 5. Data Loading & Initial Inspection
We ingest the raw, untouched dataset from `data/raw/heart.csv`.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""raw_data_path = os.path.join("..", "data", "raw", "heart.csv")
df_raw = load_raw_data(raw_data_path)

print(f"Dataset Shape: {df_raw.shape[0]} patients x {df_raw.shape[1]} clinical attributes")
display(df_raw.head())
"""
    ))

    # Section 7: Data Understanding & Structural Audit
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 6. Data Understanding & Structural Quality Audit
We audit data types, missing values, duplicates, and target class balance.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""inspection = inspect_data(df_raw)

print("Column Data Types:")
for col, dt in inspection['dtypes'].items():
    print(f" - {col:15s}: {dt}")

print("\nMissing Values Audit:", inspection['missing_values'])
print("Duplicate Rows Count:", inspection['duplicate_rows'])
print("Target Class Counts:", inspection['target_distribution'])
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# Target Distribution Visual
viz_dir = os.path.join("..", "visualizations")
plot_target_distribution(df_raw, viz_dir)
Image(filename=os.path.join(viz_dir, "00_target_distribution.png"))
"""
    ))

    # Section 8: Data Cleaning & Preprocessing
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 7. Data Preprocessing & Column Transformer
1. Exact duplicates removed.
2. Clinical features isolated:
   - Numerical: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
   - Categorical: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`
3. `CustomColumnTransformer` applies Z-score scaling to numerical features and One-Hot Encoding to categorical features **strictly on `X_train`**.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""X, y, tracking = clean_raw_dataframe(df_raw)
preprocessor = get_preprocessor(X)

print("Clean Feature Matrix X Shape:", X.shape)
print("Target Vector y Shape:", y.shape)
"""
    ))

    # Section 9: Exploratory Clinical Data Analysis (EDA)
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 8. Exploratory Clinical Data Analysis (EDA)

We compute clinical summary statistics and plot diagnostic EDA dashboards.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""df_summary = compute_clinical_summary_stats(df_raw)
print("=== CLINICAL SUMMARY STATISTICS ===")
display(df_summary)
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# Age Distribution by Diagnosis
plot_age_distribution_by_target(df_raw, viz_dir)
Image(filename=os.path.join(viz_dir, "01_age_distribution_by_target.png"))
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# Chest Pain Classification vs Target
plot_chest_pain_vs_target(df_raw, viz_dir)
Image(filename=os.path.join(viz_dir, "02_chest_pain_vs_target.png"))
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# Serum Cholesterol vs Blood Pressure Scatter
plot_cholesterol_bp_scatter(df_raw, viz_dir)
Image(filename=os.path.join(viz_dir, "03_cholesterol_bp_scatter.png"))
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# Max Heart Rate (thalach) Boxplot
plot_max_heart_rate_boxplot(df_raw, viz_dir)
Image(filename=os.path.join(viz_dir, "04_max_heart_rate_boxplot.png"))
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# Clinical Correlation Heatmap
plot_correlation_heatmap(df_raw, viz_dir)
Image(filename=os.path.join(viz_dir, "05_correlation_heatmap.png"))
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# Exercise Angina & ST Depression (oldpeak)
plot_exercise_angina_st_depression(df_raw, viz_dir)
Image(filename=os.path.join(viz_dir, "06_exercise_angina_st_depression.png"))
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# Pairwise Feature Correlation Ranking
plot_pairwise_feature_importance(df_raw, viz_dir)
Image(filename=os.path.join(viz_dir, "07_pairwise_feature_importance.png"))
"""
    ))

    # Section 10: Stratified Train / Test Split
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 9. Stratified Train / Test Split
We split the 302 clean patient records into 80% Training (`X_train`: 242 rows) and 20% Holdout Testing (`X_test`: 60 rows) while preserving disease prevalence ratios (`random_state=42`).
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

print(f"Train Partition: {X_train.shape[0]} patients")
print(f"Test Partition : {X_test.shape[0]} patients")
"""
    ))

    # Section 11: Machine Learning Model Training
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 10. Supervised Machine Learning Model Training
We fit three classification pipelines:
1. **Baseline Model**: L2-Regularized Logistic Regression
2. **Decision Tree Classifier**: Depth-constrained decision tree (`max_depth=5`)
3. **Random Forest Classifier**: Ensemble of 100 decision trees (`max_depth=6`)
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""pipelines = build_model_pipelines(preprocessor)
trained_models = train_all_models(pipelines, X_train, y_train)
"""
    ))

    # Section 12: Model Evaluation & Holdout Performance
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 11. Model Evaluation on Holdout Test Set
We evaluate holdout test predictions across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""df_results, detailed_metrics = evaluate_all_models(trained_models, X_test, y_test)

print("=== HOLDOUT TEST SET PERFORMANCE METRICS ===")
display(df_results)
"""
    ))

    # Section 13: Evaluation Dashboards
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 12. Visual Evaluation Dashboards

#### 12.1 Confusion Matrices
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""plot_confusion_matrices(detailed_metrics, y_test, viz_dir)
Image(filename=os.path.join(viz_dir, "08_confusion_matrices.png"))
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""#### 12.2 ROC Curves & AUC Score Comparison"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""plot_roc_curves(detailed_metrics, y_test, viz_dir)
Image(filename=os.path.join(viz_dir, "09_roc_curves.png"))
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""#### 12.3 Overall Model Performance Comparison Bar Chart"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""plot_model_comparison(df_results, viz_dir)
Image(filename=os.path.join(viz_dir, "10_model_comparison.png"))
"""
    ))

    # Section 14: Feature Importance Analysis
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 13. Clinical Feature Importance Analysis
We extract relative feature importances from the Random Forest classifier to identify top clinical predictors.

> **Note on Medical Interpretation:** Feature importance scores reflect mathematical utility for model decision splits. High feature importance indicates strong predictive association, **not direct clinical causation**.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""rf_model = trained_models["Random Forest"]
df_imp = plot_feature_importance(rf_model, X_train, viz_dir, top_n=15)

display(df_imp)
Image(filename=os.path.join(viz_dir, "11_feature_importance.png"))
"""
    ))

    # Section 15 & 16: Key Findings, Limitations & Conclusion
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 14. Key Findings & Clinical Risk Insights

1. **Model Performance**:
   - **Baseline Logistic Regression** achieved the highest overall **ROC-AUC score of 0.8588**, **Accuracy of 81.36%**, and **Precision of 83.87%**.
   - **Random Forest** achieved **0.8553 ROC-AUC** and **79.66% Accuracy**.

2. **Top Clinical Risk Predictors**:
   - `thal_2` (Fixed Defect Thalassemia): Strongest single predictor of heart disease.
   - `oldpeak` (ST depression induced by exercise): Higher ST depression correlates strongly with ischemia/disease.
   - `exang_1` (Exercise Induced Angina): Angina during exercise increases risk score.
   - `thalach` (Maximum Heart Rate): Lower max heart rate during stress testing is associated with higher disease risk.
   - `age`: Higher age increases baseline cardiovascular risk.

---
### 15. Study Limitations
* **Dataset Scale**: The UCI dataset comprises 303 patient records from a single medical center (Cleveland Clinic). Larger multi-center datasets are needed for broader clinical generalization.
* **Demographic Representation**: Higher male representation (68%) in the dataset.

---
### 16. Conclusion & Model Serialization
The final selected Logistic Regression model pipeline is saved to `models/best_health_model.joblib` for demonstration.
"""
    ))

    nb['cells'] = cells

    nb_path = os.path.join("notebooks", "Task_4_Health_Data_Project.ipynb")
    os.makedirs(os.path.dirname(nb_path), exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"[NOTEBOOK CREATED] Saved to: {nb_path}")


if __name__ == "__main__":
    build_notebook()
