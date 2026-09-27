# Thiranex Data Science Internship — Executive Project Summary

## Overview of Completed Tasks

### Task 1: Data Cleaning & Visualization Project
* **Project Folder:** `01_Data_Cleaning_Visualization/`
* **Domain:** Retail / E-Commerce Sales Performance Analytics
* **Dataset:** UCI Online Retail Dataset (`Online_Retail.xlsx`, 541,909 raw records)
* **Key Results:**
  - Audited and cleaned 541,909 raw transactional records, removing 5,268 exact duplicates and handling 135,037 guest checkouts without losing revenue.
  - Processed 524,878 clean sales rows (96.86% data retention).
  - Analyzed £10,642,110.80 (~£10.64 Million) total net revenue across 19,960 unique orders.
  - Generated 8 high-resolution dashboard figures (Distribution, Outliers, Monthly Trends, Top Products, Regional Breakdown, Correlations, Shopping Hours).

---

### Task 2: Predictive Modeling Using Machine Learning
* **Project Folder:** `02_Predictive_Modeling/`
* **Domain:** Customer Churn Prediction & Retention Analytics
* **Dataset:** IBM Telco Customer Churn Dataset (7,043 customer accounts)
* **Key Results:**
  - Implemented `CustomColumnTransformer` and `MLPipeline` wrappers ensuring 0% data leakage.
  - Applied 80/20 Stratified Train/Test Split (5,636 train / 1,407 test records).
  - Trained Baseline Logistic Regression, Decision Tree, and Random Forest models.
  - **Random Forest achieved 0.8589 ROC-AUC and 69.65% Precision** on the holdout test set.
  - Saved model artifact to `models/best_churn_model.joblib` and created inference script `src/predict.py`.

---

### Task 3: Exploratory Data Analysis (EDA) Project
* **Project Folder:** `03_Exploratory_Data_Analysis/`
* **Domain:** World Happiness / Socioeconomic Analytics
* **Dataset:** World Happiness Report 2021 Dataset (149 countries across 10 regions)
* **Key Results:**
  - Computed univariate descriptive statistics (Mean, Median, StdDev, IQR, Skewness) for all numerical factors.
  - Computed Pearson ($r$) and Spearman ($\rho$) correlation matrices.
  - Identified Logged GDP per Capita ($r = +0.7898$) and Healthy Life Expectancy ($r = +0.7681$) as top predictors of happiness.
  - Identified Finland (7.842) as happiest nation and Afghanistan (2.523) as lowest ranked nation.
  - Saved 8 visualization figures in `visualizations/`.

---

### Task 4: Real-World Data Project (Health Domain)
* **Project Folder:** `04_Real_World_Project/`
* **Domain:** Health Risk Analysis & Heart Disease Prediction
* **Dataset:** UCI Heart Disease Dataset (303 patient records)
* **Key Results:**
  - Audited 14 clinical features and generated 8 clinical EDA figures.
  - Trained Baseline Logistic Regression, Decision Tree, and Random Forest classifiers on 80/20 stratified split.
  - **Baseline Logistic Regression achieved 81.36% Accuracy, 83.87% Precision, 81.25% Recall, and 0.8588 ROC-AUC**.
  - Identified `thal_2` (Fixed Defect Thalassemia) and `oldpeak` (Exercise ST Depression) as top diagnostic predictors.
  - Serialized model pipeline to `models/best_health_model.joblib` and created inference script `src/predict.py`.
  - Incorporated mandatory **Educational & Non-Clinical Disclaimer**.

---

## Technical Deliverables Check Matrix

| Project Folder | Raw Data | Clean Data | Notebook | Python Src | Visualizations | Joblib Model | Report | README |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01_Data_Cleaning_Visualization** | Yes | Yes | Yes | Yes | 8 PNGs | N/A | Yes | Yes |
| **02_Predictive_Modeling** | Yes | Yes | Yes | Yes | 5 PNGs | Yes | Yes | Yes |
| **03_Exploratory_Data_Analysis** | Yes | Yes | Yes | Yes | 8 PNGs | N/A | Yes | Yes |
| **04_Real_World_Project** | Yes | Yes | Yes | Yes | 12 PNGs | Yes | Yes | Yes |
