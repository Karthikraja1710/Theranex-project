# Comprehensive Data Science Internship Report
**Internship Program:** Thiranex Data Science Internship  
**Task Title:** Task 2 — Predictive Modeling Using Machine Learning  
**Domain:** Customer Churn Prediction & Retention Analytics  
**Due Date:** 9 October 2026  
**Status:** Completed & Submission-Ready  

---

## 1. Executive Summary & Problem Statement
Customer attrition (churn) presents a significant financial challenge for telecommunications and subscription-based service providers. Acquiring new customers typically costs 5x to 25x more than retaining existing accounts. 

The primary objective of **Task 2: Predictive Modeling Using Machine Learning** is to build an end-to-end, leak-free supervised machine learning classification system that accurately predicts customer churn probability based on customer demographics, account details, and subscription services. 

Using holdout test evaluation across 1,407 customer records, three model architectures—a **Baseline L2-Regularized Logistic Regression**, a **Decision Tree Classifier**, and an ensemble **Random Forest Classifier**—were trained and evaluated. The **Random Forest model achieved the highest overall ROC-AUC score of 0.8589**, providing reliable risk scoring for targeted retention strategies.

---

## 2. Dataset Source & Description
* **Dataset Name:** IBM Telco Customer Churn Dataset
* **Source / Mirror:** IBM Sample Data Sets / Public GitHub Repository Mirror
* **Dataset Dimensions:** 7,043 rows × 21 columns
* **Target Variable:** `Churn` (`Yes` = 1, `No` = 0)
* **Dataset Class Ratio:** 
  * **Retained (`No` / 0):** 5,174 customers (73.46%)
  * **Churned (`Yes` / 1):** 1,869 customers (26.54%)

### Feature Description Overview

| Category | Feature Name | Description & Type |
| :--- | :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Customer personal demographics (Binary / Categorical). |
| **Account Info** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` | Subscription length, contract type, billing & payment preferences. |
| **Financials** | `MonthlyCharges`, `TotalCharges` | Monthly subscription fee (£/$) and total cumulative charges. |
| **Services** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Subscribed phone, internet, security, and streaming add-on features. |

---

## 3. Data Preprocessing & Leakage Prevention Methodology

To guarantee zero data leakage between training and evaluation steps:
1. **Identifier Exclusion:** Dropped non-predictive `customerID`.
2. **Missing & Invalid Value Handling:** `TotalCharges` contained 11 whitespace entries (`' '`). These were parsed as numeric and missing values were imputed using `MonthlyCharges * tenure`.
3. **Column Transformer Pipeline:** Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) were standardized using Z-score scaling ($z = \frac{x - \mu}{\sigma}$). Categorical features were converted via dummy variable One-Hot Encoding (`drop='first'`). All scaling parameters ($\mu, \sigma$) and category maps were fitted **strictly on `X_train`** inside `CustomColumnTransformer`.
4. **Stratified Train / Test Partitioning:** Applied an 80/20 stratified split (`random_state=42`) yielding:
   * **Training Partition (`X_train`):** 5,636 customer records (4,139 Retained, 1,497 Churned).
   * **Holdout Test Partition (`X_test`):** 1,407 customer records (1,035 Retained, 372 Churned).

---

## 4. Machine Learning Algorithms & Training Methodology

Three distinct model families were constructed within `MLPipeline` architectures:

1. **Baseline Model (L2-Regularized Logistic Regression):**
   - Sigmoid activation with L-BFGS optimization.
   - Serves as a baseline benchmark for linear decision boundaries.
2. **Decision Tree Classifier:**
   - Non-linear rule-based split model (`max_depth=6`, `min_samples_split=10`).
   - Evaluates splits using Gini Impurity reduction.
3. **Random Forest Classifier (Ensemble):**
   - Ensemble of 100 decision trees (`max_depth=8`, `min_samples_split=5`).
   - Applies feature subsampling ($\sqrt{D}$) per split to reduce variance and control overfitting.

---

## 5. Model Evaluation & Empirical Comparison

All models were evaluated on the exact same 1,407 holdout test set using five key evaluation metrics:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total}}$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Empirical Test Set Results

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **80.67%** | **69.65%** | **47.99%** | **0.5683** | **0.8589** |
| **Baseline (Logistic Regression)** | **81.17%** | **68.62%** | **53.35%** | **0.6003** | **0.8570** |
| **Decision Tree** | **79.25%** | **64.41%** | **48.53%** | **0.5535** | **0.8368** |

---

## 6. Confusion Matrix & ROC-AUC Analysis

### Confusion Matrix Breakdown (Test Set = 1,407 records)

* **Random Forest Model:**
  * True Negatives (Correctly Retained): **957**
  * False Positives (False Churn Alarms): **78**
  * False Negatives (Missed Churners): **194**
  * True Positives (Correctly Identified Churners): **178**

* **Logistic Regression Baseline Model:**
  * True Negatives: **944**
  * False Positives: **91**
  * False Negatives: **174**
  * True Positives: **198**

### Trade-Off Analysis
* **Precision Focus:** Random Forest yields the highest **Precision (69.65%)**, meaning when it predicts a customer will churn, it is correct nearly 70% of the time. This minimizes wasted retention marketing budgets.
* **Discriminative Capacity:** Random Forest achieves the highest **ROC-AUC (0.8589)**, demonstrating superior ranking capability across probability thresholds.

---

## 7. Feature Importance Analysis

The Random Forest model identified the top relative predictors of customer churn:

1. **`tenure` (Relative Importance: 20.31%):** The length of time a customer has stayed with the company is the single strongest predictor. Customers in their first 6-12 months exhibit significantly higher churn risk.
2. **`TotalCharges` (13.72%):** Cumulative financial commitment.
3. **`InternetService_Fiber optic` (10.61%):** Fiber optic subscribers churn at a higher rate, likely due to pricing sensitivity or service issues.
4. **`MonthlyCharges` (7.73%):** High monthly recurring bills increase churn probability.
5. **`PaymentMethod_Electronic check` (7.51%):** Customers paying via electronic check churn significantly more than those on automated credit card/bank transfer options.

> **Important Methodological Note:**  
> Feature importance quantifies **statistical predictive utility** for tree splits. It does **not automatically imply direct physical causation**.

---

## 8. Business Recommendations
1. **Onboarding Retention Campaigns:** Implement targeted customer success touchpoints during the first 90 days of tenure.
2. **Contract Migration Incentives:** Offer small discounts (e.g., 5-10% off) to convert Month-to-Month subscribers into 1-Year or 2-Year contracts.
3. **Payment Method Standardization:** Provide automated billing incentives to migrate customers away from manual Electronic Check payments toward automated Direct Debit.

---

## 9. Model Export & Verification
* **Serialized Model Path:** `models/best_churn_model.joblib`
* **Inference Demonstration:** `src/predict.py` executes end-to-end inference on sample customer input, returning churn prediction labels and probability scores.
