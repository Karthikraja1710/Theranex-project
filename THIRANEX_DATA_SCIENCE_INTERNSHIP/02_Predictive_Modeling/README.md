# Customer Churn Prediction: Predictive Modeling Using Machine Learning

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-Completed-success.svg)]()

## Internship Task
**Organization:** Thiranex Data Science Internship  
**Task 2 Title:** Predictive Modeling Using Machine Learning  
**Domain:** Customer Churn Prediction & Retention Analytics  
**Due Date:** 9 October 2026  

---

## Project Overview
The objective of this project is to build an end-to-end, leak-free supervised machine learning classification system that predicts whether a customer is likely to churn (cancel their subscription). By accurately identifying high-risk accounts, businesses can deploy proactive retention strategies and minimize customer attrition.

---

## Dataset
* **Dataset Name:** IBM Telco Customer Churn Dataset
* **Source:** IBM Sample Data Sets ([GitHub Mirror](https://raw.githubusercontent.com/YuehHanChen/Telco_Customer_Churn_Analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv))
* **Size:** 7,043 customer rows × 21 columns
* **Target Column:** `Churn` (`Yes` / `No`)
* **Class Imbalance:** 73.46% Retained (`No`), 26.54% Churned (`Yes`)
* **Raw File Path:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

## Technologies Used
* **Python 3.14+**
* **Machine Learning:** Custom Vectorized Classifiers, SciPy, Joblib
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Notebook & Automation:** Jupyter Notebook, OpenPyXL

---

## Project Structure
```text
thiranex_task2_predictive_modeling/
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Original raw dataset
│   └── processed/
│       └── cleaned_telco_churn.csv                # Processed dataset
│
├── notebooks/
│   └── Task_2_Predictive_Modeling.ipynb           # Fully executed Jupyter Notebook
│
├── src/
│   ├── preprocessing.py                          # Data cleaning & ColumnTransformer
│   ├── train.py                                  # Vectorized ML algorithms & Pipeline
│   ├── evaluate.py                               # Vectorized metrics & visual plotting
│   └── predict.py                                # Inference prediction script
│
├── models/
│   └── best_churn_model.joblib                   # Serialized Random Forest pipeline
│
├── visualizations/                               # High-resolution PNG charts (300 DPI)
│   ├── 00_class_distribution.png
│   ├── 01_confusion_matrices.png
│   ├── 02_roc_curves.png
│   ├── 03_model_comparison.png
│   └── 04_feature_importance.png
│
├── reports/
│   ├── Task_2_Report.md                          # Detailed internship task report
│   └── metrics_summary.json                      # Empirical test metrics JSON
│
├── create_notebook.py                            # Notebook generation script
├── execute_notebook.py                           # Programmatic notebook runner
├── run_pipeline.py                               # Master pipeline execution script
├── README.md                                     # Project documentation
├── requirements.txt                              # Python dependencies
└── .gitignore                                    # Git ignore rules
```

---

## Data Preprocessing & Leakage Prevention
1. **Identifier Cleanup:** Removed non-predictive `customerID`.
2. **Missing TotalCharges Imputation:** Filled 11 whitespace entries with `MonthlyCharges * tenure`.
3. **Pipeline Isolation:** Numerical scaling (StandardScaler) and categorical One-Hot Encoding are fitted **strictly on `X_train`** using `CustomColumnTransformer`.
4. **Stratified Split:** 80/20 train/test split (`random_state=42`) preserving class ratios.

---

## Empirical Model Performance (Holdout Test Set = 1,407 Records)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **80.67%** | **69.65%** | **47.99%** | **0.5683** | **0.8589** |
| **Baseline (Logistic Regression)** | **81.17%** | **68.62%** | **53.35%** | **0.6003** | **0.8570** |
| **Decision Tree** | **79.25%** | **64.41%** | **48.53%** | **0.5535** | **0.8368** |

---

## Top Predictors of Churn
1. **`tenure`**: Customer subscription length (20.31% relative importance).
2. **`TotalCharges`**: Cumulative revenue (13.72%).
3. **`InternetService_Fiber optic`**: High-speed internet subscription tier (10.61%).
4. **`MonthlyCharges`**: Monthly recurring fee (7.73%).
5. **`PaymentMethod_Electronic check`**: Electronic check billing method (7.51%).

---

## How to Run

### Step 1: Install Dependencies
```bash
# Navigate to Task 2 directory
cd thiranex_task2_predictive_modeling

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run End-to-End ML Pipeline
```bash
python run_pipeline.py
```
This script loads the raw dataset, performs preprocessing, splits the data, trains all 3 models, evaluates holdout performance, saves figures to `visualizations/`, and exports `models/best_churn_model.joblib`.

### Step 3: Run Inference Script on Sample Data
```bash
python src/predict.py
```

### Step 4: Launch Jupyter Notebook
```bash
python create_notebook.py
python execute_notebook.py
jupyter notebook notebooks/Task_2_Predictive_Modeling.ipynb
```

---

## Conclusion
The predictive modeling pipeline delivers a robust, high-performing churn prediction model with **0.8589 ROC-AUC**. All artifacts are submission-ready.
