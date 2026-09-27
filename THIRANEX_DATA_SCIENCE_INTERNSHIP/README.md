# Thiranex Data Science Internship — Technical Submission Package

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0+-green.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/status-Submission--Ready-success.svg)]()

## Overview
This repository contains the complete technical submission package for the **Thiranex Data Science Internship**. It includes four comprehensive end-to-end data science projects covering data cleaning, predictive machine learning, exploratory data analysis (EDA), and healthcare applied AI.

---

## Directory Structure
```text
THIRANEX_DATA_SCIENCE_INTERNSHIP/
│
├── 01_Data_Cleaning_Visualization/     # Task 1: Retail E-Commerce Sales Cleaning & Visualization
│   ├── data/ (raw & processed)
│   ├── notebooks/ (Task_1_Data_Cleaning_Visualization.ipynb)
│   ├── src/ (data_cleaning.py, visualization.py)
│   ├── visualizations/ (8 high-res PNG charts)
│   ├── reports/ (Task_1_Report.md)
│   └── README.md
│
├── 02_Predictive_Modeling/              # Task 2: Customer Churn Machine Learning Prediction
│   ├── data/ (raw & processed)
│   ├── notebooks/ (Task_2_Predictive_Modeling.ipynb)
│   ├── src/ (preprocessing.py, train.py, evaluate.py, predict.py)
│   ├── models/ (best_churn_model.joblib)
│   ├── visualizations/ (5 high-res PNG charts)
│   ├── reports/ (Task_2_Report.md)
│   └── README.md
│
├── 03_Exploratory_Data_Analysis/        # Task 3: World Happiness & Socioeconomic EDA
│   ├── data/ (raw & processed)
│   ├── notebooks/ (Task_3_EDA.ipynb)
│   ├── src/ (analysis.py)
│   ├── visualizations/ (8 high-res PNG charts)
│   ├── reports/ (Task_3_EDA_Report.md)
│   └── README.md
│
├── 04_Real_World_Project/               # Task 4: Real-World Health Risk Analysis & Prediction
│   ├── data/ (raw & processed)
│   ├── notebooks/ (Task_4_Health_Data_Project.ipynb)
│   ├── src/ (preprocessing.py, analysis.py, train.py, evaluate.py, predict.py)
│   ├── models/ (best_health_model.joblib)
│   ├── visualizations/ (12 high-res PNG charts)
│   ├── reports/ (Final_Project_Report.md)
│   └── README.md
│
├── Final_Internship_Report/             # Comprehensive Internship Reports
│   ├── Thiranex_Data_Science_Internship_Report.md
│   └── Internship_Project_Summary.md
│
├── README.md                            # Master Submission README
├── requirements.txt                     # Global dependencies
└── .gitignore                           # Git ignore rules
```

---

## Task Summary Table

| Task Folder | Project Title | Key Dataset | Best Model / Key Finding | Status |
| :--- | :--- | :--- | :--- | :---: |
| `01_Data_Cleaning_Visualization` | **Data Cleaning & Visualization** | UCI Online Retail | 524.8K clean sales records, £10.64M net revenue analyzed. | Verified |
| `02_Predictive_Modeling` | **Predictive Modeling ML** | IBM Telco Customer Churn | **Random Forest (0.8589 ROC-AUC, 69.65% Precision)** | Verified |
| `03_Exploratory_Data_Analysis` | **Exploratory Data Analysis** | World Happiness 2021 | Logged GDP per Capita ($r = +0.790$) is top happiness predictor. | Verified |
| `04_Real_World_Project` | **Real-World Health Project** | UCI Heart Disease | **Logistic Regression (81.36% Acc, 0.8588 ROC-AUC)** | Verified |

---

## How to Run the Projects

### Global Environment Installation:
```bash
cd THIRANEX_DATA_SCIENCE_INTERNSHIP
pip install -r requirements.txt
```

### Running Individual Tasks:

#### Task 1: Data Cleaning & Visualization
```bash
cd 01_Data_Cleaning_Visualization
python run_pipeline.py
jupyter notebook notebooks/Task_1_Data_Cleaning_Visualization.ipynb
```

#### Task 2: Predictive Modeling Using Machine Learning
```bash
cd 02_Predictive_Modeling
python run_pipeline.py
python src/predict.py
jupyter notebook notebooks/Task_2_Predictive_Modeling.ipynb
```

#### Task 3: Exploratory Data Analysis (EDA)
```bash
cd 03_Exploratory_Data_Analysis
python run_pipeline.py
jupyter notebook notebooks/Task_3_EDA.ipynb
```

#### Task 4: Real-World Health Data Project
```bash
cd 04_Real_World_Project
python run_pipeline.py
python src/predict.py
jupyter notebook notebooks/Task_4_Health_Data_Project.ipynb
```

---

## Verification & Compliance
* **Zero Data Leakage:** Preprocessing transformations (scaling and one-hot encoding) are fitted strictly on `X_train`.
* **Reproducibility:** Random seeds fixed (`random_state=42`) across all splits and algorithms.
* **Pre-Rendered Notebooks:** All Jupyter Notebooks contain fully rendered markdown, tables, and inline visual charts.
* **Ethical Disclosure:** Mandatory non-clinical educational disclaimers incorporated in health project deliverables.
