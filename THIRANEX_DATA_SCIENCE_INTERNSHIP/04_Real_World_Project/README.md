# Health Risk Analysis and Prediction

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-Completed-success.svg)]()

## Internship
**Organization:** Thiranex Data Science Internship  

## Task
**Task 4 Title:** Real-World Data Project — Health Domain  
**Due Date:** 23 October 2026  

---

## Objective
The objective of this project is to perform clinical exploratory data analysis (EDA), investigate physiological risk factors, and build an end-to-end, leak-free supervised machine learning classification pipeline to predict heart disease diagnosis (`target`: 1 = Disease Present, 0 = Normal / Healthy).

---

## Dataset
* **Dataset Name:** UCI Heart Disease Dataset (Cleveland Clinic Database)
* **Source:** UCI Machine Learning Repository ([GitHub Raw Mirror](https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv))
* **Sample Size:** 303 patient records × 14 clinical attributes
* **Target Distribution:** 54.5% Disease Present (`1`), 45.5% Normal / Healthy (`0`)
* **Raw File Path:** `data/raw/heart.csv`

---

## ⚠️ Mandatory Educational Disclaimer
> **THIS PROJECT IS BUILT STRICTLY FOR EDUCATIONAL INTERNSHIP DEMONSTRATION PURPOSES.**  
> The machine learning models and predictive outputs presented in this codebase **CANNOT AND MUST NOT** replace diagnostic evaluation by a licensed medical physician or clinical diagnostic equipment.

---

## Technologies Used
* **Python 3.14+**
* **Machine Learning:** Custom Vectorized Classifiers, SciPy, Joblib
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Notebook & Automation:** Jupyter Notebook, OpenPyXL

---

## Project Architecture
```text
thiranex_task4_real_world_health_project/
│
├── data/
│   ├── raw/
│   │   └── heart.csv                          # Original raw dataset
│   └── processed/
│       └── cleaned_heart_disease.csv          # Cleaned dataset
│
├── notebooks/
│   └── Task_4_Health_Data_Project.ipynb      # Fully executed Jupyter Notebook
│
├── src/
│   ├── preprocessing.py                     # Data audit & CustomColumnTransformer
│   ├── analysis.py                          # Clinical EDA & visualization routines
│   ├── train.py                             # Vectorized ML classifiers & MLPipeline
│   ├── evaluate.py                          # Metric evaluation & plotting
│   └── predict.py                           # Inference script on sample profile
│
├── models/
│   └── best_health_model.joblib             # Serialized model artifact
│
├── visualizations/                          # High-resolution PNG figures (300 DPI)
│   ├── 00_target_distribution.png
│   ├── 01_age_distribution_by_target.png
│   ├── 02_chest_pain_vs_target.png
│   ├── 03_cholesterol_bp_scatter.png
│   ├── 04_max_heart_rate_boxplot.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_exercise_angina_st_depression.png
│   ├── 07_pairwise_feature_importance.png
│   ├── 08_confusion_matrices.png
│   ├── 09_roc_curves.png
│   ├── 10_model_comparison.png
│   └── 11_feature_importance.png
│
├── reports/
│   ├── Final_Project_Report.md              # Comprehensive internship report
│   └── summary_metrics.json                 # JSON summary log
│
├── create_notebook.py                       # Notebook generator script
├── execute_notebook.py                      # Programmatic notebook executor
├── run_pipeline.py                          # Master pipeline script
├── README.md                                # Project documentation
├── requirements.txt                         # Project dependencies
└── .gitignore                               # Git ignore rules
```

---

## Methodology & Machine Learning Results

### Leakage-Free Preprocessing
* Numerical features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) scaled via Z-score standardization.
* Categorical features (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) One-Hot Encoded.
* Transformations fitted **strictly on `X_train`** inside `CustomColumnTransformer`.
* 80/20 Stratified Train/Test Split (`random_state=42`).

### Holdout Test Performance (59 Patients)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Logistic Regression)** | **81.36%** | **83.87%** | **81.25%** | **0.8254** | **0.8588** |
| **Random Forest** | **79.66%** | **83.33%** | **78.12%** | **0.8065** | **0.8553** |
| **Decision Tree** | **76.27%** | **76.47%** | **81.25%** | **0.7879** | **0.7882** |

---

## How to Run

### Step 1: Install Dependencies
```bash
cd thiranex_task4_real_world_health_project
pip install -r requirements.txt
```

### Step 2: Run Master Pipeline
```bash
python run_pipeline.py
```

### Step 3: Run Inference Script
```bash
python src/predict.py
```

### Step 4: Launch Jupyter Notebook
```bash
python create_notebook.py
python execute_notebook.py
jupyter notebook notebooks/Task_4_Health_Data_Project.ipynb
```

---

## Limitations & Future Improvements
* **Cohort Scale:** 303 patient records from a single clinic. Multi-center clinical datasets are required for generalizability.
* **Future Work:** Integrate multi-center clinical trials data and cross-validation parameter optimization.

---

## Conclusion
The pipeline successfully demonstrates applied health data science with an **81.36% Accuracy and 0.8588 ROC-AUC** model. All deliverables are submission-ready.
