# Comprehensive Data Science Internship Report
**Internship Program:** Thiranex Data Science Internship  
**Task Title:** Task 4 — Real-World Data Project (Health Domain)  
**Project Title:** Health Risk Analysis and Prediction  
**Due Date:** 23 October 2026  
**Status:** Completed & Submission-Ready  

---

## 1. Executive Summary & Problem Statement
Cardiovascular diseases (CVDs) represent the single largest cause of death globally. Early risk stratification allows clinicians to initiate preventative treatments before irreversible cardiac events occur.

The primary objective of **Task 4: Real-World Data Project (Health Domain)** is to build an end-to-end data science and machine learning pipeline that analyzes patient clinical attributes, uncovers physiological risk factors, and trains supervised classification models to predict heart disease risk (`target`: 1 = Disease Present, 0 = Normal / Healthy).

Using holdout test evaluation across 59 patient records, three model architectures—**Baseline L2-Regularized Logistic Regression**, a **Decision Tree Classifier**, and a **Random Forest Classifier**—were trained and evaluated. **Baseline Logistic Regression achieved the highest performance with 81.36% Accuracy, 83.87% Precision, 81.25% Recall, and an ROC-AUC of 0.8588**.

***
> ### ⚠️ MANDATORY EDUCATIONAL & ETHICAL DISCLAIMER
> **THIS PROJECT IS BUILT STRICTLY FOR EDUCATIONAL INTERNSHIP DEMONSTRATION PURPOSES.**  
> The machine learning models and predictive outputs presented in this report **CANNOT AND MUST NOT** replace diagnostic evaluation by a licensed medical physician or clinical diagnostic equipment.
***

---

## 2. Dataset Source & Description
* **Dataset Name:** UCI Heart Disease Dataset (Cleveland Clinic Database)
* **Source:** UCI Machine Learning Repository ([GitHub Raw Mirror](https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv))
* **Sample Size:** 303 patient records (302 after deduplication) × 14 clinical attributes.
* **Target Variable:** `target` (1 = Heart Disease Present [54.5%], 0 = Normal / Healthy [45.5%]).

### Clinical Feature Specifications

| Feature | Category | Clinical Description & Scale |
| :--- | :--- | :--- |
| `age` | Demographic | Patient age in years (29 to 77 years). |
| `sex` | Demographic | Patient biological sex (1 = Male, 0 = Female). |
| `cp` | Clinical Symptom | Chest pain type (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal, 3 = Asymptomatic). |
| `trestbps` | Vital Sign | Resting blood pressure in mm Hg upon hospital admission. |
| `chol` | Laboratory | Serum cholesterol level in mg/dl. |
| `fbs` | Laboratory | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False). |
| `restecg` | Diagnostic | Resting electrocardiographic results (0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy). |
| `thalach` | Stress Test | Maximum heart rate achieved during exercise stress testing (bpm). |
| `exang` | Symptom | Exercise-induced angina (1 = Yes, 0 = No). |
| `oldpeak` | Diagnostic | ST depression induced by exercise relative to rest. |
| `slope` | Diagnostic | Slope of peak exercise ST segment (0 = Upsloping, 1 = Flat, 2 = Downsloping). |
| `ca` | Imaging | Number of major fluoroscopy-colored vessels (0 to 3). |
| `thal` | Pathology | Thalassemia nuclear scan (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect). |

---

## 3. Data Cleaning & Preprocessing Methodology

1. **Deduplication:** 1 exact duplicate record removed, yielding 302 clean patient records.
2. **Missingness Audit:** 0 missing values detected across all 14 clinical features.
3. **Leakage-Free Pipeline Transformation:** 
   - Numerical clinical features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) were standardized using Z-score scaling.
   - Categorical clinical features (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) were One-Hot Encoded (`drop='first'`).
   - Transformations were fitted **strictly on `X_train`** inside `CustomColumnTransformer`.
4. **Stratified Train / Test Partitioning:** Applied an 80/20 stratified split (`random_state=42`) yielding:
   - **Training Set (`X_train`):** 243 patient records (132 Disease, 111 Normal).
   - **Holdout Testing Set (`X_test`):** 59 patient records (32 Disease, 27 Normal).

---

## 4. Exploratory Data Analysis (EDA) Summary

1. **Age vs. Disease Risk:** Patients with heart disease exhibit a higher median age (56 years) compared to healthy individuals.
2. **Chest Pain Symptom Indicator:** Patients with non-anginal pain (`cp = 2`) and atypical angina (`cp = 1`) show significantly higher heart disease rates than asymptomatic patients.
3. **Exercise Stress Testing Dynamics:** Patients diagnosed with heart disease exhibit significantly lower maximum heart rate (`thalach` mean: 139 bpm vs. 158 bpm in normal group) and higher exercise-induced ST depression (`oldpeak`).

---

## 5. Machine Learning Evaluation & Model Comparison

All three models were evaluated on the 59 holdout test set patients:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total}}$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Holdout Test Performance Metrics

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Logistic Regression)** | **81.36%** | **83.87%** | **81.25%** | **0.8254** | **0.8588** |
| **Random Forest** | **79.66%** | **83.33%** | **78.12%** | **0.8065** | **0.8553** |
| **Decision Tree** | **76.27%** | **76.47%** | **81.25%** | **0.7879** | **0.7882** |

---

## 6. Confusion Matrix & Clinical Trade-Off Analysis

* **Baseline Logistic Regression Model (Test Set = 59 patients):**
  * True Negatives (Correctly Identified Normal): **22**
  * False Positives (False Alarm / Healthy diagnosed as Disease): **5**
  * False Negatives (Missed Disease Cases): **6**
  * True Positives (Correctly Identified Heart Disease): **26**

### Clinical Trade-Off Discussion
In healthcare AI applications, **Recall (Sensitivity)** is prioritized over Precision to minimize False Negatives (un-diagnosed sick patients). Both Logistic Regression and Decision Tree achieved an **81.25% Recall**, correctly diagnosing 26 out of 32 diseased test patients. Logistic Regression offered superior overall specificity and precision (83.87%).

---

## 7. Feature Importance & Clinical Predictors

Top clinical predictors identified by Random Forest feature importance:
1. **`thal_2` (Fixed Defect Thalassemia):** 12.76% relative importance.
2. **`oldpeak` (Exercise ST Depression):** 9.96% relative importance.
3. **`exang_1` (Exercise Induced Angina):** 9.18% relative importance.
4. **`thalach` (Max Heart Rate Achieved):** 9.02% relative importance.
5. **`age` (Patient Age):** 7.76% relative importance.

> **Note on Medical Interpretability:** Model feature importance reflects mathematical node split utility. High feature importance signifies strong diagnostic association, **not direct physiological causation**.

---

## 8. Dashboard Visualizations Summary
All figures are saved in `visualizations/` as 300 DPI high-resolution figures:
1. `00_target_distribution.png`: Target heart disease class count & percentage.
2. `01_age_distribution_by_target.png`: Age KDE & histogram by Target.
3. `02_chest_pain_vs_target.png`: Chest Pain Type vs Heart Disease.
4. `03_cholesterol_bp_scatter.png`: Serum Cholesterol vs Blood Pressure.
5. `04_max_heart_rate_boxplot.png`: Max Heart Rate Boxplot.
6. `05_correlation_heatmap.png`: Pearson Correlation Heatmap.
7. `06_exercise_angina_st_depression.png`: Exercise Angina & ST Depression.
8. `07_pairwise_feature_importance.png`: Pearson correlation feature ranking.
9. `08_confusion_matrices.png`: Confusion Matrix heatmaps.
10. `09_roc_curves.png`: Overlay ROC curves with AUC values.
11. `10_model_comparison.png`: Model Performance Comparison Bar Chart.
12. `11_feature_importance.png`: Random Forest Feature Importance chart.

---

## 9. Ethical Considerations & Limitations
* **Sample Size Limitation:** The 303-patient dataset represents a single clinical cohort. Larger multi-center cohorts are necessary before considering clinical translation.
* **Non-Clinical Deployment:** This software is purely an academic artifact for internship evaluation and must never be deployed for clinical triage or diagnostic decisions.

---

## 10. Conclusion & Saved Artifacts
* **Best Model:** Baseline Logistic Regression (ROC-AUC: 0.8588).
* **Serialized Model Path:** `models/best_health_model.joblib`.
* **Inference Demonstration:** `src/predict.py` executes successfully.
