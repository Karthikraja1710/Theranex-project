# Thiranex Data Science Internship — Final Comprehensive Report

---

## 1. Student Introduction
* **Intern Candidate:** Data Science Intern
* **Program:** Applied Data Science & Machine Learning Internship
* **Organization:** Thiranex
* **Domain:** Data Science & Applied AI
* **Submission Date:** September / October 2026

---

## 2. Internship Organization: Thiranex
**Thiranex** provides specialized experiential learning and technical training in Data Science, Artificial Intelligence, and Software Engineering. The internship program challenges candidates to complete end-to-end industry projects covering raw data cleaning, machine learning modeling, exploratory data analysis, and domain-specific applied healthcare data science.

---

## 3. Internship Domain: Data Science
The internship focuses on practical, applied Data Science methodology, enforcing strict data quality audits, leakage-free machine learning pipelines, scientific exploratory data analysis, statistical metric evaluations, and professional visual storytelling.

---

## 4. Internship Duration & Schedule
* **Program Period:** August 2026 – October 2026
* **Task 1 Due Date:** 2 October 2026
* **Task 2 Due Date:** 9 October 2026
* **Task 3 Due Date:** 16 October 2026
* **Task 4 Due Date:** 23 October 2026

---

## 5. Internship Objectives
1. **Master Practical Data Cleaning**: Handle real-world data quality flaws including missing values, duplicate records, non-positive quantities/prices, text anomalies, and extreme outliers.
2. **Build Leakage-Free ML Systems**: Implement strict `ColumnTransformer` and `Pipeline` architectures ensuring zero data leakage between training and holdout test partitions.
3. **Conduct Rigorous EDA & Statistical Testing**: Analyze distributions, evaluate Pearson ($r$) and Spearman ($\rho$) correlation matrices, and quantify regional/demographic disparities.
4. **Apply Domain Healthcare AI Ethically**: Train risk prediction models on clinical health data while incorporating mandatory educational disclaimers.
5. **Deliver Submission-Ready Artifacts**: Produce clean Python packages, fully pre-executed Jupyter Notebooks, high-resolution PNG figures, and detailed markdown reports.

---

## Summary Matrix of Internship Tasks

| Task | Project | Technologies | Main Work | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1** | **Data Cleaning & Visualization** | Python, Pandas, NumPy, Matplotlib, Seaborn | Audited 541.9K retail records, cleaned missing descriptions/guest checkouts, engineered 8 temporal/financial features, generated 8 figures. | Preserved 96.86% data retention; analyzed £10.64M net revenue across 524.8K clean sales transactions. |
| **Task 2** | **Predictive Modeling Using Machine Learning** | Python, Scikit-Learn, SciPy, Joblib | Built leakage-free pipelines for Baseline Logistic Regression, Decision Tree, and Random Forest on 7,043 Telco churn records. | **Random Forest achieved 0.8589 ROC-AUC** and **69.65% Precision** on 1,407 test records; saved `best_churn_model.joblib`. |
| **Task 3** | **Exploratory Data Analysis (EDA)** | Python, Pandas, Seaborn, SciPy.stats | Conducted statistical EDA on World Happiness 2021 dataset (149 countries), computing Pearson ($r$) and Spearman ($\rho$) correlation metrics. | Identified Logged GDP per Capita ($r = +0.790$) and Health ($r = +0.768$) as top predictors of global well-being. |
| **Task 4** | **Real-World Health Data Project** | Python, Scikit-Learn, SciPy, Joblib | Analyzed UCI Heart Disease dataset (303 patients), trained 3 classifiers, evaluated recall/precision trade-offs, added ethical disclaimers. | **Baseline Logistic Regression achieved 81.36% Accuracy and 0.8588 ROC-AUC**; saved `best_health_model.joblib`. |

---

## 6. Task 1 Summary — Data Cleaning & Visualization Project
* **Domain**: Retail / E-Commerce Sales Performance & Customer Analytics
* **Dataset**: UCI Online Retail Dataset (`Online_Retail.xlsx`, 541,909 raw records)
* **Key Achievements**:
  - Identified and removed 5,268 exact duplicate rows.
  - Imputed missing product descriptions (`UNSPECIFIED PRODUCT`) and handled missing customer IDs (`CustomerID = -1`, `IsGuest = 1`) to preserve 25% of total sales volume.
  - Filtered order cancellations (`InvoiceNo` starting with `'C'`) and non-positive quantities/prices.
  - Engineered derived temporal and financial metrics (`TotalAmount`, `YearMonth`, `DayOfWeek`, `InvoiceHour`).
  - Saved clean processed dataset `cleaned_retail_data.csv` (524,878 rows, 96.86% retention).
  - Output 8 publication-grade dashboard charts (300 DPI) in `visualizations/`.

---

## 7. Task 2 Summary — Predictive Modeling Using Machine Learning
* **Domain**: Customer Churn Prediction & Retention Analytics
* **Dataset**: IBM Telco Customer Churn Dataset (7,043 customer accounts)
* **Key Achievements**:
  - Implemented `CustomColumnTransformer` for Z-score scaling and One-Hot Encoding fitted strictly on `X_train`.
  - Applied 80/20 Stratified Split (`random_state=42`) creating 5,636 train and 1,407 holdout test records.
  - Trained Baseline Logistic Regression, Decision Tree, and Random Forest models.
  - Evaluated performance on test set: Random Forest achieved **0.8589 ROC-AUC** and **69.65% Precision**.
  - Quantified top predictors: `tenure` (20.31% importance), `TotalCharges` (13.72%), and `Fiber Optic Internet` (10.61%).
  - Serialized model artifact to `models/best_churn_model.joblib` and created inference script `src/predict.py`.

---

## 8. Task 3 Summary — Exploratory Data Analysis (EDA) Project
* **Domain**: World Happiness & Socioeconomic Analytics
* **Dataset**: World Happiness Report 2021 Dataset (149 countries across 10 regions)
* **Key Achievements**:
  - Computed univariate descriptive statistics (Mean, Median, StdDev, IQR, Skewness) for all factors.
  - Evaluated Pearson ($r$) and Spearman ($\rho$) correlation matrices with Happiness Score.
  - Highest positive correlation: Logged GDP per Capita ($r = +0.7898$, $\rho = +0.8131$) and Healthy Life Expectancy ($r = +0.7681$).
  - Identified Finland (7.842) as happiest nation and Afghanistan (2.523) as lowest ranked nation.
  - Formulated clear distinction between statistical correlation and physical causation.
  - Saved 8 visualization charts in `visualizations/`.

---

## 9. Task 4 Summary — Real-World Data Project (Health Domain)
* **Domain**: Health Risk Analysis & Heart Disease Prediction
* **Dataset**: UCI Heart Disease Dataset (303 patient records)
* **Key Achievements**:
  - Audited 14 clinical attributes (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `cp`, `exang`, `thal`).
  - Generated 8 clinical EDA figures investigating age, chest pain type, max heart rate, and ST depression.
  - Trained Baseline Logistic Regression, Decision Tree, and Random Forest classifiers on 80/20 stratified split.
  - **Baseline Logistic Regression achieved 81.36% Accuracy, 83.87% Precision, 81.25% Recall, and 0.8588 ROC-AUC**.
  - Identified `thal_2` (Fixed Defect Thalassemia) and `oldpeak` (Exercise ST Depression) as top diagnostic predictors.
  - Serialized pipeline to `models/best_health_model.joblib` and created inference script `src/predict.py`.
  - Incorporated mandatory **Educational & Non-Clinical Disclaimer**.

---

## 10. Technologies Used
* **Programming Language:** Python 3.14+
* **Data Engineering & Analytics:** Pandas, NumPy, SciPy
* **Machine Learning & Pipeline Architecture:** Scikit-Learn, Custom Vectorized Classifiers, CustomColumnTransformer, MLPipeline, Joblib
* **Data Visualization:** Matplotlib, Seaborn
* **Execution & Automation:** Jupyter Notebook, OpenPyXL, Nbformat, Nbconvert, PowerShell

---

## 11. Skills Gained
* **Data Quality Auditing**: Systematic identification of missingness patterns, duplicate records, and invalid values.
* **Leakage-Free Pipeline Design**: Constructing preprocessors that fit scaling and encoding parameters exclusively on training splits.
* **Imbalanced Classification Evaluation**: Evaluating models using Precision, Recall, F1-Score, and ROC-AUC rather than relying on raw accuracy alone.
* **Statistical Correlation Metrics**: Distinguishing linear (Pearson) vs rank (Spearman) relationship structures.
* **Publication-Quality Visual Storytelling**: Designing clean, non-misleading multi-panel charts with custom palettes.
* **Ethical Healthcare AI Disclosure**: Communicating model limitations and non-clinical educational disclaimers.

---

## 12. Challenges Encountered & Solutions Implemented

| Challenge Encountered | Root Cause | Solution Implemented |
| :--- | :--- | :--- |
| **System DLL Block on C-Extensions** | Windows Application Control (AppLocker) blocked un-signed Cython `.pyd` files inside `.venv`. | Developed pure, high-performance vectorized ML classifier algorithms in `src/train.py` utilizing NumPy matrix operations and SciPy optimization. |
| **Data Leakage in Preprocessing** | Standard scaling on entire dataset before train/test split leaks test mean/variance into training. | Enforced `CustomColumnTransformer` and `MLPipeline` wrappers that execute `fit_transform` strictly on `X_train`. |
| **Class Imbalance in Churn Data** | Churned customers represent only 26.5% of total dataset, creating high baseline accuracy bias. | Applied stratified sampling (`stratify=y`) and prioritized Precision, Recall, F1-Score, and ROC-AUC metrics over raw accuracy. |
| **Missing Customer IDs in Sales Data** | Dropping 135K missing `CustomerID` rows would destroy 25% of total sales metrics. | Imputed missing IDs with `-1` and assigned a binary flag `IsGuest = 1` to retain macro-revenue analysis while isolating registered accounts. |

---

## 13. Overall Learning Outcomes
Through the Thiranex Data Science Internship, I gained comprehensive end-to-end expertise in applied data science. From ingesting raw, un-sanitized CSV/Excel datasets to delivering fully pre-executed Jupyter Notebooks and serialized production models, I learned to prioritize data integrity, algorithmic reproducibility, and transparent reporting.

---

## 14. Future Improvements
1. **Hyperparameter Optimization**: Incorporate Bayesian optimization (Optuna / GridSearchCV) for hyperparameter tuning.
2. **Multi-Year Panel Datasets**: Extend the World Happiness EDA to multi-year longitudinal panels (2015–2026).
3. **Multi-Center Clinical Datasets**: Validate the heart disease risk prediction model across larger, multi-hospital healthcare cohorts.

---

## 15. Conclusion
The four internship projects demonstrate complete technical rigor, zero data leakage, empirical metric tracking, and publication-ready presentation. All code scripts, visualizations, trained models, executed notebooks, and detailed reports are organized and ready for submission.
