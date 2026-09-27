# World Happiness & Socioeconomic Exploratory Data Analysis (EDA) Project

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0+-green.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/status-Completed-success.svg)]()

## Internship Task
**Organization:** Thiranex Data Science Internship  
**Task 3 Title:** Exploratory Data Analysis (EDA) Project  
**Domain:** World Happiness & Socioeconomic Analytics  
**Due Date:** 16 October 2026  

---

## Project Overview
This project performs an in-depth statistical and visual Exploratory Data Analysis (EDA) on the **World Happiness Report 2021 dataset**. It investigates national subjective well-being across 149 countries, measures linear (Pearson) and rank (Spearman) correlations across socioeconomic factors (GDP, social support, health, freedom, generosity, corruption), and analyzes regional disparities.

---

## Dataset
* **Dataset Name:** World Happiness Report 2021 Dataset
* **Source:** Sustainable Development Solutions Network (SDSN) ([GitHub Mirror](https://raw.githubusercontent.com/anuragg130/DS-Visualization---World-Happiness-Report-2021/main/world-happiness-report-2021.csv))
* **Year:** 2021
* **Scope:** 149 countries across 10 global geographical regions
* **Raw File Path:** `data/raw/world-happiness-report-2021.csv`

---

## Technologies Used
* **Python 3.14+**
* **Data Manipulation & Analysis:** Pandas, NumPy, SciPy
* **Visualization:** Matplotlib, Seaborn
* **Environment & Automation:** Jupyter Notebook, OpenPyXL

---

## Project Structure
```text
thiranex_task3_eda/
│
├── data/
│   ├── raw/
│   │   └── world-happiness-report-2021.csv   # Original unchanged raw dataset
│   └── processed/
│       └── cleaned_happiness_2021.csv         # Cleaned analysis dataset
│
├── notebooks/
│   └── Task_3_EDA.ipynb                       # Fully executed Jupyter Notebook
│
├── src/
│   └── analysis.py                            # Data audit, stats & plotting module
│
├── visualizations/                            # High-resolution PNG figures (300 DPI)
│   ├── 01_happiness_distribution.png
│   ├── 02_regional_happiness_boxplot.png
│   ├── 03_gdp_vs_happiness_scatter.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_top10_bottom10_countries.png
│   ├── 06_regional_mean_happiness.png
│   ├── 07_pairwise_factors_scatter.png
│   └── 08_correlation_comparison.png
│
├── reports/
│   ├── Task_3_EDA_Report.md                   # Comprehensive internship report
│   └── summary_metrics.json                   # Summary metrics JSON log
│
├── create_notebook.py                         # Notebook builder script
├── execute_notebook.py                        # Programmatic notebook executor
├── run_pipeline.py                            # Master analysis pipeline
├── README.md                                  # Project documentation
├── requirements.txt                           # Project dependencies
└── .gitignore                                 # Git ignore rules
```

---

## Key Statistical Findings & Insights

| Socioeconomic Factor | Pearson ($r$) | Spearman ($\rho$) | Relationship Impact |
| :--- | :--- | :--- | :--- |
| **Logged GDP per Capita** | **+0.7898** | **+0.8131** | Strongest positive predictor |
| **Healthy Life Expectancy** | **+0.7681** | **+0.7801** | Very strong positive association |
| **Social Support** | **+0.7569** | **+0.7842** | Crucial social buffer |
| **Freedom to Make Choices** | **+0.6078** | **+0.6033** | Moderate positive impact |
| **Corruption Perception** | **-0.4211** | **-0.3298** | Moderate negative correlation |
| **Generosity** | **-0.0178** | **-0.0694** | Negligible linear correlation |

* **Happiest Country:** **Finland (7.842)** (Western Europe)
* **Unhappiest Country:** **Afghanistan (2.523)** (South Asia)
* **Top Region:** **North America and ANZ (7.128)** & **Western Europe (6.915)**
* **Lowest Region:** **South Asia (4.442)** & **Sub-Saharan Africa (4.494)**

---

## How to Run

### Step 1: Install Dependencies
```bash
cd thiranex_task3_eda
pip install -r requirements.txt
```

### Step 2: Run Master Analysis Pipeline
```bash
python run_pipeline.py
```
This script loads the raw dataset, calculates univariate/bivariate statistics, exports `data/processed/cleaned_happiness_2021.csv`, generates all 8 figures into `visualizations/`, and logs metrics to `reports/summary_metrics.json`.

### Step 3: Launch Jupyter Notebook
```bash
python create_notebook.py
python execute_notebook.py
jupyter notebook notebooks/Task_3_EDA.ipynb
```

---

## Conclusion
The EDA pipeline provides a reproducible workflow for analyzing global happiness. All project artifacts are submission-ready.
