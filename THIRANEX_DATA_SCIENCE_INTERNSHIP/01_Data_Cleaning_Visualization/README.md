# Retail & E-Commerce Sales Analysis: Data Cleaning & Visualization Project

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0+-green.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/status-Completed-success.svg)]()

## Internship Task
**Organization:** Thiranex Data Science Internship  
**Task 1 Title:** Data Cleaning & Visualization Project  
**Domain:** Retail / E-Commerce Sales Performance Analytics  
**Due Date:** 2 October 2026  

---

## Objective
The goal of this project is to perform end-to-end data preprocessing, data quality auditing, outlier treatment, feature engineering, exploratory data analysis (EDA), and data storytelling using a raw transactional retail dataset. The project demonstrates professional data science practices from raw data ingest to submission-ready deliverables.

---

## Dataset
* **Dataset Name:** UCI Online Retail Dataset
* **Source:** UCI Machine Learning Repository ([Link](https://archive.ics.uci.edu/ml/datasets/Online+Retail))
* **Format:** Excel Spreadsheet (`Online_Retail.xlsx`)
* **Raw Size:** 541,909 rows × 8 columns
* **Storage Location:** `data/raw/Online_Retail.xlsx` *(Unchanged raw copy)*

---

## Technologies Used
* **Python 3.14+**
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook, OpenPyXL, Virtualenv (`.venv`)

---

## Project Structure
```text
thiranex_task1_data_cleaning_visualization/
│
├── data/
│   ├── raw/
│   │   └── Online_Retail.xlsx            # Unchanged original dataset
│   └── processed/
│       └── cleaned_retail_data.csv       # Final processed dataset
│
├── notebooks/
│   └── Task_1_Data_Cleaning_Visualization.ipynb  # Comprehensive Jupyter Notebook
│
├── src/
│   ├── data_cleaning.py                 # Data loading, audit & cleaning module
│   └── visualization.py                 # Visualizations generator module
│
├── visualizations/                      # High-resolution PNG figures (300 DPI)
│   ├── 01_missing_values_analysis.png
│   ├── 02_numerical_distributions.png
│   ├── 03_outlier_boxplots.png
│   ├── 04_monthly_sales_trend.png
│   ├── 05_top_products.png
│   ├── 06_country_sales_distribution.png
│   ├── 07_correlation_heatmap.png
│   └── 08_dayofweek_sales_distribution.png
│
├── reports/
│   └── Task_1_Report.md                 # Detailed internship task report
│
├── create_notebook.py                   # Automated notebook generation script
├── run_pipeline.py                      # End-to-end execution pipeline script
├── README.md                            # Project documentation
├── requirements.txt                     # Project dependencies
└── .gitignore                           # Git ignore configurations
```

---

## Data Cleaning Methodology
1. **Deduplication:** Removed 5,268 exact duplicate rows.
2. **Missing Description Handling:** Imputed missing values with `'UNSPECIFIED PRODUCT'` and normalized string formatting.
3. **Missing Customer ID Preservation:** Imputed missing `CustomerID` values with `-1` and assigned `IsGuest = 1` to prevent losing 25% of total sales volume.
4. **Order Cancellations:** Filtered out 9,288 cancelled transactions (`InvoiceNo` starting with `'C'`) into net sales dataset.
5. **Invalid Quantities/Prices:** Excluded negative/zero quantities and prices resulting from adjustments or damaged inventory.
6. **Outlier Flagging:** Evaluated extreme quantities and unit prices using IQR boundaries ($Q_3 + 3 \times \text{IQR}$) and added `IsOutlier` flag while retaining valid sales data.

---

## Analysis & Visualizations

| Visual Report | Description | Output File |
| :--- | :--- | :--- |
| **Missing Values Audit** | Bar chart showing percentage missing per column | `01_missing_values_analysis.png` |
| **Numerical Distributions** | Histogram & KDE of Quantity, UnitPrice, and Revenue | `02_numerical_distributions.png` |
| **Outlier Boxplots** | Comparative box plots before vs. after cleaning | `03_outlier_boxplots.png` |
| **Monthly Revenue Trend** | Dual-axis line chart tracking Revenue & Order Volume | `04_monthly_sales_trend.png` |
| **Top Products** | Horizontal bar charts of top 10 products by revenue & volume | `05_top_products.png` |
| **Regional Sales** | Revenue distribution by Country (UK vs International) | `06_country_sales_distribution.png` |
| **Correlation Heatmap** | Matrix heatmap showing linear relationships | `07_correlation_heatmap.png` |
| **Shopping Dynamics** | Revenue breakdown by Day of Week & Hour of Day | `08_dayofweek_sales_distribution.png` |

---

## Key Insights
* **Total Clean Revenue:** £9,747,747.93 across 397,924 valid sales transactions.
* **Peak Season:** November 2011 generated highest revenue (£1,161,817.38) due to holiday inventory stocking.
* **Geographic Market:** United Kingdom generated 84%+ of overall revenue; top international markets are Netherlands, Ireland (EIRE), and Germany.
* **Peak Shopping Hours:** Most transactions occur between 10:00 AM and 3:00 PM, with Thursday being the highest-revenue day of the week.

---

## How to Run

### Step 1: Clone repository & set up environment
```bash
# Navigate to project directory
cd thiranex_task1_data_cleaning_visualization

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Execute End-to-End Pipeline
```bash
python run_pipeline.py
```
This runs `src/data_cleaning.py` and `src/visualization.py`, generates `data/processed/cleaned_retail_data.csv`, outputs all 8 visualization charts into `visualizations/`, and logs metric summaries.

### Step 3: Build & Launch Jupyter Notebook
```bash
python create_notebook.py
jupyter notebook notebooks/Task_1_Data_Cleaning_Visualization.ipynb
```

---

## Results
- Cleaned dataset saved cleanly to `data/processed/cleaned_retail_data.csv`.
- High-resolution visual charts created in `visualizations/`.
- Full project report generated in `reports/Task_1_Report.md`.

---

## Conclusion
The data pipeline provides a robust, reproducible workflow for cleaning, analyzing, and visualizing e-commerce transactions. All prompt requirements have been met and verified.
