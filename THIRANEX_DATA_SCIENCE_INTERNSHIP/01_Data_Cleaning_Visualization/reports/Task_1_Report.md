# Comprehensive Data Science Internship Report
**Internship Program:** Thiranex Data Science Internship  
**Task Title:** Task 1 — Data Cleaning & Visualization Project  
**Domain:** Retail / E-Commerce Sales Performance & Customer Analytics  
**Due Date:** 2 October 2026  
**Status:** Completed & Submission-Ready  

---

## 1. Executive Summary & Objective
This report details the execution of **Task 1: Data Cleaning & Visualization Project** as part of the Thiranex Data Science Internship. The primary objective is to take a real-world, raw transactional e-commerce dataset (the UCI Online Retail dataset), conduct an exhaustive data quality audit, perform systematic data cleaning and feature engineering, analyze key performance metrics, and build publication-grade visual dashboards.

By addressing data quality issues—such as missing customer attributes, order cancellations, negative prices/quantities, duplicate entries, and extreme statistical outliers—this project transforms raw transactional logs into reliable analytics-ready data assets.

---

## 2. Dataset Source & Description
* **Original Dataset Source:** UCI Machine Learning Repository — [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
* **Provider:** UCI Machine Learning Repository / E-commerce retailer based in the UK.
* **Period Covered:** 01/12/2010 to 09/12/2011 (1 year of transactions).
* **Dataset Characteristics:**
  * **Raw Record Count:** 541,909 rows × 8 columns.
  * **Data Attributes:**
    1. `InvoiceNo`: 6-digit integral number assigned to each transaction. Starts with 'C' for cancellations.
    2. `StockCode`: Product/item code.
    3. `Description`: Product name/description.
    4. `Quantity`: Quantity of each product per transaction.
    5. `InvoiceDate`: Timestamp of order creation.
    6. `UnitPrice`: Product price per unit in Sterling (£).
    7. `CustomerID`: 5-digit identifier unique to registered customers.
    8. `Country`: Name of the country where customer resides.

---

## 3. Tools & Technology Stack
* **Language:** Python 3.14+
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Environment & Automation:** Jupyter Notebook, Python Virtual Environment (`.venv`), OpenPyXL

---

## 4. Data Quality Audit & Cleaning Methodology

### 4.1 Initial Data Quality Issues Identified
1. **Duplicate Records:** 5,268 exact duplicate rows identified across all columns.
2. **Missing Product Descriptions:** 1,454 rows missing product description strings.
3. **Missing Customer IDs:** 135,080 rows (~24.9% of dataset) missing `CustomerID`.
4. **Order Cancellations:** 9,288 cancelled transactions prefixed with `'C'` in `InvoiceNo`.
5. **Invalid / Non-Positive Values:** 10,624 rows with `Quantity <= 0` and 2,517 rows with `UnitPrice <= 0` (representing damaged goods, test transactions, or inventory adjustments).

### 4.2 Cleaning & Preprocessing Decisions

| Step | Action Taken | Rationale & Impact | Rows Affected |
| :--- | :--- | :--- | :--- |
| **1. Deduplication** | Dropped exact duplicate rows (`df.drop_duplicates()`) | Eliminates double-counted transactions. | -5,268 rows |
| **2. Description Imputation** | Filled `NaN` with `'UNSPECIFIED PRODUCT'`, converted to uppercase and trimmed whitespace | Preserves row integrity while standardizing text searchability. | 1,454 rows imputed |
| **3. CustomerID Preservation** | Imputed missing `CustomerID` with `-1` and created binary flag `IsGuest = 1` | **Critical Decision:** Deleting 135K rows would distort total revenue analysis by 25%. Retaining guest sales enables accurate macro-revenue reporting. | 135,080 rows flagged |
| **4. Cancellations Handling** | Filtered out orders starting with `'C'` into separate net sales dataset (`IsCancelled = 1`) | Prevents negative sales skewing performance charts while preserving cancellation tracking. | -9,288 rows filtered |
| **5. Invalid Quantities & Prices** | Filtered `Quantity > 0` and `UnitPrice > 0` | Excludes system write-offs, sample giveaways, and negative adjustments. | -1,336 additional rows |

---

## 5. Outlier Detection & Handling
Outliers were analyzed using the **Interquartile Range (IQR)** method on numerical metrics:
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Upper Limit} = Q_3 + 3 \times \text{IQR}$$

* **Quantity IQR Upper Limit:** 37 units (Extreme outliers identified: ~26,840 records).
* **UnitPrice IQR Upper Limit:** £8.85 (Extreme outliers identified: ~36,220 records).

**Handling Rationale:** E-commerce retail datasets naturally exhibit high variance due to bulk B2B purchases (e.g., wholesalers purchasing 1,000+ units of party supplies). Rather than blindly removing high-value orders and understating revenue, we tagged records using an `IsOutlier` flag and used log-scaling & percentile clipping for visualizations.

---

## 6. Feature Engineering
Derived features created to enable rich temporal and financial dimensions:
* **`TotalAmount`** $= \text{Quantity} \times \text{UnitPrice}$ (Rounded to 2 decimal places).
* **`InvoiceYear`**, **`InvoiceMonth`**, **`InvoiceMonthName`**, **`YearMonth`**: For monthly trend analysis.
* **`DayOfWeek`**, **`DayOfWeekNum`**, **`InvoiceDay`**: For weekly demand distribution.
* **`InvoiceHour`**: For hourly peak customer activity.

---

## 7. Exploratory Data Analysis & Empirical Insights

### Key Metrics Summary
* **Total Clean Sales Revenue:** £9,747,747.93 (~£9.75 Million)
* **Total Clean Units Sold:** 5,170,000+ units
* **Total Unique Transactions:** 18,536 invoices
* **Unique Registered Customers:** 4,339 accounts
* **Average Order Value (AOV):** £525.88

### Analytical Insights
1. **Seasonality Spike:** Revenue grows steadily throughout Q1-Q3 and surges dramatically in November 2011 (£1.16M revenue), driven by pre-holiday inventory restocking for Black Friday and Christmas.
2. **Top Performing Products:**
   * Highest Revenue: `DOTCOM POSTAGE`, `REGENCY CAKESTAND 3 TIER`, `PARTY BUNTING`.
   * Highest Volume: `PAPER CRAFT , LITTLE BIRDIE`, `MEDIUM CERAMIC TOP STORAGE JAR`.
3. **Geographic Distribution:**
   * **Domestic Market:** United Kingdom accounts for over **84%** of overall sales volume and revenue (£8.18M).
   * **Top International Markets:** Netherlands (£285K+), EIRE / Ireland (£263K+), Germany (£228K+), and France (£197K+).
4. **Shopping Hours & Days:**
   * Peak purchasing hours occur between **10:00 AM and 3:00 PM** GMT.
   * **Thursday** generates the highest total daily revenue of any day of the week.

---

## 8. Dashboard Visualizations Summary
All charts were saved to `visualizations/` as 300 DPI high-resolution figures:
1. `01_missing_values_analysis.png`: Missingness bar chart before cleaning.
2. `02_numerical_distributions.png`: Histograms & KDE of Quantity, UnitPrice, and TotalAmount.
3. `03_outlier_boxplots.png`: Outlier distribution comparison before vs after cleaning.
4. `04_monthly_sales_trend.png`: Dual-axis plot of Monthly Revenue (£) and Order Count.
5. `05_top_products.png`: Horizontal bar charts of Top 10 products by revenue and volume.
6. `06_country_sales_distribution.png`: Regional revenue breakdown (UK vs International).
7. `07_correlation_heatmap.png`: Correlation matrix of key numerical features.
8. `08_dayofweek_sales_distribution.png`: Day-of-week and hourly revenue dynamics.

---

## 9. Challenges & Limitations
* **Missing Customer IDs:** Approximately 25% of orders lacked customer identifiers, preventing complete RFM customer segmentation across all orders.
* **Dataset Period:** The dataset terminates in early December 2011, making December 2011 incomplete compared to full months.

---

## 10. Conclusion & Recommendations
The data cleaning and visualization pipeline successfully transformed messy raw e-commerce data into actionable insights. 

**Business Recommendations:**
1. **Inventory Preparation:** Increase inventory stock for top-selling lines starting in September to support November demand peaks.
2. **International Expansion:** Target marketing campaigns toward Netherlands, EIRE, and Germany, which show strong organic demand.
3. **Customer Registration Drive:** Offer guest checkout incentives (e.g., 5% discount on account creation) to reduce guest transactions and improve customer lifetime value tracking.
