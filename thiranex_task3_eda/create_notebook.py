"""
Notebook Builder Script for Task 3: Exploratory Data Analysis (EDA)
Generates notebooks/Task_3_EDA.ipynb with markdown and code cells.
"""

import os
import nbformat as nbf


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Header
    cells.append(nbf.v4.new_markdown_cell(
"""# Thiranex Data Science Internship — Task 3
## Project Title: World Happiness & Socioeconomic Exploratory Data Analysis (EDA)
**Author:** Data Science Intern  
**Domain:** World Happiness / Socioeconomic Analysis  
**Dataset:** World Happiness Report 2021 Dataset  
**Due Date:** 16 October 2026  
***

### 1. Executive Summary & Project Objectives
Understanding the global drivers of subjective well-being is vital for policymakers, economists, and international development organizations. The **World Happiness Report 2021** quantifies national happiness scores (Cantril Ladder) across 149 nations based on economic output, social support, health, freedom, generosity, and public trust.

**Core Objectives:**
1. **Data Understanding & Quality Audit**: Structural inspection of 149 country records across 9 core socioeconomic dimensions.
2. **Univariate Analysis**: Statistical distribution metrics (mean, median, IQR, skewness) for all numerical variables.
3. **Bivariate & Multivariate Analysis**: Quantify relationships between happiness scores and key drivers using Pearson ($r$) and Spearman ($\rho$) correlation matrices.
4. **Regional Comparison**: Analyze socioeconomic disparities across 10 world regions.
5. **Data Storytelling**: Generate publication-grade visual reports and business/policy recommendations.
"""
    ))

    # Section 2: Setup & Imports
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 2. Environment Setup & Library Imports
We import core scientific computing libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats`), along with modular functions from `src/analysis.py`.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

# Add project root to sys.path
sys.path.append(os.path.abspath(".."))

from src.analysis import (
    load_raw_data,
    inspect_data,
    clean_dataframe,
    compute_univariate_stats,
    compute_correlations,
    plot_happiness_distribution,
    plot_regional_boxplot,
    plot_gdp_vs_happiness,
    plot_correlation_heatmap,
    plot_top10_bottom10_countries,
    plot_regional_mean_happiness,
    plot_key_drivers_scatter_grid,
    plot_correlation_comparison
)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
%matplotlib inline

print("Task 3 Libraries and analysis module successfully imported!")
"""
    ))

    # Section 3: Data Loading
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 3. Data Loading & Ingestion
We load the original raw dataset from `data/raw/world-happiness-report-2021.csv`.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""raw_data_path = os.path.join("..", "data", "raw", "world-happiness-report-2021.csv")
df_raw = load_raw_data(raw_data_path)

print(f"Dataset Shape: {df_raw.shape[0]} countries x {df_raw.shape[1]} features")
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# Preview head and tail
print("--- Top 5 Records ---")
display(df_raw.head())
"""
    ))

    # Section 4: Data Preprocessing & Cleaning
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 4. Data Quality Audit & Cleaning
We audit data types, verify missing values, rename columns to clean standard conventions, and save `cleaned_happiness_2021.csv`.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""inspection = inspect_data(df_raw)
print("Duplicate Rows:", inspection['duplicate_rows'])

df_clean, tracking = clean_dataframe(df_raw)
processed_path = os.path.join("..", "data", "processed", "cleaned_happiness_2021.csv")
df_clean.to_csv(processed_path, index=False)

print("Cleaned Dataset Shape:", df_clean.shape)
display(df_clean.head())
"""
    ))

    # Section 5: Univariate Statistical Analysis
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 5. Univariate Statistical Analysis
We compute comprehensive descriptive statistics (Mean, Median, StdDev, Min, Max, Q25, Q75, IQR, Skewness) for each numerical variable.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""df_univariate = compute_univariate_stats(df_clean)

print("=== UNIVARIATE STATISTICAL SUMMARY ===")
display(df_univariate)
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# 1. Happiness Distribution Plot
viz_dir = os.path.join("..", "visualizations")
plot_happiness_distribution(df_clean, viz_dir)

Image(filename=os.path.join(viz_dir, "01_happiness_distribution.png"))
"""
    ))

    # Section 6: Bivariate & Regional Analysis
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 6. Bivariate & Regional Disparity Analysis

We analyze how Happiness Scores vary across world regions and examine economic output (`LogGDP`) as a primary predictor.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# 2. Regional Happiness Boxplot
plot_regional_boxplot(df_clean, viz_dir)
Image(filename=os.path.join(viz_dir, "02_regional_happiness_boxplot.png"))
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# 3. Logged GDP vs Happiness Scatter Plot with OLS Regression Trendline
plot_gdp_vs_happiness(df_clean, viz_dir)
Image(filename=os.path.join(viz_dir, "03_gdp_vs_happiness_scatter.png"))
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# 4. Top 10 vs Bottom 10 Countries
plot_top10_bottom10_countries(df_clean, viz_dir)
Image(filename=os.path.join(viz_dir, "05_top_bottom_countries_bar.png"))
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# 5. Regional Mean Happiness Comparison
plot_regional_mean_happiness(df_clean, viz_dir)
Image(filename=os.path.join(viz_dir, "06_regional_mean_happiness.png"))
"""
    ))

    # Section 7: Multivariate Correlation Analysis
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 7. Multivariate Correlation Analysis
We compute Pearson ($r$, linear relationship) and Spearman ($\rho$, monotonic rank relationship) correlation matrices.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""pearson_corr, spearman_corr = compute_correlations(df_clean)

print("=== PEARSON CORRELATION MATRIX ===")
display(pearson_corr)
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# 6. Correlation Heatmap
plot_correlation_heatmap(df_clean, viz_dir)
Image(filename=os.path.join(viz_dir, "04_correlation_heatmap.png"))
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# 7. Pairwise Factors Scatter Grid
plot_key_drivers_scatter_grid(df_clean, viz_dir)
Image(filename=os.path.join(viz_dir, "07_pairwise_factors_scatter.png"))
"""
    ))

    cells.append(nbf.v4.new_code_cell(
r"""# 8. Pearson vs Spearman Correlation Comparison Chart
plot_correlation_comparison(df_clean, viz_dir)
Image(filename=os.path.join(viz_dir, "08_correlation_comparison.png"))
"""
    ))

    # Section 8: Insights & Conclusion
    cells.append(nbf.v4.new_markdown_cell(
"""---
### 8. Key Empirical Findings & Discussion

1. **Top & Bottom Rankings**:
   - **Happiest Country**: **Finland (7.842)** (followed by Denmark, Switzerland, Iceland, Netherlands).
   - **Unhappiest Country**: **Afghanistan (2.523)** (followed by Zimbabwe, Rwanda, Botswana, Malawi).
   - **Highest Ranked Region**: **North America and ANZ (7.128)** & **Western Europe (6.915)**.
   - **Lowest Ranked Region**: **South Asia (4.442)** & **Sub-Saharan Africa (4.494)**.

2. **Primary Predictors**:
   - **Logged GDP per Capita**: Highest correlation ($r = +0.7898$, $\rho = +0.8131$). Economic prosperity provides essential healthcare, security, and infrastructure.
   - **Healthy Life Expectancy**: Strongly correlated ($r = +0.7681$).
   - **Social Support**: Strongly correlated ($r = +0.7569$). Having trusted friends/family in times of need is vital.
   - **Freedom to Make Life Choices**: Moderately correlated ($r = +0.6078$).
   - **Perception of Corruption**: Negatively correlated ($r = -0.4211$). High institutional corruption reduces public well-being.

3. **Methodological Note on Correlation vs. Causation**:
   - High correlation ($r = 0.79$) indicates strong predictive association, **not direct causation**. For instance, high GDP per capita enables better healthcare infrastructure, which in turn enhances life expectancy and social security.

---
### 9. Analytical Conclusion
The exploratory data analysis successfully identifies economic prosperity, social safety nets, and physical health as the primary pillars of global human happiness.
"""
    ))

    nb['cells'] = cells

    nb_path = os.path.join("notebooks", "Task_3_EDA.ipynb")
    os.makedirs(os.path.dirname(nb_path), exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"[NOTEBOOK CREATED] Saved to: {nb_path}")


if __name__ == "__main__":
    build_notebook()
