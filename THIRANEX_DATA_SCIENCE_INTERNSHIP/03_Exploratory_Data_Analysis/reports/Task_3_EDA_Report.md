# Comprehensive Data Science Internship Report
**Internship Program:** Thiranex Data Science Internship  
**Task Title:** Task 3 — Exploratory Data Analysis (EDA) Project  
**Domain:** World Happiness & Socioeconomic Analytics  
**Due Date:** 16 October 2026  
**Status:** Completed & Submission-Ready  

---

## 1. Introduction & Problem Statement
Subjective well-being and national happiness serve as comprehensive indicators of national progress, moving beyond simple economic metrics like Gross Domestic Product (GDP).

The primary objective of **Task 3: Exploratory Data Analysis (EDA) Project** is to conduct an in-depth, rigorous statistical and exploratory investigation of the **World Happiness Report 2021 dataset**. By analyzing national happiness scores (Cantril Ladder) across 149 countries, this project identifies key socioeconomic drivers, evaluates regional disparities, measures linear and rank correlations, and communicates policy-relevant insights.

---

## 2. Dataset Source & Description
* **Dataset Name:** World Happiness Report 2021 Dataset
* **Source:** Sustainable Development Solutions Network (SDSN) / Gallup World Poll ([GitHub Mirror](https://raw.githubusercontent.com/anuragg130/DS-Visualization---World-Happiness-Report-2021/main/world-happiness-report-2021.csv))
* **Dataset Year:** 2021
* **Scope:** 149 countries across 10 global geographical regions.

### Major Variables Analyzed

| Feature Name | Clean Standard Name | Description & Scale |
| :--- | :--- | :--- |
| `Ladder score` | `HappinessScore` | Subjective well-being score on Cantril Ladder scale (0 = Worst possible, 10 = Best possible). |
| `Logged GDP per capita` | `LogGDP` | Natural logarithm of Purchasing Power Parity (PPP) GDP per capita. |
| `Social support` | `SocialSupport` | National average binary response (0 or 1) to having relatives/friends to rely on. |
| `Healthy life expectancy` | `HealthyLifeExpectancy` | Expected years of healthy life at birth (World Health Organization data). |
| `Freedom to make life choices` | `Freedom` | National average binary response regarding freedom of personal life choices. |
| `Generosity` | `Generosity` | Residual of national average response to donating money to charity. |
| `Perceptions of corruption` | `CorruptionPerception` | National average perception of corruption in government and business (0 to 1). |
| `Regional indicator` | `Region` | Geographic classification of countries into 10 world regions. |

---

## 3. Data Preparation & Structural Quality Audit

* **Raw Dimensions:** 149 rows × 20 columns.
* **Missing Values Audit:** Zero missing values across key analysis variables ($N = 149$ complete records).
* **Exact Duplicate Rows:** 0 duplicate rows detected.
* **Cleaning Transformations:** Standardized column naming conventions and saved cleaned data to `data/processed/cleaned_happiness_2021.csv`.

---

## 4. Univariate Statistical Analysis

Summary statistical metrics computed across all 149 nations:

| Socioeconomic Variable | Mean | Median | Std Dev | Min | Q25 | Q75 | Max | IQR | Skewness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Happiness Score** | 5.5328 | 5.5340 | 1.0739 | 2.5230 | 4.8520 | 6.2550 | 7.8420 | 1.4030 | -0.1043 |
| **Logged GDP per Capita** | 9.4322 | 9.5690 | 1.1586 | 6.6350 | 8.5410 | 10.4210 | 11.6470 | 1.8800 | -0.3520 |
| **Social Support** | 0.8147 | 0.8320 | 0.1149 | 0.4630 | 0.7500 | 0.9050 | 0.9830 | 0.1550 | -0.9378 |
| **Healthy Life Expectancy** | 64.9928 | 66.6030 | 6.7620 | 48.4780 | 59.8020 | 69.6000 | 76.9530 | 9.7980 | -0.5220 |
| **Freedom** | 0.7916 | 0.8040 | 0.1133 | 0.3820 | 0.7180 | 0.8770 | 0.9700 | 0.1590 | -0.7548 |
| **Generosity** | -0.0151 | -0.0360 | 0.1507 | -0.2880 | -0.1260 | 0.0790 | 0.5420 | 0.2050 | +1.0100 |
| **Corruption Perception** | 0.7274 | 0.7810 | 0.1792 | 0.0820 | 0.6670 | 0.8450 | 0.9390 | 0.1780 | -1.5775 |

---

## 5. Bivariate & Multivariate Correlation Analysis

Both **Pearson ($r$, linear correlation)** and **Spearman ($\rho$, monotonic rank correlation)** coefficients were calculated between `HappinessScore` and socioeconomic drivers:

| Socioeconomic Factor | Pearson ($r$) | Spearman ($\rho$) | Relationship Strength & Direction |
| :--- | :--- | :--- | :--- |
| **Logged GDP per Capita** | **+0.7898** | **+0.8131** | Very Strong Positive Linear & Monotonic |
| **Healthy Life Expectancy** | **+0.7681** | **+0.7801** | Very Strong Positive |
| **Social Support** | **+0.7569** | **+0.7842** | Very Strong Positive |
| **Freedom to Make Choices** | **+0.6078** | **+0.6033** | Moderate-to-Strong Positive |
| **Corruption Perception** | **-0.4211** | **-0.3298** | Moderate Negative |
| **Generosity** | **-0.0178** | **-0.0694** | Negligible / Weak Linear |

All major correlations ($r > 0.40$) are statistically significant at $p < 0.001$.

---

## 6. Regional Disparities & Country Rankings

### Top 5 Happiest Nations (2021)
1. **Finland** (Western Europe) — Score: **7.842**
2. **Denmark** (Western Europe) — Score: **7.620**
3. **Switzerland** (Western Europe) — Score: **7.571**
4. **Iceland** (Western Europe) — Score: **7.554**
5. **Netherlands** (Western Europe) — Score: **7.464**

### Bottom 5 Lowest Happiness Nations (2021)
1. **Afghanistan** (South Asia) — Score: **2.523**
2. **Zimbabwe** (Sub-Saharan Africa) — Score: **3.145**
3. **Rwanda** (Sub-Saharan Africa) — Score: **3.415**
4. **Botswana** (Sub-Saharan Africa) — Score: **3.467**
5. **Malawi** (Sub-Saharan Africa) — Score: **3.600**

### Regional Performance Summary
* **Highest Mean Happiness Region:** **North America and ANZ** (Mean Score: **7.128**) & **Western Europe** (Mean Score: **6.915**).
* **Lowest Mean Happiness Region:** **South Asia** (Mean Score: **4.442**) & **Sub-Saharan Africa** (Mean Score: **4.494**).

---

## 7. Key Analytical Insights

1. **Economic Prosperity & Health as Core Pillars:** Logged GDP per capita ($r = 0.790$) and Healthy Life Expectancy ($r = 0.768$) exhibit the strongest positive association with national happiness. Nations with log GDP > 10.0 consistently report happiness scores above 6.5.
2. **Social Safety Nets Mitigate Disparities:** Social support ($r = 0.757$) is essential. Countries with high social support scores (>0.90) buffer against economic shocks.
3. **Corruption Impairs Well-being:** Public corruption perceptions ($r = -0.421$) inversely impact happiness. Nordic countries (Denmark, Finland, Sweden) report corruption perception scores below 0.25 alongside top-tier happiness ranks.

> **Methodological Note on Correlation vs. Causation:**  
> While Logged GDP per Capita strongly correlates with Happiness ($r = 0.79$), GDP alone does not directly "cause" happiness. Instead, higher economic output enables investments in healthcare, social security, and institutional transparency.

---

## 8. Dashboard Visualizations Summary
All figures are saved in `visualizations/` as 300 DPI high-resolution assets:
1. `01_happiness_distribution.png`: Histogram & KDE distribution of Happiness Score.
2. `02_regional_happiness_boxplot.png`: Box plots comparing Happiness Scores across World Regions.
3. `03_gdp_vs_happiness_scatter.png`: Scatter plot with OLS linear trendline for GDP vs Happiness.
4. `04_correlation_heatmap.png`: Pearson Correlation Matrix heatmap.
5. `05_top10_bottom10_countries.png`: Side-by-side bar chart of Top 10 vs Bottom 10 nations.
6. `06_regional_mean_happiness.png`: Mean Happiness Score by Region.
7. `07_pairwise_factors_scatter.png`: 2x2 grid of scatter plots for key drivers.
8. `08_correlation_comparison.png`: Bar chart comparing Pearson ($r$) vs Spearman ($\rho$) correlation metrics.

---

## 9. Limitations & Future Work
* **Cross-Sectional Dataset:** The 2021 dataset provides a single snapshot in time. Longitudinal multi-year panel analysis can track post-pandemic recovery trends.
* **Self-Reported Measures:** Cantril Ladder surveys reflect subjective self-evaluations, which can be influenced by cultural communication norms.
