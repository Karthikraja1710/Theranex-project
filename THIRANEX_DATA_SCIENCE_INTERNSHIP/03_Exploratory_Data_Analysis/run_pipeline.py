"""
Master Pipeline Script for Task 3: Exploratory Data Analysis (EDA)
World Happiness / Socioeconomic Analysis (2021)
"""

import os
import json
import pandas as pd
import numpy as np

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


def run_pipeline():
    project_root = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(project_root, "data", "raw", "world-happiness-report-2021.csv")
    processed_path = os.path.join(project_root, "data", "processed", "cleaned_happiness_2021.csv")
    viz_dir = os.path.join(project_root, "visualizations")
    reports_dir = os.path.join(project_root, "reports")

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("==================================================")
    print("STEP 1: INGESTING RAW WORLD HAPPINESS 2021 DATASET")
    print("==================================================")
    df_raw = load_raw_data(raw_path)

    print("\n==================================================")
    print("STEP 2: STRUCTURAL QUALITY AUDIT")
    print("==================================================")
    inspection = inspect_data(df_raw)
    print(f"Raw Shape: {inspection['num_rows']} rows x {inspection['num_cols']} columns")
    print(f"Duplicate Rows: {inspection['duplicate_rows']}")

    print("\n==================================================")
    print("STEP 3: DATA PREPROCESSING & STANDARDIZATION")
    print("==================================================")
    df_clean, tracking = clean_dataframe(df_raw)
    df_clean.to_csv(processed_path, index=False)
    print(f"[INFO] Cleaned dataset saved to: {processed_path}")
    print(f"[INFO] Analysis Columns: {list(df_clean.columns)}")

    print("\n==================================================")
    print("STEP 4: UNIVARIATE STATISTICAL ANALYSIS")
    print("==================================================")
    df_univariate = compute_univariate_stats(df_clean)
    print(df_univariate.to_string(index=False))

    print("\n==================================================")
    print("STEP 5: BIVARIATE & MULTIVARIATE CORRELATIONS")
    print("==================================================")
    pearson_corr, spearman_corr = compute_correlations(df_clean)
    print("\nPearson Correlation with HappinessScore:")
    print(pearson_corr['HappinessScore'].sort_values(ascending=False).to_string())

    print("\n==================================================")
    print("STEP 6: GENERATING EDA VISUALIZATIONS")
    print("==================================================")
    plot_happiness_distribution(df_clean, viz_dir)
    plot_regional_boxplot(df_clean, viz_dir)
    plot_gdp_vs_happiness(df_clean, viz_dir)
    plot_correlation_heatmap(df_clean, viz_dir)
    plot_top10_bottom10_countries(df_clean, viz_dir)
    plot_regional_mean_happiness(df_clean, viz_dir)
    plot_key_drivers_scatter_grid(df_clean, viz_dir)
    plot_correlation_comparison(df_clean, viz_dir)

    print("\n==================================================")
    print("STEP 7: HIGHEST & LOWEST RANKED COUNTRIES & REGIONS")
    print("==================================================")
    top_country = df_clean.nlargest(1, 'HappinessScore').iloc[0]
    bottom_country = df_clean.nsmallest(1, 'HappinessScore').iloc[0]

    regional_means = df_clean.groupby('Region')['HappinessScore'].mean().sort_values(ascending=False)
    top_region = regional_means.index[0]
    top_region_score = regional_means.iloc[0]
    bottom_region = regional_means.index[-1]
    bottom_region_score = regional_means.iloc[-1]

    print(f"Happiest Country: {top_country['Country']} ({top_country['Region']}) - Score: {top_country['HappinessScore']:.3f}")
    print(f"Unhappiest Country: {bottom_country['Country']} ({bottom_country['Region']}) - Score: {bottom_country['HappinessScore']:.3f}")
    print(f"Highest Ranked Region: {top_region} (Mean Score: {top_region_score:.3f})")
    print(f"Lowest Ranked Region: {bottom_region} (Mean Score: {bottom_region_score:.3f})")

    # Log summary to JSON
    summary_path = os.path.join(reports_dir, "summary_metrics.json")
    summary = {
        "dataset_year": 2021,
        "total_countries": len(df_clean),
        "total_regions": tracking["num_regions"],
        "top_happiest_country": {
            "country": top_country['Country'],
            "region": top_country['Region'],
            "score": round(float(top_country['HappinessScore']), 3)
        },
        "bottom_happiest_country": {
            "country": bottom_country['Country'],
            "region": bottom_country['Region'],
            "score": round(float(bottom_country['HappinessScore']), 3)
        },
        "top_region": {
            "region": top_region,
            "mean_score": round(float(top_region_score), 3)
        },
        "bottom_region": {
            "region": bottom_region,
            "mean_score": round(float(bottom_region_score), 3)
        },
        "univariate_stats": df_univariate.to_dict(orient="records"),
        "pearson_correlations_with_happiness": pearson_corr['HappinessScore'].round(4).to_dict(),
        "spearman_correlations_with_happiness": spearman_corr['HappinessScore'].round(4).to_dict()
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SUCCESS] Summary metrics saved to: {summary_path}")
    print("Task 3 EDA Pipeline executed successfully!")


if __name__ == "__main__":
    run_pipeline()
