"""
Pipeline Runner Script for Thiranex Internship Task 1
Data Cleaning, Feature Engineering, Visualizations & Insights Generation
"""

import os
import json
import pandas as pd
import numpy as np

from src.data_cleaning import load_raw_data, inspect_data, clean_dataset, engineer_features, save_cleaned_data
from src.visualization import (
    plot_missing_values,
    plot_numerical_distributions,
    plot_outlier_boxplots,
    plot_monthly_sales_trend,
    plot_top_products,
    plot_country_sales,
    plot_correlation_heatmap,
    plot_dayofweek_hourly_sales
)


def run_pipeline():
    project_root = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(project_root, "data", "raw", "Online_Retail.xlsx")
    processed_path = os.path.join(project_root, "data", "processed", "cleaned_retail_data.csv")
    viz_dir = os.path.join(project_root, "visualizations")

    print("==================================================")
    print("STEP 1: LOADING RAW DATASET")
    print("==================================================")
    df_raw = load_raw_data(raw_path)

    print("\n==================================================")
    print("STEP 2: DATA UNDERSTANDING & QUALITY CHECK")
    print("==================================================")
    raw_inspection = inspect_data(df_raw)
    print(f"Raw Shape: {raw_inspection['num_rows']} rows, {raw_inspection['num_cols']} columns")
    print("\nColumn Data Types:")
    for col, dtype in raw_inspection['dtypes'].items():
        print(f" - {col}: {dtype}")
    
    print("\nMissing Values:")
    for col, count in raw_inspection['missing_values'].items():
        pct = raw_inspection['missing_percentage'][col]
        print(f" - {col}: {count:,} missing ({pct:.2f}%)")
    
    print(f"\nExact Duplicate Rows: {raw_inspection['duplicate_rows']:,}")

    # Generate plot 1: Missing values before cleaning
    plot_missing_values(df_raw, viz_dir)

    print("\n==================================================")
    print("STEP 3: EXECUTING DATA CLEANING")
    print("==================================================")
    df_cleaned, tracking_stats = clean_dataset(df_raw)

    print(json.dumps(tracking_stats, indent=2))

    print("\n==================================================")
    print("STEP 4: FEATURE ENGINEERING")
    print("==================================================")
    df_final = engineer_features(df_cleaned)
    print(f"Features created: TotalAmount, InvoiceYear, InvoiceMonth, DayOfWeek, InvoiceHour, etc.")
    print(f"Final dataset shape: {df_final.shape}")

    print("\n==================================================")
    print("STEP 5: SAVING CLEANED DATASET")
    print("==================================================")
    save_cleaned_data(df_final, processed_path)

    print("\n==================================================")
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("==================================================")
    plot_numerical_distributions(df_final, viz_dir)
    plot_outlier_boxplots(df_raw, df_final, viz_dir)
    plot_monthly_sales_trend(df_final, viz_dir)
    plot_top_products(df_final, viz_dir)
    plot_country_sales(df_final, viz_dir)
    plot_correlation_heatmap(df_final, viz_dir)
    plot_dayofweek_hourly_sales(df_final, viz_dir)

    print("\n==================================================")
    print("STEP 7: DERIVING AUTOMATED NUMERICAL INSIGHTS")
    print("==================================================")
    total_rev = df_final['TotalAmount'].sum()
    total_qty = df_final['Quantity'].sum()
    total_orders = df_final['InvoiceNo'].nunique()
    total_customers = df_final[df_final['CustomerID'] != -1]['CustomerID'].nunique()
    avg_order_val = total_rev / total_orders if total_orders > 0 else 0

    monthly_rev = df_final.groupby('YearMonth')['TotalAmount'].sum()
    peak_month = monthly_rev.idxmax()
    peak_month_rev = monthly_rev.max()
    lowest_month = monthly_rev.idxmin()
    lowest_month_rev = monthly_rev.min()

    top_prod_rev = df_final.groupby('Description')['TotalAmount'].sum().idxmax()
    top_prod_rev_val = df_final.groupby('Description')['TotalAmount'].sum().max()
    
    top_prod_qty = df_final.groupby('Description')['Quantity'].sum().idxmax()
    top_prod_qty_val = df_final.groupby('Description')['Quantity'].sum().max()

    top_country = df_final.groupby('Country')['TotalAmount'].sum().idxmax()
    top_country_rev = df_final.groupby('Country')['TotalAmount'].sum().max()
    top_country_pct = (top_country_rev / total_rev) * 100

    non_uk_top = df_final[df_final['Country'] != 'United Kingdom'].groupby('Country')['TotalAmount'].sum().idxmax()
    non_uk_top_rev = df_final[df_final['Country'] != 'United Kingdom'].groupby('Country')['TotalAmount'].sum().max()

    print(f"Total Revenue Generated: £{total_rev:,.2f}")
    print(f"Total Products Sold (Units): {total_qty:,}")
    print(f"Total Unique Orders: {total_orders:,}")
    print(f"Unique Registered Customers: {total_customers:,}")
    print(f"Average Order Value (AOV): £{avg_order_val:,.2f}")
    print(f"Peak Revenue Month: {peak_month} (£{peak_month_rev:,.2f})")
    print(f"Lowest Revenue Month: {lowest_month} (£{lowest_month_rev:,.2f})")
    print(f"Top Product by Revenue: {top_prod_rev} (£{top_prod_rev_val:,.2f})")
    print(f"Top Product by Volume: {top_prod_qty} ({top_prod_qty_val:,} units)")
    print(f"Top Domestic Market: {top_country} (£{top_country_rev:,.2f}, {top_country_pct:.2f}% of total revenue)")
    print(f"Top International Market: {non_uk_top} (£{non_uk_top_rev:,.2f})")

    # Save summary stats to json for notebook & report referencing
    summary = {
        "raw_rows": df_raw.shape[0],
        "raw_cols": df_raw.shape[1],
        "clean_rows": df_final.shape[0],
        "clean_cols": df_final.shape[1],
        "total_revenue": round(float(total_rev), 2),
        "total_quantity": int(total_qty),
        "total_orders": int(total_orders),
        "unique_customers": int(total_customers),
        "avg_order_value": round(float(avg_order_val), 2),
        "peak_month": peak_month,
        "peak_month_revenue": round(float(peak_month_rev), 2),
        "lowest_month": lowest_month,
        "lowest_month_revenue": round(float(lowest_month_rev), 2),
        "top_product_revenue": top_prod_rev,
        "top_product_revenue_value": round(float(top_prod_rev_val), 2),
        "top_product_quantity": top_prod_qty,
        "top_product_quantity_value": int(top_prod_qty_val),
        "top_country": top_country,
        "top_country_revenue": round(float(top_country_rev), 2),
        "top_country_pct": round(float(top_country_pct), 2),
        "top_international_country": non_uk_top,
        "top_international_revenue": round(float(non_uk_top_rev), 2),
        "cleaning_tracking": tracking_stats
    }

    with open(os.path.join(project_root, "reports", "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nPipeline executed successfully! Summary saved to reports/summary_metrics.json")


if __name__ == "__main__":
    run_pipeline()
