"""
Data Cleaning Module for Thiranex Internship Task 1
Retail / E-commerce Sales Analysis
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw dataset from Excel or CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")
    
    print(f"[INFO] Loading raw dataset from: {file_path}")
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Use .xlsx or .csv")
    
    print(f"[INFO] Raw dataset loaded successfully. Shape: {df.shape}")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Return comprehensive structural and data quality information."""
    inspection_results = {
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_values": {col: df[col].nunique() for col in df.columns}
    }
    return inspection_results


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Perform thorough data cleaning based on UCI Online Retail dataset specifics.
    Returns cleaned DataFrame and a dictionary tracking statistics at each step.
    """
    tracking = {}
    initial_count = len(df)
    tracking["initial_rows"] = initial_count

    df_clean = df.copy()

    # Step 1: Remove exact duplicate rows
    dup_count = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    tracking["duplicates_removed"] = int(dup_count)
    tracking["rows_after_duplicates"] = len(df_clean)

    # Step 2: Handle missing Descriptions
    missing_desc = df_clean["Description"].isnull().sum()
    df_clean["Description"] = df_clean["Description"].fillna("Unspecified Product").astype(str).str.strip().str.upper()
    tracking["missing_description_count"] = int(missing_desc)

    # Step 3: Handle missing CustomerIDs
    # CustomerID is missing for non-registered/guest checkouts.
    # Rather than dropping ~25% of data, we handle CustomerID gracefully.
    # We assign -1 to missing CustomerID to retain sales data while enabling customer-level filtering.
    missing_cust = df_clean["CustomerID"].isnull().sum()
    df_clean["IsGuest"] = df_clean["CustomerID"].isnull().astype(int)
    df_clean["CustomerID"] = df_clean["CustomerID"].fillna(-1).astype(int)
    tracking["missing_customer_id_count"] = int(missing_cust)

    # Step 4: Handle Cancellations (InvoiceNo starting with 'C')
    df_clean["InvoiceNo"] = df_clean["InvoiceNo"].astype(str).str.strip()
    df_clean["IsCancelled"] = df_clean["InvoiceNo"].str.startswith("C").astype(int)
    cancellation_count = df_clean["IsCancelled"].sum()
    tracking["cancellations_count"] = int(cancellation_count)

    # Filter out cancelled orders for primary sales performance analysis
    # (Note: Cancellations often have negative quantities)
    df_valid_orders = df_clean[df_clean["IsCancelled"] == 0].copy()
    tracking["rows_after_excluding_cancellations"] = len(df_valid_orders)

    # Step 5: Filter non-positive Quantity and UnitPrice
    invalid_qty = (df_valid_orders["Quantity"] <= 0).sum()
    invalid_price = (df_valid_orders["UnitPrice"] <= 0).sum()
    tracking["invalid_quantity_count"] = int(invalid_qty)
    tracking["invalid_price_count"] = int(invalid_price)

    df_valid_sales = df_valid_orders[(df_valid_orders["Quantity"] > 0) & (df_valid_orders["UnitPrice"] > 0)].copy()
    tracking["rows_after_valid_qty_price"] = len(df_valid_sales)

    # Step 6: Convert data types
    df_valid_sales["InvoiceDate"] = pd.to_datetime(df_valid_sales["InvoiceDate"])
    df_valid_sales["StockCode"] = df_valid_sales["StockCode"].astype(str).str.strip().str.upper()
    df_valid_sales["Country"] = df_valid_sales["Country"].astype(str).str.strip()

    # Step 7: Detect outliers using IQR method for Quantity and UnitPrice
    q1_qty, q3_qty = df_valid_sales["Quantity"].quantile(0.25), df_valid_sales["Quantity"].quantile(0.75)
    iqr_qty = q3_qty - q1_qty
    upper_qty_bound = q3_qty + 3 * iqr_qty  # using 3*IQR for extreme outlier threshold
    
    q1_price, q3_price = df_valid_sales["UnitPrice"].quantile(0.25), df_valid_sales["UnitPrice"].quantile(0.75)
    iqr_price = q3_price - q1_price
    upper_price_bound = q3_price + 3 * iqr_price

    outliers_qty = (df_valid_sales["Quantity"] > upper_qty_bound).sum()
    outliers_price = (df_valid_sales["UnitPrice"] > upper_price_bound).sum()
    
    tracking["iqr_quantity_upper_limit"] = float(upper_qty_bound)
    tracking["iqr_quantity_outlier_count"] = int(outliers_qty)
    tracking["iqr_price_upper_limit"] = float(upper_price_bound)
    tracking["iqr_price_outlier_count"] = int(outliers_price)

    # Retain all valid business transactions while adding an outlier flag
    df_valid_sales["IsOutlier"] = (
        (df_valid_sales["Quantity"] > upper_qty_bound) | 
        (df_valid_sales["UnitPrice"] > upper_price_bound)
    ).astype(int)

    tracking["final_clean_rows"] = len(df_valid_sales)
    tracking["total_rows_removed"] = initial_count - len(df_valid_sales)
    tracking["retention_rate_pct"] = round((len(df_valid_sales) / initial_count) * 100, 2)

    return df_valid_sales, tracking


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for temporal and financial analysis."""
    df_fe = df.copy()

    # Financial feature
    df_fe["TotalAmount"] = (df_fe["Quantity"] * df_fe["UnitPrice"]).round(2)

    # Temporal features
    df_fe["InvoiceYear"] = df_fe["InvoiceDate"].dt.year
    df_fe["InvoiceMonth"] = df_fe["InvoiceDate"].dt.month
    df_fe["InvoiceMonthName"] = df_fe["InvoiceDate"].dt.strftime("%b")
    df_fe["YearMonth"] = df_fe["InvoiceDate"].dt.strftime("%Y-%m")
    df_fe["DayOfWeek"] = df_fe["InvoiceDate"].dt.day_name()
    df_fe["DayOfWeekNum"] = df_fe["InvoiceDate"].dt.dayofweek
    df_fe["InvoiceDay"] = df_fe["InvoiceDate"].dt.day
    df_fe["InvoiceHour"] = df_fe["InvoiceDate"].dt.hour

    return df_fe


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """Save cleaned dataset to CSV file and verify reloading."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Cleaned dataset saved to: {output_path}")

    # Reload check
    df_reload = pd.read_csv(output_path)
    assert len(df_reload) == len(df), "Saved dataset row count mismatch!"
    print(f"[VERIFIED] Successfully reloaded {len(df_reload)} rows from saved CSV.")
