"""
Preprocessing Module for Task 2: Predictive Modeling
IBM Telco Customer Churn Prediction
Handles raw data ingestion, structural audit, data cleaning, feature transformation,
and stratified train-test splitting without data leakage.
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw Telco Customer Churn dataset from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")
    
    print(f"[INFO] Loading raw Telco dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[INFO] Raw dataset loaded successfully. Shape: {df.shape}")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Return structural quality metrics and target distribution."""
    missing = df.isnull().sum().to_dict()
    blank_spaces_in_total_charges = 0
    if "TotalCharges" in df.columns:
        blank_spaces_in_total_charges = int((df["TotalCharges"].astype(str).str.strip() == "").sum())

    inspection = {
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": missing,
        "blank_total_charges_count": blank_spaces_in_total_charges,
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df["Churn"].value_counts().to_dict() if "Churn" in df.columns else {}
    }
    return inspection


def clean_raw_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Clean raw DataFrame:
    1. Remove duplicate rows.
    2. Convert TotalCharges whitespace to float & fill missing values.
    3. Convert target Churn ('Yes'/'No') to 1/0.
    4. Separate feature matrix X and target y.
    """
    df_clean = df.copy()
    tracking = {}

    dup_count = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    tracking["duplicates_removed"] = int(dup_count)

    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = df_clean["TotalCharges"].astype(str).str.strip()
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors='coerce')
        
        missing_tc = df_clean["TotalCharges"].isnull().sum()
        tracking["missing_total_charges_count"] = int(missing_tc)
        
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["MonthlyCharges"] * df_clean["tenure"])

    if "Churn" not in df_clean.columns:
        raise KeyError("Target column 'Churn' not found in dataset!")

    y = df_clean["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    tracking["target_churn_rate_pct"] = round((y.mean()) * 100, 2)
    tracking["total_churned_customers"] = int(y.sum())
    tracking["total_retained_customers"] = int((y == 0).sum())

    drop_cols = ["customerID", "Churn"]
    feature_cols = [c for c in df_clean.columns if c not in drop_cols]
    X = df_clean[feature_cols].copy()

    return X, y, tracking


class CustomColumnTransformer:
    """
    Applies numerical scaling and categorical one-hot encoding
    to feature matrices without data leakage.
    """
    def __init__(self, num_cols: list[str], cat_cols: list[str]):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.num_means_ = {}
        self.num_stds_ = {}
        self.cat_categories_ = {}
        self.feature_names_out_ = []

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        
        # Fit numerical standard scalers
        for col in self.num_cols:
            vals = X[col].astype(float).values
            mean_val = np.nanmean(vals)
            std_val = np.nanstd(vals)
            if std_val == 0 or np.isnan(std_val):
                std_val = 1.0
            self.num_means_[col] = mean_val
            self.num_stds_[col] = std_val

        # Fit categorical one-hot categories
        feature_names = list(self.num_cols)
        for col in self.cat_cols:
            cats = sorted(X[col].astype(str).unique().tolist())
            # Drop first category to avoid multicollinearity
            if len(cats) > 1:
                cats = cats[1:]
            self.cat_categories_[col] = cats
            for c in cats:
                feature_names.append(f"{col}_{c}")

        self.feature_names_out_ = np.array(feature_names)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        transformed_parts = []

        # Scale numerical features
        for col in self.num_cols:
            vals = X[col].astype(float).values
            mean_val = self.num_means_[col]
            std_val = self.num_stds_[col]
            scaled = (vals - mean_val) / std_val
            # Impute NaN if any
            scaled = np.nan_to_num(scaled, nan=0.0)
            transformed_parts.append(scaled.reshape(-1, 1))

        # One-hot encode categorical features
        for col in self.cat_cols:
            cats = self.cat_categories_[col]
            col_vals = X[col].astype(str).values
            for c in cats:
                binary_col = (col_vals == c).astype(float)
                transformed_parts.append(binary_col.reshape(-1, 1))

        return np.hstack(transformed_parts)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> np.ndarray:
        return self.feature_names_out_


def get_preprocessor(X: pd.DataFrame) -> CustomColumnTransformer:
    """Return preprocessor configured for X's numerical and categorical columns."""
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"[INFO] Numerical features ({len(num_cols)}): {num_cols}")
    print(f"[INFO] Categorical features ({len(cat_cols)}): {cat_cols}")

    return CustomColumnTransformer(num_cols=num_cols, cat_cols=cat_cols)


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Perform stratified train-test split using reproducible numpy indexing.
    """
    np.random.seed(random_state)
    y_arr = y.values
    indices = np.arange(len(y_arr))

    # Stratified sampling by class
    idx_0 = indices[y_arr == 0]
    idx_1 = indices[y_arr == 1]

    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)

    n_test_0 = int(len(idx_0) * test_size)
    n_test_1 = int(len(idx_1) * test_size)

    test_idx = np.concatenate([idx_0[:n_test_0], idx_1[:n_test_1]])
    train_idx = np.concatenate([idx_0[n_test_0:], idx_1[n_test_1:]])

    np.random.shuffle(test_idx)
    np.random.shuffle(train_idx)

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    print(f"[INFO] Data split into Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows (Test ratio: {test_size})")
    return X_train, X_test, y_train, y_test
