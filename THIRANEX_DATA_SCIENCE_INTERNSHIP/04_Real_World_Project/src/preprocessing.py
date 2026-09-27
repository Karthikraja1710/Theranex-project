"""
Preprocessing Module for Task 4: Real-World Health Data Project
UCI Heart Disease Risk Analysis & Prediction
Handles raw data ingestion, structural quality audit, leakage-free feature transformation,
and stratified train-test splitting.
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw UCI Heart Disease dataset from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw file not found at: {file_path}")
    print(f"[INFO] Loading raw Heart Disease dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[INFO] Raw dataset loaded successfully. Shape: {df.shape}")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Perform structural quality audit."""
    inspection = {
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df['target'].value_counts().to_dict() if 'target' in df.columns else {}
    }
    return inspection


def clean_raw_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Clean raw DataFrame:
    1. Remove duplicate rows.
    2. Separate feature matrix X and binary target y (1 = Disease Present, 0 = Normal).
    """
    df_clean = df.copy()
    tracking = {}

    dup_count = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    tracking["duplicates_removed"] = int(dup_count)

    if "target" not in df_clean.columns:
        raise KeyError("Target column 'target' not found in dataset!")

    y = df_clean["target"].astype(int)
    tracking["total_records"] = len(df_clean)
    tracking["target_disease_rate_pct"] = round((y.mean()) * 100, 2)
    tracking["total_disease_cases"] = int(y.sum())
    tracking["total_normal_cases"] = int((y == 0).sum())

    feature_cols = [c for c in df_clean.columns if c != "target"]
    X = df_clean[feature_cols].copy()

    return X, y, tracking


class CustomColumnTransformer:
    """
    Applies numerical Z-score scaling and categorical One-Hot encoding
    to clinical features without data leakage.
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
        
        for col in self.num_cols:
            vals = X[col].astype(float).values
            mean_val = np.nanmean(vals)
            std_val = np.nanstd(vals)
            if std_val == 0 or np.isnan(std_val):
                std_val = 1.0
            self.num_means_[col] = mean_val
            self.num_stds_[col] = std_val

        feature_names = list(self.num_cols)
        for col in self.cat_cols:
            cats = sorted(X[col].astype(str).unique().tolist())
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

        for col in self.num_cols:
            vals = X[col].astype(float).values
            mean_val = self.num_means_[col]
            std_val = self.num_stds_[col]
            scaled = (vals - mean_val) / std_val
            scaled = np.nan_to_num(scaled, nan=0.0)
            transformed_parts.append(scaled.reshape(-1, 1))

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
    """Construct ColumnTransformer for numerical scaling and categorical encoding."""
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # Filter to existing columns
    num_cols = [c for c in num_cols if c in X.columns]
    cat_cols = [c for c in cat_cols if c in X.columns]

    print(f"[INFO] Numerical Clinical Features ({len(num_cols)}): {num_cols}")
    print(f"[INFO] Categorical Clinical Features ({len(cat_cols)}): {cat_cols}")

    return CustomColumnTransformer(num_cols=num_cols, cat_cols=cat_cols)


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Perform stratified train-test split."""
    np.random.seed(random_state)
    y_arr = y.values
    indices = np.arange(len(y_arr))

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

    print(f"[INFO] Stratified Split -> Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows (Test ratio: {test_size})")
    return X_train, X_test, y_train, y_test
