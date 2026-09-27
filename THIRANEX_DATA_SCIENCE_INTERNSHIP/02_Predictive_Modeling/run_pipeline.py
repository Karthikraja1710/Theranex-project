"""
Master Pipeline Script for Task 2: Predictive Modeling
Orchestrates raw data loading, inspection, leakage-free preprocessing pipelines,
stratified train-test split, model training (Baseline, Decision Tree, Random Forest),
evaluation, visualization generation, and model artifact serialization.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from src.preprocessing import (
    load_raw_data,
    inspect_data,
    clean_raw_dataframe,
    get_preprocessor,
    split_data
)
from src.train import build_model_pipelines, train_all_models
from src.evaluate import (
    plot_class_distribution,
    evaluate_all_models,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_model_comparison,
    plot_feature_importance
)


def run_pipeline():
    project_root = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(project_root, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    processed_path = os.path.join(project_root, "data", "processed", "cleaned_telco_churn.csv")
    models_dir = os.path.join(project_root, "models")
    viz_dir = os.path.join(project_root, "visualizations")
    reports_dir = os.path.join(project_root, "reports")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("==================================================")
    print("STEP 1: INGESTING RAW DATASET")
    print("==================================================")
    df_raw = load_raw_data(raw_path)

    print("\n==================================================")
    print("STEP 2: DATA UNDERSTANDING & QUALITY AUDIT")
    print("==================================================")
    inspection = inspect_data(df_raw)
    print(f"Dataset Shape: {inspection['num_rows']} rows x {inspection['num_cols']} columns")
    print(f"Blank TotalCharges count: {inspection['blank_total_charges_count']}")
    print(f"Exact Duplicate rows: {inspection['duplicate_rows']}")
    print(f"Target 'Churn' distribution: {inspection['target_distribution']}")

    plot_class_distribution(df_raw['Churn'], viz_dir)

    print("\n==================================================")
    print("STEP 3: PREPROCESSING & FEATURE ISOLATION")
    print("==================================================")
    X, y, tracking = clean_raw_dataframe(df_raw)

    # Save processed dataframe copy for inspection
    df_processed = X.copy()
    df_processed['Churn'] = y
    df_processed.to_csv(processed_path, index=False)
    print(f"[INFO] Processed dataset saved to: {processed_path}")

    preprocessor = get_preprocessor(X)

    print("\n==================================================")
    print("STEP 4: STRATIFIED TRAIN / TEST SPLIT")
    print("==================================================")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    print("\n==================================================")
    print("STEP 5: MODEL TRAINING (STRICT LEAKAGE PREVENTION)")
    print("==================================================")
    pipelines = build_model_pipelines(preprocessor)
    trained_models = train_all_models(pipelines, X_train, y_train)

    print("\n==================================================")
    print("STEP 6: MODEL EVALUATION ON HOLDOUT TEST SET")
    print("==================================================")
    df_results, detailed_metrics = evaluate_all_models(trained_models, X_test, y_test)
    print("\nModel Comparison Table:")
    print(df_results.to_string(index=False))

    print("\n==================================================")
    print("STEP 7: GENERATING EVALUATION VISUALIZATIONS")
    print("==================================================")
    plot_confusion_matrices(detailed_metrics, y_test, viz_dir)
    plot_roc_curves(detailed_metrics, y_test, viz_dir)
    plot_model_comparison(df_results, viz_dir)

    # Feature Importance for Random Forest
    rf_model = trained_models["Random Forest"]
    df_imp = plot_feature_importance(rf_model, X_train, viz_dir, top_n=15)
    print("\nTop 5 Important Features (Random Forest):")
    print(df_imp.head(5).to_string(index=False))

    print("\n==================================================")
    print("STEP 8: SELECTING & SAVING BEST MODEL ARTIFACT")
    print("==================================================")
    best_model_name = df_results.sort_values(by="ROC-AUC", ascending=False).iloc[0]["Model"]
    best_pipeline = trained_models[best_model_name]
    best_auc = df_results.sort_values(by="ROC-AUC", ascending=False).iloc[0]["ROC-AUC"]

    best_model_path = os.path.join(models_dir, "best_churn_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"[SUCCESS] Selected Best Model: {best_model_name} (ROC-AUC: {best_auc:.4f})")
    print(f"[SUCCESS] Model artifact saved to: {best_model_path}")

    summary_path = os.path.join(reports_dir, "metrics_summary.json")
    summary = {
        "dataset_name": "IBM Telco Customer Churn Dataset",
        "total_records": len(df_raw),
        "total_features": X.shape[1],
        "train_records": len(X_train),
        "test_records": len(X_test),
        "churn_rate_pct": tracking["target_churn_rate_pct"],
        "results_table": df_results.to_dict(orient="records"),
        "best_model": best_model_name,
        "best_roc_auc": float(best_auc),
        "top_features": df_imp.head(10).to_dict(orient="records")
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SUCCESS] Metrics summary saved to: {summary_path}")
    print("\nTask 2 Master Pipeline executed successfully!")


if __name__ == "__main__":
    run_pipeline()
