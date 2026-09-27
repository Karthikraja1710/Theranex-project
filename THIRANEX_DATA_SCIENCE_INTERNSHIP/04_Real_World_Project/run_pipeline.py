"""
Master Pipeline Script for Task 4: Real-World Health Data Project
UCI Heart Disease Risk Analysis and Prediction
Orchestrates raw data loading, inspection, EDA, leakage-free preprocessing pipelines,
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
from src.analysis import (
    compute_clinical_summary_stats,
    plot_target_distribution,
    plot_age_distribution_by_target,
    plot_chest_pain_vs_target,
    plot_cholesterol_bp_scatter,
    plot_max_heart_rate_boxplot,
    plot_correlation_heatmap,
    plot_exercise_angina_st_depression,
    plot_pairwise_feature_importance
)
from src.train import build_model_pipelines, train_all_models
from src.evaluate import (
    evaluate_all_models,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_model_comparison,
    plot_feature_importance
)


def run_pipeline():
    project_root = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(project_root, "data", "raw", "heart.csv")
    processed_path = os.path.join(project_root, "data", "processed", "cleaned_heart_disease.csv")
    models_dir = os.path.join(project_root, "models")
    viz_dir = os.path.join(project_root, "visualizations")
    reports_dir = os.path.join(project_root, "reports")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("==================================================")
    print("STEP 1: INGESTING RAW UCI HEART DISEASE DATASET")
    print("==================================================")
    df_raw = load_raw_data(raw_path)

    print("\n==================================================")
    print("STEP 2: DATA UNDERSTANDING & QUALITY AUDIT")
    print("==================================================")
    inspection = inspect_data(df_raw)
    print(f"Dataset Shape: {inspection['num_rows']} rows x {inspection['num_cols']} columns")
    print(f"Duplicate rows: {inspection['duplicate_rows']}")
    print(f"Target distribution: {inspection['target_distribution']}")

    plot_target_distribution(df_raw, viz_dir)

    print("\n==================================================")
    print("STEP 3: PREPROCESSING & FEATURE ISOLATION")
    print("==================================================")
    X, y, tracking = clean_raw_dataframe(df_raw)

    df_processed = X.copy()
    df_processed['target'] = y
    df_processed.to_csv(processed_path, index=False)
    print(f"[INFO] Processed dataset saved to: {processed_path}")

    preprocessor = get_preprocessor(X)

    print("\n==================================================")
    print("STEP 4: CLINICAL EXPLORATORY DATA ANALYSIS (EDA)")
    print("==================================================")
    df_summary = compute_clinical_summary_stats(df_raw)
    print(df_summary.to_string(index=False))

    plot_age_distribution_by_target(df_raw, viz_dir)
    plot_chest_pain_vs_target(df_raw, viz_dir)
    plot_cholesterol_bp_scatter(df_raw, viz_dir)
    plot_max_heart_rate_boxplot(df_raw, viz_dir)
    plot_correlation_heatmap(df_raw, viz_dir)
    plot_exercise_angina_st_depression(df_raw, viz_dir)
    plot_pairwise_feature_importance(df_raw, viz_dir)

    print("\n==================================================")
    print("STEP 5: STRATIFIED TRAIN / TEST SPLIT")
    print("==================================================")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    print("\n==================================================")
    print("STEP 6: MACHINE LEARNING MODEL TRAINING")
    print("==================================================")
    pipelines = build_model_pipelines(preprocessor)
    trained_models = train_all_models(pipelines, X_train, y_train)

    print("\n==================================================")
    print("STEP 7: MODEL EVALUATION ON HOLDOUT TEST SET")
    print("==================================================")
    df_results, detailed_metrics = evaluate_all_models(trained_models, X_test, y_test)
    print("\nModel Comparison Table:")
    print(df_results.to_string(index=False))

    print("\n==================================================")
    print("STEP 8: GENERATING EVALUATION VISUALIZATIONS")
    print("==================================================")
    plot_confusion_matrices(detailed_metrics, y_test, viz_dir)
    plot_roc_curves(detailed_metrics, y_test, viz_dir)
    plot_model_comparison(df_results, viz_dir)

    rf_model = trained_models["Random Forest"]
    df_imp = plot_feature_importance(rf_model, X_train, viz_dir, top_n=15)
    print("\nTop 5 Important Clinical Features (Random Forest):")
    print(df_imp.head(5).to_string(index=False))

    print("\n==================================================")
    print("STEP 9: SELECTING & SAVING BEST MODEL ARTIFACT")
    print("==================================================")
    best_model_name = df_results.sort_values(by="ROC-AUC", ascending=False).iloc[0]["Model"]
    best_pipeline = trained_models[best_model_name]
    best_auc = df_results.sort_values(by="ROC-AUC", ascending=False).iloc[0]["ROC-AUC"]

    best_model_path = os.path.join(models_dir, "best_health_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"[SUCCESS] Selected Best Model: {best_model_name} (ROC-AUC: {best_auc:.4f})")
    print(f"[SUCCESS] Model artifact saved to: {best_model_path}")

    summary_path = os.path.join(reports_dir, "summary_metrics.json")
    summary = {
        "dataset_name": "UCI Heart Disease Dataset",
        "total_records": len(df_raw),
        "total_features": X.shape[1],
        "train_records": len(X_train),
        "test_records": len(X_test),
        "disease_prevalence_pct": tracking["target_disease_rate_pct"],
        "results_table": df_results.to_dict(orient="records"),
        "best_model": best_model_name,
        "best_roc_auc": float(best_auc),
        "top_clinical_features": df_imp.head(10).to_dict(orient="records")
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SUCCESS] Summary metrics saved to: {summary_path}")
    print("\nTask 4 Master Pipeline executed successfully!")


if __name__ == "__main__":
    run_pipeline()
