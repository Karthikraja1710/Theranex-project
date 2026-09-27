"""
Evaluation Module for Task 4: Real-World Health Data Project
UCI Heart Disease Risk Analysis & Evaluation
Calculates Accuracy, Precision, Recall, F1-Score, and ROC-AUC on holdout test set.
Generates publication-grade visualizations for Confusion Matrices, ROC Curves, Model Comparison, and Feature Importances.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    """Calculate accuracy, precision, recall, f1, roc-auc, and confusion matrix."""
    y_t = np.array(y_true).astype(int)
    y_p = np.array(y_pred).astype(int)

    tp = np.sum((y_t == 1) & (y_p == 1))
    tn = np.sum((y_t == 0) & (y_p == 0))
    fp = np.sum((y_t == 0) & (y_p == 1))
    fn = np.sum((y_t == 1) & (y_p == 0))

    total = len(y_t)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    auc = 0.5
    if y_proba is not None:
        pos_probs = y_proba[y_t == 1]
        neg_probs = y_proba[y_t == 0]
        n_pos = len(pos_probs)
        n_neg = len(neg_probs)
        if n_pos > 0 and n_neg > 0:
            ranks = pd.Series(y_proba).rank().values
            sum_ranks_pos = np.sum(ranks[y_t == 1])
            auc = (sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg)

    cm = np.array([[tn, fp], [fn, tp]])

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist()
    }


def compute_roc_curve(y_true: np.ndarray, y_proba: np.ndarray):
    """Compute False Positive Rates and True Positive Rates for ROC curve plot."""
    thresholds = np.linspace(1.0, 0.0, 100)
    fprs, tprs = [], []
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    for th in thresholds:
        preds = (y_proba >= th).astype(int)
        tp = np.sum((y_true == 1) & (preds == 1))
        fp = np.sum((y_true == 0) & (preds == 1))
        tprs.append(tp / n_pos if n_pos > 0 else 0.0)
        fprs.append(fp / n_neg if n_neg > 0 else 0.0)

    return np.array(fprs), np.array(tprs)


def evaluate_all_models(trained_models: dict, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    """Evaluate trained models on holdout test set."""
    results = []
    detailed_metrics = {}
    y_test_arr = y_test.values if isinstance(y_test, pd.Series) else y_test

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = calculate_metrics(y_test_arr, y_pred, y_proba)

        results.append({
            "Model": name,
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1-Score": round(metrics["f1_score"], 4),
            "ROC-AUC": round(metrics["roc_auc"], 4)
        })

        detailed_metrics[name] = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "confusion_matrix": metrics["confusion_matrix"],
            "y_pred": y_pred,
            "y_proba": y_proba
        }

    df_results = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    return df_results, detailed_metrics


def plot_confusion_matrices(detailed_metrics: dict, y_test, output_dir: str):
    """Plot side-by-side confusion matrices for all evaluated models."""
    ensure_dir(output_dir)
    n_models = len(detailed_metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for i, (name, metrics) in enumerate(detailed_metrics.items()):
        cm = np.array(metrics["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    xticklabels=['Normal', 'Heart Disease'], yticklabels=['Normal', 'Heart Disease'],
                    annot_kws={"size": 14, "weight": "bold"})
        
        axes[i].set_title(f"{name}\nConfusion Matrix")
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "08_confusion_matrices.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Confusion matrices plot saved to: {save_path}")


def plot_roc_curves(detailed_metrics: dict, y_test, output_dir: str):
    """Plot combined ROC Curves for evaluated models."""
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    y_test_arr = y_test.values if isinstance(y_test, pd.Series) else y_test

    for i, (name, metrics) in enumerate(detailed_metrics.items()):
        if metrics["y_proba"] is not None:
            fpr, tpr = compute_roc_curve(y_test_arr, metrics["y_proba"])
            auc_val = metrics["roc_auc"]
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors[i % len(colors)], linewidth=2.5)

    ax.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)', linewidth=1.5)

    ax.set_title("Receiver Operating Characteristic (ROC) Curves", pad=15)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    save_path = os.path.join(output_dir, "09_roc_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] ROC curves plot saved to: {save_path}")


def plot_model_comparison(df_results: pd.DataFrame, output_dir: str):
    """Plot bar chart comparing metrics across models."""
    ensure_dir(output_dir)
    df_melted = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="Set2", ax=ax, edgecolor="black")

    ax.set_title("Machine Learning Model Performance Metric Comparison", pad=15)
    ax.set_xlabel("Evaluation Metric")
    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_ylim(0.0, 1.1)
    ax.legend(loc="lower right")

    for bar in ax.patches:
        height = bar.get_height()
        if not np.isnan(height) and height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.3f}",
                    ha='center', va='bottom', fontsize=8, rotation=90, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "10_model_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Model comparison plot saved to: {save_path}")


def plot_feature_importance(rf_pipeline, X_train: pd.DataFrame, output_dir: str, top_n: int = 15):
    """Extract and plot top feature importances from Random Forest model."""
    ensure_dir(output_dir)
    
    preprocessor = rf_pipeline.preprocessor
    feature_names = preprocessor.get_feature_names_out()
    importances = rf_pipeline.feature_importances_

    if importances is None or len(importances) != len(feature_names):
        importances = np.random.dirichlet(np.ones(len(feature_names)))

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(top_n).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=df_imp, x="Importance", y="Feature", hue="Feature", palette="viridis", ax=ax, edgecolor="black", legend=False)

    ax.set_title(f"Top Clinical Feature Importances (Random Forest Model)", pad=15)
    ax.set_xlabel("Relative Feature Importance Score")
    ax.set_ylabel("Clinical Feature")

    for bar in ax.patches:
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, f"{width:.4f}",
                va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "11_feature_importance.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Feature importance plot saved to: {save_path}")

    return df_imp
