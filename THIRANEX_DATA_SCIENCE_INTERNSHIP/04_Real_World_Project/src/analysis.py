"""
EDA & Visual Analysis Module for Task 4: Real-World Health Data Project
UCI Heart Disease Risk Analysis & Visualization
Generates summary statistics, bivariate clinical analysis, and publication-quality figures.
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


def compute_clinical_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for clinical variables."""
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    num_cols = [c for c in num_cols if c in df.columns]
    stats_list = []

    for col in num_cols:
        series = df[col].dropna()
        stats_list.append({
            "Feature": col,
            "Mean": round(series.mean(), 2),
            "StdDev": round(series.std(), 2),
            "Median": round(series.median(), 2),
            "Min": round(series.min(), 2),
            "Q25 (25%)": round(series.quantile(0.25), 2),
            "Q75 (75%)": round(series.quantile(0.75), 2),
            "Max": round(series.max(), 2),
            "IQR": round(series.quantile(0.75) - series.quantile(0.25), 2)
        })

    return pd.DataFrame(stats_list)


def plot_target_distribution(df: pd.DataFrame, output_dir: str):
    """Plot distribution of Heart Disease Target (0 = Normal, 1 = Disease Present)."""
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = df['target'].value_counts().sort_index()
    labels = ['Normal / Healthy (0)', 'Heart Disease Present (1)']
    colors = ['#2ca02c', '#d62728']

    bars = ax.bar(labels, counts, color=colors, edgecolor='black', alpha=0.85, width=0.5)
    total = len(df)

    for bar, count in zip(bars, counts):
        pct = (count / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"{count} Patients\n({pct:.1f}%)", ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_title("Target Class Distribution: Heart Disease Diagnosis", pad=15)
    ax.set_ylabel("Number of Patients")
    ax.set_ylim(0, max(counts) * 1.25)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "00_target_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Target distribution plot saved to: {save_path}")


def plot_age_distribution_by_target(df: pd.DataFrame, output_dir: str):
    """Plot age distribution KDE grouped by Heart Disease status."""
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(data=df, x='age', hue='target', kde=True, ax=ax, palette={0: '#2ca02c', 1: '#d62728'},
                 bins=15, edgecolor='black', alpha=0.6)

    ax.set_title("Patient Age Distribution by Heart Disease Status", pad=15)
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Patient Count")

    # Legend customization
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(title="Diagnosis", labels=['Heart Disease (1)', 'Normal (0)'], loc='upper left')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "01_age_distribution_by_target.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Age distribution plot saved to: {save_path}")


def plot_chest_pain_vs_target(df: pd.DataFrame, output_dir: str):
    """Plot Chest Pain Type (cp) vs Heart Disease prevalence."""
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 6))

    cp_labels = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-Anginal Pain', 3: 'Asymptomatic'}
    df_plot = df.copy()
    df_plot['ChestPain'] = df_plot['cp'].map(cp_labels)

    sns.countplot(data=df_plot, x='ChestPain', hue='target', palette={0: '#2ca02c', 1: '#d62728'}, ax=ax, edgecolor='black')

    ax.set_title("Chest Pain Type vs. Heart Disease Prevalence", pad=15)
    ax.set_xlabel("Chest Pain Classification")
    ax.set_ylabel("Patient Count")
    ax.legend(title="Diagnosis", labels=['Normal (0)', 'Heart Disease (1)'], loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "02_chest_pain_vs_target.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Chest pain vs target plot saved to: {save_path}")


def plot_cholesterol_bp_scatter(df: pd.DataFrame, output_dir: str):
    """Plot Serum Cholesterol vs Resting Blood Pressure scatter plot by Target."""
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(data=df, x='trestbps', y='chol', hue='target', style='target',
                    palette={0: '#2ca02c', 1: '#d62728'}, s=70, alpha=0.8, edgecolor='black', ax=ax)

    ax.set_title("Serum Cholesterol vs. Resting Blood Pressure by Heart Disease Status", pad=15)
    ax.set_xlabel("Resting Blood Pressure (mm Hg)")
    ax.set_ylabel("Serum Cholesterol (mg/dl)")
    ax.legend(title="Diagnosis", labels=['Normal (0)', 'Heart Disease (1)'], loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "03_cholesterol_bp_scatter.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Cholesterol vs BP scatter plot saved to: {save_path}")


def plot_max_heart_rate_boxplot(df: pd.DataFrame, output_dir: str):
    """Plot Maximum Heart Rate (thalach) boxplot across Target."""
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.boxplot(data=df, x='target', y='thalach', hue='target', palette={0: '#2ca02c', 1: '#d62728'}, ax=ax, fliersize=4, legend=False)
    sns.stripplot(data=df, x='target', y='thalach', color='black', alpha=0.3, jitter=0.2, ax=ax)

    ax.set_xticklabels(['Normal (0)', 'Heart Disease (1)'])
    ax.set_title("Maximum Heart Rate Achieved (thalach) by Diagnosis", pad=15)
    ax.set_xlabel("Diagnosis Group")
    ax.set_ylabel("Max Heart Rate (bpm)")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "04_max_heart_rate_boxplot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Max heart rate boxplot saved to: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """Plot annotated Pearson correlation heatmap of clinical features."""
    ensure_dir(output_dir)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=1, ax=ax, cbar_kws={"shrink": .8})
    ax.set_title("Correlation Heatmap of Clinical Attributes", pad=15)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "05_correlation_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Correlation heatmap saved to: {save_path}")


def plot_exercise_angina_st_depression(df: pd.DataFrame, output_dir: str):
    """Plot ST depression (oldpeak) and Exercise Induced Angina (exang) vs Target."""
    ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ST Depression Boxplot
    sns.boxplot(data=df, x='target', y='oldpeak', hue='target', palette={0: '#2ca02c', 1: '#d62728'}, ax=axes[0], legend=False)
    axes[0].set_xticklabels(['Normal (0)', 'Heart Disease (1)'])
    axes[0].set_title("ST Depression Induced by Exercise (oldpeak)")
    axes[0].set_xlabel("Diagnosis Group")
    axes[0].set_ylabel("ST Depression Value")

    # Exercise Angina Countplot
    sns.countplot(data=df, x='exang', hue='target', palette={0: '#2ca02c', 1: '#d62728'}, ax=axes[1], edgecolor='black')
    axes[1].set_xticklabels(['No Exercise Angina (0)', 'Exercise Angina (1)'])
    axes[1].set_title("Exercise Induced Angina Prevalence")
    axes[1].set_xlabel("Exercise Angina")
    axes[1].set_ylabel("Patient Count")
    axes[1].legend(title="Diagnosis", labels=['Normal (0)', 'Heart Disease (1)'], loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "06_exercise_angina_st_depression.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Exercise angina & ST depression plot saved to: {save_path}")


def plot_pairwise_feature_importance(df: pd.DataFrame, output_dir: str):
    """Plot correlation ranking of clinical variables with target diagnosis."""
    ensure_dir(output_dir)
    corr_target = df.corr()['target'].drop('target').sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in corr_target.values]
    bars = ax.barh(corr_target.index, corr_target.values, color=colors, edgecolor='black', alpha=0.85)

    ax.axvline(0, color='black', linewidth=1)
    ax.set_title("Pearson Correlation of Clinical Features with Heart Disease Diagnosis", pad=15)
    ax.set_xlabel("Correlation Coefficient with Target (Heart Disease)")
    ax.set_ylabel("Clinical Feature")

    for bar in bars:
        width = bar.get_width()
        x_pos = width + 0.01 if width >= 0 else width - 0.05
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "07_pairwise_feature_importance.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Pairwise feature correlation plot saved to: {save_path}")
