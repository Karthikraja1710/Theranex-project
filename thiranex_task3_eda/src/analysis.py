"""
Analysis & Visualization Module for Task 3: Exploratory Data Analysis (EDA)
World Happiness / Socioeconomic Analysis (2021 Dataset)
Provides data loading, structural auditing, statistical analysis, correlation metrics,
and publication-quality visualization routines.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Global visual styling
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


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw World Happiness Report 2021 CSV dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw file not found at: {file_path}")
    print(f"[INFO] Loading raw World Happiness dataset from: {file_path}")
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
        "duplicate_rows": int(df.duplicated().sum())
    }
    return inspection


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean dataset:
    1. Filter relevant key variables.
    2. Rename columns to intuitive clean names.
    3. Check missing values and data types.
    """
    column_mapping = {
        'Country name': 'Country',
        'Regional indicator': 'Region',
        'Ladder score': 'HappinessScore',
        'Logged GDP per capita': 'LogGDP',
        'Social support': 'SocialSupport',
        'Healthy life expectancy': 'HealthyLifeExpectancy',
        'Freedom to make life choices': 'Freedom',
        'Generosity': 'Generosity',
        'Perceptions of corruption': 'CorruptionPerception'
    }

    df_clean = df[list(column_mapping.keys())].copy()
    df_clean = df_clean.rename(columns=column_mapping)

    tracking = {
        "initial_count": len(df),
        "final_count": len(df_clean),
        "missing_values": df_clean.isnull().sum().to_dict(),
        "num_regions": df_clean['Region'].nunique()
    }

    return df_clean, tracking


def compute_univariate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for numerical socioeconomic variables."""
    num_cols = ['HappinessScore', 'LogGDP', 'SocialSupport', 'HealthyLifeExpectancy', 'Freedom', 'Generosity', 'CorruptionPerception']
    stats_list = []

    for col in num_cols:
        series = df[col].dropna()
        mean_val = series.mean()
        median_val = series.median()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        skew_val = series.skew()

        stats_list.append({
            "Variable": col,
            "Mean": round(mean_val, 4),
            "Median": round(median_val, 4),
            "StdDev": round(std_val, 4),
            "Min": round(min_val, 4),
            "Q25 (25%)": round(q25, 4),
            "Q75 (75%)": round(q75, 4),
            "Max": round(max_val, 4),
            "IQR": round(iqr, 4),
            "Skewness": round(skew_val, 4)
        })

    return pd.DataFrame(stats_list)


def compute_correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson (linear) and Spearman (rank) correlation matrices with p-values."""
    num_cols = ['HappinessScore', 'LogGDP', 'SocialSupport', 'HealthyLifeExpectancy', 'Freedom', 'Generosity', 'CorruptionPerception']
    df_num = df[num_cols].dropna()

    pearson_corr = df_num.corr(method='pearson')
    spearman_corr = df_num.corr(method='spearman')

    return pearson_corr, spearman_corr


# ==================================================
# VISUALIZATION ROUTINES
# ==================================================

def plot_happiness_distribution(df: pd.DataFrame, output_dir: str):
    """Plot distribution (Histogram + KDE) of Happiness Score."""
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(df['HappinessScore'], kde=True, ax=ax, color='#2b5c8f', bins=20, edgecolor='black', alpha=0.7)
    mean_val = df['HappinessScore'].mean()
    median_val = df['HappinessScore'].median()

    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean Score: {mean_val:.3f}')
    ax.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median Score: {median_val:.3f}')

    ax.set_title("Distribution of World Happiness Score (Ladder Score - 2021)", pad=15)
    ax.set_xlabel("Happiness Score (0 to 10 Scale)")
    ax.set_ylabel("Number of Countries")
    ax.legend(loc='upper left')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "01_happiness_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Happiness distribution plot saved to: {save_path}")


def plot_regional_boxplot(df: pd.DataFrame, output_dir: str):
    """Plot boxplot comparing Happiness Score distribution across World Regions."""
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(12, 7))

    region_order = df.groupby('Region')['HappinessScore'].median().sort_values(ascending=False).index

    sns.boxplot(data=df, y='Region', x='HappinessScore', hue='Region', order=region_order, palette='Spectral', ax=ax, fliersize=4, legend=False)
    sns.stripplot(data=df, y='Region', x='HappinessScore', order=region_order, color='black', alpha=0.4, jitter=0.2, ax=ax)

    ax.set_title("Happiness Score Distribution Across World Regions (2021)", pad=15)
    ax.set_xlabel("Happiness Score (0 to 10 Scale)")
    ax.set_ylabel("World Region")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "02_regional_happiness_boxplot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Regional happiness boxplot saved to: {save_path}")


def plot_gdp_vs_happiness(df: pd.DataFrame, output_dir: str):
    """Plot scatter plot of Happiness Score vs Logged GDP per Capita with regression line."""
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.regplot(data=df, x='LogGDP', y='HappinessScore', ax=ax, color='#1f77b4',
                scatter_kws={'alpha': 0.7, 's': 50, 'edgecolor': 'black'},
                line_kws={'color': 'darkred', 'linewidth': 2.5, 'label': 'OLS Trendline'})

    r_val, p_val = stats.pearsonr(df['LogGDP'].dropna(), df['HappinessScore'].dropna())
    ax.text(0.05, 0.90, f"Pearson r = {r_val:.3f} (p < 0.001)", transform=ax.transAxes,
            fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="yellow", ec="black", lw=1))

    ax.set_title("Economic Output vs Happiness: Logged GDP per Capita vs. Ladder Score", pad=15)
    ax.set_xlabel("Logged GDP per Capita")
    ax.set_ylabel("Happiness Score")
    ax.legend(loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "03_gdp_vs_happiness_scatter.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] GDP vs Happiness scatter plot saved to: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """Plot annotated correlation heatmap of key socioeconomic factors."""
    ensure_dir(output_dir)
    num_cols = ['HappinessScore', 'LogGDP', 'SocialSupport', 'HealthyLifeExpectancy', 'Freedom', 'Generosity', 'CorruptionPerception']
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=1.5, ax=ax, cbar_kws={"shrink": .8})
    ax.set_title("Pearson Correlation Matrix of Socioeconomic Factors", pad=15)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "04_correlation_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Correlation heatmap saved to: {save_path}")


def plot_top10_bottom10_countries(df: pd.DataFrame, output_dir: str):
    """Plot horizontal bar chart of Top 10 happiest vs Bottom 10 unhappiest countries."""
    ensure_dir(output_dir)
    top10 = df.nlargest(10, 'HappinessScore')[['Country', 'HappinessScore']].sort_values(by='HappinessScore', ascending=True)
    bottom10 = df.nsmallest(10, 'HappinessScore')[['Country', 'HappinessScore']].sort_values(by='HappinessScore', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Top 10
    sns.barplot(data=top10, x='HappinessScore', y='Country', hue='Country', palette='Greens_r', ax=axes[0], edgecolor='black', legend=False)
    axes[0].set_title("Top 10 Happiest Countries (2021)")
    axes[0].set_xlabel("Happiness Score (0 to 10 Scale)")
    axes[0].set_ylabel("Country")
    axes[0].set_xlim(0, 10)
    for bar in axes[0].patches:
        width = bar.get_width()
        axes[0].text(width + 0.1, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center', fontweight='bold', fontsize=9)

    # Bottom 10
    sns.barplot(data=bottom10, x='HappinessScore', y='Country', hue='Country', palette='Reds_r', ax=axes[1], edgecolor='black', legend=False)
    axes[1].set_title("Bottom 10 Lowest Happiness Countries (2021)")
    axes[1].set_xlabel("Happiness Score (0 to 10 Scale)")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, 10)
    for bar in axes[1].patches:
        width = bar.get_width()
        axes[1].text(width + 0.1, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "05_top_bottom_countries_bar.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Top/Bottom 10 countries bar chart saved to: {save_path}")


def plot_regional_mean_happiness(df: pd.DataFrame, output_dir: str):
    """Plot bar chart comparing Mean Happiness Score by Region."""
    ensure_dir(output_dir)
    regional = df.groupby('Region')['HappinessScore'].mean().sort_values(ascending=False).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=regional, x='HappinessScore', y='Region', hue='Region', palette='viridis', ax=ax, edgecolor='black', legend=False)

    ax.set_title("Mean Happiness Score by World Region (2021)", pad=15)
    ax.set_xlabel("Mean Happiness Score")
    ax.set_ylabel("World Region")
    ax.set_xlim(0, 8.5)

    for bar in ax.patches:
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "06_regional_mean_happiness.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Regional mean happiness plot saved to: {save_path}")


def plot_key_drivers_scatter_grid(df: pd.DataFrame, output_dir: str):
    """Plot 2x2 grid of scatter plots for key drivers vs Happiness Score."""
    ensure_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    factors = [
        ('SocialSupport', 'Social Support', '#2ca02c', axes[0, 0]),
        ('HealthyLifeExpectancy', 'Healthy Life Expectancy', '#ff7f0e', axes[0, 1]),
        ('Freedom', 'Freedom to Make Life Choices', '#9467bd', axes[1, 0]),
        ('CorruptionPerception', 'Perception of Corruption', '#d62728', axes[1, 1])
    ]

    for col, name, color, ax in factors:
        sns.regplot(data=df, x=col, y='HappinessScore', ax=ax, color=color,
                    scatter_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'black'},
                    line_kws={'color': 'black', 'linewidth': 2})
        r_val, _ = stats.pearsonr(df[col].dropna(), df['HappinessScore'].dropna())
        ax.set_title(f"{name} vs Happiness\n(Pearson r = {r_val:.3f})")
        ax.set_xlabel(name)
        ax.set_ylabel("Happiness Score")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "07_pairwise_factors_scatter.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Pairwise factors scatter grid saved to: {save_path}")


def plot_correlation_comparison(df: pd.DataFrame, output_dir: str):
    """Plot bar chart contrasting Pearson vs Spearman correlation for each factor with HappinessScore."""
    ensure_dir(output_dir)
    num_cols = ['LogGDP', 'SocialSupport', 'HealthyLifeExpectancy', 'Freedom', 'Generosity', 'CorruptionPerception']

    pearson_corrs = [df['HappinessScore'].corr(df[col], method='pearson') for col in num_cols]
    spearman_corrs = [df['HappinessScore'].corr(df[col], method='spearman') for col in num_cols]

    df_comp = pd.DataFrame({
        "Factor": num_cols,
        "Pearson (r)": pearson_corrs,
        "Spearman (rho)": spearman_corrs
    }).melt(id_vars="Factor", var_name="Correlation Method", value_name="Coefficient")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_comp, x="Factor", y="Coefficient", hue="Correlation Method", palette="Set2", ax=ax, edgecolor="black")

    ax.set_title("Comparison of Linear (Pearson) vs. Rank (Spearman) Correlations with Happiness Score", pad=15)
    ax.set_xlabel("Socioeconomic Factor")
    ax.set_ylabel("Correlation Coefficient (-1.0 to +1.0)")
    ax.axhline(0, color='black', linewidth=1)

    for bar in ax.patches:
        height = bar.get_height()
        if not np.isnan(height):
            y_pos = height + 0.02 if height >= 0 else height - 0.05
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{height:.3f}", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "08_correlation_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Correlation comparison chart saved to: {save_path}")
