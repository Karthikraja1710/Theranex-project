"""
Visualization Module for Thiranex Internship Task 1
Retail / E-commerce Sales Analysis
Creates professional, publication-quality visualizations using Matplotlib & Seaborn.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set global aesthetic configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'


def ensure_output_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)


def plot_missing_values(raw_df: pd.DataFrame, output_dir: str):
    """Plot missing values in raw dataset before cleaning."""
    ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 6))

    missing = raw_df.isnull().sum()
    missing_pct = (missing / len(raw_df)) * 100
    missing_df = pd.DataFrame({'Missing_Count': missing, 'Percentage': missing_pct})
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(by='Missing_Count', ascending=True)

    if missing_df.empty:
        ax.text(0.5, 0.5, "No Missing Values Found", fontsize=16, ha='center', va='center')
    else:
        colors = sns.color_palette("viridis", len(missing_df))
        bars = ax.barh(missing_df.index, missing_df['Missing_Count'], color=colors, edgecolor='black', alpha=0.85)
        
        for bar, pct in zip(bars, missing_df['Percentage']):
            width = bar.get_width()
            ax.text(width + (max(missing_df['Missing_Count']) * 0.01), bar.get_y() + bar.get_height()/2,
                    f"{int(width):,} ({pct:.1f}%)", va='center', fontweight='bold', color='#333333')

        ax.set_title("Missing Value Analysis (Before Cleaning)", pad=15)
        ax.set_xlabel("Number of Missing Records")
        ax.set_ylabel("Dataset Column")
        ax.set_xlim(0, max(missing_df['Missing_Count']) * 1.25)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "01_missing_values_analysis.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Missing values plot saved to: {save_path}")


def plot_numerical_distributions(df: pd.DataFrame, output_dir: str):
    """Plot distributions of key numerical variables (Quantity, UnitPrice, TotalAmount)."""
    ensure_output_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ('Quantity', 'Quantity per Item', '#2b5c8f'),
        ('UnitPrice', 'Unit Price (£)', '#d95f02'),
        ('TotalAmount', 'Total Transaction Amount (£)', '#7570b3')
    ]

    for i, (col, title, color) in enumerate(metrics):
        # Clip top 1% for visualization clarity to avoid distortion from extreme values
        q99 = df[col].quantile(0.99)
        data_to_plot = df[df[col] <= q99][col]

        sns.histplot(data_to_plot, kde=True, ax=axes[i], color=color, bins=30, edgecolor='black', alpha=0.7)
        mean_val = data_to_plot.mean()
        median_val = data_to_plot.median()

        axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
        axes[i].axvline(median_val, color='green', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.2f}')

        axes[i].set_title(f"Distribution of {title}\n(Clipped at 99th Percentile)")
        axes[i].set_xlabel(title)
        axes[i].set_ylabel("Frequency")
        axes[i].legend(loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "02_numerical_distributions.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Numerical distributions plot saved to: {save_path}")


def plot_outlier_boxplots(df_raw: pd.DataFrame, df_clean: pd.DataFrame, output_dir: str):
    """Plot boxplots demonstrating outlier detection and filtering impact."""
    ensure_output_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Raw vs Clean Quantity
    sns.boxplot(y=df_raw[df_raw['Quantity'] > 0]['Quantity'], ax=axes[0, 0], color='#ff7f0e', fliersize=3)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title("Raw Data: Quantity Distribution (Log Scale)")
    axes[0, 0].set_ylabel("Quantity (Log Scale)")

    sns.boxplot(y=df_clean['Quantity'], ax=axes[0, 1], color='#1f77b4', fliersize=3)
    axes[0, 1].set_title("Cleaned Data: Quantity Distribution (Retained Valid Sales)")
    axes[0, 1].set_ylabel("Quantity")

    # Raw vs Clean UnitPrice
    sns.boxplot(y=df_raw[df_raw['UnitPrice'] > 0]['UnitPrice'], ax=axes[1, 0], color='#d62728', fliersize=3)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title("Raw Data: Unit Price Distribution (Log Scale)")
    axes[1, 0].set_ylabel("Unit Price (£ Log Scale)")

    sns.boxplot(y=df_clean['UnitPrice'], ax=axes[1, 1], color='#2ca02c', fliersize=3)
    axes[1, 1].set_title("Cleaned Data: Unit Price Distribution (Retained Valid Sales)")
    axes[1, 1].set_ylabel("Unit Price (£)")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "03_outlier_boxplots.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Outlier boxplots saved to: {save_path}")


def plot_monthly_sales_trend(df: pd.DataFrame, output_dir: str):
    """Plot monthly total revenue and invoice count over time."""
    ensure_output_dir(output_dir)
    
    # Group by YearMonth
    monthly = df.groupby('YearMonth').agg(
        TotalRevenue=('TotalAmount', 'sum'),
        TotalOrders=('InvoiceNo', 'nunique')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(14, 6))

    color_rev = '#1f77b4'
    ax1.set_xlabel('Year-Month', fontweight='bold')
    ax1.set_ylabel('Total Revenue (£)', color=color_rev, fontweight='bold')
    line1 = ax1.plot(monthly['YearMonth'], monthly['TotalRevenue'], marker='o', color=color_rev, linewidth=2.5, label='Monthly Revenue (£)')
    ax1.tick_params(axis='y', labelcolor=color_rev)
    plt.xticks(rotation=45)

    # Format y-axis for revenue
    ax1.yaxis.set_major_formatter('£{x:,.0f}')

    # Secondary axis for total orders
    ax2 = ax1.twinx()
    color_ord = '#ff7f0e'
    ax2.set_ylabel('Total Orders', color=color_ord, fontweight='bold')
    line2 = ax2.plot(monthly['YearMonth'], monthly['TotalOrders'], marker='s', color=color_ord, linewidth=2.5, linestyle='--', label='Total Orders')
    ax2.tick_params(axis='y', labelcolor=color_ord)

    # Annotate peak revenue month
    peak_idx = monthly['TotalRevenue'].idxmax()
    peak_month = monthly.loc[peak_idx, 'YearMonth']
    peak_rev = monthly.loc[peak_idx, 'TotalRevenue']
    ax1.annotate(f'Peak: £{peak_rev:,.0f}',
                 xy=(peak_idx, peak_rev),
                 xytext=(peak_idx - 1, peak_rev * 1.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=1))

    plt.title("Monthly E-Commerce Sales Revenue & Order Volume Trend", pad=15)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "04_monthly_sales_trend.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Monthly sales trend plot saved to: {save_path}")


def plot_top_products(df: pd.DataFrame, output_dir: str):
    """Plot Top 10 products by revenue and by volume sold."""
    ensure_output_dir(output_dir)
    
    top_revenue = df.groupby('Description')['TotalAmount'].sum().nlargest(10).reset_index()
    top_quantity = df.groupby('Description')['Quantity'].sum().nlargest(10).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Top by Revenue
    sns.barplot(data=top_revenue, x='TotalAmount', y='Description', hue='Description', palette='Blues_r', ax=axes[0], edgecolor='black', legend=False)
    axes[0].set_title("Top 10 Products by Total Revenue (£)")
    axes[0].set_xlabel("Total Revenue (£)")
    axes[0].set_ylabel("Product Description")
    for bar in axes[0].patches:
        width = bar.get_width()
        axes[0].text(width * 1.01, bar.get_y() + bar.get_height()/2, f"£{width:,.0f}", va='center', fontsize=9, fontweight='bold')

    # Top by Quantity
    sns.barplot(data=top_quantity, x='Quantity', y='Description', hue='Description', palette='Greens_r', ax=axes[1], edgecolor='black', legend=False)
    axes[1].set_title("Top 10 Products by Total Quantity Sold")
    axes[1].set_xlabel("Quantity Sold (Units)")
    axes[1].set_ylabel("")
    for bar in axes[1].patches:
        width = bar.get_width()
        axes[1].text(width * 1.01, bar.get_y() + bar.get_height()/2, f"{int(width):,}", va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "05_top_products.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Top products plot saved to: {save_path}")


def plot_country_sales(df: pd.DataFrame, output_dir: str):
    """Plot regional sales distribution (excluding UK for non-UK comparison)."""
    ensure_output_dir(output_dir)
    
    country_rev = df.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Chart 1: Top 10 Countries overall (including UK)
    top10_overall = country_rev.head(10)
    sns.barplot(data=top10_overall, x='TotalAmount', y='Country', hue='Country', palette='Purples_r', ax=axes[0], edgecolor='black', legend=False)
    axes[0].set_title("Top 10 Countries by Total Revenue (Inc. UK)")
    axes[0].set_xlabel("Total Revenue (£)")
    axes[0].set_ylabel("Country")
    for bar in axes[0].patches:
        width = bar.get_width()
        axes[0].text(width * 1.01, bar.get_y() + bar.get_height()/2, f"£{width:,.0f}", va='center', fontsize=9, fontweight='bold')

    # Chart 2: Top 10 International Markets (Excluding UK)
    non_uk = country_rev[country_rev['Country'] != 'United Kingdom'].head(10)
    sns.barplot(data=non_uk, x='TotalAmount', y='Country', hue='Country', palette='YlOrRd_r', ax=axes[1], edgecolor='black', legend=False)
    axes[1].set_title("Top 10 International Markets (Excl. UK)")
    axes[1].set_xlabel("Total Revenue (£)")
    axes[1].set_ylabel("")
    for bar in axes[1].patches:
        width = bar.get_width()
        axes[1].text(width * 1.01, bar.get_y() + bar.get_height()/2, f"£{width:,.0f}", va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "06_country_sales_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Country sales distribution plot saved to: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """Plot correlation matrix of numerical variables."""
    ensure_output_dir(output_dir)
    
    num_cols = ['Quantity', 'UnitPrice', 'TotalAmount', 'InvoiceYear', 'InvoiceMonth', 'InvoiceHour', 'IsGuest', 'IsOutlier']
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=1, ax=ax, cbar_kws={"shrink": .8})
    ax.set_title("Correlation Heatmap of Numerical Features", pad=15)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "07_correlation_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Correlation heatmap saved to: {save_path}")


def plot_dayofweek_hourly_sales(df: pd.DataFrame, output_dir: str):
    """Plot sales distribution across Day of Week and Hourly shopping behavior."""
    ensure_output_dir(output_dir)

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_sales = df.groupby('DayOfWeek')['TotalAmount'].sum().reindex(day_order).dropna().reset_index()

    hourly_sales = df.groupby('InvoiceHour')['TotalAmount'].sum().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Day of Week Revenue
    sns.barplot(data=dow_sales, x='DayOfWeek', y='TotalAmount', hue='DayOfWeek', palette='crest', ax=axes[0], edgecolor='black', legend=False)
    axes[0].set_title("Total Revenue by Day of the Week")
    axes[0].set_xlabel("Day of Week")
    axes[0].set_ylabel("Total Revenue (£)")
    axes[0].yaxis.set_major_formatter('£{x:,.0f}')
    for bar in axes[0].patches:
        height = bar.get_height()
        if not np.isnan(height):
            axes[0].text(bar.get_x() + bar.get_width()/2, height * 1.01, f"£{height:,.0f}", ha='center', fontsize=9, fontweight='bold')

    # Hourly Revenue
    sns.lineplot(data=hourly_sales, x='InvoiceHour', y='TotalAmount', marker='o', color='#e74c3c', linewidth=2.5, ax=axes[1])
    axes[1].set_title("Total Revenue by Hour of Day (Shopping Peak Hours)")
    axes[1].set_xlabel("Hour of Day (24h Clock)")
    axes[1].set_ylabel("Total Revenue (£)")
    axes[1].yaxis.set_major_formatter('£{x:,.0f}')
    axes[1].set_xticks(range(6, 21))

    plt.tight_layout()
    save_path = os.path.join(output_dir, "08_dayofweek_sales_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Day of week & hourly sales plot saved to: {save_path}")
