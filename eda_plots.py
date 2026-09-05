import logging
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
 
logger = logging.getLogger(__name__)
 
sns.set_theme(style="whitegrid")
 
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
 
AQI_BANDS = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy (Sensitive)", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]
 
 
def _prep_datetime(df):
    """Internal helper: attach a real datetime column without mutating input."""
    df = df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.sort_values("dt")
 
 
def _shade_aqi_bands(ax, axis="y"):
    """Overlay faint horizontal (or vertical) bands for AQI hazard categories."""
    for low, high, label, color in AQI_BANDS:
        if axis == "y":
            ax.axhspan(low, high, color=color, alpha=0.06)
        else:
            ax.axvspan(low, high, color=color, alpha=0.06)
 
 
def plot_aqi_timeseries(df, figsize=(14, 5)):
    """Full-history AQI line plot with AQI hazard bands shaded in the background."""
    df = _prep_datetime(df)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df["dt"], df["aqi"], color="black", linewidth=0.8)
    _shade_aqi_bands(ax, axis="y")
    ax.set_title("Lahore AQI Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    fig.tight_layout()
    return fig
 
 
def plot_hourly_pattern(df, figsize=(10, 5)):
    """Boxplot of AQI grouped by hour-of-day — reveals diurnal (traffic-driven) cycles."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x="hour", y="aqi", ax=ax, color="#4c72b0")
    ax.set_title("AQI Distribution by Hour of Day")
    ax.set_xlabel("Hour (0-23)")
    ax.set_ylabel("AQI")
    fig.tight_layout()
    return fig
 
 
def plot_weekly_pattern(df, figsize=(9, 5)):
    """Boxplot of AQI grouped by day-of-week."""
    fig, ax = plt.subplots(figsize=figsize)
    order = list(range(7))
    sns.boxplot(data=df, x="day_of_week", y="aqi", order=order, ax=ax, color="#55a868")
    ax.set_xticks(order)
    ax.set_xticklabels(DAY_NAMES)
    ax.set_title("AQI Distribution by Day of Week")
    ax.set_xlabel("")
    ax.set_ylabel("AQI")
    fig.tight_layout()
    return fig
 
 
def plot_monthly_pattern(df, figsize=(10, 5)):
    """Boxplot of AQI grouped by month. Only meaningful once history spans 2+ months."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x="month", y="aqi", ax=ax, color="#c44e52")
    ax.set_title("AQI Distribution by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("AQI")
    fig.tight_layout()
    return fig
 
 
 
def plot_aqi_distribution(df, figsize=(9, 5)):
    """Histogram + KDE of AQI values, with a skewness annotation."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(df["aqi"].dropna(), kde=True, ax=ax, color="#4c72b0")
    skew = df["aqi"].skew()
    ax.set_title(f"AQI Distribution (skewness = {skew:.2f})")
    ax.set_xlabel("AQI")
    fig.tight_layout()
    return fig
 
 
def plot_pollutant_distributions(df, figsize=(11, 5)):
    """Side-by-side histograms of PM2.5 and PM10."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.histplot(df["pm25"].dropna(), kde=True, ax=axes[0], color="#dd8452")
    axes[0].set_title("PM2.5 Distribution")
    sns.histplot(df["pm10"].dropna(), kde=True, ax=axes[1], color="#8172b3")
    axes[1].set_title("PM10 Distribution")
    fig.tight_layout()
    return fig
 
 
def plot_correlation_heatmap(df, figsize=(9, 7)):
    """Correlation heatmap across all numeric feature columns."""
    numeric_cols = [
        "aqi", "pm25", "pm10", "temperature", "humidity",
        "aqi_change_rate", "hour", "day_of_week", "month",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[numeric_cols].corr()
 
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return fig
 
 
def plot_aqi_vs_pm(df, figsize=(11, 5)):
    """Scatter of AQI against PM2.5 and PM10 — sanity check that AQI tracks its known drivers."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].scatter(df["pm25"], df["aqi"], s=8, alpha=0.4, color="#dd8452")
    axes[0].set_xlabel("PM2.5")
    axes[0].set_ylabel("AQI")
    axes[0].set_title("AQI vs PM2.5")
 
    axes[1].scatter(df["pm10"], df["aqi"], s=8, alpha=0.4, color="#8172b3")
    axes[1].set_xlabel("PM10")
    axes[1].set_ylabel("AQI")
    axes[1].set_title("AQI vs PM10")
    fig.tight_layout()
    return fig
 
 
def plot_change_rate_distribution(df, figsize=(9, 5)):
    """
    Histogram of aqi_change_rate. Note: this is the RAW feature-store value
    (contains NaNs for cold-start / >3h gaps) — NOT the train.py-imputed
    version, so there is no artificial zero-spike here. If you want to show
    what the model actually trains on, apply train._impute_change_rate()
    first and pass that dataframe in instead.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(df["aqi_change_rate"].dropna(), kde=True, ax=ax, color="#64b5cd")
    n_null = df["aqi_change_rate"].isna().sum()
    ax.set_title(f"AQI Change Rate Distribution ({n_null} null / stale-gap rows excluded)")
    ax.set_xlabel("AQI points per hour")
    fig.tight_layout()
    return fig
 

 
def plot_missingness_over_time(df, figsize=(14, 3)):
    """
    Visualizes gaps in the hourly series by plotting the time delta between
    consecutive rows. Spikes above 1h reveal missed fetch cycles or upstream
    staleness exclusions.
    """
    df = _prep_datetime(df)
    gap_hours = df["dt"].diff().dt.total_seconds() / 3600.0
 
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df["dt"], gap_hours, color="#c44e52", linewidth=0.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="expected (1h)")
    ax.set_title("Gap Between Consecutive Readings")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hours since previous reading")
    ax.legend()
    fig.tight_layout()
    return fig
 
 
def plot_train_val_test_variance(df, train_frac=0.70, val_frac=0.15, figsize=(9, 5)):
    """
    Boxplot of AQI across the chronological train/val/test split (same
    fractions as train.py's TRAIN_FRAC/VAL_FRAC), to visually explain
    whether the test window has narrower AQI variance than train — the
    known driver of low test R² noted in project history.
    """
    df = _prep_datetime(df).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
 
    split_labels = np.empty(n, dtype=object)
    split_labels[:train_end] = "train"
    split_labels[train_end:val_end] = "val"
    split_labels[val_end:] = "test"
    df["split"] = split_labels
 
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x="split", y="aqi", order=["train", "val", "test"], ax=ax,
                palette={"train": "#4c72b0", "val": "#dd8452", "test": "#55a868"})
    ax.set_title("AQI Variance Across Chronological Train/Val/Test Splits")
    fig.tight_layout()
    return fig
