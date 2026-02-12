"""
Exploratory Data Analysis: patterns, sentiment, keyword frequency, distributions.
Visualizations with Matplotlib and Seaborn.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROCESSED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Style
sns.set_theme(style="whitegrid", palette="muted")
OUT_DIR = DATA_PROCESSED / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load single source of truth from processed."""
    csv_path = DATA_PROCESSED / "single_source_of_truth.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Run transform.py first to create single_source_of_truth.csv")
    return pd.read_csv(csv_path)


def plot_sentiment_distribution(df: pd.DataFrame) -> str:
    """Histogram of sentiment_compound if present."""
    if "sentiment_compound" not in df.columns:
        return ""
    fig, ax = plt.subplots(figsize=(8, 4))
    df["sentiment_compound"].dropna().hist(bins=20, ax=ax, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Sentiment (compound)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Customer Sentiment")
    path = OUT_DIR / "sentiment_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_word_count_distribution(df: pd.DataFrame) -> str:
    """Distribution of word_count in feedback/reviews."""
    col = "word_count" if "word_count" in df.columns else None
    if col is None:
        return ""
    fig, ax = plt.subplots(figsize=(8, 4))
    df[col].dropna().hist(bins=15, ax=ax, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Word count")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Text Length (word count)")
    path = OUT_DIR / "word_count_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_source_type_counts(df: pd.DataFrame) -> str:
    """Bar chart of source_type (internal vs public)."""
    if "source_type" not in df.columns:
        return ""
    counts = df["source_type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["steelblue", "coral"], edgecolor="black")
    ax.set_xlabel("Source type")
    ax.set_ylabel("Count")
    ax.set_title("Data by Source Type")
    plt.xticks(rotation=15)
    path = OUT_DIR / "source_type_counts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_time_patterns(df: pd.DataFrame) -> str:
    """Count by month (bar)."""
    if "month" not in df.columns:
        return ""
    fig, ax = plt.subplots(figsize=(8, 4))
    df["month"].dropna().value_counts().sort_index().plot(kind="bar", ax=ax, color="teal", alpha=0.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.set_title("Feedback/Reviews by Month")
    path = OUT_DIR / "feedback_by_month.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_line_trend(df: pd.DataFrame, date_col: str = "date") -> str:
    """Line chart: trend of counts over time (e.g. feedback volume)."""
    if date_col not in df.columns:
        return ""
    s = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if s.empty:
        return ""
    daily = s.dt.date.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(daily)), daily.values, marker="o", markersize=4)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Count")
    ax.set_title("Trend of Interactions Over Time (line chart)")
    path = OUT_DIR / "trend_line.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_scatter(df: pd.DataFrame, x_col: str = "word_count", y_col: str = "sentiment_compound") -> str:
    """Scatter plot for potential relationships between two variables."""
    xc = x_col if x_col in df.columns else None
    yc = y_col if y_col in df.columns else None
    if not xc or not yc:
        return ""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[xc].fillna(0), df[yc].fillna(0), alpha=0.6, edgecolors="k", linewidths=0.3)
    ax.set_xlabel(xc)
    ax.set_ylabel(yc)
    ax.set_title(f"Relationship: {xc} vs {yc} (scatter)")
    path = OUT_DIR / "scatter_wordcount_sentiment.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_boxplot_noise(df: pd.DataFrame, col: str = "sentiment_compound") -> str:
    """Boxplot to identify outliers/noise in data."""
    if col not in df.columns:
        return ""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(df[col].dropna(), vert=True)
    ax.set_ylabel(col)
    ax.set_title(f"Distribution and Outliers ({col}) – noise detection")
    path = OUT_DIR / "boxplot_sentiment_noise.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_segment_distribution(segments_path: Path = None) -> str:
    """Bar chart: distribution of customers in identified segments (K-Means)."""
    base = DATA_PROCESSED
    path = segments_path or (base / "customers_with_segments.csv")
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    if "segment" not in df.columns:
        return ""
    counts = df["segment"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#3498db", "#e74c3c"], edgecolor="black")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Number of customers")
    ax.set_title("Distribution of Customers in Identified Segments")
    path_out = OUT_DIR / "segment_distribution.png"
    fig.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path_out)


def keyword_frequency(df: pd.DataFrame, text_col: str = "text_clean", top_n: int = 15) -> pd.Series:
    """Top N words by frequency (simple split)."""
    if text_col not in df.columns:
        return pd.Series()
    all_words = " ".join(df[text_col].fillna("").astype(str)).split()
    from collections import Counter
    return pd.Series(Counter(all_words)).nlargest(top_n)


def get_practical_insights(df: pd.DataFrame, top_keywords: dict) -> dict:
    """Answer test.md practical insight questions (what topics, sentiment, time pattern, which source)."""
    insights = {}
    if "text_clean" in df.columns and top_keywords:
        insights["topics_discussed"] = list(top_keywords.keys())[:10]
    if "sentiment_compound" in df.columns:
        m = df["sentiment_compound"].mean()
        pos = (df["sentiment_compound"] > 0.05).sum()
        neg = (df["sentiment_compound"] < -0.05).sum()
        insights["sentiment_towards_products"] = f"Mean compound={m:.3f}; positive count={pos}, negative count={neg}"
    if "month" in df.columns:
        by_month = df["month"].value_counts().sort_index()
        insights["time_pattern"] = f"Peak month(s): {by_month.idxmax()} (count={by_month.max()})"
    if "source_type" in df.columns:
        by_source = df["source_type"].value_counts()
        insights["best_insight_source"] = by_source.to_dict()
    return insights


def run_eda() -> dict:
    """Run all EDA: plots (histogram, bar, line, scatter, boxplot, segment), keyword freq, practical insights."""
    df = load_data()
    results = {"rows": len(df), "columns": list(df.columns)}

    paths = []
    for plot_fn in [
        plot_sentiment_distribution,
        plot_word_count_distribution,
        plot_source_type_counts,
        plot_time_patterns,
        plot_line_trend,
        plot_scatter,
        plot_boxplot_noise,
    ]:
        p = plot_fn(df)
        if p:
            paths.append(p)
    p = plot_segment_distribution()
    if p:
        paths.append(p)

    results["plot_paths"] = paths
    kw = keyword_frequency(df)
    if len(kw) > 0:
        results["top_keywords"] = kw.to_dict()
    results["practical_insights"] = get_practical_insights(df, results.get("top_keywords", {}))
    return results


if __name__ == "__main__":
    r = run_eda()
    print("EDA done. Rows:", r["rows"])
    print("Plots saved:", r.get("plot_paths", []))
    if "top_keywords" in r:
        print("Top keywords:", list(r["top_keywords"].keys())[:10])
