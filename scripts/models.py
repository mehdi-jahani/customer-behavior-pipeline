"""
Modeling: Churn (Logistic Regression, Naive Bayes), Segmentation (K-Means),
Sentiment from VADER or optional Hugging Face pre-trained. Evaluation: K-Fold CV, Precision, Recall, F1.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROCESSED

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

OUT_DIR = DATA_PROCESSED / "models"
PLOTS_DIR = DATA_PROCESSED / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import joblib
except ImportError:
    joblib = None


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load customers_clean, orders_clean, single_source_of_truth."""
    base = DATA_PROCESSED
    customers = pd.read_csv(base / "customers_clean.csv") if (base / "customers_clean.csv").exists() else pd.DataFrame()
    orders = pd.read_csv(base / "orders_clean.csv") if (base / "orders_clean.csv").exists() else pd.DataFrame()
    sst = pd.read_csv(base / "single_source_of_truth.csv") if (base / "single_source_of_truth.csv").exists() else pd.DataFrame()
    return customers, orders, sst


def build_churn_dataset(customers: pd.DataFrame, orders: pd.DataFrame, last_days: int = 120) -> pd.DataFrame:
    """
    Churn = 1 if customer has no order in the last `last_days` days (from max order_date).
    Features: order_count, total_amount, day_of_week, month, season, channel, region (encoded).
    """
    if orders.empty or customers.empty:
        return pd.DataFrame()
    orders = orders.dropna(subset=["customer_id"]).copy()
    orders["customer_id"] = orders["customer_id"].astype(int)
    max_date = pd.to_datetime(orders["order_date"]).max()
    cutoff = max_date - pd.Timedelta(days=last_days)
    last_orders = orders.groupby("customer_id")["order_date"].max().reset_index()
    last_orders.columns = ["customer_id", "last_order_date"]
    last_orders["last_order_date"] = pd.to_datetime(last_orders["last_order_date"])
    last_orders["churn"] = (last_orders["last_order_date"] < cutoff).astype(int)
    agg = orders.groupby("customer_id").agg(
        order_count=("order_id", "count"),
        total_amount=("amount", "sum"),
    ).reset_index()
    merged = customers.merge(last_orders[["customer_id", "churn"]], on="customer_id", how="inner")
    merged = merged.merge(agg, on="customer_id", how="left")
    merged["order_count"] = merged["order_count"].fillna(0)
    merged["total_amount"] = merged["total_amount"].fillna(0)
    # Encode categorical
    for col in ["channel", "region"]:
        if col in merged.columns:
            merged[col] = merged[col].astype(str).fillna("unknown")
            merged[col + "_code"] = pd.Categorical(merged[col]).codes
    return merged


def get_churn_features_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """Numeric features and churn target for classification. Returns (X, y, feature_names)."""
    feat_cols = ["order_count", "total_amount", "day_of_week", "month", "season"]
    feat_cols = [c for c in feat_cols if c in df.columns]
    cat_code = [c for c in df.columns if c.endswith("_code")]
    feat_cols = feat_cols + cat_code
    X = df[feat_cols].fillna(0).values
    y = df["churn"].values
    return X, y, feat_cols


def train_churn_models(X: np.ndarray, y: np.ndarray, feat_cols: list[str], cv: int = 3) -> dict:
    """Logistic Regression and Naive Bayes with K-Fold CV; return metrics, coefficients, save joblib."""
    if len(np.unique(y)) < 2:
        return {"error": "Need both classes for classification"}
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}
    for name, model in [
        ("logistic", LogisticRegression(max_iter=500, random_state=42)),
        ("naive_bayes", GaussianNB()),
    ]:
        y_pred = cross_val_predict(model, Xs, y, cv=kf)
        results[name] = {
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "cv_report": classification_report(y, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }
        if name == "logistic":
            model.fit(Xs, y)
            results[name]["coefficients"] = model.coef_.tolist()
            results[name]["feature_importance"] = dict(zip(feat_cols, model.coef_[0].tolist()))
            if joblib:
                joblib.dump({"model": model, "scaler": scaler}, OUT_DIR / "churn_logistic.joblib")
        if name == "naive_bayes" and joblib:
            model.fit(Xs, y)
            joblib.dump({"model": model, "scaler": scaler}, OUT_DIR / "churn_naive_bayes.joblib")
    return results


def run_segmentation(customers: pd.DataFrame, orders: pd.DataFrame, n_clusters: int = 3) -> dict:
    """K-Means on order_count, total_amount. Aggregate orders if not in customers."""
    if not orders.empty and "customer_id" in orders.columns:
        ord_agg = orders.dropna(subset=["customer_id"]).copy()
        ord_agg["customer_id"] = ord_agg["customer_id"].astype(int)
        ord_agg = ord_agg.groupby("customer_id").agg(
            order_count=("order_id", "count"),
            total_amount=("amount", "sum"),
        ).reset_index()
        customers = customers.merge(ord_agg, on="customer_id", how="left")
        customers["order_count"] = customers["order_count"].fillna(0)
        customers["total_amount"] = customers["total_amount"].fillna(0)
    feat = ["order_count", "total_amount"]
    feat = [c for c in feat if c in customers.columns]
    if not feat:
        return {}
    X = customers[feat].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    customers = customers.copy()
    customers["segment"] = labels
    seg_counts = customers["segment"].value_counts().sort_index()
    # Save customers with segments for EDA and plot segment distribution (test.md: key viz)
    seg_path = DATA_PROCESSED / "customers_with_segments.csv"
    customers.to_csv(seg_path, index=False)
    if joblib:
        joblib.dump({"model": km, "scaler": scaler}, OUT_DIR / "kmeans_segment.joblib")
    # Plot: distribution of customers in identified segments
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    seg_counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#3498db", "#e74c3c"], edgecolor="black")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Number of customers")
    ax.set_title("Distribution of Customers in Identified Segments")
    plot_path = PLOTS_DIR / "segment_distribution.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return {
        "n_clusters": n_clusters,
        "inertia": float(km.inertia_),
        "segment_counts": seg_counts.to_dict(),
        "labels_sample": labels[:20].tolist(),
        "customers_with_segments_path": str(seg_path),
        "segment_plot_path": str(plot_path),
    }


def run_pretrained_sentiment_if_available(texts: list[str], max_samples: int = 20) -> dict | None:
    """If Hugging Face transformers available, run pre-trained sentiment-analysis (no training on small data)."""
    try:
        from transformers import pipeline
        pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        sample = texts[:max_samples] if texts else []
        if not sample:
            return None
        out = pipe([t[:512] for t in sample])
        return {"sample_predictions": out, "note": "Pre-trained Hugging Face sentiment (distilbert)"}
    except Exception:
        return None


def run_all_models() -> dict:
    """Run churn, segmentation, and optional pre-trained sentiment."""
    customers, orders, sst = load_processed()
    summary = {}
    # Churn
    churn_df = build_churn_dataset(customers, orders)
    if len(churn_df) >= 20 and churn_df["churn"].nunique() >= 2:
        X, y, feat_cols = get_churn_features_target(churn_df)
        summary["churn"] = train_churn_models(X, y, feat_cols, cv=3)
    else:
        summary["churn"] = {"note": "Not enough data or one class only"}
    # Segmentation
    if not customers.empty:
        summary["segmentation"] = run_segmentation(customers, orders, n_clusters=3)
    # Optional: pre-trained sentiment (test.md: use Hugging Face when no large dataset)
    if not sst.empty and "text_clean" in sst.columns:
        texts = sst["text_clean"].dropna().astype(str).tolist()
        hf = run_pretrained_sentiment_if_available(texts)
        if hf:
            summary["pretrained_sentiment"] = hf
    return summary


if __name__ == "__main__":
    summary = run_all_models()
    print("Churn:", summary.get("churn", {}))
    print("Segmentation:", summary.get("segmentation", {}))
