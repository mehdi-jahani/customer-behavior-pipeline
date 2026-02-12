"""
Transform and clean data: iterative cleaning, missing values, normalization,
text NLP (stopwords, stemming), and feature engineering.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_PROCESSED

# NLTK
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    for res in ("stopwords", "vader_lexicon"):
        try:
            nltk.data.find(f"corpora/{res}")
        except (LookupError, Exception):
            nltk.download(res, quiet=True)
    try:
        from nltk.stem import WordNetLemmatizer
    except Exception:
        WordNetLemmatizer = None
except ImportError:
    stopwords = None
    PorterStemmer = None
    WordNetLemmatizer = None
    SentimentIntensityAnalyzer = None


def normalize_dates(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    """Parse and normalize date columns to datetime."""
    out = df.copy()
    for col in date_columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def fill_missing_numeric(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Fill numeric NaNs with column median or mean."""
    out = df.copy()
    num = out.select_dtypes(include=[np.number])
    for c in num.columns:
        if out[c].isna().any():
            if strategy == "median":
                out[c] = out[c].fillna(out[c].median())
            else:
                out[c] = out[c].fillna(out[c].mean())
    return out


def clean_text_column(series: pd.Series) -> pd.Series:
    """Lowercase, remove non-alphanumeric (keep spaces), collapse spaces."""
    def _clean(s):
        if pd.isna(s) or not isinstance(s, str):
            return ""
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()
    return series.map(_clean)


def remove_stopwords_and_stem(text: str, stem: bool = True, use_lemmatize: bool = False) -> str:
    """Remove stopwords and apply Porter stemming or WordNet lemmatization (NLTK)."""
    if not text or (stopwords is None and PorterStemmer is None and (WordNetLemmatizer is None or not use_lemmatize)):
        return text
    words = text.split()
    try:
        sw = set(stopwords.words("english"))
        words = [w for w in words if w not in sw]
    except Exception:
        pass
    if use_lemmatize and WordNetLemmatizer is not None:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]
    elif stem and PorterStemmer is not None:
        ps = PorterStemmer()
        words = [ps.stem(w) for w in words]
    return " ".join(words)


# Simple positive/negative keyword lists (include stemmed forms for Stemming pipeline)
_POSITIVE_KEYWORDS = {"good", "great", "love", "amaz", "best", "excellent", "happy", "beautiful", "beauti", "recommend", "perfect", "excel"}
_NEGATIVE_KEYWORDS = {"bad", "poor", "slow", "disappoint", "broken", "missing", "late", "worst", "never"}


def _has_emoji(s: str) -> int:
    """Return 1 if text contains emoji (Unicode ranges), else 0."""
    if not s or not isinstance(s, str):
        return 0
    for c in s:
        if (
            (0x1F300 <= ord(c) <= 0x1F9FF)
            or (0x2600 <= ord(c) <= 0x26FF)
            or (0x2700 <= ord(c) <= 0x27BF)
        ):
            return 1
    return 0


def add_text_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Add: text length, word count, has_emoji, positive/negative keyword counts, sentiment (NLTK)."""
    out = df.copy()
    if text_col not in out.columns:
        return out
    txt = out[text_col].fillna("").astype(str)
    out["text_length"] = txt.str.len()
    out["word_count"] = txt.str.split().str.len().fillna(0)
    out["has_emoji"] = txt.map(_has_emoji)
    # Count positive/negative keywords (stemmed words may match sets above)
    def count_keywords(s, positive=True):
        words = set(s.lower().split()) if s else set()
        return sum(1 for w in words if w in (_POSITIVE_KEYWORDS if positive else _NEGATIVE_KEYWORDS))
    out["positive_keyword_count"] = txt.map(lambda x: count_keywords(x, True))
    out["negative_keyword_count"] = txt.map(lambda x: count_keywords(x, False))
    if SentimentIntensityAnalyzer is not None:
        sia = SentimentIntensityAnalyzer()
        scores = txt.apply(lambda x: sia.polarity_scores(x))
        out["sentiment_neg"] = [s["neg"] for s in scores]
        out["sentiment_neu"] = [s["neu"] for s in scores]
        out["sentiment_pos"] = [s["pos"] for s in scores]
        out["sentiment_compound"] = [s["compound"] for s in scores]
    return out


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add day_of_week, month, season (1-4) from date column."""
    out = df.copy()
    if date_col not in out.columns:
        return out
    d = pd.to_datetime(out[date_col], errors="coerce")
    out["day_of_week"] = d.dt.dayofweek
    out["month"] = d.dt.month
    # Simple season: 1=Dec-Feb, 2=Mar-May, 3=Jun-Aug, 4=Sep-Nov
    out["season"] = ((d.dt.month % 12 + 3) // 3).fillna(0).astype(int)
    return out


def transform_customers(customers: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich customer table."""
    df = customers.copy()
    df = normalize_dates(df, ["signup_date"])
    df = fill_missing_numeric(df)
    for col in ["channel", "region"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].dropna()) else "unknown")
    # Feature: has customer interacted via channel X? (simple one-hot style)
    if "channel" in df.columns:
        for ch in ["email", "instagram", "website", "referral"]:
            df[f"channel_{ch}"] = (df["channel"].astype(str).str.lower() == ch).astype(int)
    df = add_time_features(df, "signup_date")
    return df


def transform_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich orders."""
    df = orders.copy()
    df = normalize_dates(df, ["order_date"])
    df = fill_missing_numeric(df)
    # Drop rows with missing critical customer_id if desired (or keep and flag)
    df = add_time_features(df, "order_date")
    return df


def drop_rows_missing_key_columns(df: pd.DataFrame, key_columns: list[str], min_fill_ratio: float = 0.5) -> pd.DataFrame:
    """Drop rows that are missing key columns (with care). Keep row if at least min_fill_ratio of key_columns are non-null."""
    if not key_columns:
        return df
    existing = [c for c in key_columns if c in df.columns]
    if not existing:
        return df
    keep = df[existing].notna().mean(axis=1) >= min_fill_ratio
    return df.loc[keep].copy()


def transform_feedback(feedback: pd.DataFrame, use_lemmatization: bool = False) -> pd.DataFrame:
    """Clean text (stopwords + Stemming or Lemmatization), add text and time features."""
    df = feedback.copy()
    df = normalize_dates(df, ["date"])
    if "text" in df.columns:
        df["text_clean"] = clean_text_column(df["text"])
        df["text_clean"] = df["text_clean"].apply(lambda t: remove_stopwords_and_stem(t, stem=not use_lemmatization, use_lemmatize=use_lemmatization))
        df = add_text_features(df, "text_clean")
    df = add_time_features(df, "date")
    return df


def build_single_source_of_truth(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    feedback: pd.DataFrame,
    sentiment_public: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Create one combined dataset for analysis: one row per feedback/review with
    optional merge to customer/order aggregates. If sentiment_public exists, include it.
    """
    # Aggregate orders per customer (use consistent type for merge)
    orders_ = orders.dropna(subset=["customer_id"]).copy()
    orders_["customer_id"] = orders_["customer_id"].astype(int)
    order_agg = orders_.groupby("customer_id").agg(
        order_count=("order_id", "count"),
        total_amount=("amount", "sum"),
    ).reset_index()
    customers_enriched = customers.merge(order_agg, on="customer_id", how="left")
    customers_enriched["order_count"] = customers_enriched["order_count"].fillna(0)
    customers_enriched["total_amount"] = customers_enriched["total_amount"].fillna(0)

    # Feedback with optional customer merge (by email if we had it; here we keep feedback as main)
    feedback_enhanced = feedback.copy()
    feedback_enhanced["source_type"] = "internal_feedback"

    rows = [feedback_enhanced]

    if sentiment_public is not None and len(sentiment_public) > 0:
        pub = sentiment_public.copy()
        if "text" in pub.columns and "label" in pub.columns:
            pub["text_clean"] = clean_text_column(pub["text"])
            pub = add_text_features(pub, "text_clean")
            pub["source_type"] = "public_sentiment"
            pub["date"] = pd.NaT
            rows.append(pub)

    combined = pd.concat(rows, ignore_index=True)
    return combined


def run_all(raw_dir: Path = None, out_dir: Path = None) -> pd.DataFrame:
    """Load raw CSVs, transform, build single source of truth, return combined DF."""
    raw_dir = raw_dir or DATA_RAW
    out_dir = out_dir or DATA_PROCESSED

    customers = pd.read_csv(raw_dir / "customers.csv") if (raw_dir / "customers.csv").exists() else pd.DataFrame()
    orders = pd.read_csv(raw_dir / "orders.csv") if (raw_dir / "orders.csv").exists() else pd.DataFrame()
    feedback = pd.read_csv(raw_dir / "feedback.csv") if (raw_dir / "feedback.csv").exists() else pd.DataFrame()
    sentiment_public = None
    if (raw_dir / "sentiment_public.csv").exists():
        sentiment_public = pd.read_csv(raw_dir / "sentiment_public.csv")

    if customers.size:
        customers = transform_customers(customers)
        customers.to_csv(out_dir / "customers_clean.csv", index=False)
    if orders.size:
        orders = transform_orders(orders)
        orders.to_csv(out_dir / "orders_clean.csv", index=False)
    if feedback.size:
        # Optional: drop rows missing key info (with care - keep if at least half key cols present)
        feedback = drop_rows_missing_key_columns(feedback, ["text", "date"], min_fill_ratio=0.5)
        feedback = transform_feedback(feedback)
        feedback.to_csv(out_dir / "feedback_clean.csv", index=False)

    combined = build_single_source_of_truth(customers, orders, feedback, sentiment_public)
    combined.to_csv(out_dir / "single_source_of_truth.csv", index=False)
    if combined.shape[1] <= 50:  # Parquet if columns not too many
        combined.to_parquet(out_dir / "single_source_of_truth.parquet", index=False)
    print(f"Saved single_source_of_truth: {len(combined)} rows to {out_dir}")
    return combined


if __name__ == "__main__":
    run_all()
