"""
Download a public dataset (e.g. sentiment/reviews) for analysis.
Uses a direct CSV URL so we don't require the 'datasets' library.
Alternative: use Hugging Face datasets API if transformers/datasets are installed.
"""
import pandas as pd
import requests
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW


def download_csv_from_url(url: str, output_name: str = "public_dataset.csv") -> pd.DataFrame:
    """Download CSV from URL and save to data/raw."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    path = DATA_RAW / output_name
    path.write_bytes(r.content)
    df = pd.read_csv(path)
    print(f"Downloaded {len(df)} rows to {path}")
    return df


def get_sample_sentiment_data() -> pd.DataFrame:
    """
    Use a small public sentiment/review dataset (e.g. from UCI or similar).
    Fallback: create minimal in-memory data if URL fails.
    """
    # Small, well-known CSV (Kaggle/UCI style - sentiment or reviews)
    url = "https://raw.githubusercontent.com/clairett/pytorch-sentiment-analysis/master/data/SST2/train.tsv"
    # SST-2 is label \t sentence; might need sep='\t' and column names
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        path = DATA_RAW / "sentiment_public.csv"
        path.write_bytes(r.content)
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
        # Limit size for quick runs
        df = df.head(500).copy()
        df.to_csv(DATA_RAW / "sentiment_public.csv", index=False)
        print(f"Saved {len(df)} rows to sentiment_public.csv")
        return df
    except Exception as e:
        print(f"Download failed ({e}). Creating minimal synthetic sentiment data.")
        df = pd.DataFrame({
            "label": [1, 0, 1, 0, 1],
            "text": [
                "Great product, love it!",
                "Poor quality, disappointed.",
                "Amazing and fast delivery.",
                "Not worth the price.",
                "Will buy again.",
            ],
        })
        df.to_csv(DATA_RAW / "sentiment_public.csv", index=False)
        return df


if __name__ == "__main__":
    get_sample_sentiment_data()
