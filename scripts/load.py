"""
Load (save) cleaned data to standard formats: CSV and Parquet.
Single source of truth is produced by transform.run_all and saved here.
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROCESSED


def save_cleaned(df: pd.DataFrame, name: str = "single_source_of_truth") -> None:
    """Save DataFrame to data/processed as CSV and Parquet."""
    csv_path = DATA_PROCESSED / f"{name}.csv"
    parquet_path = DATA_PROCESSED / f"{name}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {name}: CSV -> {csv_path}, Parquet -> {parquet_path}")


def load_cleaned(name: str = "single_source_of_truth") -> pd.DataFrame:
    """Load from Parquet if exists, else CSV."""
    parquet_path = DATA_PROCESSED / f"{name}.parquet"
    csv_path = DATA_PROCESSED / f"{name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No {name}.parquet or {name}.csv in {DATA_PROCESSED}")


if __name__ == "__main__":
    # Run transform and then save (transform.run_all already saves; this is for standalone load usage)
    from scripts.transform import run_all
    df = run_all()
    print(df.shape)
    print(df.head())
