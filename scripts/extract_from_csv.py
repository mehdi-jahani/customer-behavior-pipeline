"""
Extract data from local CSV files.
Use for: internal customer list, orders, or any CSV data source.
"""
import pandas as pd
from pathlib import Path
import sys

# Allow running from project root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW


def extract_csv(file_name: str, **read_kwargs) -> pd.DataFrame:
    """
    Load a CSV file from data/raw.
    read_kwargs are passed to pandas.read_csv (e.g. sep=',', encoding='utf-8').
    """
    path = DATA_RAW / file_name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **read_kwargs)


if __name__ == "__main__":
    # Example: list CSVs in data/raw and load first one
    csvs = list(DATA_RAW.glob("*.csv"))
    if csvs:
        df = extract_csv(csvs[0].name)
        print(f"Loaded {len(df)} rows from {csvs[0].name}")
        print(df.head())
    else:
        print("No CSV files in data/raw. Add CSVs or run synthetic data generator.")
