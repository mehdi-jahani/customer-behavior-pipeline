"""
Extract text content from web pages using BeautifulSoup.
Use for: blog posts, news, or any public page (respect robots.txt and terms of use).
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW

# Polite defaults
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DataProject/1.0; +research)"}


def extract_page_text(url: str, timeout: int = 10) -> str:
    """Fetch URL and return main text content (paragraphs)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove script/style
        for tag in soup(["script", "style"]):
            tag.decompose()
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        return text.strip() or soup.get_text(separator=" ", strip=True)[:5000]
    except Exception as e:
        return f"[Error: {e}]"


def extract_urls_to_df(urls: list[str], source_label: str = "web") -> pd.DataFrame:
    """Fetch multiple URLs and return a DataFrame with columns: source, url, text, fetched_at."""
    rows = []
    for url in urls:
        text = extract_page_text(url)
        rows.append({
            "source": source_label,
            "url": url,
            "text": text,
            "fetched_at": datetime.utcnow().isoformat(),
        })
    return pd.DataFrame(rows)


def run_example_and_save() -> pd.DataFrame:
    """
    Example: fetch a few public pages (e.g. Wikipedia) and save to CSV.
    Replace URLs with your own targets (craft blogs, trend articles, etc.).
    """
    # Safe public URLs for demo (replace with craft/lifestyle articles if needed)
    urls = [
        "https://en.wikipedia.org/wiki/Handicraft",
        "https://en.wikipedia.org/wiki/E-commerce",
    ]
    df = extract_urls_to_df(urls, source_label="wikipedia")
    out = DATA_RAW / "web_content.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    return df


if __name__ == "__main__":
    run_example_and_save()
