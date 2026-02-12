"""
Generate synthetic internal data (customers, orders, feedback) for demo/offline use.
Simulates: email list, limited orders, scattered text feedback - as in test.md.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW

# Reproducible noise
RNG = np.random.default_rng(42)


def generate_customers(n: int = 200) -> pd.DataFrame:
    """Synthetic customer list (email, signup date, some missing fields)."""
    base = datetime(2023, 1, 1)
    dates = [base + timedelta(days=int(x)) for x in RNG.integers(0, 400, n)]
    # Intentionally add missing values (noisy data)
    channels = ["email", "instagram", "website", "referral", None]
    ch = RNG.choice(channels, n, p=[0.35, 0.25, 0.2, 0.1, 0.1])
    region = RNG.choice(["North", "South", "Central", None], n, p=[0.3, 0.3, 0.3, 0.1])
    df = pd.DataFrame({
        "customer_id": range(1000, 1000 + n),
        "email": [f"user{i}@example.com" for i in range(n)],
        "signup_date": dates,
        "channel": ch,
        "region": region,
    })
    return df


def generate_orders(n: int = 150) -> pd.DataFrame:
    """Limited past orders (customer_id, order_date, amount) - no full details."""
    customers = RNG.integers(1000, 1200, n)
    base = datetime(2023, 6, 1)
    dates = [base + timedelta(days=int(x)) for x in RNG.integers(0, 400, n)]
    amount = np.round(RNG.lognormal(3, 1.2, n), 2)
    # Some missing customer_id (noisy)
    mask = RNG.random(n) < 0.05
    customers = customers.astype(float)
    customers[mask] = np.nan
    return pd.DataFrame({
        "order_id": range(5000, 5000 + n),
        "customer_id": customers,
        "order_date": dates,
        "amount": amount,
    })


def generate_feedback(n: int = 80) -> pd.DataFrame:
    """Scattered text feedback (email/contact form style)."""
    texts = [
        "Love the handmade soap, will buy again!",
        "Delivery was slow but product is good.",
        "Not satisfied with the quality.",
        "Amazing craftsmanship, highly recommend.",
        "The vase broke during shipping.",
        "Great customer service, thank you.",
        "Expected better for the price.",
        "Perfect gift for my mother.",
        "Order arrived late.",
        "Beautiful design, very happy.",
        "Some items were missing from my order.",
        "Best online purchase this year!",
    ]
    base = datetime(2023, 9, 1)
    dates = [base + timedelta(days=int(x)) for x in RNG.integers(0, 120, n)]
    source = RNG.choice(["email", "contact_form", "email"], n)
    # Many missing email (noisy)
    email = [f"user{RNG.integers(0, 200)}@example.com" if RNG.random() > 0.3 else None for _ in range(n)]
    text = RNG.choice(texts, n)
    return pd.DataFrame({
        "feedback_id": range(1, n + 1),
        "date": dates,
        "source": source,
        "customer_email": email,
        "text": text,
    })


def run(output_dir: Path = None) -> None:
    output_dir = output_dir or DATA_RAW
    customers = generate_customers()
    orders = generate_orders()
    feedback = generate_feedback()
    customers.to_csv(output_dir / "customers.csv", index=False)
    orders.to_csv(output_dir / "orders.csv", index=False)
    feedback.to_csv(output_dir / "feedback.csv", index=False)
    print(f"Written: customers.csv ({len(customers)}), orders.csv ({len(orders)}), feedback.csv ({len(feedback)})")


if __name__ == "__main__":
    run()
