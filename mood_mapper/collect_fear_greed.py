"""
collect_fear_greed.py
---------------------
Enhancement: Scrapes CNN Money's Fear & Greed Index historical data
and saves it as an additional market-sentiment feature.

The Fear & Greed Index (0–100) measures 7 market indicators:
  market momentum, stock price strength, stock price breadth,
  put/call options ratio, junk bond demand, market volatility (VIX),
  and safe-haven demand. Score < 25 = Extreme Fear, > 75 = Extreme Greed.

CNN's public API endpoint returns ~2 years of historical daily values.

Usage:
    python collect_fear_greed.py

Output: data/raw/fear_greed.csv
Columns: date, fear_greed_value, fear_greed_rating
"""

import os
import logging

import requests
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# CNN's undocumented but stable public API endpoint
FNG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://edition.cnn.com/markets/fear-and-greed",
}


def rating_from_value(v: float) -> str:
    """Map numeric score to categorical label."""
    if v <= 25:  return "Extreme Fear"
    if v <= 45:  return "Fear"
    if v <= 55:  return "Neutral"
    if v <= 75:  return "Greed"
    return "Extreme Greed"


def fetch_fear_greed() -> pd.DataFrame:
    """
    Fetches CNN Fear & Greed JSON and parses the 'fear_and_greed_historical'
    array, which contains daily { x (epoch ms), y (score), rating } entries.
    """
    log.info(f"Fetching Fear & Greed data from CNN ...")
    resp = requests.get(FNG_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    data = resp.json()

    # The JSON structure: { "fear_and_greed": {...}, "fear_and_greed_historical": { "data": [...] } }
    historical = data.get("fear_and_greed_historical", {}).get("data", [])

    if not historical:
        raise ValueError(
            "Unexpected JSON structure from CNN API -- the endpoint may have changed.\n"
            f"Keys found: {list(data.keys())}"
        )

    records = []
    for point in historical:
        # x is epoch milliseconds
        date = pd.to_datetime(point["x"], unit="ms").strftime("%Y-%m-%d")
        value = float(point["y"])
        records.append({
            "date":               date,
            "fear_greed_value":   value,
            "fear_greed_rating":  rating_from_value(value),
        })

    df = (
        pd.DataFrame(records)
        .sort_values("date")
        .drop_duplicates(subset="date")
        .reset_index(drop=True)
    )

    # CNN returns the last ~2 years of data from today.
    # Filter to our configured start date but always allow up to today's date
    # (using config.END_DATE would cut out data if END_DATE < today - 2yr)
    from datetime import date as _date
    today_str = _date.today().strftime("%Y-%m-%d")
    df = df[(df["date"] >= config.START_DATE) & (df["date"] <= today_str)]

    return df


def collect():
    os.makedirs(config.RAW_DIR, exist_ok=True)

    df = fetch_fear_greed()

    out = f"{config.RAW_DIR}/fear_greed.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved {len(df)} rows -> {out}")
    log.info("\n" + df.head().to_string(index=False))
    log.info(f"\nScore range: {df['fear_greed_value'].min():.1f} – {df['fear_greed_value'].max():.1f}")
    log.info(f"Rating distribution:\n{df['fear_greed_rating'].value_counts().to_string()}")


if __name__ == "__main__":
    collect()
