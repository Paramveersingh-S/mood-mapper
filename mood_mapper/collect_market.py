"""
collect_market.py
-----------------
Downloads SPY daily OHLCV data via yfinance for the configured date range
and saves to data/raw/spy_prices.csv.

Usage:
    python collect_market.py

Output columns:
    Date, Open, High, Low, Close, Volume
    (auto_adjust=True means Close is already split/dividend-adjusted)
"""

import os
import logging

import pandas as pd
import yfinance as yf

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def collect():
    os.makedirs(config.RAW_DIR, exist_ok=True)

    log.info(
        f"Downloading {config.MARKET_TICKER} "
        f"from {config.START_DATE} to {config.END_DATE} ..."
    )

    ticker = yf.Ticker(config.MARKET_TICKER)
    df = ticker.history(
        start=config.START_DATE,
        end=config.END_DATE,
        interval="1d",
        auto_adjust=True,   # adjusts for splits & dividends -> use Close directly
    )

    if df.empty:
        raise ValueError(
            f"No data returned for {config.MARKET_TICKER}. "
            "Check the ticker symbol and date range in config.py."
        )

    # Flatten multi-level columns that yfinance sometimes returns
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]

    # Reset index -> Date becomes a plain column, strip timezone info
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")

    # Keep only the columns we need
    cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols]

    df.to_csv(config.RAW_MARKET_FILE, index=False)
    log.info(f"Saved {len(df)} rows -> {config.RAW_MARKET_FILE}")
    log.info("\n" + df.head().to_string(index=False))


if __name__ == "__main__":
    collect()
