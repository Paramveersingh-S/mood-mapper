"""
engineer_features.py
--------------------
Merges daily Reddit sentiment with SPY OHLCV prices, engineers
time-series features, and builds the binary classification label.
Optionally merges CNN Fear & Greed Index if data/raw/fear_greed.csv exists.

Label definition:
    label = 1  if SPY Close(t+1) > SPY Close(t)   ("up next day")
    label = 0  otherwise

Feature rationale:
    sentiment_7d_mean  -- week-long mood trend, smooths daily noise
    sentiment_lag3     -- does Reddit predict the market 3 days ahead?
    sentiment_delta    -- mood momentum (getting more/less bullish?)
    volume_7d_std      -- unusual post-activity bursts -> volatility signal
    spy_vol_5d         -- rolling realised volatility as market context

Usage:
    python engineer_features.py

Output: data/processed/features.csv
"""

import os
import logging

import pandas as pd
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(config.DAILY_SENTIMENT_FILE):
        raise FileNotFoundError(f"{config.DAILY_SENTIMENT_FILE} missing -- run score_sentiment.py first.")
    if not os.path.exists(config.RAW_MARKET_FILE):
        raise FileNotFoundError(f"{config.RAW_MARKET_FILE} missing -- run collect_market.py first.")

    sentiment = pd.read_csv(config.DAILY_SENTIMENT_FILE, parse_dates=["date"])
    market    = pd.read_csv(config.RAW_MARKET_FILE,      parse_dates=["Date"])
    market    = market.rename(columns={"Date": "date"})
    return sentiment, market


def engineer(sentiment: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    # -- 1. Merge on trading dates only (inner join drops weekends from sentiment) --
    df = pd.merge(
        sentiment,
        market[["date", "Close", "Volume"]],
        on="date",
        how="inner",
    ).sort_values("date").reset_index(drop=True)

    # -- 2. Market context features --------------------------------------------
    df["spy_return"]  = df["Close"].pct_change()           # daily log-approx return
    df["spy_vol_5d"]  = df["spy_return"].rolling(5).std()  # 5-day realised volatility

    # -- 3. Sentiment rolling / lag features -----------------------------------
    # 7-day rolling mean: captures sustained mood shifts vs single-day spikes
    df["sentiment_7d_mean"] = df["mean_compound"].rolling(7, min_periods=1).mean()
    # 3-day lag: tests predictive power of Reddit mood on prices 3 days out
    df["sentiment_lag3"]    = df["mean_compound"].shift(3)
    # Daily delta: mood momentum -- are people getting more or less bullish?
    df["sentiment_delta"]   = df["mean_compound"].diff()
    # Rolling std of post volume: activity surges often precede volatility
    df["volume_7d_std"]     = df["post_volume"].rolling(7, min_periods=1).std()

    # -- 4. Binary label (shift -1: today's features -> tomorrow's direction) --
    # CRITICAL: shift(-1) aligns next-day Close with today's feature row.
    # The last row will be NaN and is dropped below -- no future leakage.
    df["label"] = (df["Close"].shift(-1) > df["Close"]).astype("Int64")

    # -- 5. Fear & Greed Index (optional) -------------------------------------
    fng_path = f"{config.RAW_DIR}/fear_greed.csv"
    if os.path.exists(fng_path):
        fng = pd.read_csv(fng_path, parse_dates=["date"])
        df = pd.merge(df, fng[["date", "fear_greed_value"]], on="date", how="left")
        # Normalise to [0, 1]
        df["fear_greed_norm"] = df["fear_greed_value"] / 100.0
        # Forward-fill weekends/holidays, then back-fill early dates,
        # then fill any remaining NaN with 0.5 (neutral) so no rows are dropped
        df["fear_greed_norm"]  = df["fear_greed_norm"].ffill().bfill().fillna(0.5)
        df["fear_greed_value"] = df["fear_greed_value"].ffill().bfill().fillna(50.0)
        log.info(f"Fear & Greed Index merged [OK] (dates may not overlap -- NaN filled with neutral 0.5)")
    else:
        log.info("Fear & Greed file not found -- skipping (run collect_fear_greed.py to add it).")

    # -- 6. Drop NaN rows introduced by rolling windows and shift -------------
    df = df.dropna().reset_index(drop=True)

    up_pct = df["label"].mean()
    log.info(f"Class balance -- Up: {up_pct:.1%}  Down: {1 - up_pct:.1%}  (baseline: always-up = {up_pct:.1%})")

    return df


def save(df: pd.DataFrame):
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    df.to_csv(config.FEATURES_FILE, index=False)
    log.info(f"Saved {len(df)} rows, {df.shape[1]} columns -> {config.FEATURES_FILE}")
    log.info(
        "\n"
        + df[["date", "mean_compound", "sentiment_7d_mean", "spy_return", "label"]]
        .head()
        .to_string(index=False)
    )


def run():
    sentiment, market = load_data()
    df = engineer(sentiment, market)
    save(df)


if __name__ == "__main__":
    run()
