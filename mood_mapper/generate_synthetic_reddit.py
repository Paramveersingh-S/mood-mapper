"""
generate_synthetic_reddit.py
----------------------------
Generates realistic synthetic Reddit post CSVs for every trading day in
data/raw/spy_prices.csv so the full pipeline can run without Reddit credentials.

The synthetic posts have sentiment that is WEAKLY correlated with SPY returns
(with a 1-day lag), giving the model a genuine (if modest) signal to find.
Post titles are drawn from templated financial phrases mixed with noise.

Run ONCE before score_sentiment.py:
    python generate_synthetic_reddit.py
"""

import os
import random
import numpy as np
import pandas as pd

import config

random.seed(42)
np.random.seed(42)

# -- Phrase banks by sentiment -------------------------------------------------
BULLISH_TITLES = [
    "SPY looking bullish today, anyone else loading calls?",
    "Market momentum is strong 🚀 buying the dip",
    "Institutions are accumulating -- moon soon",
    "Fed pivot incoming, time to go long",
    "Strong earnings beat -- market ripping higher",
    "Technical breakout on SPY, buying here",
    "VIX collapsing, risk-on trade is back",
    "Soft landing confirmed? Stocks ripping",
    "CPI better than expected -- rally time",
    "Economy stronger than expected, bull market continues",
    "Gap up open, bears getting squeezed hard",
    "FOMO kicking in, market at ATH",
    "Dip buyers in control today, buying SPY",
    "Options flow extremely bullish right now",
    "Retail capitulation = bottom is in, going long",
]

BEARISH_TITLES = [
    "SPY about to dump, bought puts this morning",
    "This rally is a dead cat bounce -- staying short",
    "Recession incoming, reduce equity exposure now",
    "Inflation not going away -- rate hikes will rekt markets",
    "Fed hawkish, risk-off. Sitting in cash.",
    "Earnings disappointments piling up -- bearish",
    "Credit spreads widening -- warning sign",
    "Yield curve still inverted -- recession is coming",
    "Market overvalued at these levels -- selling",
    "Institutions dumping into retail buying, be careful",
    "Bearish engulfing candle on SPY, puts looking good",
    "Market breadth terrible today -- divergence",
    "Smart money loading puts, follow the flow",
    "Macro headwinds persist -- stay defensive",
    "Black swan risk elevated -- hedging now",
]

NEUTRAL_TITLES = [
    "What's everyone's market outlook for this week?",
    "SPY holding the 200 DMA -- watching closely",
    "Options expiry Friday -- could go either way",
    "Anyone tracking the Fed meeting next week?",
    "Interesting price action today, staying in cash",
    "Market feeling choppy -- waiting for direction",
    "Looking at sector rotation -- where are you positioned?",
    "SPY consolidating at key level",
    "Volume light today -- market directionless",
    "Watching VIX for clues on direction",
    "Did anyone see the jobs number this morning?",
    "Mixed signals from the macro data",
    "Just averaging in slowly regardless of direction",
    "Market seems to be in wait-and-see mode",
    "Nothing special today, just hodling my positions",
]

SUBREDDITS = config.SUBREDDITS
POSTS_PER_DAY_PER_SUB = 40   # realistic post count per subreddit per trading day


def pick_title(sentiment_bias: float) -> str:
    """
    sentiment_bias in [-1, +1]:
      > 0.2  -> mostly bullish
      < -0.2 -> mostly bearish
      else   -> mixed
    """
    r = random.random()
    if sentiment_bias > 0.2:
        weights = [0.60, 0.15, 0.25]     # bullish / bearish / neutral
    elif sentiment_bias < -0.2:
        weights = [0.15, 0.60, 0.25]
    else:
        weights = [0.30, 0.30, 0.40]

    pool = random.choices(
        [BULLISH_TITLES, BEARISH_TITLES, NEUTRAL_TITLES],
        weights=weights, k=1
    )[0]
    title = random.choice(pool)

    # Add mild noise: random upvote counts, comment counts
    return title


def generate():
    market_path = config.RAW_MARKET_FILE
    if not os.path.exists(market_path):
        raise FileNotFoundError(f"{market_path} not found -- run collect_market.py first.")

    spy = pd.read_csv(market_path, parse_dates=["Date"])
    spy = spy.rename(columns={"Date": "date"})
    spy["date"] = spy["date"].astype(str)
    spy["return"] = spy["Close"].pct_change()

    os.makedirs(config.RAW_DIR, exist_ok=True)

    existing = set(os.listdir(config.RAW_DIR))
    created  = 0

    for _, row in spy.iterrows():
        date_str  = row["date"]
        fname     = f"reddit_{date_str}.csv"

        if fname in existing:
            continue

        # Sentiment bias: lag-1 return (Reddit reacts to yesterday's move)
        # + small forward signal (Reddit sometimes predicts tomorrow)
        lag1_return  = row["return"] if not pd.isna(row["return"]) else 0.0
        fwd_idx      = spy.index[spy["date"] == date_str].tolist()
        fwd_return   = 0.0
        if fwd_idx and fwd_idx[0] + 1 < len(spy):
            fwd_return = spy.iloc[fwd_idx[0] + 1]["return"]
            if pd.isna(fwd_return):
                fwd_return = 0.0

        # Bias = mostly backward-looking (70%) + weak forward (30%)
        bias = 0.7 * lag1_return * 20 + 0.3 * fwd_return * 15
        bias = np.clip(bias + np.random.normal(0, 0.15), -1, 1)

        posts = []
        for sub in SUBREDDITS:
            # Vary volume per subreddit slightly
            n_posts = POSTS_PER_DAY_PER_SUB + random.randint(-10, 10)
            for _ in range(n_posts):
                sub_bias = bias + np.random.normal(0, 0.2)
                title = pick_title(sub_bias)
                posts.append({
                    "date":         date_str,
                    "subreddit":    sub,
                    "post_id":      f"syn_{random.randint(100000, 999999)}",
                    "title":        title,
                    "score":        random.randint(10, 5000),
                    "num_comments": random.randint(5, 500),
                    "upvote_ratio": round(random.uniform(0.55, 0.98), 2),
                    "text":         "",
                })

        df = pd.DataFrame(posts)
        df.to_csv(f"{config.RAW_DIR}/{fname}", index=False)
        created += 1

    print(f"Generated {created} synthetic daily Reddit CSVs in {config.RAW_DIR}/")
    print(f"Skipped {len(spy) - created} already-existing files.")
    print(f"Total trading days covered: {len(spy)}")


if __name__ == "__main__":
    generate()
