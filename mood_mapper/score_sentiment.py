"""
score_sentiment.py
------------------
Loads all raw Reddit CSVs, scores each post with VADER (enhanced with the
WSB custom lexicon from config.py), then aggregates scores to daily features.

Why VADER over TextBlob?
  VADER handles capitalisation ("MOON" > "moon"), punctuation ("!!!"),
  emoji-adjacent slang, and negation ("not bullish") -- all common on WSB.

Usage:
    python score_sentiment.py

Output: data/processed/daily_sentiment.csv
Columns:
    date, mean_compound, std_compound, post_volume, pos_ratio, neg_ratio
"""

import os
import glob
import logging

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_analyzer() -> SentimentIntensityAnalyzer:
    """
    Create a VADER analyser pre-loaded with WSB slang from config.
    We update the lexicon in-place -- values are compound-score adjustments.
    """
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(config.VADER_CUSTOM_LEXICON)
    return analyzer


def score_text(analyzer: SentimentIntensityAnalyzer, title: str, text: str) -> float:
    """
    Score a post by blending title (70%) and body text (30%).
    The title is more deliberate and signal-dense than the body,
    hence the higher weight. Posts with no body default body score to 0.
    """
    title_score = analyzer.polarity_scores(str(title))["compound"]
    text_score  = analyzer.polarity_scores(str(text))["compound"] if text else 0.0
    return 0.7 * title_score + 0.3 * text_score


def aggregate_daily(df: pd.DataFrame) -> pd.Series:
    """
    Collapse post-level compound scores into daily aggregate features.
    ddof=0 for std so single-post days still return 0.0 instead of NaN.
    Pos/neg ratios use VADER's standard +/-0.05 thresholds for neutrality.
    """
    return pd.Series({
        "mean_compound": df["compound"].mean(),
        "std_compound":  df["compound"].std(ddof=0),
        "post_volume":   len(df),
        "pos_ratio":     (df["compound"] >  0.05).mean(),
        "neg_ratio":     (df["compound"] < -0.05).mean(),
    })


def score():
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    raw_files = sorted(glob.glob(f"{config.RAW_DIR}/reddit_*.csv"))
    if not raw_files:
        raise FileNotFoundError(
            f"No raw Reddit CSVs found in {config.RAW_DIR}/.\n"
            "Run collect_reddit.py first."
        )

    log.info(f"Scoring {len(raw_files)} daily files ...")
    analyzer  = build_analyzer()
    all_posts = []

    for fpath in raw_files:
        try:
            df = pd.read_csv(fpath, dtype=str)
            df["text"]  = df["text"].fillna("")
            df["title"] = df["title"].fillna("")
            df["compound"] = df.apply(
                lambda r: score_text(analyzer, r["title"], r["text"]), axis=1
            )
            all_posts.append(df)
        except Exception as e:
            log.warning(f"Skipping {fpath}: {e}")

    if not all_posts:
        raise ValueError("No posts could be scored. Check your raw data files.")

    posts_df = pd.concat(all_posts, ignore_index=True)
    log.info(f"Total posts scored: {len(posts_df):,}")

    daily = (
        posts_df.groupby("date")
        .apply(aggregate_daily, include_groups=False)
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily.to_csv(config.DAILY_SENTIMENT_FILE, index=False)
    log.info(f"Saved {len(daily)} daily rows -> {config.DAILY_SENTIMENT_FILE}")
    log.info("\n" + daily.head().to_string(index=False))


if __name__ == "__main__":
    score()
