"""
score_bert.py
-------------
Stretch Goal 1: FinancialBERT sentiment scorer as an alternative to VADER.

FinancialBERT (ProsusAI/finbert) is a BERT model fine-tuned on financial text
from TRC2 and financial phrasebank. It classifies each sentence as:
  positive / negative / neutral

Unlike VADER (rule-based), FinancialBERT:
  [OK] Understands financial context ("beat earnings" = positive)
  [OK] Handles long-form text better
  [X] Slower (~10-50x slower than VADER on CPU)
  [X] Requires ~500MB model download on first run

Install requirements before running:
    pip install transformers torch

Usage:
    python score_bert.py

Output: data/processed/daily_sentiment_bert.csv
Columns: date, mean_compound_bert, std_compound_bert, post_volume

The output format matches daily_sentiment.csv so you can swap it directly
into engineer_features.py by pointing DAILY_SENTIMENT_FILE to the BERT version.
"""

import os
import glob
import logging

import pandas as pd
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BERT_SENTIMENT_FILE = "data/processed/daily_sentiment_bert.csv"
FINBERT_MODEL       = "ProsusAI/finbert"   # ~500MB download on first use
BATCH_SIZE          = 32                    # lower if you run out of VRAM/RAM
MAX_LEN             = 128                   # max tokens per text chunk


def load_pipeline():
    """
    Lazy-load the FinBERT pipeline. The first call downloads the model weights.
    Requires: pip install transformers torch
    """
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "transformers not installed.\n"
            "Run: pip install transformers torch"
        )

    log.info(f"Loading FinBERT model '{FINBERT_MODEL}' ...")
    pipe = pipeline(
        task="text-classification",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        max_length=MAX_LEN,
        truncation=True,
        batch_size=BATCH_SIZE,
        device=-1,     # -1 = CPU; change to 0 for CUDA GPU
    )
    log.info("FinBERT loaded [OK]")
    return pipe


def label_to_score(label: str) -> float:
    """
    Map FinBERT's categorical output to a numeric score compatible with VADER's
    compound score range [-1, +1], so the rest of the pipeline is unchanged.
    """
    mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    return mapping.get(label.lower(), 0.0)


def score_texts(pipe, texts: list[str]) -> list[float]:
    """Batch-score a list of strings and return compound-equivalent floats."""
    if not texts:
        return []
    results = pipe(texts)
    scores  = []
    for r in results:
        raw_score = label_to_score(r["label"]) * r["score"]   # weighted by confidence
        scores.append(raw_score)
    return scores


def build_combined_text(title: str, text: str) -> str:
    """Concatenate title and body with a separator for a single inference pass."""
    body = str(text).strip() if text else ""
    combined = f"{title}. {body}"[:512]   # FinBERT max practical input
    return combined


def score():
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    raw_files = sorted(glob.glob(f"{config.RAW_DIR}/reddit_*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No raw Reddit CSVs found in {config.RAW_DIR}/")

    pipe = load_pipeline()

    log.info(f"Scoring {len(raw_files)} daily files with FinBERT ...")
    all_posts = []

    for fpath in raw_files:
        try:
            df = pd.read_csv(fpath, dtype=str)
            df["text"]  = df["text"].fillna("")
            df["title"] = df["title"].fillna("")
            df["combined"] = df.apply(
                lambda r: build_combined_text(r["title"], r["text"]), axis=1
            )
            texts = df["combined"].tolist()
            df["compound_bert"] = score_texts(pipe, texts)
            all_posts.append(df)
        except Exception as e:
            log.warning(f"Skipping {fpath}: {e}")

    if not all_posts:
        raise ValueError("No posts could be scored.")

    posts_df = pd.concat(all_posts, ignore_index=True)
    log.info(f"Total posts scored: {len(posts_df):,}")

    # Aggregate to daily (same structure as VADER output for drop-in replacement)
    daily = (
        posts_df.groupby("date")
        .apply(lambda g: pd.Series({
            "mean_compound_bert": g["compound_bert"].mean(),
            "std_compound_bert":  g["compound_bert"].std(ddof=0),
            "post_volume":        len(g),
            "pos_ratio_bert":     (g["compound_bert"] > 0.1).mean(),
            "neg_ratio_bert":     (g["compound_bert"] < -0.1).mean(),
        }), include_groups=False)
        .reset_index()
        .rename(columns={
            "mean_compound_bert": "mean_compound",
            "std_compound_bert":  "std_compound",
        })
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily.to_csv(BERT_SENTIMENT_FILE, index=False)
    log.info(f"Saved {len(daily)} daily rows -> {BERT_SENTIMENT_FILE}")
    log.info("\n" + daily.head().to_string(index=False))

    log.info(
        "\nTo use FinBERT scores in the pipeline, set in config.py:\n"
        f"  DAILY_SENTIMENT_FILE = '{BERT_SENTIMENT_FILE}'\n"
        "Then re-run: python engineer_features.py && python train_model.py && python evaluate.py"
    )


if __name__ == "__main__":
    score()
