"""
analyze_subreddits.py
---------------------
Enhancement: Compare each subreddit's sentiment predictive power separately.

Trains a walk-forward logistic regression for each subreddit independently
and ranks them by mean AUC. Also plots sentiment overlap across subreddits.

Usage:
    python analyze_subreddits.py

Outputs:
    reports/subreddit_auc_comparison.png
    reports/subreddit_sentiment_distributions.png
    reports/subreddit_cv_results.csv
"""

import os
import glob
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-darkgrid")
COLOURS = {"wallstreetbets": "#e74c3c", "investing": "#3498db", "stocks": "#2ecc71"}


def build_analyzer() -> SentimentIntensityAnalyzer:
    a = SentimentIntensityAnalyzer()
    a.lexicon.update(config.VADER_CUSTOM_LEXICON)
    return a


def score_text(analyzer, title: str, text: str) -> float:
    t = analyzer.polarity_scores(str(title))["compound"]
    b = analyzer.polarity_scores(str(text))["compound"] if text else 0.0
    return 0.7 * t + 0.3 * b


def build_subreddit_daily(subreddit: str, analyzer) -> pd.DataFrame:
    """Aggregate daily sentiment for a single subreddit."""
    raw_files = sorted(glob.glob(f"{config.RAW_DIR}/reddit_*.csv"))
    rows = []
    for fpath in raw_files:
        try:
            df = pd.read_csv(fpath, dtype=str)
            df = df[df["subreddit"] == subreddit].copy()
            if df.empty:
                continue
            df["text"]  = df["text"].fillna("")
            df["title"] = df["title"].fillna("")
            df["compound"] = df.apply(lambda r: score_text(analyzer, r["title"], r["text"]), axis=1)
            date = df["date"].iloc[0]
            rows.append({
                "date":          date,
                "mean_compound": df["compound"].mean(),
                "std_compound":  df["compound"].std(ddof=0),
                "post_volume":   len(df),
            })
        except Exception as e:
            log.warning(f"Skipping {fpath}: {e}")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def walk_forward_auc(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> list[float]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for train_idx, test_idx in tscv.split(X):
        if len(np.unique(y[test_idx])) < 2:
            continue  # can't compute AUC with single class in test fold
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        model = LogisticRegression(C=config.LOGREG_C, max_iter=1000, random_state=config.RANDOM_STATE)
        model.fit(X_tr, y[train_idx])
        prob = model.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y[test_idx], prob))
    return aucs


def analyze():
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    # Load market labels
    market = pd.read_csv(config.RAW_MARKET_FILE, parse_dates=["Date"])
    market = market.rename(columns={"Date": "date"})
    market["label"] = (market["Close"].shift(-1) > market["Close"]).astype(int)
    market = market.dropna(subset=["label"])
    market["date"] = market["date"].dt.strftime("%Y-%m-%d")

    analyzer = build_analyzer()
    all_results = []
    sub_daily_dfs = {}

    for sub in config.SUBREDDITS:
        log.info(f"Processing r/{sub} ...")
        sub_df = build_subreddit_daily(sub, analyzer)
        if sub_df.empty:
            log.warning(f"No data for r/{sub} -- skipping.")
            continue

        sub_daily_dfs[sub] = sub_df

        merged = pd.merge(sub_df, market[["date", "label", "Close"]], on="date", how="inner")
        if len(merged) < 20:
            log.warning(f"Too few rows for r/{sub} ({len(merged)}) -- skipping CV.")
            continue

        # Rolling features
        merged["sent_7d"] = merged["mean_compound"].rolling(7, min_periods=1).mean()
        merged["sent_lag3"] = merged["mean_compound"].shift(3)
        merged = merged.dropna()

        X = merged[["mean_compound", "std_compound", "post_volume", "sent_7d", "sent_lag3"]].values
        y = merged["label"].values.astype(int)

        aucs = walk_forward_auc(X, y)
        mean_auc = np.mean(aucs) if aucs else float("nan")
        log.info(f"  r/{sub}: Mean AUC = {mean_auc:.3f}  (folds: {aucs})")

        for i, auc in enumerate(aucs, 1):
            all_results.append({"subreddit": sub, "fold": i, "auc": auc})

    if not all_results:
        log.error("No CV results -- ensure raw Reddit CSVs exist.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{config.REPORTS_DIR}/subreddit_cv_results.csv", index=False)

    # -- Plot 1: AUC comparison bar chart --------------------------------------
    summary = results_df.groupby("subreddit")["auc"].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [COLOURS.get(s, "#95a5a6") for s in summary["subreddit"]]
    bars = ax.bar(summary["subreddit"], summary["mean"], yerr=summary["std"],
                  color=colors, edgecolor="white", linewidth=0.5,
                  capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "white"})
    ax.axhline(0.55, color="#f39c12", linestyle="--", linewidth=1.5, label="Target (0.55)")
    ax.axhline(0.50, color="grey",   linestyle=":",  linewidth=1.0, label="Random (0.50)")
    for bar, val in zip(bars, summary["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, color="white")
    ax.set_ylabel("Mean AUC-ROC", fontsize=12)
    ax.set_title("Subreddit Predictive Power Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0.35, 0.85)
    ax.legend(framealpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{config.REPORTS_DIR}/subreddit_auc_comparison.png", dpi=150)
    plt.close(fig)
    log.info(f"Saved -> {config.REPORTS_DIR}/subreddit_auc_comparison.png")

    # -- Plot 2: Sentiment distributions per subreddit ------------------------
    if sub_daily_dfs:
        fig, ax = plt.subplots(figsize=(8, 4))
        for sub, df in sub_daily_dfs.items():
            color = COLOURS.get(sub, "#95a5a6")
            ax.hist(df["mean_compound"], bins=30, alpha=0.55,
                    color=color, label=f"r/{sub}", edgecolor="none")
        ax.axvline(0, color="white", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Daily Mean VADER Compound Score", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Sentiment Score Distribution by Subreddit", fontsize=14, fontweight="bold")
        ax.legend(framealpha=0.25)
        fig.tight_layout()
        fig.savefig(f"{config.REPORTS_DIR}/subreddit_sentiment_distributions.png", dpi=150)
        plt.close(fig)
        log.info(f"Saved -> {config.REPORTS_DIR}/subreddit_sentiment_distributions.png")

    log.info("\n-- Subreddit Ranking ------------------------------------------")
    log.info("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    analyze()
