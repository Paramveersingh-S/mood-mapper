"""
evaluate.py
-----------
Generates all report charts and prints final metrics.

Charts produced:
    reports/walk_forward_auc.png       -- AUC per CV fold (bar)
    reports/feature_importance.png     -- logistic regression coefficients
    reports/confusion_matrix.png       -- confusion matrix on last fold
    reports/sentiment_vs_returns.png   -- scatter: today's sentiment vs next-day SPY return

Usage:
    python evaluate.py
"""

import os
import logging

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")   # headless backend -- no display required
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch

import config
from train_model import FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# -- Consistent colour palette --------------------------------------------------
C_GREEN  = "#2ecc71"
C_RED    = "#e74c3c"
C_BLUE   = "#3498db"
C_ORANGE = "#f39c12"

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlepad": 12})


# -----------------------------------------------------------------------------
def plot_walk_forward_auc(cv_df: pd.DataFrame, out_path: str):
    """Bar chart of AUC per fold with target line at 0.55."""
    fig, ax = plt.subplots(figsize=(8, 4))

    colors = [C_GREEN if v >= 0.55 else C_RED for v in cv_df["auc"]]
    bars = ax.bar(cv_df["fold"], cv_df["auc"], color=colors, edgecolor="white", linewidth=0.6, zorder=3)

    ax.axhline(0.55,              color=C_ORANGE, linestyle="--", linewidth=1.5, label="Target (0.55)", zorder=4)
    ax.axhline(0.50,              color="grey",   linestyle=":",  linewidth=1.0, label="Random (0.50)", zorder=4)
    ax.axhline(cv_df["auc"].mean(), color="white", linestyle="-", linewidth=1.8,
               label=f"Mean AUC ({cv_df['auc'].mean():.3f})", zorder=4)

    # Annotate bar values
    for bar, val in zip(bars, cv_df["auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="white")

    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("Walk-Forward Cross-Validation -- AUC per Fold", fontsize=14, fontweight="bold")
    ax.set_ylim(0.3, 1.0)
    ax.set_xticks(cv_df["fold"])
    ax.legend(framealpha=0.25, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved -> {out_path}")


# -----------------------------------------------------------------------------
def plot_feature_importance(model, feature_names: list[str], out_path: str):
    """Horizontal bar chart of logistic regression coefficients."""
    coefs = model.coef_[0]
    imp_df = (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        .sort_values("coefficient")
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [C_GREEN if c > 0 else C_RED for c in imp_df["coefficient"]]
    ax.barh(imp_df["feature"], imp_df["coefficient"], color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="white", linewidth=0.8)
    ax.set_xlabel("Logistic Regression Coefficient (L2-regularised, C=0.1)", fontsize=10)
    ax.set_title("Feature Importance -- Mood-to-Market Mapper", fontsize=14, fontweight="bold")

    legend_elements = [
        Patch(facecolor=C_GREEN, label="Positive -> predicts UP"),
        Patch(facecolor=C_RED,   label="Negative -> predicts DOWN"),
    ]
    ax.legend(handles=legend_elements, framealpha=0.25, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved -> {out_path}")


# -----------------------------------------------------------------------------
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    """Heatmap confusion matrix for the last CV fold."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred Down", "Pred Up"],
        yticklabels=["True Down", "True Up"],
        ax=ax, linewidths=0.5, linecolor="grey",
    )
    ax.set_title("Confusion Matrix -- Last CV Fold", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved -> {out_path}")


# -----------------------------------------------------------------------------
def plot_sentiment_vs_returns(df: pd.DataFrame, out_path: str):
    """Scatter of today's compound sentiment vs next-day SPY return."""
    # next-day return = shift(-1) on spy_return
    next_day_return = df["spy_return"].shift(-1).dropna()
    aligned_df = df.loc[next_day_return.index].copy()
    aligned_df["next_day_return"] = next_day_return

    colors = aligned_df["label"].map({1: C_GREEN, 0: C_RED})

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        aligned_df["mean_compound"],
        aligned_df["next_day_return"],
        c=colors, alpha=0.45, edgecolors="none", s=22,
    )
    ax.axhline(0, color="white", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="white", linewidth=0.6, linestyle="--")
    ax.set_xlabel("VADER Compound Sentiment Score (today)", fontsize=11)
    ax.set_ylabel("SPY Return (next trading day)", fontsize=11)
    ax.set_title("Reddit Sentiment vs Next-Day SPY Return", fontsize=13, fontweight="bold")

    legend_elements = [
        Patch(facecolor=C_GREEN, label="SPY Up next day"),
        Patch(facecolor=C_RED,   label="SPY Down next day"),
    ]
    ax.legend(handles=legend_elements, framealpha=0.25, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved -> {out_path}")


# -----------------------------------------------------------------------------
def evaluate():
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    # -- Load saved artefacts --------------------------------------------------
    bundle = joblib.load(config.MODEL_FILE)
    model, features = bundle["model"], bundle["features"]

    cv_df = pd.read_csv(f"{config.MODELS_DIR}/cv_results.csv")
    df    = pd.read_csv(config.FEATURES_FILE, parse_dates=["date"])

    X = df[features].values
    y = df["label"].values.astype(int)

    # -- Reproduce last-fold predictions for confusion matrix ------------------
    tscv = TimeSeriesSplit(n_splits=config.CV_N_SPLITS)
    last_train_idx, last_test_idx = list(tscv.split(X))[-1]

    X_train_last = X[last_train_idx]
    X_test_last  = X[last_test_idx]
    y_test_last  = y[last_test_idx]

    scaler_last = StandardScaler().fit(X_train_last)
    y_pred_last = model.predict(scaler_last.transform(X_test_last))

    # -- Generate all plots ----------------------------------------------------
    plot_walk_forward_auc(cv_df, f"{config.REPORTS_DIR}/walk_forward_auc.png")
    plot_feature_importance(model, features, f"{config.REPORTS_DIR}/feature_importance.png")
    plot_confusion_matrix(y_test_last, y_pred_last, f"{config.REPORTS_DIR}/confusion_matrix.png")
    plot_sentiment_vs_returns(df, f"{config.REPORTS_DIR}/sentiment_vs_returns.png")

    # -- Final summary ---------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("  FINAL EVALUATION RESULTS")
    log.info("=" * 60)
    log.info(f"  Mean AUC ({config.CV_N_SPLITS} folds): {cv_df['auc'].mean():.4f} +/- {cv_df['auc'].std():.4f}")
    log.info(f"  Mean Accuracy:         {cv_df['accuracy'].mean():.4f}")
    log.info(f"  Mean F1:               {cv_df['f1'].mean():.4f}")
    log.info(f"  Target AUC > 0.55:     {'[OK] ACHIEVED' if cv_df['auc'].mean() > 0.55 else '[X] Not yet reached'}")
    log.info("=" * 60)


if __name__ == "__main__":
    evaluate()
