"""
compare_models.py
-----------------
Enhancement: Walk-forward CV comparison of Logistic Regression vs XGBoost.

Both models are evaluated with the same TimeSeriesSplit folds and the same
StandardScaler preprocessing (fit on train only). Results are plotted side
by side and saved to reports/.

Install XGBoost before running:
    pip install xgboost

Usage:
    python compare_models.py

Outputs:
    reports/model_comparison_auc.png    -- AUC per fold, LR vs XGB
    reports/model_comparison_summary.png -- mean AUC + accuracy + F1 bar chart
    models/xgb_model.pkl                -- trained XGBoost model
    models/model_comparison.csv         -- per-fold metrics for both models
"""

import os
import logging

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

import config
from train_model import get_feature_cols, load_features, train_final_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-darkgrid")
MODEL_COLORS = {"LogisticRegression": "#3498db", "XGBoost": "#e67e22"}


def run_cv_for_model(model_factory, X: np.ndarray, y: np.ndarray, model_name: str) -> list[dict]:
    """Generic walk-forward CV runner -- accepts any sklearn-compatible model factory."""
    tscv = TimeSeriesSplit(n_splits=config.CV_N_SPLITS)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = model_factory()
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        results.append({
            "model":    model_name,
            "fold":     fold,
            "n_train":  len(train_idx),
            "n_test":   len(test_idx),
            "auc":      roc_auc_score(y_test, y_prob),
            "accuracy": accuracy_score(y_test, y_pred),
            "f1":       f1_score(y_test, y_pred, zero_division=0),
        })
        log.info(
            f"  [{model_name}] Fold {fold}: "
            f"AUC={results[-1]['auc']:.3f}  Acc={results[-1]['accuracy']:.3f}"
        )

    return results


def plot_fold_comparison(df: pd.DataFrame):
    """Line chart showing AUC per fold for each model."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for model_name, grp in df.groupby("model"):
        color = MODEL_COLORS.get(model_name, "#95a5a6")
        ax.plot(grp["fold"], grp["auc"], marker="o", linewidth=2,
                color=color, label=model_name, markersize=6)

    ax.axhline(0.55, color="#f39c12", linestyle="--", linewidth=1.2, label="Target (0.55)")
    ax.axhline(0.50, color="grey",   linestyle=":",  linewidth=1.0, label="Random (0.50)")
    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("Walk-Forward AUC: Logistic Regression vs XGBoost", fontsize=14, fontweight="bold")
    ax.set_xticks(df["fold"].unique())
    ax.set_ylim(0.3, 1.0)
    ax.legend(framealpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{config.REPORTS_DIR}/model_comparison_auc.png", dpi=150)
    plt.close(fig)
    log.info(f"Saved -> {config.REPORTS_DIR}/model_comparison_auc.png")


def plot_summary_comparison(df: pd.DataFrame):
    """Grouped bar chart comparing mean AUC, accuracy, and F1."""
    metrics = ["auc", "accuracy", "f1"]
    summary = df.groupby("model")[metrics].mean().reset_index()

    x  = np.arange(len(metrics))
    w  = 0.35
    models = summary["model"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (_, row) in enumerate(summary.iterrows()):
        model_name = row["model"]
        color = MODEL_COLORS.get(model_name, "#95a5a6")
        bars = ax.bar(x + i * w, [row[m] for m in metrics], w,
                      label=model_name, color=color, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, [row[m] for m in metrics]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, color="white")

    ax.axhline(0.55, color="#f39c12", linestyle="--", linewidth=1.0, label="AUC target")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(["Mean AUC", "Mean Accuracy", "Mean F1"], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Comparison Summary", fontsize=14, fontweight="bold")
    ax.legend(framealpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{config.REPORTS_DIR}/model_comparison_summary.png", dpi=150)
    plt.close(fig)
    log.info(f"Saved -> {config.REPORTS_DIR}/model_comparison_summary.png")


def compare():
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR,  exist_ok=True)

    if not XGB_AVAILABLE:
        log.error("XGBoost not installed. Run: pip install xgboost")
        return

    df = load_features()
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values
    y = df["label"].values.astype(int)

    log.info(f"Dataset: {len(df)} samples, {X.shape[1]} features")
    log.info(f"Baseline (always-up): {y.mean():.3f}\n")

    # -- Logistic Regression ---------------------------------------------------
    log.info("-- Logistic Regression ------------------------------------------")
    lr_factory = lambda: LogisticRegression(
        C=config.LOGREG_C, penalty="l2", solver="lbfgs",
        max_iter=1000, random_state=config.RANDOM_STATE
    )
    lr_results = run_cv_for_model(lr_factory, X, y, "LogisticRegression")

    # -- XGBoost ---------------------------------------------------------------
    log.info("\n-- XGBoost ------------------------------------------------------")
    xgb_factory = lambda: XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
        verbosity=0,
    )
    xgb_results = run_cv_for_model(xgb_factory, X, y, "XGBoost")

    all_results = pd.DataFrame(lr_results + xgb_results)
    all_results.to_csv(f"{config.MODELS_DIR}/model_comparison.csv", index=False)

    # -- Summary table ---------------------------------------------------------
    summary = all_results.groupby("model")[["auc", "accuracy", "f1"]].mean()
    log.info("\n-- Summary ------------------------------------------------------")
    log.info("\n" + summary.to_string())

    winner = summary["auc"].idxmax()
    log.info(f"\n  🏆 Better mean AUC: {winner}")

    # -- Save XGBoost final model on all data ----------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    xgb_final = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
        eval_metric="logloss", random_state=config.RANDOM_STATE, verbosity=0,
    )
    xgb_final.fit(X_scaled, y)
    joblib.dump({"model": xgb_final, "scaler": scaler, "features": feature_cols},
                f"{config.MODELS_DIR}/xgb_model.pkl")
    log.info(f"XGBoost model saved -> {config.MODELS_DIR}/xgb_model.pkl")

    # -- Plots -----------------------------------------------------------------
    plot_fold_comparison(all_results)
    plot_summary_comparison(all_results)


if __name__ == "__main__":
    compare()
