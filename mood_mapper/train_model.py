"""
train_model.py
--------------
Walk-forward cross-validation with TimeSeriesSplit + L2 Logistic Regression.

Why TimeSeriesSplit and NOT train_test_split?
  train_test_split shuffles rows randomly, so training data contains rows
  from the FUTURE relative to some test rows -- pure data leakage for time series.
  TimeSeriesSplit always trains on the past and tests on the future:
    Fold 1: train [0..N1]   test [N1..N2]
    Fold 2: train [0..N2]   test [N2..N3]
    ...

Why fit StandardScaler on train only?
  If we fit on the full dataset, test rows' statistics (mean/std) bleed into
  the scaler -- another subtle leakage. Each fold gets its own fresh scaler
  fitted exclusively on that fold's training slice.

Usage:
    python train_model.py

Outputs:
    models/logistic_model.pkl   -- final model (trained on all data)
    models/cv_results.csv       -- per-fold AUC, accuracy, F1
"""

import os
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Base feature list -- fear_greed_norm added automatically if it exists in the CSV
BASE_FEATURES = [
    "mean_compound",      # raw daily sentiment
    "std_compound",       # sentiment polarisation / disagreement
    "post_volume",        # number of posts (attention signal)
    "pos_ratio",          # fraction of bullish posts
    "neg_ratio",          # fraction of bearish posts
    "sentiment_7d_mean",  # week-long mood trend
    "sentiment_lag3",     # 3-day lagged sentiment (predictive lead)
    "sentiment_delta",    # mood momentum
    "volume_7d_std",      # post-activity burst signal
    "spy_vol_5d",         # market realised volatility context
]
OPTIONAL_FEATURES = ["fear_greed_norm"]   # added if collect_fear_greed.py was run

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    extra = [f for f in OPTIONAL_FEATURES if f in df.columns]
    cols = BASE_FEATURES + extra
    if extra:
        log.info(f"Optional features included: {extra}")
    return cols

# Keep backward-compatible alias
FEATURE_COLS = BASE_FEATURES


def load_features() -> pd.DataFrame:
    if not os.path.exists(config.FEATURES_FILE):
        raise FileNotFoundError(
            f"{config.FEATURES_FILE} not found. Run engineer_features.py first."
        )
    return pd.read_csv(config.FEATURES_FILE, parse_dates=["date"])


def walk_forward_cv(X: np.ndarray, y: np.ndarray) -> list[dict]:
    """Run TimeSeriesSplit CV; return per-fold metrics."""
    tscv    = TimeSeriesSplit(n_splits=config.CV_N_SPLITS)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # -- Scaler: fit on TRAIN only ----------------------------------------
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)          # never fit on test!

        model = LogisticRegression(
            C=config.LOGREG_C,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=config.RANDOM_STATE,
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        result = {
            "fold":     fold,
            "n_train":  len(train_idx),
            "n_test":   len(test_idx),
            "auc":      roc_auc_score(y_test, y_prob),
            "accuracy": accuracy_score(y_test, y_pred),
            "f1":       f1_score(y_test, y_pred, zero_division=0),
        }
        results.append(result)
        log.info(
            f"  Fold {fold}: AUC={result['auc']:.3f}  "
            f"Acc={result['accuracy']:.3f}  F1={result['f1']:.3f}  "
            f"(train={result['n_train']}, test={result['n_test']})"
        )

    return results


def train_final_model(
    X: np.ndarray, y: np.ndarray
) -> tuple[LogisticRegression, StandardScaler]:
    """
    Train on ALL available data -- this is the production-ready model.
    A fresh scaler is fitted on the full dataset.
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        C=config.LOGREG_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_scaled, y)
    return model, scaler


def train():
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    df = load_features()
    feature_cols = get_feature_cols(df)          # auto-includes fear_greed_norm if present
    X  = df[feature_cols].values
    y  = df["label"].values.astype(int)

    log.info(f"Dataset: {len(df)} samples, {X.shape[1]} features: {feature_cols}")
    log.info(f"Baseline (always-up): {y.mean():.3f}")

    # -- Walk-forward CV -------------------------------------------------------
    log.info(f"\nRunning {config.CV_N_SPLITS}-fold walk-forward cross-validation ...")
    cv_results = walk_forward_cv(X, y)
    cv_df      = pd.DataFrame(cv_results)

    log.info("\n-- Cross-Validation Summary --------------------------------------")
    log.info("\n" + cv_df.to_string(index=False))
    log.info(f"\n  Mean AUC:      {cv_df['auc'].mean():.4f} +/- {cv_df['auc'].std():.4f}")
    log.info(f"  Mean Accuracy: {cv_df['accuracy'].mean():.4f}")
    log.info(f"  Mean F1:       {cv_df['f1'].mean():.4f}")
    log.info(f"  Target AUC > 0.55: {'[OK] ACHIEVED' if cv_df['auc'].mean() > 0.55 else '[X] Not yet'}")

    cv_df.to_csv(f"{config.MODELS_DIR}/cv_results.csv", index=False)

    # -- Final model on all data -----------------------------------------------
    log.info("\nTraining final model on full dataset ...")
    model, scaler = train_final_model(X, y)

    bundle = {"model": model, "scaler": scaler, "features": feature_cols}
    joblib.dump(bundle, config.MODEL_FILE)
    log.info(f"Model saved -> {config.MODEL_FILE}")

    return cv_df


if __name__ == "__main__":
    train()
