"""
paper_trade.py
--------------
Stretch Goal 3: Simulate paper-trading SPY for 30 days using model predictions.

Strategy:
  - Each day, if the model predicts "up" -> BUY (hold SPY for that day)
  - If model predicts "down" -> CASH (sit out, 0% return that day)
  - Compare cumulative returns vs Buy-and-Hold benchmark

Uses the LAST 30 trading days in the features dataset (no re-fitting needed --
the final model was trained on prior data, so these are out-of-sample days
relative to later training folds).

Usage:
    python paper_trade.py

Outputs:
    reports/paper_trade_equity_curve.png
    reports/paper_trade_results.csv
    Terminal: total return, Sharpe ratio, win rate, max drawdown
"""

import os
import logging

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TRADE_DAYS     = 30      # number of recent days to simulate
STARTING_VALUE = 10_000  # USD starting portfolio value


def max_drawdown(equity: pd.Series) -> float:
    """Peak-to-trough maximum drawdown as a fraction."""
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min()


def sharpe_ratio(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe Ratio (risk-free rate assumed = 0)."""
    if daily_returns.std() == 0:
        return 0.0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(periods_per_year)


def simulate():
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    # -- Load model & features -------------------------------------------------
    if not os.path.exists(config.MODEL_FILE):
        raise FileNotFoundError(f"{config.MODEL_FILE} not found -- run train_model.py first.")
    if not os.path.exists(config.FEATURES_FILE):
        raise FileNotFoundError(f"{config.FEATURES_FILE} not found -- run engineer_features.py first.")

    bundle  = joblib.load(config.MODEL_FILE)
    model   = bundle["model"]
    scaler  = bundle["scaler"]
    features = bundle["features"]

    df = pd.read_csv(config.FEATURES_FILE, parse_dates=["date"])

    # Use the last TRADE_DAYS rows for the simulation
    sim_df = df.tail(TRADE_DAYS).copy().reset_index(drop=True)
    if len(sim_df) < TRADE_DAYS:
        log.warning(f"Only {len(sim_df)} rows available -- using all of them.")

    X_sim = scaler.transform(sim_df[features].values)
    sim_df["pred_up"]  = model.predict(X_sim)
    sim_df["pred_prob"] = model.predict_proba(X_sim)[:, 1]

    # -- Compute daily returns -------------------------------------------------
    # spy_return is today's return; strategy return = spy_return if pred_up else 0
    sim_df["strategy_return"] = sim_df["spy_return"] * sim_df["pred_up"]
    sim_df["bnh_return"]      = sim_df["spy_return"]   # buy-and-hold

    # -- Cumulative equity curves ----------------------------------------------
    sim_df["strategy_equity"] = STARTING_VALUE * (1 + sim_df["strategy_return"]).cumprod()
    sim_df["bnh_equity"]      = STARTING_VALUE * (1 + sim_df["bnh_return"]).cumprod()

    # -- Metrics ---------------------------------------------------------------
    strat_total  = (sim_df["strategy_equity"].iloc[-1] / STARTING_VALUE - 1) * 100
    bnh_total    = (sim_df["bnh_equity"].iloc[-1]      / STARTING_VALUE - 1) * 100
    strat_sharpe = sharpe_ratio(sim_df["strategy_return"])
    bnh_sharpe   = sharpe_ratio(sim_df["bnh_return"])
    strat_dd     = max_drawdown(sim_df["strategy_equity"]) * 100
    bnh_dd       = max_drawdown(sim_df["bnh_equity"]) * 100

    # Win rate: days where we were in market AND SPY went up
    in_market = sim_df["pred_up"] == 1
    win_rate  = (sim_df.loc[in_market, "spy_return"] > 0).mean() * 100 if in_market.any() else 0.0
    days_in   = in_market.sum()

    log.info("\n" + "=" * 60)
    log.info("  PAPER TRADE SIMULATION RESULTS (last 30 trading days)")
    log.info("=" * 60)
    log.info(f"  {'Metric':<30} {'Strategy':>10} {'Buy&Hold':>10}")
    log.info(f"  {'-'*50}")
    log.info(f"  {'Total Return':<30} {strat_total:>9.2f}%  {bnh_total:>9.2f}%")
    log.info(f"  {'Annualised Sharpe':<30} {strat_sharpe:>10.3f}  {bnh_sharpe:>10.3f}")
    log.info(f"  {'Max Drawdown':<30} {strat_dd:>9.2f}%  {bnh_dd:>9.2f}%")
    log.info(f"  {'Days in Market':<30} {days_in:>10}  {TRADE_DAYS:>10}")
    log.info(f"  {'Win Rate (when invested)':<30} {win_rate:>9.1f}%")
    log.info("=" * 60)

    # -- Save CSV --------------------------------------------------------------
    out_csv = f"{config.REPORTS_DIR}/paper_trade_results.csv"
    sim_df[["date", "pred_up", "pred_prob", "spy_return",
            "strategy_return", "strategy_equity", "bnh_equity"]].to_csv(out_csv, index=False)
    log.info(f"Saved CSV -> {out_csv}")

    # -- Plot equity curves ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    # Top panel: equity curves
    ax1.plot(sim_df["date"], sim_df["strategy_equity"], color="#2ecc71", linewidth=2.5,
             label=f"Strategy  {strat_total:+.1f}%")
    ax1.plot(sim_df["date"], sim_df["bnh_equity"],      color="#3498db", linewidth=2.0,
             linestyle="--", label=f"Buy & Hold  {bnh_total:+.1f}%")
    ax1.axhline(STARTING_VALUE, color="grey", linewidth=0.7, linestyle=":")
    ax1.set_ylabel(f"Portfolio Value (USD, start=${STARTING_VALUE:,})", fontsize=11)
    ax1.set_title(
        f"Paper Trade: Mood-to-Market Strategy vs Buy & Hold\n"
        f"(last {len(sim_df)} trading days -- model: Logistic Regression)",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(framealpha=0.25, fontsize=10)
    ax1.fill_between(
        sim_df["date"],
        sim_df["strategy_equity"],
        sim_df["bnh_equity"],
        where=sim_df["strategy_equity"] >= sim_df["bnh_equity"],
        alpha=0.12, color="#2ecc71", label="Strategy ahead",
    )
    ax1.fill_between(
        sim_df["date"],
        sim_df["strategy_equity"],
        sim_df["bnh_equity"],
        where=sim_df["strategy_equity"] < sim_df["bnh_equity"],
        alpha=0.12, color="#e74c3c",
    )

    # Bottom panel: prediction probability
    ax2.bar(sim_df["date"], sim_df["pred_prob"], color="#f39c12", alpha=0.7, width=0.8)
    ax2.axhline(0.5, color="white", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("P(up)", fontsize=10)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    out_png = f"{config.REPORTS_DIR}/paper_trade_equity_curve.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log.info(f"Saved chart -> {out_png}")


if __name__ == "__main__":
    simulate()
