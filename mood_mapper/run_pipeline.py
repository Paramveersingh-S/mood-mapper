"""
run_pipeline.py
---------------
Orchestrates all Mood-to-Market Mapper steps in the correct dependency order.
Logs progress to both stdout and pipeline.log.

Usage:
    # Full run (collect Reddit + market, then model)
    python run_pipeline.py

    # Skip collection (re-use cached CSVs, re-run model only)
    python run_pipeline.py --skip-collect
"""

import argparse
import logging
import sys
import time


def setup_logging():
    import sys
    # Force UTF-8 output so box-drawing chars work on Windows cp1252 consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)-20s] %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", mode="w", encoding="utf-8"),
        ],
    )


def run_step(step_num: int, name: str, fn):
    log = logging.getLogger("pipeline")
    separator = "=" * 62
    log.info(f"\n{separator}")
    log.info(f"  STEP {step_num}: {name}")
    log.info(separator)
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    log.info(f"  [OK] {name} -- done in {elapsed:.1f}s")


def main():
    setup_logging()
    log = logging.getLogger("pipeline")

    parser = argparse.ArgumentParser(
        description="Mood-to-Market Mapper -- end-to-end ML pipeline"
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip Reddit + market collection (use already-downloaded CSVs)",
    )
    args = parser.parse_args()

    # -- Import modules (deferred so logging is set up first) ------------------
    import collect_market
    import score_sentiment
    import engineer_features
    import train_model
    import evaluate

    step = 1

    if not args.skip_collect:
        import collect_reddit
        run_step(step, "Collect Reddit Posts",   collect_reddit.collect);  step += 1
        run_step(step, "Collect Market Data",    collect_market.collect);  step += 1
    else:
        log.info("--skip-collect: skipping Reddit scrape, refreshing market data only.")
        run_step(step, "Collect Market Data (refresh)", collect_market.collect); step += 1

    run_step(step, "Score Sentiment (VADER)",          score_sentiment.score);      step += 1
    run_step(step, "Engineer Features",                engineer_features.run);      step += 1
    run_step(step, "Train Model (Walk-Forward CV)",    train_model.train);          step += 1
    run_step(step, "Evaluate & Generate Reports",      evaluate.evaluate);          step += 1

    log.info(f"\n{'=' * 62}")
    log.info("  Pipeline complete!")
    log.info("  Charts saved to:  reports/")
    log.info("  Model saved to:   models/logistic_model.pkl")
    log.info("  Full log:         pipeline.log")
    log.info(f"{'=' * 62}")


if __name__ == "__main__":
    main()
