# config.py -- Central configuration for Mood-to-Market Mapper

# -- Reddit API credentials ----------------------------------------------------
# Get yours at https://www.reddit.com/prefs/apps  (create a "script" app)
REDDIT_CLIENT_ID     = "YOUR_CLIENT_ID"
REDDIT_CLIENT_SECRET = "YOUR_CLIENT_SECRET"
REDDIT_USER_AGENT    = "mood_mapper_bot/0.1 by YOUR_USERNAME"

# -- Subreddits to scrape ------------------------------------------------------
SUBREDDITS    = ["wallstreetbets", "investing", "stocks"]
POSTS_PER_SUB = 100   # top posts to fetch per subreddit per day

# -- Date range ----------------------------------------------------------------
START_DATE = "2023-01-01"
END_DATE   = "2024-12-31"

# -- Market data ---------------------------------------------------------------
MARKET_TICKER = "SPY"

# -- File paths (all relative -- never hardcoded absolutes) ---------------------
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
REPORTS_DIR   = "reports"

RAW_REDDIT_TEMPLATE  = f"{RAW_DIR}/reddit_{{date}}.csv"   # format with date string
RAW_MARKET_FILE      = f"{RAW_DIR}/spy_prices.csv"
DAILY_SENTIMENT_FILE = f"{PROCESSED_DIR}/daily_sentiment.csv"
FEATURES_FILE        = f"{PROCESSED_DIR}/features.csv"
MODEL_FILE           = f"{MODELS_DIR}/logistic_model.pkl"

# -- Model hyperparameters -----------------------------------------------------
LOGREG_C     = 0.1   # L2 regularisation (lower = stronger penalty, less overfit)
CV_N_SPLITS  = 5     # TimeSeriesSplit folds
RANDOM_STATE = 42

# -- VADER custom lexicon -- WSB slang injected at runtime ----------------------
# Values are VADER compound-score adjustments in the range [-4, +4].
VADER_CUSTOM_LEXICON = {
    "moon":     2.0,
    "mooning":  2.0,
    "rocket":   1.5,
    "bullish":  1.5,
    "yolo":     0.5,
    "tendies":  1.0,
    "apes":     0.5,
    "squeeze":  0.8,
    "rekt":    -2.0,
    "dump":    -1.5,
    "crash":   -2.0,
    "bearish": -1.5,
    "puts":    -0.8,
    "short":   -0.5,
    "calls":    0.8,
    "hodl":     0.5,
    "baghold": -1.5,
    "fud":     -1.0,
}
