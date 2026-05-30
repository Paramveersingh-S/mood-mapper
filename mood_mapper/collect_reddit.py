"""
collect_reddit.py
-----------------
Scrapes top posts from configured subreddits for each calendar day
and saves them to data/raw/reddit_YYYY-MM-DD.csv.

Requires valid Reddit API credentials set in config.py.
Get credentials at: https://www.reddit.com/prefs/apps  (create a "script" type app)

Usage:
    python collect_reddit.py

Output columns per CSV:
    date, subreddit, post_id, title, score, num_comments, upvote_ratio, text
"""

import os
import time
import logging
from datetime import datetime, timedelta

import pandas as pd
import praw

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_reddit_client() -> praw.Reddit:
    """Initialise a read-only PRAW client using credentials from config."""
    return praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
    )


def fetch_top_posts(
    reddit: praw.Reddit,
    subreddit_name: str,
    date: datetime,
    limit: int,
) -> list[dict]:
    """
    Fetch top posts for a given subreddit on a specific day.
    Uses Reddit's CloudSearch syntax to filter by Unix timestamp window.

    Note: Reddit's free API only keeps ~1000 posts per listing; for historical
    data beyond ~2 weeks you may need Pushshift (api.pushshift.io) or Arctic.
    """
    sub = reddit.subreddit(subreddit_name)
    start_ts = int(date.timestamp())
    end_ts   = int((date + timedelta(days=1)).timestamp())

    posts = []
    try:
        for post in sub.search(
            query=f"timestamp:{start_ts}..{end_ts}",
            sort="top",
            syntax="cloudsearch",
            limit=limit,
        ):
            posts.append({
                "date":         date.strftime("%Y-%m-%d"),
                "subreddit":    subreddit_name,
                "post_id":      post.id,
                "title":        post.title,
                "score":        post.score,
                "num_comments": post.num_comments,
                "upvote_ratio": post.upvote_ratio,
                # Truncate body to 500 chars -- enough for sentiment, avoids huge CSVs
                "text": post.selftext[:500] if post.selftext else "",
            })
    except Exception as e:
        log.warning(f"Error fetching r/{subreddit_name} on {date.date()}: {e}")

    return posts


def date_range(start: str, end: str):
    """Yield datetime objects for each calendar day in [start, end]."""
    current = datetime.strptime(start, "%Y-%m-%d")
    stop    = datetime.strptime(end,   "%Y-%m-%d")
    while current <= stop:
        yield current
        current += timedelta(days=1)


def collect():
    os.makedirs(config.RAW_DIR, exist_ok=True)
    reddit = build_reddit_client()

    for day in date_range(config.START_DATE, config.END_DATE):
        date_str = day.strftime("%Y-%m-%d")
        out_path = config.RAW_REDDIT_TEMPLATE.format(date=date_str)

        if os.path.exists(out_path):
            log.info(f"Skipping {date_str} -- already collected.")
            continue

        daily_posts: list[dict] = []
        for sub_name in config.SUBREDDITS:
            posts = fetch_top_posts(reddit, sub_name, day, config.POSTS_PER_SUB)
            daily_posts.extend(posts)
            time.sleep(1)  # polite delay -- Reddit rate limit is ~60 req/min

        if daily_posts:
            df = pd.DataFrame(daily_posts)
            df.to_csv(out_path, index=False)
            log.info(f"Saved {len(df):>4} posts -> {out_path}")
        else:
            log.warning(f"No posts found for {date_str} (weekend / holiday?)")


if __name__ == "__main__":
    collect()
