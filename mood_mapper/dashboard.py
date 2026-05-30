"""
dashboard.py
------------
Stretch Goal 2: Streamlit dashboard showing live Reddit sentiment,
model predictions, and SPY performance.

Install Streamlit before running:
    pip install streamlit

Usage:
    streamlit run dashboard.py

Features:
  - Live sentiment gauge for today
  - Historical sentiment trend (30-day chart)
  - Model's probability of SPY going up tomorrow
  - SPY price chart with sentiment overlay
  - Subreddit breakdown comparison
  - Latest Reddit posts table
"""

import os
import glob
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

import config

# -- Page config ---------------------------------------------------------------
if STREAMLIT_AVAILABLE:
    st.set_page_config(
        page_title="Mood-to-Market Mapper",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# -- Custom CSS ----------------------------------------------------------------
CUSTOM_CSS = """
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #0f0f1a 100%);
    }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .bull-signal  { color: #2ecc71; font-weight: 700; font-size: 1.3rem; }
    .bear-signal  { color: #e74c3c; font-weight: 700; font-size: 1.3rem; }
    .neutral-signal { color: #f39c12; font-weight: 700; font-size: 1.3rem; }
    h1 { background: linear-gradient(90deg, #2ecc71, #3498db);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
"""

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", showline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", showline=False),
    margin=dict(l=20, r=20, t=40, b=20),
)


# -- Data loaders (cached for performance) -------------------------------------
@st.cache_data(ttl=3600)
def load_features_df():
    if not os.path.exists(config.FEATURES_FILE):
        return None
    return pd.read_csv(config.FEATURES_FILE, parse_dates=["date"])


@st.cache_resource
def load_model_bundle():
    if not os.path.exists(config.MODEL_FILE):
        return None
    return joblib.load(config.MODEL_FILE)


@st.cache_data(ttl=3600)
def load_cv_results():
    path = f"{config.MODELS_DIR}/cv_results.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_raw_posts(days_back: int = 7) -> pd.DataFrame:
    """Load the most recent raw Reddit posts for the posts table."""
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    files  = sorted(glob.glob(f"{config.RAW_DIR}/reddit_*.csv"))
    recent = [f for f in files if os.path.basename(f) >= f"reddit_{cutoff}"]
    if not recent:
        return pd.DataFrame()
    dfs = []
    for f in recent[-7:]:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# -- Chart builders -------------------------------------------------------------
def sentiment_gauge(prob: float) -> go.Figure:
    """Semicircular gauge showing probability of SPY going up tomorrow."""
    color = "#2ecc71" if prob > 0.55 else ("#e74c3c" if prob < 0.45 else "#f39c12")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "P(SPY Up Tomorrow) %", "font": {"size": 16, "color": "white"}},
        delta={"reference": 50, "suffix": "%"},
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 45],  "color": "rgba(231,76,60,0.15)"},
                {"range": [45, 55], "color": "rgba(243,156,18,0.15)"},
                {"range": [55, 100],"color": "rgba(46,204,113,0.15)"},
            ],
            "threshold": {
                "line":  {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280)
    return fig


def sentiment_trend_chart(df: pd.DataFrame, days: int = 30) -> go.Figure:
    """Line chart of daily sentiment over the last N days."""
    recent = df.tail(days).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent["mean_compound"],
        mode="lines+markers", name="Daily Sentiment",
        line=dict(color="#3498db", width=2),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(52,152,219,0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent["sentiment_7d_mean"],
        mode="lines", name="7-Day Rolling Mean",
        line=dict(color="#f39c12", width=2, dash="dot"),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(**PLOTLY_LAYOUT, title="Reddit Sentiment Trend", height=300,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def spy_chart(df: pd.DataFrame, days: int = 60) -> go.Figure:
    """SPY close price with sentiment colorized background."""
    recent = df.tail(days).copy()
    fig = go.Figure()

    # Candlestick-style close line
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent["Close"],
        mode="lines", name="SPY Close",
        line=dict(color="#2ecc71", width=2),
    ))

    # Sentiment as bar on secondary axis
    fig.add_trace(go.Bar(
        x=recent["date"],
        y=recent["mean_compound"],
        name="Sentiment",
        marker_color=[
            "rgba(46,204,113,0.5)" if v > 0 else "rgba(231,76,60,0.5)"
            for v in recent["mean_compound"]
        ],
        yaxis="y2",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="SPY Price vs Reddit Sentiment",
        height=350,
        yaxis=dict(title="SPY Close ($)", gridcolor="rgba(255,255,255,0.08)"),
        yaxis2=dict(title="Sentiment", overlaying="y", side="right",
                    range=[-1, 1], showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def auc_fold_chart(cv_df: pd.DataFrame) -> go.Figure:
    colors = ["#2ecc71" if v >= 0.55 else "#e74c3c" for v in cv_df["auc"]]
    fig = go.Figure(go.Bar(
        x=[f"Fold {i}" for i in cv_df["fold"]],
        y=cv_df["auc"],
        marker_color=colors,
        text=[f"{v:.3f}" for v in cv_df["auc"]],
        textposition="auto",
    ))
    fig.add_hline(y=0.55, line_dash="dash", line_color="#f39c12",
                  annotation_text="Target 0.55")
    fig.add_hline(y=0.50, line_dash="dot", line_color="grey",
                  annotation_text="Random 0.50")
    fig.update_layout(**PLOTLY_LAYOUT, title="Walk-Forward CV -- AUC per Fold", height=280)
    return fig


# -- Main app ------------------------------------------------------------------
def main():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not installed. Run: pip install streamlit plotly")
        return

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # -- Sidebar ---------------------------------------------------------------
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/b/b3/Reddit_Logo.png", width=60)
        st.title("Mood-to-Market")
        st.caption("Reddit sentiment -> SPY direction")
        st.divider()
        days_trend = st.slider("Sentiment trend window (days)", 7, 90, 30)
        days_spy   = st.slider("SPY chart window (days)", 14, 120, 60)
        st.divider()
        st.caption(
            f"Subreddits: {', '.join(f'r/{s}' for s in config.SUBREDDITS)}\n\n"
            f"Model: Logistic Regression (C={config.LOGREG_C})\n\n"
            f"Target: Mean AUC > 0.55"
        )

    # -- Title -----------------------------------------------------------------
    st.title("📈 Mood-to-Market Mapper")
    st.caption(
        f"Using Reddit sentiment from **{config.START_DATE}** to **{config.END_DATE}** "
        f"to predict SPY direction | Model: L2 Logistic Regression"
    )

    # -- Load data -------------------------------------------------------------
    df     = load_features_df()
    bundle = load_model_bundle()
    cv_df  = load_cv_results()

    if df is None or bundle is None:
        st.error(
            "⚠️ Model or features not found.\n\n"
            "Run the pipeline first:\n```bash\npython run_pipeline.py --skip-collect\n```"
        )
        st.stop()

    model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]

    # Latest prediction
    latest      = df.tail(1)
    X_latest    = scaler.transform(latest[features].values)
    pred_up     = model.predict(X_latest)[0]
    pred_prob   = model.predict_proba(X_latest)[0][1]
    latest_date = latest["date"].iloc[0].strftime("%Y-%m-%d")
    latest_sent = latest["mean_compound"].iloc[0]

    # -- Row 1: Key metrics ----------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Latest Sentiment",
            f"{latest_sent:+.3f}",
            delta=f"{latest['sentiment_delta'].iloc[0]:+.3f} vs prev day",
        )
    with col2:
        signal = "🟢 BULLISH" if pred_up else "🔴 BEARISH"
        st.metric("Tomorrow's Signal", signal, delta=f"Prob = {pred_prob:.1%}")
    with col3:
        mean_auc = cv_df["auc"].mean() if cv_df is not None else float("nan")
        st.metric("Mean CV AUC", f"{mean_auc:.3f}",
                  delta="[OK] Above target" if mean_auc > 0.55 else "Below target")
    with col4:
        st.metric("Post Volume (today)", f"{int(latest['post_volume'].iloc[0]):,}",
                  delta=f"7d avg: {latest['volume_7d_std'].iloc[0]:.0f} std")

    st.divider()

    # -- Row 2: Gauge + Sentiment trend ---------------------------------------
    col_g, col_t = st.columns([1, 2])
    with col_g:
        st.plotly_chart(sentiment_gauge(pred_prob), use_container_width=True)
    with col_t:
        st.plotly_chart(sentiment_trend_chart(df, days=days_trend), use_container_width=True)

    # -- Row 3: SPY chart ------------------------------------------------------
    st.plotly_chart(spy_chart(df, days=days_spy), use_container_width=True)

    # -- Row 4: CV AUC + Feature importance ------------------------------------
    col_cv, col_fi = st.columns(2)

    with col_cv:
        if cv_df is not None:
            st.plotly_chart(auc_fold_chart(cv_df), use_container_width=True)
        else:
            st.info("CV results not found -- run train_model.py")

    with col_fi:
        coefs = model.coef_[0]
        fi_df = pd.DataFrame({"Feature": features, "Coefficient": coefs}) \
                  .sort_values("Coefficient")
        colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in fi_df["Coefficient"]]
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Coefficient"], y=fi_df["Feature"],
            orientation="h", marker_color=colors,
        ))
        fig_fi.update_layout(**PLOTLY_LAYOUT, title="Feature Importance (Coefficients)", height=280)
        st.plotly_chart(fig_fi, use_container_width=True)

    # -- Row 5: Recent posts table ----------------------------------------------
    st.subheader("🗞️ Recent Reddit Posts")
    posts_df = load_raw_posts(days_back=7)
    if not posts_df.empty:
        show_cols = [c for c in ["date", "subreddit", "title", "score", "num_comments"] if c in posts_df.columns]
        top_posts = posts_df.sort_values("score", ascending=False).head(20)[show_cols]
        st.dataframe(top_posts, use_container_width=True, hide_index=True)
    else:
        st.info("No recent posts found -- run collect_reddit.py first.")

    # -- Footer ----------------------------------------------------------------
    st.divider()
    st.caption(
        f"Last updated: **{latest_date}** | "
        f"Model trained on {len(df)} days | "
        f"Data: r/{', r/'.join(config.SUBREDDITS)}"
    )


if __name__ == "__main__":
    if not STREAMLIT_AVAILABLE:
        print("Install streamlit first:  pip install streamlit plotly")
    else:
        main()
