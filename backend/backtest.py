"""
backtest.py
-----------
Correlates daily sentiment scores against next-day stock price returns.

Methodology (same as quant funds analyzing alternative data):
  1. For each trading day, compute an aggregate sentiment score from that day's headlines
  2. Fetch OHLCV price data for the same period
  3. Align sentiment score (day T) with next-day return (day T+1)
  4. Compute Pearson correlation and identify signal days

This is the module that turns the project from a classifier into signal research.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional


def get_price_returns(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily closing prices and compute next-day returns.
    Returns a DataFrame with columns: date, close, next_day_return
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end)

    if hist.empty:
        return pd.DataFrame()

    hist = hist[["Close"]].copy()
    hist.index = hist.index.tz_localize(None)
    hist["date"] = hist.index.strftime("%Y-%m-%d")
    hist["close"] = hist["Close"].round(2)

    # Next-day return: (close[T+1] - close[T]) / close[T]
    hist["next_day_return"] = hist["Close"].pct_change().shift(-1).round(4)

    return hist[["date", "close", "next_day_return"]].dropna()


def compute_backtest(
    symbol: str,
    timeline: list[dict],  # list of TimelinePoint dicts from the /timeline endpoint
) -> dict:
    """
    Given a sentiment timeline and a symbol, compute backtesting statistics.

    Returns a dict with:
      - daily_data: list of {date, sentiment_score, close, next_day_return, return_pct}
      - correlation: Pearson r between sentiment score and next-day return
      - signal_days: top 5 days where sentiment most strongly preceded a move
      - stats: summary statistics
    """
    if not timeline:
        return {"error": "No timeline data provided."}

    # Date range from timeline
    dates = [p["date"] for p in timeline]
    start = dates[0]
    # Extend end by 2 days to capture the next-day return for the last sentiment day
    end = (datetime.strptime(dates[-1], "%Y-%m-%d") + timedelta(days=3)).strftime("%Y-%m-%d")

    price_df = get_price_returns(symbol, start=start, end=end)
    if price_df.empty:
        return {"error": f"Could not fetch price data for {symbol}."}

    # Build sentiment score lookup by date
    sentiment_by_date = {p["date"]: p["score"] for p in timeline}

    # Merge: only keep days where we have BOTH sentiment and a next-day return
    merged = []
    for _, row in price_df.iterrows():
        date_str = row["date"]
        if date_str in sentiment_by_date:
            score = sentiment_by_date[date_str]
            ret = float(row["next_day_return"])
            merged.append({
                "date": date_str,
                "sentiment_score": round(score, 3),
                "close": float(row["close"]),
                "next_day_return": ret,
                "return_pct": round(ret * 100, 2),
            })

    if len(merged) < 3:
        return {"error": "Not enough overlapping data points for backtesting. Try a longer date range."}

    scores  = np.array([d["sentiment_score"] for d in merged])
    returns = np.array([d["next_day_return"]  for d in merged])

    # Pearson correlation
    correlation = float(np.corrcoef(scores, returns)[0, 1])

    # Signal days: where sentiment score had largest absolute value and return confirmed direction
    # A "confirmed" signal = sentiment positive + positive next-day return, or negative + negative
    signal_days = []
    for d in merged:
        s = d["sentiment_score"]
        r = d["next_day_return"]
        # Only include days where signal and return agree in direction
        if (s > 0.1 and r > 0) or (s < -0.1 and r < 0):
            signal_days.append({**d, "signal_strength": round(abs(s) * abs(r), 4)})

    signal_days = sorted(signal_days, key=lambda x: x["signal_strength"], reverse=True)[:5]

    # Summary stats
    positive_days = [d for d in merged if d["sentiment_score"] > 0.1]
    negative_days = [d for d in merged if d["sentiment_score"] < -0.1]

    avg_return_after_positive = np.mean([d["next_day_return"] for d in positive_days]) if positive_days else 0
    avg_return_after_negative = np.mean([d["next_day_return"] for d in negative_days]) if negative_days else 0

    stats = {
        "total_days": len(merged),
        "positive_sentiment_days": len(positive_days),
        "negative_sentiment_days": len(negative_days),
        "avg_next_day_return_after_positive": round(float(avg_return_after_positive) * 100, 3),
        "avg_next_day_return_after_negative": round(float(avg_return_after_negative) * 100, 3),
        "signal_accuracy": round(len(signal_days) / max(len(positive_days) + len(negative_days), 1), 3),
    }

    return {
        "symbol": symbol,
        "correlation": round(correlation, 3),
        "correlation_strength": _correlation_label(correlation),
        "daily_data": merged,
        "signal_days": signal_days,
        "stats": stats,
    }


def _correlation_label(r: float) -> str:
    """Human-readable interpretation of Pearson r."""
    abs_r = abs(r)
    direction = "positive" if r > 0 else "negative"
    if abs_r >= 0.5:
        return f"Strong {direction} correlation"
    elif abs_r >= 0.3:
        return f"Moderate {direction} correlation"
    elif abs_r >= 0.1:
        return f"Weak {direction} correlation"
    else:
        return "No meaningful correlation"
