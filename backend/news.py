"""
news.py
-------
Fetches financial news headlines from two sources:

Primary:   NewsAPI (newsapi.org) — requires free API key, gives 30-day history
Fallback:  yfinance               — no key needed, ~10 recent headlines per ticker

The fetch_headlines() function tries NewsAPI first. If no key is configured
or the request fails, it falls back to yfinance automatically.
"""

import os
import requests
import yfinance as yf
from datetime import datetime, timedelta, timezone
from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # loads .env from the current working directory

from cache import news_cache, name_cache

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  # set in .env
NEWSAPI_URL = "https://newsapi.org/v2/everything"


# ── NewsAPI ───────────────────────────────────────────────────────────────────

def fetch_newsapi(query: str, days: int = 30, max_articles: int = 50) -> list[dict]:
    """
    Fetch headlines from NewsAPI for a given query string.
    Returns a list of dicts with keys: headline, source, published_at
    """
    if not NEWSAPI_KEY:
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    params = {
        "q": query,
        "sortBy": "publishedAt",
        "from": from_date,
        "language": "en",
        "pageSize": min(max_articles, 100),
        "apiKey": NEWSAPI_KEY,
    }

    try:
        response = requests.get(NEWSAPI_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])

        return [
            {
                "headline": a["title"],
                "source": a.get("source", {}).get("name"),
                "published_at": a.get("publishedAt"),
            }
            for a in articles
            if a.get("title") and a["title"] != "[Removed]"
        ]
    except Exception as e:
        print(f"NewsAPI error for '{query}': {e}")
        return []


# ── yfinance fallback ─────────────────────────────────────────────────────────

def fetch_yfinance(symbol: str) -> list[dict]:
    """
    Fetch recent news from yfinance.
    Returns ~10 recent headlines. No date filtering — timestamps vary by ticker.
    """
    try:
        ticker = yf.Ticker(symbol)
        news_items = ticker.news or []

        results = []
        for item in news_items:
            title = item.get("title", "")
            if not title:
                continue

            # Convert unix timestamp to ISO string if available
            published_at = None
            ts = item.get("providerPublishTime")
            if ts:
                published_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

            results.append({
                "headline": title,
                "source": item.get("publisher"),
                "published_at": published_at,
            })

        return results
    except Exception as e:
        print(f"yfinance error for '{symbol}': {e}")
        return []


# ── Company name lookup ───────────────────────────────────────────────────────

def get_company_name(symbol: str) -> Optional[str]:
    """Returns the long name for a ticker symbol, or None if not found. Cached 24h."""
    cached = name_cache.get(symbol)
    if cached is not None:
        return cached
    try:
        info = yf.Ticker(symbol).info
        name = info.get("longName") or info.get("shortName")
        name_cache.set(symbol, name)
        return name
    except Exception:
        return None


# ── Unified fetch interface ───────────────────────────────────────────────────

def fetch_headlines(symbol: str, days: int = 30, max_articles: int = 50) -> list[dict]:
    """
    Main entry point. Tries NewsAPI first (rich history), falls back to yfinance.
    Always returns a list of dicts: {headline, source, published_at}
    """
    # NewsAPI: search by ticker symbol + company name for better coverage
    company_name = get_company_name(symbol)
    query = f"{symbol} {company_name}" if company_name else symbol

    cache_key = f"{symbol}:{days}:{max_articles}"
    cached = news_cache.get(cache_key)
    if cached is not None:
        print(f"Cache hit for {symbol}")
        return cached

    articles = fetch_newsapi(query, days=days, max_articles=max_articles)

    if articles:
        print(f"NewsAPI: {len(articles)} articles for {symbol}")
    else:
        articles = fetch_yfinance(symbol)
        print(f"yfinance fallback: {len(articles)} articles for {symbol}")

    news_cache.set(cache_key, articles)
    return articles
