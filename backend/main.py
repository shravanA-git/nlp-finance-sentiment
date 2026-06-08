"""
main.py
-------
FastAPI backend for the NLP Finance Sentiment Dashboard.

Endpoints:
  POST /analyze                    — score a single headline
  GET  /ticker/{symbol}            — live headlines + aggregate sentiment
  GET  /ticker/{symbol}/timeline   — daily sentiment breakdown (30 days)
  GET  /watchlist                  — score all tracked tickers
  POST /watchlist                  — add a ticker to the watchlist
  DELETE /watchlist/{symbol}       — remove a ticker from the watchlist
  GET  /health                     — server + model status check

Run locally:
  uvicorn main:app --reload --port 8000

Then open: http://localhost:8000/docs  (interactive API explorer, free with FastAPI)
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
from datetime import datetime, timezone

from backtest import compute_backtest
from database import init_db, get_watchlist, add_to_watchlist, remove_from_watchlist
from schemas import (
    AnalyzeRequest, AnalyzeResponse,
    TickerResponse, ScoredHeadline,
    TimelineResponse, TimelinePoint,
    WatchlistResponse, WatchlistItem, WatchlistAddRequest,
    SentimentAggregate, ErrorResponse,
)
from model import SentimentModel, get_model
from news import fetch_headlines, get_company_name


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NLP Finance Sentiment API",
    description="BERT-based financial news sentiment analysis. Built by Shravan Anand.",
    version="1.0.0",
)

# Allow the React frontend (any localhost port during dev) to call this API.
# In production, replace "*" with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Watchlist is now backed by SQLite via database.py


# ── Startup: load model eagerly so first request isn't slow ──────────────────

@app.on_event("startup")
async def startup_event():
    """Load model and initialize database at startup."""
    init_db()
    get_model()


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_aggregate(scored: list[ScoredHeadline]) -> SentimentAggregate:
    """Compute aggregate sentiment stats from a list of scored headlines."""
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for item in scored:
        counts[item.sentiment.label] += 1

    total = sum(counts.values()) or 1
    dominant = max(counts, key=counts.get)
    # Weighted score: ranges from -1 (all negative) to +1 (all positive)
    score = (counts["positive"] - counts["negative"]) / total

    return SentimentAggregate(
        positive=counts["positive"],
        neutral=counts["neutral"],
        negative=counts["negative"],
        total=total,
        dominant=dominant,
        score=round(score, 3),
    )


def score_headlines(raw: list[dict], model: SentimentModel) -> list[ScoredHeadline]:
    """Take raw news dicts and return scored headline objects."""
    results = []
    for item in raw:
        sentiment = model.predict(item["headline"])
        results.append(ScoredHeadline(
            headline=item["headline"],
            source=item.get("source"),
            published_at=item.get("published_at"),
            sentiment=sentiment,
        ))
    return results


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/backtest/{symbol}")
def get_backtest(
    symbol: str,
    days: int = 30,
    model: SentimentModel = Depends(get_model),
):
    """
    Correlate daily sentiment scores against next-day price returns.

    Fetches headline sentiment for each day in the window, aligns with
    price data from yfinance, and computes Pearson correlation + signal days.

    This is the same methodology quant funds use when evaluating alternative data.
    """
    symbol = symbol.upper()

    # Reuse timeline logic to get daily sentiment
    raw = fetch_headlines(symbol, days=days, max_articles=100)
    if not raw:
        raise HTTPException(status_code=404, detail=f"No news found for {symbol}.")

    seen = set()
    raw = [a for a in raw if not (a["headline"] in seen or seen.add(a["headline"]))]

    scored = score_headlines(raw, model)

    # Build timeline (same logic as /timeline endpoint)
    from collections import defaultdict
    by_date = defaultdict(list)
    for item in scored:
        if item.published_at:
            try:
                dt = datetime.fromisoformat(item.published_at.replace("Z", "+00:00"))
                by_date[dt.strftime("%Y-%m-%d")].append(item)
            except ValueError:
                pass

    timeline_dicts = []
    for date_str in sorted(by_date.keys()):
        agg = build_aggregate(by_date[date_str])
        timeline_dicts.append({"date": date_str, "score": agg.score})

    result = compute_backtest(symbol, timeline_dicts)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@app.get("/")
def root():
    return {"name": "FinSentiment API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health_check():
    """Check that the server and model are running."""
    return {"status": "ok", "model": "ProsusAI/finbert"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_headline(
    request: AnalyzeRequest,
    model: SentimentModel = Depends(get_model),
):
    """
    Score a single financial headline.

    Example request body:
        {"headline": "Apple reports record quarterly earnings"}

    Returns the sentiment label and confidence scores for all three classes.
    """
    sentiment = model.predict(request.headline)
    return AnalyzeResponse(headline=request.headline, sentiment=sentiment)


@app.get("/ticker/{symbol}", response_model=TickerResponse)
def get_ticker_sentiment(
    symbol: str,
    days: int = 7,
    max_articles: int = 20,
    model: SentimentModel = Depends(get_model),
):
    """
    Fetch recent news for a ticker and return scored headlines + aggregate.

    - symbol: stock ticker (e.g. AAPL, TSLA, NVDA)
    - days: how many days back to search (default 7, max 30 with NewsAPI)
    - max_articles: max headlines to return (default 20)
    """
    symbol = symbol.upper()
    raw = fetch_headlines(symbol, days=days, max_articles=max_articles)

    if not raw:
        raise HTTPException(
            status_code=404,
            detail=f"No news found for {symbol}. Check the ticker symbol."
        )

    # Deduplicate by headline text
    seen = set()
    raw = [a for a in raw if not (a["headline"] in seen or seen.add(a["headline"]))]

    scored = score_headlines(raw, model)
    aggregate = build_aggregate(scored)
    company_name = get_company_name(symbol)

    return TickerResponse(
        symbol=symbol,
        company_name=company_name,
        headlines=scored,
        aggregate=aggregate,
    )


@app.get("/ticker/{symbol}/timeline", response_model=TimelineResponse)
def get_ticker_timeline(
    symbol: str,
    days: int = 30,
    model: SentimentModel = Depends(get_model),
):
    """
    Return daily sentiment breakdown for a ticker over the past N days.
    Requires NewsAPI key for full history (yfinance only has ~10 recent articles).

    Each point in the timeline has:
    - date: "YYYY-MM-DD"
    - positive / neutral / negative counts
    - score: weighted daily sentiment (-1 to +1)
    """
    symbol = symbol.upper()
    raw = fetch_headlines(symbol, days=days, max_articles=100)

    if not raw:
        raise HTTPException(status_code=404, detail=f"No news found for {symbol}.")

    scored = score_headlines(raw, model)

    # Group by calendar date
    by_date: dict[str, list[ScoredHeadline]] = defaultdict(list)
    undated_count = 0

    for item in scored:
        if item.published_at:
            try:
                # Parse ISO datetime and extract date string
                dt = datetime.fromisoformat(item.published_at.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
                by_date[date_str].append(item)
            except ValueError:
                undated_count += 1
        else:
            undated_count += 1

    if undated_count > 0:
        print(f"{undated_count} headlines had no publish date and were excluded from the timeline.")

    # Build timeline sorted by date ascending
    timeline = []
    for date_str in sorted(by_date.keys()):
        day_headlines = by_date[date_str]
        agg = build_aggregate(day_headlines)
        timeline.append(TimelinePoint(
            date=date_str,
            positive=agg.positive,
            neutral=agg.neutral,
            negative=agg.negative,
            score=agg.score,
        ))

    overall = build_aggregate(scored)

    return TimelineResponse(
        symbol=symbol,
        days=days,
        timeline=timeline,
        overall_aggregate=overall,
    )


@app.get("/watchlist", response_model=WatchlistResponse)
def watchlist_get(model: SentimentModel = Depends(get_model)):
    """Return current sentiment for all tickers in the watchlist."""
    items = []
    for symbol in get_watchlist():
        raw = fetch_headlines(symbol, days=7, max_articles=10)
        if not raw:
            continue
        seen = set()
        raw = [a for a in raw if not (a["headline"] in seen or seen.add(a["headline"]))]
        scored = score_headlines(raw, model)
        aggregate = build_aggregate(scored)
        items.append(WatchlistItem(
            symbol=symbol,
            aggregate=aggregate,
            headline_count=len(scored),
            latest_headline=scored[0].headline if scored else None,
        ))
    return WatchlistResponse(tickers=items)


@app.post("/watchlist", status_code=201)
def watchlist_add(request: WatchlistAddRequest):
    """Add a ticker to the watchlist."""
    symbol = request.symbol.upper()
    if not add_to_watchlist(symbol):
        raise HTTPException(status_code=409, detail=f"{symbol} is already in your watchlist.")
    return {"message": f"{symbol} added.", "watchlist": get_watchlist()}


@app.delete("/watchlist/{symbol}")
def watchlist_remove(symbol: str):
    """Remove a ticker from the watchlist."""
    symbol = symbol.upper()
    if not remove_from_watchlist(symbol):
        raise HTTPException(status_code=404, detail=f"{symbol} not found in watchlist.")
    return {"message": f"{symbol} removed.", "watchlist": get_watchlist()}
