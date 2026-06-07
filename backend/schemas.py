"""
schemas.py
----------
Pydantic models defining the shape of every API request and response.
Keeping these separate makes it easy to version the API later.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


# ── Requests ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    headline: str = Field(..., min_length=3, max_length=1000,
                          example="Apple reports record quarterly earnings, beating estimates by 15%")


class WatchlistAddRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, example="AAPL")


# ── Core sentiment result (reused across responses) ───────────────────────────

class SentimentScore(BaseModel):
    label: str                          # "positive" | "neutral" | "negative"
    confidence: float                   # probability of the top label (0–1)
    scores: dict[str, float]            # all three class probabilities


class ScoredHeadline(BaseModel):
    headline: str
    source: Optional[str] = None
    published_at: Optional[str] = None  # ISO datetime string
    sentiment: SentimentScore


# ── Aggregate sentiment summary ───────────────────────────────────────────────

class SentimentAggregate(BaseModel):
    positive: int                       # count of positive headlines
    neutral: int
    negative: int
    total: int
    dominant: str                       # whichever label has the most
    score: float                        # weighted score: (pos - neg) / total, range -1 to 1


# ── Endpoint responses ────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    headline: str
    sentiment: SentimentScore


class TickerResponse(BaseModel):
    symbol: str
    company_name: Optional[str] = None
    headlines: list[ScoredHeadline]
    aggregate: SentimentAggregate


class TimelinePoint(BaseModel):
    date: str                           # "YYYY-MM-DD"
    positive: int
    neutral: int
    negative: int
    score: float                        # daily weighted score


class TimelineResponse(BaseModel):
    symbol: str
    days: int                           # how many days of data
    timeline: list[TimelinePoint]
    overall_aggregate: SentimentAggregate


class WatchlistItem(BaseModel):
    symbol: str
    aggregate: SentimentAggregate
    headline_count: int
    latest_headline: Optional[str] = None


class WatchlistResponse(BaseModel):
    tickers: list[WatchlistItem]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
