"""
cache.py
--------
Simple in-memory TTL (time-to-live) cache for news fetches and scored results.

Why: FinBERT inference + NewsAPI calls together take ~5–10 seconds per ticker.
Caching means repeat requests within the TTL window return instantly.

TTL defaults:
  - News + scores: 15 minutes (prices and sentiment change, but not that fast)
  - Company name:  24 hours   (stable)
"""

import time
from typing import Any, Optional


class TTLCache:
    def __init__(self, ttl_seconds: int = 900):
        self._store: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        if key in self._store:
            value, expires_at = self._store[key]
            if time.time() < expires_at:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (value, time.time() + self._ttl)

    def clear(self) -> None:
        self._store.clear()

    def size(self) -> int:
        # Prune expired entries and return count
        now = time.time()
        self._store = {k: v for k, v in self._store.items() if v[1] > now}
        return len(self._store)


# Module-level singletons
news_cache    = TTLCache(ttl_seconds=900)   # 15 min
name_cache    = TTLCache(ttl_seconds=86400) # 24 hours
backtest_cache = TTLCache(ttl_seconds=1800)  # 30 min
