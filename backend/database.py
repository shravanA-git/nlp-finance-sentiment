"""
database.py
-----------
SQLite-backed watchlist storage using Python's built-in sqlite3.
No external database server required — the db lives in watchlist.db.

Why SQLite: zero config, ships with Python, persists across server restarts,
and is exactly what you'd use in a real small-scale production app before
graduating to Postgres.
"""

import sqlite3
import os

DB_PATH = os.getenv("DB_PATH", "watchlist.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Called once at startup."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                symbol TEXT PRIMARY KEY,
                added_at TEXT DEFAULT (datetime('now'))
            )
        """)
        # Seed with defaults if empty
        existing = conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
        if existing == 0:
            for symbol in ["AAPL", "MSFT", "NVDA"]:
                conn.execute("INSERT OR IGNORE INTO watchlist (symbol) VALUES (?)", (symbol,))
        conn.commit()


def get_watchlist() -> list[str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT symbol FROM watchlist ORDER BY added_at").fetchall()
    return [row["symbol"] for row in rows]


def add_to_watchlist(symbol: str) -> bool:
    """Returns False if symbol already exists."""
    try:
        with get_conn() as conn:
            conn.execute("INSERT INTO watchlist (symbol) VALUES (?)", (symbol,))
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def remove_from_watchlist(symbol: str) -> bool:
    """Returns False if symbol wasn't in the list."""
    with get_conn() as conn:
        cursor = conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
        conn.commit()
    return cursor.rowcount > 0
