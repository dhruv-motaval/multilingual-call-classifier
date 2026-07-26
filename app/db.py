"""SQLite log of the actions the agent takes on each call.

No ORM on purpose. The point is to give the agent's tool calls a real,
inspectable side effect instead of just printing to stdout.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "tickets.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            reason TEXT,
            customer_name TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def log_action(action: str, reason: str = "", customer_name: str = "") -> int:
    init_db()
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO tickets (action, reason, customer_name, created_at) VALUES (?, ?, ?, ?)",
        (action, reason, customer_name, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    ticket_id = cur.lastrowid
    conn.close()
    return ticket_id


def fetch_all() -> list[dict]:
    init_db()
    conn = _connect()
    rows = conn.execute("SELECT * FROM tickets ORDER BY id").fetchall()
    conn.close()
    return [dict(row) for row in rows]
