"""Dedicated trade ledger — data/trades.db (V1.1)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from scout_auto_os.storage.db import now_kst

TRADES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity REAL NOT NULL,
    trade_size_usdt REAL NOT NULL,
    leverage INTEGER NOT NULL DEFAULT 1,
    entry_reason TEXT,
    exit_time TEXT,
    exit_price REAL,
    exit_reason TEXT,
    realized_pnl_usdt REAL,
    realized_pnl_pct REAL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_status ON trades(symbol, status);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);
"""


class TradeRecordDB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(TRADES_TABLE_SQL)
        self.conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self.conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        cur = self.conn.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    def insert_open(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        trade_size_usdt: float,
        leverage: int,
        entry_reason: str,
    ) -> int:
        ts = now_kst()
        cur = self.execute(
            """INSERT INTO trades
            (symbol, side, entry_time, entry_price, quantity, trade_size_usdt, leverage,
             entry_reason, status, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                symbol.upper(), side.upper(), ts, entry_price, quantity,
                trade_size_usdt, leverage, entry_reason, "OPEN", ts, ts,
            ),
        )
        return int(cur.lastrowid)

    def close_latest_open(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        realized_pnl_pct: float,
        realized_pnl_usdt: float,
        status: str = "CLOSED",
    ) -> int | None:
        row = self.fetchone(
            """SELECT id FROM trades
            WHERE symbol=? AND status='OPEN'
            ORDER BY entry_time DESC LIMIT 1""",
            (symbol.upper(),),
        )
        if not row:
            return None
        ts = now_kst()
        self.execute(
            """UPDATE trades SET
            exit_time=?, exit_price=?, exit_reason=?,
            realized_pnl_usdt=?, realized_pnl_pct=?, status=?, updated_at=?
            WHERE id=?""",
            (
                ts, exit_price, exit_reason,
                realized_pnl_usdt, realized_pnl_pct, status, ts, row["id"],
            ),
        )
        return int(row["id"])

    def open_trades(self) -> list[dict]:
        return self.fetchall(
            "SELECT * FROM trades WHERE status='OPEN' ORDER BY entry_time"
        )

    def closed_between(self, start_kst: str, end_kst: str) -> list[dict]:
        return self.fetchall(
            """SELECT * FROM trades
            WHERE status IN ('CLOSED', 'CLOSED_BY_USER')
            AND exit_time >= ? AND exit_time < ?
            ORDER BY exit_time""",
            (start_kst, end_kst),
        )
