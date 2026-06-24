"""Dedicated trade ledger — data/trades.db (V1.1 + V1.2 review fields)."""

from __future__ import annotations

import json
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
"""

V12_COLUMNS = [
    ("scan_rank", "INTEGER"),
    ("score", "REAL"),
    ("expected_ev", "REAL"),
    ("reason_1h", "TEXT"),
    ("reason_2h", "TEXT"),
    ("range_pct", "REAL"),
    ("top5_snapshot_id", "TEXT"),
    ("entry_context_json", "TEXT"),
    ("max_profit_pct_during_hold", "REAL"),
    ("max_drawdown_pct_during_hold", "REAL"),
    ("exit_quality_score", "REAL"),
    ("exit_comment", "TEXT"),
    ("missed_more_upside_pct_30m_after_exit", "REAL"),
    ("post_exit_30m_return", "REAL"),
    ("post_exit_2h_return", "REAL"),
]


class TradeRecordDB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(TRADES_TABLE_SQL)
        self.conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        existing = {r[1] for r in self.conn.execute("PRAGMA table_info(trades)")}
        for name, typ in V12_COLUMNS:
            if name not in existing:
                self.conn.execute(f"ALTER TABLE trades ADD COLUMN {name} {typ}")
        self.conn.commit()
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_status ON trades(symbol, status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)"
        )
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
        context: dict | None = None,
    ) -> int:
        ts = now_kst()
        ctx = context or {}
        cur = self.execute(
            """INSERT INTO trades
            (symbol, side, entry_time, entry_price, quantity, trade_size_usdt, leverage,
             entry_reason, status, created_at, updated_at,
             scan_rank, score, expected_ev, reason_1h, reason_2h, range_pct,
             top5_snapshot_id, entry_context_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                symbol.upper(), side.upper(), ts, entry_price, quantity,
                trade_size_usdt, leverage, entry_reason, "OPEN", ts, ts,
                ctx.get("scan_rank"), ctx.get("score"), ctx.get("expected_ev"),
                ctx.get("reason_1h"), ctx.get("reason_2h"), ctx.get("range_pct"),
                ctx.get("top5_snapshot_id"), json.dumps(ctx, ensure_ascii=False),
            ),
        )
        return int(cur.lastrowid)

    def update_entry_context(self, symbol: str, context: dict) -> int | None:
        row = self.fetchone(
            """SELECT id FROM trades WHERE symbol=? AND status='OPEN'
            ORDER BY entry_time DESC LIMIT 1""",
            (symbol.upper(),),
        )
        if not row:
            return None
        ts = now_kst()
        self.execute(
            """UPDATE trades SET
            scan_rank=?, score=?, expected_ev=?, reason_1h=?, reason_2h=?,
            range_pct=?, top5_snapshot_id=?, entry_context_json=?, updated_at=?
            WHERE id=?""",
            (
                context.get("scan_rank"), context.get("score"), context.get("expected_ev"),
                context.get("reason_1h"), context.get("reason_2h"), context.get("range_pct"),
                context.get("top5_snapshot_id"),
                json.dumps(context, ensure_ascii=False), ts, row["id"],
            ),
        )
        return int(row["id"])

    def close_latest_open(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        realized_pnl_pct: float,
        realized_pnl_usdt: float,
        status: str = "CLOSED",
        exit_analysis: dict | None = None,
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
        ea = exit_analysis or {}
        self.execute(
            """UPDATE trades SET
            exit_time=?, exit_price=?, exit_reason=?,
            realized_pnl_usdt=?, realized_pnl_pct=?, status=?, updated_at=?,
            max_profit_pct_during_hold=?, max_drawdown_pct_during_hold=?,
            exit_quality_score=?, exit_comment=?,
            missed_more_upside_pct_30m_after_exit=?, post_exit_30m_return=?, post_exit_2h_return=?
            WHERE id=?""",
            (
                ts, exit_price, exit_reason,
                realized_pnl_usdt, realized_pnl_pct, status, ts,
                ea.get("max_profit_pct_during_hold"),
                ea.get("max_drawdown_pct_during_hold"),
                ea.get("exit_quality_score"),
                ea.get("exit_comment"),
                ea.get("missed_more_upside_pct_30m_after_exit"),
                ea.get("post_exit_30m_return"),
                ea.get("post_exit_2h_return"),
                row["id"],
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

    def all_closed(self) -> list[dict]:
        return self.fetchall(
            """SELECT * FROM trades WHERE status IN ('CLOSED', 'CLOSED_BY_USER')
            ORDER BY exit_time DESC"""
        )

    def stats_since(self, since_kst: str | None = None) -> dict:
        if since_kst:
            closed = self.fetchall(
                """SELECT * FROM trades WHERE status IN ('CLOSED','CLOSED_BY_USER')
                AND exit_time >= ?""",
                (since_kst,),
            )
            entries = self.fetchall(
                "SELECT * FROM trades WHERE entry_time >= ?", (since_kst,),
            )
        else:
            closed = self.all_closed()
            entries = self.fetchall("SELECT * FROM trades ORDER BY entry_time")
        open_rows = self.open_trades()
        wins = [t for t in closed if (t.get("realized_pnl_usdt") or 0) > 0]
        losses = [t for t in closed if (t.get("realized_pnl_usdt") or 0) <= 0]
        realized = sum(t.get("realized_pnl_usdt") or 0 for t in closed)
        return {
            "total_entries": len(entries),
            "open_trades": len(open_rows),
            "closed_trades": len(closed),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": round(len(wins) / len(closed) * 100, 2) if closed else 0.0,
            "realized_pnl_usdt": round(realized, 4),
            "best_trade": max(closed, key=lambda x: x.get("realized_pnl_usdt") or 0, default=None),
            "worst_trade": min(closed, key=lambda x: x.get("realized_pnl_usdt") or 0, default=None),
        }
