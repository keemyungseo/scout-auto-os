"""SQLite table schemas and row helpers."""

from __future__ import annotations

TABLES_SQL = """
CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    source TEXT NOT NULL,
    engine TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL,
    unrealized_pnl_pct REAL DEFAULT 0,
    realized_pnl_pct REAL,
    status TEXT NOT NULL,
    manual_lock INTEGER DEFAULT 0,
    auto_manage INTEGER DEFAULT 1,
    exit_reason TEXT,
    last_update_time TEXT,
    a6_score REAL,
    expected_ev REAL,
    exit_plan TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    position_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    price REAL NOT NULL,
    quantity_pct REAL DEFAULT 100,
    pnl_pct REAL,
    fee_pct REAL DEFAULT 0,
    timestamp TEXT NOT NULL,
    reason TEXT,
    engine TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    rank INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    a6_score REAL,
    reason TEXT,
    entry_price REAL,
    expected_ev REAL,
    trend_alive TEXT,
    acceleration TEXT,
    volume_state TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    message TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS manual_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    details TEXT
);

CREATE TABLE IF NOT EXISTS daily_reports (
    report_date TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS engine_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    module TEXT NOT NULL,
    event_type TEXT NOT NULL,
    details TEXT
);
"""
