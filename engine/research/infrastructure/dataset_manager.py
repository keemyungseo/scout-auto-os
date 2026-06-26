"""SQLite history database + Parquet export."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    scan_kst TEXT PRIMARY KEY,
    scan_date TEXT NOT NULL,
    symbol_count INTEGER NOT NULL,
    regime_market TEXT,
    regime_volatility TEXT,
    regime_structure TEXT,
    regime_dynamics TEXT,
    regime_ecology TEXT,
    archived_at TEXT NOT NULL,
    label_ready INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_kst TEXT NOT NULL,
    symbol TEXT NOT NULL,
    rank_pred INTEGER,
    pred_score REAL,
    in_top50 INTEGER DEFAULT 0,
    features_json TEXT NOT NULL,
    max_up_4h REAL,
    UNIQUE(scan_kst, symbol)
);

CREATE TABLE IF NOT EXISTS forward_labels (
    scan_kst TEXT NOT NULL,
    symbol TEXT NOT NULL,
    return_2h REAL,
    return_minus_dd REAL,
    max_drawdown_2h REAL,
    max_up_2h REAL,
    min_return_2h REAL,
    mfe_2h REAL,
    mae_2h REAL,
    intrabar_sharpe REAL,
    sharpe_contribution REAL,
    return_30m REAL,
    return_1h REAL,
    return_4h REAL,
    labeled_at TEXT NOT NULL,
    PRIMARY KEY (scan_kst, symbol)
);

CREATE TABLE IF NOT EXISTS dataset_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_candidates_scan ON candidates(scan_kst);
CREATE INDEX IF NOT EXISTS idx_labels_scan ON forward_labels(scan_kst);
CREATE INDEX IF NOT EXISTS idx_scans_date ON scans(scan_date);
"""


class HistoryDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def set_meta(self, key: str, value: str) -> None:
        now = datetime.now(KST).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO dataset_meta (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, now),
            )

    def get_meta(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM dataset_meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else default

    def upsert_scan(
        self,
        scan_kst: str,
        scan_date: str,
        symbol_count: int,
        regimes: dict[str, str],
        label_ready: bool = False,
    ) -> None:
        now = datetime.now(KST).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scans
                (scan_kst, scan_date, symbol_count, regime_market, regime_volatility,
                 regime_structure, regime_dynamics, regime_ecology, archived_at, label_ready)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scan_kst, scan_date, symbol_count,
                    regimes.get("market_simple", ""),
                    regimes.get("volatility", ""),
                    regimes.get("structure", ""),
                    regimes.get("dynamics", ""),
                    regimes.get("market_ecology", ""),
                    now, 1 if label_ready else 0,
                ),
            )

    def upsert_candidate(
        self,
        scan_kst: str,
        symbol: str,
        rank_pred: int,
        pred_score: float,
        in_top50: bool,
        features: dict,
        max_up_4h: float | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO candidates
                (scan_kst, symbol, rank_pred, pred_score, in_top50, features_json, max_up_4h)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scan_kst, symbol, rank_pred, pred_score,
                    1 if in_top50 else 0,
                    json.dumps(features, ensure_ascii=False),
                    max_up_4h,
                ),
            )

    def upsert_label(self, scan_kst: str, symbol: str, labels: dict) -> None:
        now = datetime.now(KST).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO forward_labels
                (scan_kst, symbol, return_2h, return_minus_dd, max_drawdown_2h,
                 max_up_2h, min_return_2h, mfe_2h, mae_2h, intrabar_sharpe,
                 sharpe_contribution, return_30m, return_1h, return_4h, labeled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scan_kst, symbol,
                    labels.get("return_2h"),
                    labels.get("return_minus_dd"),
                    labels.get("max_drawdown_2h"),
                    labels.get("max_up_2h"),
                    labels.get("min_return_2h"),
                    labels.get("mfe_2h"),
                    labels.get("mae_2h"),
                    labels.get("intrabar_sharpe"),
                    labels.get("sharpe_contribution"),
                    labels.get("return_30m"),
                    labels.get("return_1h"),
                    labels.get("return_4h"),
                    now,
                ),
            )

    def scan_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]

    def sample_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]

    def labeled_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM forward_labels").fetchone()[0]

    def calendar_days(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MIN(scan_date) AS s, MAX(scan_date) AS e FROM scans",
            ).fetchone()
            if not row or not row["s"]:
                return 0
            s = datetime.strptime(row["s"], "%Y-%m-%d").date()
            e = datetime.strptime(row["e"], "%Y-%m-%d").date()
            return (e - s).days + 1

    def date_range(self) -> tuple[str, str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MIN(scan_date) AS s, MAX(scan_date) AS e FROM scans",
            ).fetchone()
            return (row["s"] or "", row["e"] or "")

    def regime_counts(self, column: str) -> dict[str, int]:
        allowed = {
            "regime_market", "regime_volatility", "regime_structure",
            "regime_dynamics", "regime_ecology",
        }
        if column not in allowed:
            return {}
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {column} AS r, COUNT(*) AS n FROM scans GROUP BY {column}",
            ).fetchall()
            return {r["r"]: r["n"] for r in rows if r["r"]}

    def scans_in_window(self, days: int) -> int:
        start, end = self.date_range()
        if not end:
            return 0
        end_d = datetime.strptime(end, "%Y-%m-%d").date()
        cut = (end_d - timedelta(days=days - 1)).isoformat()
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM scans WHERE scan_date >= ?", (cut,),
            ).fetchone()[0]

    def export_parquet(self, out_dir: Path) -> list[Path]:
        try:
            import pandas as pd
        except ImportError:
            return []

        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        with self._connect() as conn:
            for table in ("scans", "candidates", "forward_labels"):
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                p = out_dir / f"{table}.parquet"
                df.to_parquet(p, index=False)
                paths.append(p)

            scans_df = pd.read_sql_query("SELECT * FROM scans", conn)
            if not scans_df.empty:
                for period, freq in (("daily", "D"), ("weekly", "W"), ("monthly", "ME")):
                    sdf = scans_df.copy()
                    sdf["scan_date"] = pd.to_datetime(sdf["scan_date"])
                    agg = sdf.set_index("scan_date").resample(freq).agg({
                        "scan_kst": "count",
                        "symbol_count": "sum",
                        "label_ready": "sum",
                    }).rename(columns={"scan_kst": "scan_count"})
                    p = out_dir / f"calendar_{period}.parquet"
                    agg.reset_index().to_parquet(p, index=False)
                    paths.append(p)
        return paths

    def all_scan_keys(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT scan_kst FROM scans").fetchall()
            return {r["scan_kst"] for r in rows}
