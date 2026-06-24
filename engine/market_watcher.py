"""Market data adapter + scan scheduler."""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from scout_auto_os.engine.strategy_core import run_a6_scan
from scout_auto_os.storage.db import Database, now_kst

if TYPE_CHECKING:
    from scout_auto_os.engine.live_data import LiveDataEngine
    from scout_auto_os.engine.expected_ev_engine import ExpectedEVLogger

KST = timezone(timedelta(hours=9))


class DataAdapter(ABC):
    @abstractmethod
    def get_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    def get_bars(self, symbol: str, scan_kst: str) -> list:
        ...


class MockAdapter(DataAdapter):
    """Uses phase19 kline cache."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def get_price(self, symbol: str) -> float:
        return 0.0

    def get_bars(self, symbol: str, scan_kst: str) -> list:
        from scout_research_r006_pilot_execution_engine import load_forward_bars
        return load_forward_bars(symbol, scan_kst)


class CacheAdapter(DataAdapter):
    """Phase19 cache fallback."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        import scout_phase19_winner_ranking_dna as p19
        import scout_phase16_human_blind_test as p16
        p19.CACHE_DIR = cache_dir
        p16.CACHE_DIR = cache_dir

    def get_price(self, symbol: str) -> float:
        from scout_research_r006_pilot_execution_engine import load_forward_bars
        now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
        bars = load_forward_bars(symbol, now)
        return bars[-1].c if bars else 0.0

    def get_bars(self, symbol: str, scan_kst: str) -> list:
        from scout_research_r006_pilot_execution_engine import load_forward_bars
        return load_forward_bars(symbol, scan_kst)


class LiveDataAdapter(DataAdapter):
    """Live websocket + REST adapter (R014). Falls back to cache."""

    def __init__(self, live_engine: LiveDataEngine, cache_dir: Path) -> None:
        self.live = live_engine
        self._fallback = CacheAdapter(cache_dir)

    def get_price(self, symbol: str) -> float:
        if self.live.enabled:
            px = self.live.get_price(symbol)
            if px > 0:
                return px
        return self._fallback.get_price(symbol)

    def get_bars(self, symbol: str, scan_kst: str) -> list:
        if self.live.enabled:
            bars = self.live.get_forward_bars(symbol, scan_kst)
            if bars:
                return bars
        return self._fallback.get_bars(symbol, scan_kst)


# Backward compatibility alias
LiveAdapter = CacheAdapter


class MarketWatcher:
    def __init__(
        self,
        config: dict,
        db: Database,
        csv_dir: Path,
        adapter: DataAdapter,
        live_engine: LiveDataEngine | None = None,
        ev_logger: ExpectedEVLogger | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.csv_dir = csv_dir
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.adapter = adapter
        self.live_engine = live_engine
        self.ev_logger = ev_logger
        self.cache_dir = Path(config["data"]["kline_cache_dir"])
        self.last_scan_time: str | None = None
        self.last_top5: list[dict] = []

    def run_scan(self, scan_kst: str | None = None, paper_fast: bool = False) -> list[dict]:
        if scan_kst is None:
            scan_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
        top_n = int(self.config["long_engine"].get("max_candidates_alert", 5))
        max_sym = 80 if paper_fast else 0
        top5 = run_a6_scan(
            scan_kst,
            self.cache_dir,
            top_n=top_n,
            max_symbols=max_sym,
            live_engine=self.live_engine,
            ev_logger=self.ev_logger,
        )
        ts = now_kst()
        self.last_scan_time = ts
        self.last_top5 = top5

        for row in top5:
            self.db.execute(
                """INSERT INTO candidates
                (timestamp, rank, symbol, a6_score, reason, entry_price, expected_ev,
                 trend_alive, acceleration, volume_state)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    ts, row["rank"], row["symbol"], row["a6_score"], row["reason"],
                    row["entry_price"], row["expected_ev"], row["trend_alive"],
                    row["acceleration"], row["volume_state"],
                ),
            )

        path = self.csv_dir / "top5_candidates.csv"
        write = not path.exists()
        fields = [
            "timestamp", "rank", "symbol", "a6_score", "reason", "entry_price",
            "expected_ev", "remaining_ev", "trend_alive", "acceleration", "volume_state",
        ]
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if write:
                w.writeheader()
            for row in top5:
                w.writerow({"timestamp": ts, **row})

        self.db.log_event("market_watcher", "scan_complete", {"count": len(top5), "scan_kst": scan_kst})
        return top5
