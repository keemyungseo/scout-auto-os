"""
SCOUT Research Engine V1 — virtual candidate evaluation, no orders.

Runs in a separate daemon thread. Never calls order execution APIs.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from scout_auto_os.engine.research.feature_league import compute_feature_league
from scout_auto_os.engine.research.formula_league import compute_formula_league, random_baseline_stats
from scout_auto_os.engine.research.forward_tracker import ForwardTracker
from scout_auto_os.engine.research.position_evolution import PositionEvolutionStore
from scout_auto_os.engine.research.report import build_daily_report_payload, build_research_report_text
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.scanner import run_research_scan
from scout_auto_os.engine.research.settings import research_enabled
from scout_auto_os.engine.research.state_evolution import analyze_evolution
from scout_auto_os.engine.research.state_league import compute_state_league
from scout_auto_os.engine.research.storage import ResearchStore
from scout_auto_os.engine.research.zero_base.runner import ZeroBaseRunner
from scout_auto_os.engine.research.zero_base.storage import ZeroBaseStore
from scout_auto_os.storage.db import now_kst

KST = timezone(timedelta(hours=9))

FORBIDDEN_IMPORTS = (
    "create_order",
    "futures_create_order",
    "market_buy",
    "market_sell",
    "paper_entry",
    "_live_entry",
)


class ResearchEngine:
    """Background research loop — fully isolated from LIVE order path."""

    def __init__(
        self,
        config: dict,
        data_dir: Path,
        cache_dir: Path,
        price_fn: Callable[[str], float],
        live_engine=None,
        live_top5_fn: Callable[[], list[str]] | None = None,
        telegram_send_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self.config = config
        rcfg = config.get("research", {})
        self.enabled = research_enabled(config)
        self.scan_interval_sec = int(rcfg.get("scan_interval_min", 5)) * 60
        self.top_n = int(rcfg.get("top_n", 20))
        self.workers = int(rcfg.get("workers", 8))
        self.max_symbols = int(rcfg.get("max_symbols", 0))
        self.symbol_list = list(rcfg.get("scan_symbols") or [])
        self.forward_tick_sec = int(rcfg.get("forward_tick_sec", 60))
        self.rest_base = config.get("live_data", {}).get("rest_base", "https://fapi.binance.com")

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.price_fn = price_fn
        self.live_engine = live_engine
        self.live_top5_fn = live_top5_fn or (lambda: [])
        self.telegram_send = telegram_send_fn

        self.store = ResearchStore(data_dir)
        self.forward = ForwardTracker(self.store.root / "forward_pending.jsonl")
        self.evolution = PositionEvolutionStore(self.store.root)
        self.zero_base = ZeroBaseStore(data_dir)
        zb_cfg = config.get("research", {}).get("zero_base", {})
        self.zero_base_enabled = str(os.getenv("ZEROBASE_ENABLED", zb_cfg.get("enabled", "true"))).lower() in ("1", "true", "yes")
        self.zero_base_max_scans = int(zb_cfg.get("max_scans_per_refresh", 20))
        self.zero_base_random_draws = int(zb_cfg.get("random_draws", 100))
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_scan_time: str | None = None
        self._last_report_date: str | None = None
        self._last_zero_base_date: str | None = None
        self._scan_count = 0
        self._lock = threading.Lock()
        print(
            "[RESEARCH] init "
            f"enabled={self.enabled} "
            f"env={os.getenv('RESEARCH_ENABLED', '(unset)')} "
            f"interval_sec={self.scan_interval_sec} top_n={self.top_n}"
        )

    def start(self) -> None:
        if not self.enabled:
            print(
                "[RESEARCH] engine disabled — set RESEARCH_ENABLED=true in container .env "
                f"(current env={os.getenv('RESEARCH_ENABLED', '(unset)')})"
            )
            return
        try:
            self._assert_no_order_access()
        except Exception as exc:
            print(f"[RESEARCH FATAL] order guard failed: {exc}")
            traceback.print_exc()
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop_wrapper, daemon=True, name="ResearchEngine")
        self._thread.start()
        print("[RESEARCH] engine started")

    def stop(self) -> None:
        self._running = False

    @staticmethod
    def _assert_no_order_access() -> None:
        """Document-level guard — research modules must not import execution."""
        research_dir = Path(__file__).resolve().parent / "research"
        tokens = FORBIDDEN_IMPORTS
        for path in [Path(__file__), *research_dir.glob("*.py")]:
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#") or "FORBIDDEN_IMPORTS" in line:
                    continue
                for token in tokens:
                    if token in line and ("import" in line or "(" in line):
                        raise RuntimeError(f"Research must not reference order function: {token} in {path.name}")

    def snapshot(self) -> dict:
        with self._lock:
            snap = self.store.snapshot()
            snap.update({
                "enabled": self.enabled,
                "last_scan_time": self._last_scan_time,
                "scan_count": self._scan_count,
                "forward_pending": self.forward.pending_count,
                "zero_base": self.zero_base.snapshot(),
            })
            return snap

    def _loop_wrapper(self) -> None:
        try:
            print(f"[RESEARCH] thread loop starting forward_pending={self.forward.pending_count}")
            self._loop()
        except Exception as exc:
            print(f"[RESEARCH FATAL] thread crashed: {exc}")
            traceback.print_exc()

    def _loop(self) -> None:
        next_scan = time.time()
        next_forward = time.time()
        # First scan soon after boot (do not wait full interval)
        next_scan = time.time() + 5
        while self._running:
            try:
                now = time.time()
                if now >= next_forward:
                    self._tick_forward()
                    next_forward = now + self.forward_tick_sec
                if now >= next_scan:
                    self.run_scan_once()
                    next_scan = now + self.scan_interval_sec
                self._maybe_daily_report()
            except Exception as exc:
                print(f"[RESEARCH ERROR] loop: {exc}")
                traceback.print_exc()
            time.sleep(1)

    @research_safe("forward_tick")
    def _tick_forward(self) -> None:
        pending_before = self.forward.pending_count
        completed = self.forward.update(self.price_fn)
        pending_after = self.forward.pending_count
        print(f"[RESEARCH] forward pending={pending_after} (completed={len(completed)})")
        if completed:
            self.store.append_forward(completed)
            self._refresh_leagues()
            print("[RESEARCH] forward result updated")

    @research_safe("scan")
    def run_scan_once(self, scan_kst: str | None = None) -> dict | None:
        scan_kst = scan_kst or now_kst()
        live_syms = set(self.live_top5_fn())

        result = run_research_scan(
            scan_kst,
            self.cache_dir,
            self.rest_base,
            top_n=self.top_n,
            workers=self.workers,
            live_engine=self.live_engine,
            max_symbols=self.max_symbols,
            symbol_list=self.symbol_list or None,
        )
        if not result.get("candidates"):
            print(f"[RESEARCH] scan complete — no candidates (symbols={result.get('total_symbols', 0)})")
            return result

        for c in result["candidates"]:
            c["selected_by_live_engine"] = c["symbol"] in live_syms

        self.store.append_scan({
            "scan_time_kst": result["scan_time_kst"],
            "total_symbols": result["total_symbols"],
            "market_regime": result["market_regime"],
            "btc_1h_return": result["btc_1h_return"],
            "btc_4h_return": result["btc_4h_return"],
            "alt_market_strength": result["alt_market_strength"],
            "top20_symbols": result["top20_symbols"],
        })
        self.store.append_candidates(result["candidates"])
        self.store.append_formula_picks(scan_kst, result.get("formula_picks", {}))

        placeholders = []
        for c in result["candidates"]:
            self.forward.register(
                c["scan_time_kst"], c["symbol"], c["rank"], float(c["current_price"]),
            )
            placeholders.append(
                self.forward.placeholders(
                    c["scan_time_kst"], c["symbol"], c["rank"], float(c["current_price"]),
                )
            )
        self.store.append_forward(placeholders)

        with self._lock:
            self._last_scan_time = scan_kst
            self._scan_count += 1

        print(f"[RESEARCH] scan saved top_n={len(result['candidates'])} forward_pending={self.forward.pending_count}")
        return result

    @research_safe("leagues")
    def _refresh_leagues(self) -> None:
        forward_rows = self.store.read_forward_all()
        complete = [r for r in forward_rows if r.get("return_2h") not in ("", None)]
        if not complete:
            return

        picks: list[dict] = []
        if self.store.picks_path.exists():
            for line in self.store.picks_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    picks.append(json.loads(line))

        formula = compute_formula_league(complete, picks)
        self.store.write_formula_league(formula)
        print("[RESEARCH] formula league updated")

        candidates = self._read_candidates()
        features = compute_feature_league(candidates, complete)
        self.store.write_feature_league(features)
        print("[RESEARCH] feature league updated")

        state_league, blind_state = compute_state_league(complete, self.cache_dir)
        if state_league:
            self.store.write_state_league(state_league)
            print(f"[RESEARCH] state league updated n_formulas={len(state_league)}")

        record_time = now_kst()
        live_n = self.evolution.ingest_live_reviews(self.data_dir, record_time)
        replay_n = self.evolution.build_replay_checkpoints(complete, record_time)
        if live_n or replay_n:
            print(f"[RESEARCH] evolution ingested live={live_n} replay={replay_n}")

        evolution = analyze_evolution(self.evolution.read_all())
        self.store.write_state_proposals(evolution)

        report = build_daily_report_payload(
            self.store, formula, features, complete, len(candidates),
            state_league=state_league,
            state_evolution=evolution,
            blind_state=blind_state,
            zero_base=self.zero_base.snapshot(),
        )
        self.store.write_report(report)

    @research_safe("zerobase")
    def _refresh_zero_base(self) -> dict:
        if not self.zero_base_enabled:
            return {}
        runner = ZeroBaseRunner(
            self.data_dir,
            random_draws=self.zero_base_random_draws,
        )
        result = runner.run(max_scans=self.zero_base_max_scans)
        return {
            "champion_board_top": result.get("champion_board", []),
            "a6": result.get("a6", {}),
            "random_stats": result.get("random_stats", {}),
            "better_than_a6": [c.get("engine") for c in result.get("better_than_a6", [])[:5]],
            "meta": result.get("meta", {}),
        }

    def _read_candidates(self) -> list[dict]:
        path = self.store.candidates_path
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @research_safe("daily_report")
    def _maybe_daily_report(self) -> None:
        hour = int(self.config.get("loop", {}).get("daily_report_hour_kst", 8))
        now = datetime.now(KST)
        today = now.strftime("%Y%m%d")
        if now.hour < hour or self._last_report_date == today:
            return
        if not self.telegram_send:
            self._last_report_date = today
            return
        if self.zero_base_enabled and self._last_zero_base_date != today:
            self._refresh_zero_base()
            self._last_zero_base_date = today
        self._refresh_leagues()
        snap = self.store.snapshot()
        report = snap.get("report") or {}
        if not report:
            self._last_report_date = today
            return
        text = build_research_report_text(report)
        if self.telegram_send(text):
            print("[RESEARCH REPORT] sent")
        self._last_report_date = today

    def force_report(self) -> str:
        if self.zero_base_enabled:
            self._refresh_zero_base()
        self._refresh_leagues()
        report = self.store.snapshot().get("report") or {}
        return build_research_report_text(report)
