"""Scout Auto OS — paper trading main loop."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _data_dir() -> Path:
    env = os.environ.get("SCOUT_DATA_DIR", "").strip()
    if env:
        return Path(env)
    return ROOT / "scout_auto_os" / "data"

from scout_auto_os.engine.alert_manager import AlertManager
from scout_auto_os.engine.bot_control import BotControl
from scout_auto_os.engine.predator.runtime_shadow import ValueGateRuntimeShadow
from scout_auto_os.engine.daily_trade_report import DailyTradeReportService
from scout_auto_os.engine.dashboard_api import DashboardAPI
from scout_auto_os.engine.entry_block_summary import EntryBlockSummary
from scout_auto_os.engine.entry_quality_guard import EntryQualityGuard
from scout_auto_os.engine.execution_engine import ExecutionEngine
from scout_auto_os.engine.expected_ev_engine import ExpectedEVLogger
from scout_auto_os.engine.live_data import LiveDataEngine
from scout_auto_os.engine.manual_override import ManualOverride
from scout_auto_os.engine.market_watcher import CacheAdapter, LiveDataAdapter, MarketWatcher, MockAdapter
from scout_auto_os.engine.position_manager import PositionManager
from scout_auto_os.engine.position_report import PositionReportService
from scout_auto_os.engine.position_state_manager import PositionStateManager
from scout_auto_os.engine.position_sync import PositionSync
from scout_auto_os.engine.report_manager import ReportManager
from scout_auto_os.engine.review_layer import ReviewLayer
from scout_auto_os.engine.risk_manager import RiskManager
from scout_auto_os.engine.scout_long_engine import ScoutLongEngine
from scout_auto_os.engine.scout_reverse_shadow import ScoutReverseShadow
from scout_auto_os.engine.trade_record import TradeRecordService
from scout_auto_os.engine.research_engine import ResearchEngine
from scout_auto_os.engine.telegram_commands import TelegramCommandBot
from scout_auto_os.engine.runtime_audit.cost_tracker import CostTracker
from scout_auto_os.storage.db import Database, now_kst

KST = timezone(timedelta(hours=9))


class ScoutAutoOS:
    def __init__(self, config_path: Path, once: bool = False, fast_scan: bool = False) -> None:
        with config_path.open(encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.once = once
        self.fast_scan = fast_scan
        self.running = True
        sqlite = Path(self.config["storage"]["sqlite_path"])
        if not sqlite.is_absolute():
            sqlite = ROOT / sqlite
        self.db = Database(sqlite)
        self.csv_dir = Path(self.config["storage"]["csv_dir"])
        if not self.csv_dir.is_absolute():
            self.csv_dir = ROOT / self.csv_dir
        cache_dir = Path(self.config["data"]["kline_cache_dir"])
        if not cache_dir.is_absolute():
            cache_dir = ROOT / cache_dir

        live_log_dir = Path(self.config.get("live_data", {}).get("log_dir", "logs/live"))
        if not live_log_dir.is_absolute():
            live_log_dir = ROOT / live_log_dir
        self.ev_logger = ExpectedEVLogger(live_log_dir)

        self.live_engine = LiveDataEngine(self.config, ROOT)
        adapter_name = self.config["data"].get("adapter", "auto")
        use_live = self.config.get("live_data", {}).get("enabled", True)
        if adapter_name == "mock":
            adapter = MockAdapter(cache_dir)
        elif adapter_name == "live" or (adapter_name == "auto" and use_live):
            adapter = LiveDataAdapter(self.live_engine, cache_dir)
        else:
            adapter = CacheAdapter(cache_dir)

        self.watcher = MarketWatcher(
            self.config, self.db, self.csv_dir, adapter,
            live_engine=self.live_engine if use_live else None,
            ev_logger=self.ev_logger if use_live else None,
        )
        self._market_adapter = adapter
        self.execution = ExecutionEngine(
            self.config, self.db, self.csv_dir, ROOT,
            trade_recorder=TradeRecordService(
                _data_dir() / "trades.db",
                float(self.config.get("execution", {}).get("order_size_usdt", 5)),
                int(self.config.get("execution", {}).get("leverage", 1)),
            ),
        )
        data_dir = _data_dir()
        self.positions = PositionManager(self.config, self.db, self.csv_dir, adapter)
        self.state_manager = PositionStateManager(
            self.config,
            data_dir,
            lambda sym, et: self._market_adapter.get_bars(sym, et),
        )
        self.positions.state_manager = self.state_manager
        print("[STATE ENGINE] initialized (V1.4 state-based exit)")
        self.entry_guard = EntryQualityGuard(self.config)
        self.block_summary = EntryBlockSummary(
            interval_sec=int(self.config.get("entry_quality", {}).get("summary_interval_sec", 1800)),
        )
        self.position_sync = PositionSync(self.config, self.db, self.execution.client)
        self.position_report = PositionReportService(self.execution.client)
        self.dashboard_api = DashboardAPI(
            self.config, self.db, _data_dir(),
        )
        self.manual = ManualOverride(
            self.config, self.db, self.csv_dir, _data_dir(),
        )
        self.risk = RiskManager(self.config, self.db)
        self.alerts = AlertManager(self.config, self.db, self.csv_dir)
        self.reports = ReportManager(self.config, self.db, self.csv_dir)

        def _price_fn(symbol: str) -> float:
            if self.live_engine.enabled:
                px = self.live_engine.get_price(symbol)
                if px > 0:
                    return px
            return self._market_adapter.get_price(symbol)

        self.review = ReviewLayer(
            data_dir,
            self.execution.trade_recorder,
            self.position_report,
            _price_fn,
            self.config,
        )
        self.daily_trade_report = DailyTradeReportService(
            self.config,
            self.execution.trade_recorder.db,
            self.position_report,
            self.alerts,
            self.db,
            self.csv_dir,
            data_dir=data_dir,
            review_snapshot_fn=self.review.get_snapshot,
        )
        self.telegram_bot = TelegramCommandBot(
            self.config,
            data_dir,
            self.execution.trade_recorder.db,
            self.position_report,
            self.review.get_snapshot,
            self.db,
            self._engine_state_dict,
            research_snapshot_fn=None,
            position_state_fn=None,
        )
        self.long_engine = ScoutLongEngine(
            self.config, self.db, self.entry_guard, self.block_summary,
        )
        self.short_shadow = ScoutReverseShadow(self.config, self.db)
        self.portfolio_bridge = None
        if self.config.get("portfolio_engine", {}).get("enabled"):
            from scout_auto_os.engine.portfolio.live_bridge import PortfolioEntryBridge
            cache_dir = Path(self.config["data"]["kline_cache_dir"])
            if not cache_dir.is_absolute():
                cache_dir = ROOT / cache_dir
            pkg_root = Path(__file__).resolve().parent
            self.portfolio_bridge = PortfolioEntryBridge(
                self.config, _data_dir(), pkg_root,
                cache_dir, self.live_engine if use_live else None,
            )
            print("[PORTFOLIO ENGINE] enabled — Long3/Short3 selection active")
        self.last_update = now_kst()
        self.last_report_date: str | None = None
        self.next_snapshot = 0.0
        self.status_path = _data_dir() / "engine_status.json"
        self.metrics_path = _data_dir() / "live_metrics.json"
        self.bot_control = BotControl(_data_dir() / "bot_control.json")
        self.safety = SafetyGuard(_data_dir() / "control", _data_dir() / "bot_control.json")
        vg_shadow_cfg = self.config.get("value_gate_shadow", {})
        self.value_gate_shadow = ValueGateRuntimeShadow(
            _data_dir(),
            enabled=bool(vg_shadow_cfg.get("enabled", False)),
        )
        audit_cfg = self.config.get("runtime_audit", {})
        self.cost_tracker = CostTracker(
            _data_dir() / "runtime_audit",
            enabled=bool(audit_cfg.get("enabled", True)),
        )
        self._track_dup = bool(audit_cfg.get("track_duplicates", True))

        def _research_price_fn(symbol: str) -> float:
            if self.live_engine.enabled:
                px = self.live_engine.get_price(symbol)
                if px > 0:
                    return px
            return self._market_adapter.get_price(symbol)

        self.research = ResearchEngine(
            self.config,
            data_dir,
            cache_dir,
            _research_price_fn,
            live_engine=self.live_engine if use_live else None,
            live_top5_fn=lambda: [r["symbol"] for r in self.watcher.last_top5],
            telegram_send_fn=self.telegram_bot.send_text,
        )
        self.telegram_bot.research_snapshot_fn = self.research.snapshot
        self.telegram_bot.position_state_fn = (
            lambda: self.state_manager.summaries_for_open(self.positions.open_positions())
        )

        if use_live:
            self.live_engine.start()
            self._bootstrap_subscriptions()
            time.sleep(2)

        self._write_status("running")
        self.db.log_event("main", "started", {"mode": self.config.get("mode", "paper")})
        self.telegram_bot.start()
        print(f"[RESEARCH] calling start() enabled={self.research.enabled}")
        self.research.start()
        self.review.update_snapshot()

    def _engine_state_dict(self) -> dict:
        if self.status_path.exists():
            try:
                return json.loads(self.status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {"bot_state": "running"}

    def _bootstrap_subscriptions(self) -> None:
        syms: set[str] = {"BTCUSDT"}
        for p in self.positions.open_positions():
            syms.add(p["symbol"])
        latest = self.db.fetchone("SELECT symbol FROM candidates ORDER BY id DESC LIMIT 5")
        if latest:
            rows = self.db.fetchall(
                "SELECT DISTINCT symbol FROM candidates WHERE timestamp = "
                "(SELECT MAX(timestamp) FROM candidates) LIMIT 5"
            )
            for r in rows:
                syms.add(r["symbol"])
        self.live_engine.subscribe(sorted(syms))

    def _apply_bot_control(self) -> None:
        self.safety.sync_to_bot_control()
        ctrl = self.bot_control.load()
        self.risk.apply_control(ctrl)
        if self.bot_control.bot_stop_requested():
            # Soft stop: block new entries; keep position sync / exits running.
            self.bot_control.clear_bot_stop()

    def _write_status(self, state: str) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        ctrl = self.bot_control.load()
        payload = {
            "state": state,
            "bot_state": state,
            "mode": "paper" if self.config.get("paper_mode", True) else "real",
            "paper_mode": bool(self.config.get("paper_mode", True)),
            "last_scan": self.watcher.last_scan_time,
            "last_update": self.last_update,
            "kill_switch": self.risk.kill_switch,
            "new_entries_allowed": ctrl.get("new_entries_allowed", True),
            "live": self.live_engine.health() if self.live_engine.enabled else {"connected": False},
            "execution": {
                "order_size_usdt": self.config.get("execution", {}).get("order_size_usdt", 5),
                "slots_max": self.config.get("position", {}).get("max_long_slots", 2),
                "slots_used": self.positions.long_slots_used(),
            },
        }
        self.status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_live_metrics(self, updated: list[dict], top5: list[dict] | None = None) -> None:
        payload = {
            "timestamp": now_kst(),
            "positions": [
                {
                    "symbol": p["symbol"],
                    "current_price": p.get("current_price"),
                    "unrealized_pnl_pct": p.get("unrealized_pnl_pct"),
                    "expected_ev": p.get("expected_ev"),
                    "remaining_ev": p.get("remaining_ev"),
                    "trend_alive": p.get("trend_alive"),
                }
                for p in updated
            ],
            "top5": top5 or self.watcher.last_top5,
            "live": self.live_engine.health(),
        }
        self.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def tick_scan(self) -> None:
        try:
            self._apply_bot_control()
            if not self.running:
                return
            with self.cost_tracker.tick("tick_scan", module="a6_search"):
                self.manual.apply_events(self.positions, self.execution, self.alerts)
                self.manual.sync_manual_positions(self.positions)
                top5 = self.watcher.run_scan(paper_fast=self.fast_scan)
            if self.live_engine.enabled:
                self.live_engine.subscribe([r["symbol"] for r in top5])
            self.alerts.top5_alert(top5, self.watcher.last_scan_time or now_kst())
            self.short_shadow.observe(top5)
            occupied = self.positions.occupied_symbols()
            locked = self.manual.locked_symbols() | self.safety.locked_symbols()
            slots_avail = self.config.get("position", {}).get("max_long_slots", 2) - self.positions.long_slots_used()
            scan_time = self.watcher.last_scan_time or now_kst()
            if self.value_gate_shadow.enabled:
                can_enter, _ = self.risk.can_enter_long(self.positions.long_slots_used())
                shadow_cands = [{**r, "side": "long"} for r in top5]
                self.value_gate_shadow.on_scan(
                    scan_time, shadow_cands,
                    occupied=occupied, locked=locked, can_enter=can_enter,
                )
            self.review.prepare_scan(top5, occupied, locked, max(0, slots_avail), self.risk.kill_switch)
            occupied_before = set(occupied)
            with self.cost_tracker.tick("entry_fill", module="portfolio_slots"):
                if self.portfolio_bridge:
                    selection = self.portfolio_bridge.run_selection(scan_time)
                    self.portfolio_bridge.try_fill(
                        selection, self.positions, self.execution, self.alerts,
                        self.risk, occupied, locked,
                    )
                else:
                    self.long_engine.try_fill_slots(
                        top5, occupied, locked,
                        self.positions, self.execution, self.alerts, self.risk,
                        trade_recorder=self.execution.trade_recorder,
                    )
            self.block_summary.maybe_telegram_summary(self.alerts.entry_block_summary_alert)
            entered_symbols = self.positions.occupied_symbols() - occupied_before
            self.review.complete_scan(
                scan_time, top5, entered_symbols, occupied_before, locked,
                max(0, slots_avail), self.risk.kill_switch,
            )
            self.review.update_snapshot()
            self._write_live_metrics([], top5)
            self.dashboard_api.write(
                self.positions,
                live_health=self.live_engine.health() if self.live_engine.enabled else {},
                bot_control=self.bot_control.load(),
                engine_state="running" if self.running else "stopped",
            )
            self._write_status("running")
        except Exception as exc:
            self.alerts.error_alert("market_watcher", str(exc))
            self.db.log_event("main", "scan_error", {"error": str(exc)})

    def tick_positions(self) -> None:
        try:
            self._apply_bot_control()
            if not self.running:
                return
            self.manual.apply_events(self.positions, self.execution, self.alerts)
            sync_summary = self.position_sync.sync(self.positions, self.execution, self.alerts)
            if self.live_engine.enabled:
                for p in self.positions.open_positions():
                    self.live_engine.subscribe([p["symbol"]])
            open_count = len(self.positions.open_positions())
            with self.cost_tracker.tick("tick_positions", module="position_evaluation"):
                if self._track_dup and open_count:
                    self.cost_tracker.record_duplicate_calc(open_count)
                    self.cost_tracker.record_bar_fetch(open_count * 2)
                self.cost_tracker.record_positions_reviewed(open_count)
                updated = self.positions.update_prices()
                allow_exit, _ = self.safety.can_auto_exit()
                self.positions.check_exits(
                    self.execution, self.alerts,
                    allow_auto_exit=allow_exit,
                    symbol_exit_allowed=lambda s: self.safety.can_auto_exit(s)[0],
                )
            self.last_update = now_kst()
            self._write_live_metrics(updated)
            self.dashboard_api.write(
                self.positions, sync_summary,
                self.live_engine.health() if self.live_engine.enabled else {},
                bot_control=self.bot_control.load(),
                engine_state="running" if self.running else "stopped",
            )
            self._write_status("running")
            self.review.process_missed_winners()
        except Exception as exc:
            self.alerts.error_alert("position_manager", str(exc))

    def maybe_daily_report(self) -> None:
        hour = int(self.config["loop"].get("daily_report_hour_kst", 8))
        now = datetime.now(KST)
        today = now.strftime("%Y%m%d")
        if now.hour >= hour and self.last_report_date != today:
            yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
            self.daily_trade_report.send_for_date(yesterday)
            self.last_report_date = today

    def run(self) -> None:
        scan_iv = int(self.config["loop"].get("scan_interval_sec", 300))
        pos_iv = int(self.config["loop"].get("position_update_sec", 30))
        next_scan = time.time()
        next_pos = time.time()
        next_snapshot = time.time()
        print(f"[Scout Auto OS] paper_mode={self.config.get('paper_mode', True)} starting loop")
        if self.live_engine.enabled:
            print(f"[Live Data] websocket {'connected' if self.live_engine.connected else 'connecting...'}")
        self.tick_scan()
        if self.once:
            for _ in range(6):
                time.sleep(5)
                self.tick_positions()
                if self.live_engine.enabled and self.live_engine.connected:
                    print(f"[Live Data] price BTCUSDT={self.live_engine.get_price('BTCUSDT')}")
            self.reports.generate_daily()
            self.live_engine.stop()
            return
        while self.running:
            now = time.time()
            self._apply_bot_control()
            if not self.running:
                break
            if now >= next_scan:
                self.tick_scan()
                next_scan = now + scan_iv
            if now >= next_pos:
                self.tick_positions()
                next_pos = now + pos_iv
            if now >= next_snapshot:
                paper_pos = None
                if self.config.get("paper_mode", True):
                    paper_pos = self.positions.open_positions()
                self.review.capture_positions(paper_pos)
                self.review.update_snapshot()
                next_snapshot = now + 60
            self.maybe_daily_report()
            time.sleep(1)

    def stop(self, *_args) -> None:
        self.running = False
        self.research.stop()
        self.telegram_bot.stop()
        self.live_engine.stop()
        self._write_status("stopped")
        self.db.log_event("main", "stopped", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Scout Auto OS — Paper Trading")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--once", action="store_true", help="Single scan + position update then exit")
    parser.add_argument("--fast-scan", action="store_true", help="Limit universe for quick test")
    args = parser.parse_args()
    osys = ScoutAutoOS(Path(args.config), once=args.once, fast_scan=args.fast_scan)
    signal.signal(signal.SIGINT, osys.stop)
    signal.signal(signal.SIGTERM, osys.stop)
    osys.run()


if __name__ == "__main__":
    main()
