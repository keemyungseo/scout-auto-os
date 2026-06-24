"""Scout Auto OS — paper trading main loop."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.alert_manager import AlertManager
from scout_auto_os.engine.bot_control import BotControl
from scout_auto_os.engine.dashboard_api import DashboardAPI
from scout_auto_os.engine.execution_engine import ExecutionEngine
from scout_auto_os.engine.expected_ev_engine import ExpectedEVLogger
from scout_auto_os.engine.live_data import LiveDataEngine
from scout_auto_os.engine.manual_override import ManualOverride
from scout_auto_os.engine.market_watcher import CacheAdapter, LiveDataAdapter, MarketWatcher, MockAdapter
from scout_auto_os.engine.position_manager import PositionManager
from scout_auto_os.engine.position_sync import PositionSync
from scout_auto_os.engine.report_manager import ReportManager
from scout_auto_os.engine.risk_manager import RiskManager
from scout_auto_os.engine.scout_long_engine import ScoutLongEngine
from scout_auto_os.engine.scout_reverse_shadow import ScoutReverseShadow
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
        self.execution = ExecutionEngine(self.config, self.db, self.csv_dir, ROOT)
        self.positions = PositionManager(self.config, self.db, self.csv_dir, adapter)
        self.position_sync = PositionSync(self.config, self.db, self.execution.client)
        self.dashboard_api = DashboardAPI(
            self.config, self.db, ROOT / "scout_auto_os" / "data",
        )
        self.manual = ManualOverride(
            self.config, self.db, self.csv_dir, ROOT / "scout_auto_os" / "data",
        )
        self.risk = RiskManager(self.config, self.db)
        self.alerts = AlertManager(self.config, self.db, self.csv_dir)
        self.reports = ReportManager(self.config, self.db, self.csv_dir)
        self.long_engine = ScoutLongEngine(self.config, self.db)
        self.short_shadow = ScoutReverseShadow(self.config, self.db)
        self.last_update = now_kst()
        self.last_report_date: str | None = None
        self.status_path = ROOT / "scout_auto_os" / "data" / "engine_status.json"
        self.metrics_path = ROOT / "scout_auto_os" / "data" / "live_metrics.json"
        self.bot_control = BotControl(ROOT / "scout_auto_os" / "data" / "bot_control.json")

        if use_live:
            self.live_engine.start()
            self._bootstrap_subscriptions()
            time.sleep(2)

        self._write_status("running")
        self.db.log_event("main", "started", {"mode": self.config.get("mode", "paper")})

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
            self.manual.apply_events(self.positions, self.execution, self.alerts)
            self.manual.sync_manual_positions(self.positions)
            top5 = self.watcher.run_scan(paper_fast=self.fast_scan)
            if self.live_engine.enabled:
                self.live_engine.subscribe([r["symbol"] for r in top5])
            self.alerts.top5_alert(top5, self.watcher.last_scan_time or now_kst())
            self.short_shadow.observe(top5)
            occupied = self.positions.occupied_symbols()
            locked = self.manual.locked_symbols()
            self.long_engine.try_fill_slots(
                top5, occupied, locked,
                self.positions, self.execution, self.alerts, self.risk,
            )
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
            updated = self.positions.update_prices()
            self.positions.check_exits(self.execution, self.alerts)
            self.last_update = now_kst()
            self._write_live_metrics(updated)
            self.dashboard_api.write(
                self.positions, sync_summary,
                self.live_engine.health() if self.live_engine.enabled else {},
                bot_control=self.bot_control.load(),
                engine_state="running" if self.running else "stopped",
            )
            self._write_status("running")
        except Exception as exc:
            self.alerts.error_alert("position_manager", str(exc))

    def maybe_daily_report(self) -> None:
        hour = int(self.config["loop"].get("daily_report_hour_kst", 23))
        now = datetime.now(KST)
        today = now.strftime("%Y%m%d")
        if now.hour >= hour and self.last_report_date != today:
            self.reports.generate_daily(today)
            self.last_report_date = today

    def run(self) -> None:
        scan_iv = int(self.config["loop"].get("scan_interval_sec", 300))
        pos_iv = int(self.config["loop"].get("position_update_sec", 30))
        next_scan = time.time()
        next_pos = time.time()
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
            self.maybe_daily_report()
            time.sleep(1)

    def stop(self, *_args) -> None:
        self.running = False
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
