"""Daily trade & position report — Telegram at REPORT_TIME (V1.1)."""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.alert_manager import AlertManager
from scout_auto_os.engine.position_report import PositionReportService
from scout_auto_os.storage.db import Database, now_kst
from scout_auto_os.storage.trade_record_db import TradeRecordDB

KST = timezone(timedelta(hours=9))


class DailyTradeReportService:
    def __init__(
        self,
        config: dict,
        trade_db: TradeRecordDB,
        position_report: PositionReportService,
        alert_mgr: AlertManager,
        os_db: Database,
        csv_dir: Path,
    ) -> None:
        self.config = config
        self.trade_db = trade_db
        self.position_report = position_report
        self.alerts = alert_mgr
        self.os_db = os_db
        self.csv_dir = csv_dir
        exec_cfg = config.get("execution", {})
        self.trade_size = float(exec_cfg.get("order_size_usdt", 5))
        self.leverage = int(exec_cfg.get("leverage", 1))

    def _day_bounds(self, date_yyyy_mm_dd: str) -> tuple[str, str]:
        start = f"{date_yyyy_mm_dd} 00:00:00"
        d = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d").replace(tzinfo=KST)
        end = (d + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        return start, end

    def _hold_minutes(self, entry_time: str, exit_time: str) -> float:
        try:
            t0 = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")
            return (t1 - t0).total_seconds() / 60.0
        except ValueError:
            return 0.0

    def _anomalies(self, open_positions: list[dict], report_date: str) -> list[str]:
        flags: list[str] = []
        db_open = self.trade_db.open_trades()
        db_syms = {r["symbol"] for r in db_open}
        ex_syms = {p["symbol"] for p in open_positions}

        if not self.config.get("paper_mode", True):
            only_db = db_syms - ex_syms
            only_ex = ex_syms - db_syms
            if only_db:
                flags.append(f"DB OPEN but not on exchange: {', '.join(sorted(only_db))}")
            if only_ex:
                flags.append(f"Exchange position not in DB OPEN: {', '.join(sorted(only_ex))}")

        errors = self.os_db.fetchall(
            "SELECT * FROM engine_events WHERE event_type LIKE '%error%' AND timestamp LIKE ?",
            (f"{report_date}%",),
        )
        if errors:
            flags.append(f"Engine errors yesterday: {len(errors)}")

        if self.config.get("risk", {}).get("kill_switch"):
            flags.append("kill_switch active in config")

        return flags or ["None detected"]

    def build_report(self, report_date: str) -> str:
        """report_date: YYYY-MM-DD (typically yesterday)."""
        start, end = self._day_bounds(report_date)
        closed = self.trade_db.closed_between(start, end)
        wins = [t for t in closed if (t.get("realized_pnl_usdt") or 0) > 0]
        losses = [t for t in closed if (t.get("realized_pnl_usdt") or 0) <= 0]
        total = len(closed)
        win_rate = len(wins) / total * 100 if total else 0.0
        realized_usdt = sum(t.get("realized_pnl_usdt") or 0 for t in closed)
        holds = [self._hold_minutes(t["entry_time"], t["exit_time"]) for t in closed if t.get("exit_time")]
        avg_hold = statistics.mean(holds) if holds else 0.0

        best = max(closed, key=lambda x: x.get("realized_pnl_usdt") or 0, default=None)
        worst = min(closed, key=lambda x: x.get("realized_pnl_usdt") or 0, default=None)

        open_positions = self.position_report.fetch_open_positions()
        unrealized_usdt = sum(p["unrealized_pnl_usdt"] for p in open_positions)
        anomalies = self._anomalies(open_positions, report_date)

        mode = "PAPER" if self.config.get("paper_mode", True) else "LIVE"
        lines = [
            f"SCOUT LIVE V1.1 - DAILY REPORT",
            f"Date: {report_date} (KST) | Mode: {mode}",
            "=" * 44,
            "",
            "[Yesterday closed trades]",
            f"  Trades: {total}",
            f"  Wins / Losses: {len(wins)} / {len(losses)}",
            f"  Win rate: {win_rate:.1f}%",
            f"  Realized PnL (USDT): {realized_usdt:+.4f}",
            f"  Avg hold (min): {avg_hold:.0f}",
        ]
        if best:
            lines.append(
                f"  Best: {best['symbol']} {best.get('realized_pnl_pct', 0):+.2f}% "
                f"({best.get('realized_pnl_usdt', 0):+.4f} USDT)"
            )
        if worst:
            lines.append(
                f"  Worst: {worst['symbol']} {worst.get('realized_pnl_pct', 0):+.2f}% "
                f"({worst.get('realized_pnl_usdt', 0):+.4f} USDT)"
            )

        lines += ["", "[Open positions now]"]
        if open_positions:
            for p in open_positions:
                lines.append(
                    f"  {p['symbol']} {p['side']} entry={p['entry_price']} mark={p['mark_price']} "
                    f"upnl={p['unrealized_pnl_usdt']:+.4f} USDT ({p['unrealized_pnl_pct']:+.2f}%) "
                    f"lev={p['leverage']}x"
                )
            lines.append(f"  Total unrealized (USDT): {unrealized_usdt:+.4f}")
        else:
            lines.append("  (none)")

        lines += [
            "",
            "[Settings]",
            f"  TRADE_SIZE: {self.trade_size} USDT",
            f"  LEVERAGE: {self.leverage}x",
            "",
            "[Anomalies]",
        ]
        for a in anomalies:
            lines.append(f"  - {a}")

        return "\n".join(lines)

    def send_for_date(self, report_date: str) -> Path:
        text = self.build_report(report_date)
        stamp = report_date.replace("-", "")
        out = self.csv_dir / f"daily_trade_report_{stamp}.txt"
        out.write_text(text, encoding="utf-8")
        sent = self.alerts.send_telegram_report(text)
        if sent:
            print("[DAILY REPORT] sent")
        else:
            print("[DAILY REPORT] saved (telegram not configured or failed)")
        return out

    def send_previous_day(self) -> Path | None:
        now = datetime.now(KST)
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        return self.send_for_date(yesterday)
