"""Daily trade & position report — Telegram at REPORT_TIME (V1.1 + V1.2)."""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.alert_manager import AlertManager
from scout_auto_os.engine.position_report import PositionReportService
from scout_auto_os.engine.review.missed_winners import MissedWinnersStore
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
        data_dir: Path | None = None,
        review_snapshot_fn=None,
    ) -> None:
        self.config = config
        self.trade_db = trade_db
        self.position_report = position_report
        self.alerts = alert_mgr
        self.os_db = os_db
        self.csv_dir = csv_dir
        self.data_dir = data_dir or csv_dir.parent / "data"
        self.review_snapshot_fn = review_snapshot_fn
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

    def _yesterday_scans(self, report_date: str) -> list[dict]:
        path = self.data_dir / "scans.csv"
        if not path.exists():
            return []
        rows: list[dict] = []
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("scan_time_kst", "").startswith(report_date):
                    rows.append(row)
        return rows

    def _yesterday_missed(self, report_date: str) -> list[dict]:
        path = self.data_dir / "missed_winners.jsonl"
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            try:
                r = json.loads(line)
                if str(r.get("scan_time_kst", "")).startswith(report_date):
                    out.append(r)
            except json.JSONDecodeError:
                pass
        return out

    def _anomalies(self, open_positions: list[dict], report_date: str) -> list[str]:
        flags: list[str] = []
        db_open = self.trade_db.open_trades()
        db_syms = {r["symbol"] for r in db_open}
        ex_syms = {p["symbol"] for p in open_positions}
        if not self.config.get("paper_mode", True):
            if db_syms - ex_syms:
                flags.append(f"DB OPEN not on exchange: {', '.join(sorted(db_syms - ex_syms))}")
            if ex_syms - db_syms:
                flags.append(f"Exchange not in DB: {', '.join(sorted(ex_syms - db_syms))}")
        errors = self.os_db.fetchall(
            "SELECT * FROM engine_events WHERE event_type LIKE '%error%' AND timestamp LIKE ?",
            (f"{report_date}%",),
        )
        if errors:
            flags.append(f"Engine errors: {len(errors)}")
        snap = self.review_snapshot_fn() if self.review_snapshot_fn else {}
        for w in snap.get("warnings", []):
            flags.append(w)
        return flags or ["None detected"]

    def _review_points(self, report_date: str, entered: list, missed: list, closed: list) -> list[str]:
        pts: list[str] = []
        if missed:
            syms = ", ".join(m["symbol"] for m in missed[:3])
            pts.append(f"Missed movers yesterday: {syms}")
        if closed:
            low_q = [t for t in closed if (t.get("exit_quality_score") or 100) < 50]
            if low_q:
                pts.append(f"{len(low_q)} exits with low exit_quality_score — review timing")
        if entered:
            not_entered = [r for r in self._yesterday_scans(report_date) if r.get("selected_for_entry") != "true"]
            if not_entered:
                pts.append(f"TOP5 had {len(not_entered)} non-entry picks — check block reasons")
        while len(pts) < 3:
            pts.append("Compare scan expected_ev vs realized forward returns")
            if len(pts) >= 3:
                break
        return pts[:3]

    def build_report(self, report_date: str) -> str:
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
        if not open_positions:
            snap_file = self.data_dir / "positions_snapshot.json"
            if snap_file.exists():
                open_positions = json.loads(snap_file.read_text(encoding="utf-8")).get("positions", [])
        unrealized_usdt = sum(p.get("unrealized_pnl_usdt", 0) for p in open_positions)
        anomalies = self._anomalies(open_positions, report_date)

        scan_rows = self._yesterday_scans(report_date)
        entered_syms = {r["symbol"] for r in scan_rows if r.get("selected_for_entry") == "true"}
        missed = self._yesterday_missed(report_date)
        review_pts = self._review_points(report_date, list(entered_syms), missed, closed)

        mode = "PAPER" if self.config.get("paper_mode", True) else "LIVE"
        lines = [
            f"SCOUT LIVE V1.2 - DAILY REPORT",
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
                    f"  {p['symbol']} {p['side']} upnl={p.get('unrealized_pnl_usdt', 0):+.4f} USDT "
                    f"({p.get('unrealized_pnl_pct', 0):+.2f}%)"
                )
            lines.append(f"  Total unrealized (USDT): {unrealized_usdt:+.4f}")
        else:
            lines.append("  (none)")

        lines += ["", "[Yesterday TOP5 entered]", f"  {', '.join(sorted(entered_syms)) or '(none)'}"]
        lines += ["", "[Yesterday missed strong movers]"]
        if missed:
            for m in missed[:5]:
                lines.append(
                    f"  {m['symbol']} 2h={m.get('forward_2h_return')}% "
                    f"block={m.get('reason_not_entered')}"
                )
        else:
            lines.append("  (none recorded yet)")

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

        lines += ["", "[Today review points]"]
        for i, pt in enumerate(review_pts, 1):
            lines.append(f"  {i}. {pt}")

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
