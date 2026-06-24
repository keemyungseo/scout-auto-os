"""Daily performance report."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scout_auto_os.storage.db import Database, now_kst


class ReportManager:
    def __init__(self, config: dict, db: Database, csv_dir: Path) -> None:
        self.config = config
        self.db = db
        self.csv_dir = csv_dir
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.initial_capital = float(config.get("initial_capital", 10000))

    def generate_daily(self, report_date: str | None = None) -> Path:
        if report_date is None:
            report_date = now_kst()[:10].replace("-", "")
        date_filter = f"{report_date[:4]}-{report_date[4:6]}-{report_date[6:8]}"
        exits = self.db.fetchall(
            "SELECT * FROM trades WHERE action='EXIT' AND timestamp LIKE ?",
            (f"{date_filter}%",),
        )
        wins = [e for e in exits if (e.get("pnl_pct") or 0) > 0]
        losses = [e for e in exits if (e.get("pnl_pct") or 0) <= 0]
        total = len(exits)
        win_rate = len(wins) / total * 100 if total else 0
        avg_win = sum(e["pnl_pct"] for e in wins) / len(wins) if wins else 0
        avg_loss = sum(e["pnl_pct"] for e in losses) / len(losses) if losses else 0
        gross_win = sum(e["pnl_pct"] for e in wins)
        gross_loss = abs(sum(e["pnl_pct"] for e in losses))
        pf = gross_win / gross_loss if gross_loss else 0
        best = max(exits, key=lambda x: x.get("pnl_pct") or 0, default=None)
        worst = min(exits, key=lambda x: x.get("pnl_pct") or 0, default=None)
        candidates = self.db.fetchall(
            "SELECT * FROM candidates WHERE timestamp LIKE ? ORDER BY timestamp DESC LIMIT 50",
            (f"{date_filter}%",),
        )
        missed = [c for c in candidates if c["rank"] <= 5]
        overrides = self.db.fetchall(
            "SELECT * FROM manual_overrides WHERE timestamp LIKE ?",
            (f"{date_filter}%",),
        )
        errors = self.db.fetchall(
            "SELECT * FROM engine_events WHERE event_type LIKE '%error%' AND timestamp LIKE ?",
            (f"{date_filter}%",),
        )
        lines = [
            f"SCOUT AUTO OS DAILY REPORT - {date_filter}",
            "=" * 50,
            f"Total trades: {total}",
            f"Win rate: {win_rate:.1f}%",
            f"Avg win: {avg_win:.2f}%",
            f"Avg loss: {avg_loss:.2f}%",
            f"Profit Factor: {pf:.2f}",
            f"Max Drawdown: (paper — see equity curve)",
            f"Best trade: {best['symbol'] if best else 'n/a'} {best.get('pnl_pct', 0) if best else 0:.2f}%",
            f"Worst trade: {worst['symbol'] if worst else 'n/a'} {worst.get('pnl_pct', 0) if worst else 0:.2f}%",
            "",
            "Missed TOP5 summary:",
        ]
        seen = set()
        for c in missed:
            key = (c["timestamp"], c["symbol"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  {c['timestamp']} #{c['rank']} {c['symbol']} A6={c['a6_score']}")
        lines += ["", "Manual override events:"]
        for o in overrides:
            lines.append(f"  {o['timestamp']} {o['symbol']} {o['action']}")
        lines += ["", "Engine errors:"]
        for e in errors:
            lines.append(f"  {e['timestamp']} {e['module']} {e['event_type']}")
        lines += ["", "Next day notes:", "  - Continue paper validation", "  - Review TOP5 vs entries"]
        content = "\n".join(lines)
        out = self.csv_dir / f"daily_report_{report_date}.txt"
        out.write_text(content, encoding="utf-8")
        self.db.execute(
            "INSERT OR REPLACE INTO daily_reports (report_date, content, created_at) VALUES (?,?,?)",
            (date_filter, content, now_kst()),
        )
        return out

    def account_summary(self) -> dict:
        exits = self.db.fetchall("SELECT pnl_pct FROM trades WHERE action='EXIT'")
        realized = sum(e["pnl_pct"] or 0 for e in exits)
        open_pos = self.db.fetchall("SELECT unrealized_pnl_pct FROM positions WHERE status='OPEN'")
        unrealized = sum(p["unrealized_pnl_pct"] or 0 for p in open_pos)
        today = now_kst()[:10]
        today_exits = self.db.fetchall(
            "SELECT pnl_pct FROM trades WHERE action='EXIT' AND timestamp LIKE ?",
            (f"{today}%",),
        )
        today_ret = sum(e["pnl_pct"] or 0 for e in today_exits)
        total_ret = realized + unrealized
        equity = self.initial_capital * (1 + total_ret / 100)
        return {
            "initial_capital": self.initial_capital,
            "current_equity": equity,
            "realized_pnl_pct": realized,
            "unrealized_pnl_pct": unrealized,
            "today_return_pct": today_ret,
            "total_return_pct": total_ret,
        }
