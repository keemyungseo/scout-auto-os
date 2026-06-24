"""review_snapshot.json — always-current summary (V1.2)."""

from __future__ import annotations

import json
from pathlib import Path

from scout_auto_os.engine.review.missed_winners import MissedWinnersStore
from scout_auto_os.engine.review.scan_history import ScanHistoryStore
from scout_auto_os.storage.db import now_kst
from scout_auto_os.storage.trade_record_db import TradeRecordDB


class ReviewSnapshotBuilder:
    def __init__(
        self,
        data_dir: Path,
        trade_db: TradeRecordDB,
        since_start: str,
    ) -> None:
        self.path = data_dir / "review_snapshot.json"
        self.trade_db = trade_db
        self.since_start = since_start
        self.data_dir = data_dir

    def _last_top5(self) -> list[dict]:
        jsonl = self.data_dir / "scans.jsonl"
        if not jsonl.exists():
            return []
        lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return []
        last_id = None
        rows = []
        for line in reversed(lines):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = r.get("top5_snapshot_id")
            if last_id is None:
                last_id = sid
            if sid != last_id:
                break
            rows.append(r)
        return list(reversed(rows))

    def _warnings(self, stats: dict, open_positions: list[dict]) -> list[str]:
        w: list[str] = []
        if stats["open_trades"] > 0 and not open_positions:
            w.append("DB shows OPEN trades but no live/paper positions")
        if stats["closed_trades"] >= 5 and stats["win_rate"] < 40:
            w.append("Win rate below 40% since start")
        return w

    def _review_focus(self, stats: dict, missed: list[dict]) -> list[str]:
        focus: list[str] = []
        if missed:
            focus.append(f"Review {len(missed)} recent missed winners for pattern gaps")
        if stats["open_trades"]:
            focus.append(f"Monitor {stats['open_trades']} open trade(s) for exit timing")
        if stats["closed_trades"] >= 3:
            focus.append("Compare exit_quality_score vs max_profit_pct_during_hold")
        while len(focus) < 3:
            focus.append("Continue LOO-style scan vs entry alignment review")
            if len(focus) >= 3:
                break
        return focus[:3]

    def build(
        self,
        open_positions: list[dict],
        unrealized_usdt: float,
        missed_recent: list[dict],
        scan_count: int,
    ) -> dict:
        stats = self.trade_db.stats_since(self.since_start)
        last_entries = self.trade_db.fetchall(
            "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 5"
        )
        last_exits = self.trade_db.all_closed()[:5]
        warnings = self._warnings(stats, open_positions)

        payload = {
            "updated_at_kst": now_kst(),
            "since_start_time": self.since_start,
            "total_scans": scan_count,
            "total_entries": stats["total_entries"],
            "open_trades": stats["open_trades"],
            "closed_trades": stats["closed_trades"],
            "win_count": stats["win_count"],
            "loss_count": stats["loss_count"],
            "win_rate": stats["win_rate"],
            "realized_pnl_usdt": stats["realized_pnl_usdt"],
            "unrealized_pnl_usdt": round(unrealized_usdt, 4),
            "total_pnl_usdt": round(stats["realized_pnl_usdt"] + unrealized_usdt, 4),
            "best_trade": stats["best_trade"],
            "worst_trade": stats["worst_trade"],
            "current_positions": open_positions,
            "last_top5": self._last_top5(),
            "last_entries": last_entries,
            "last_exits": last_exits,
            "recent_missed_winners": missed_recent,
            "warnings": warnings,
            "suggested_review_focus": self._review_focus(stats, missed_recent),
        }
        return payload

    def write(self, payload: dict) -> None:
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print("[REVIEW] review_snapshot updated")
