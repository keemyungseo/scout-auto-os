"""Position snapshots — 1min interval (V1.2)."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from scout_auto_os.engine.position_report import PositionReportService
from scout_auto_os.storage.db import now_kst
from scout_auto_os.storage.trade_record_db import TradeRecordDB

HISTORY_FIELDS = [
    "snapshot_time_kst", "symbol", "side", "entry_time", "entry_price", "mark_price",
    "quantity", "leverage", "margin_used_usdt", "unrealized_pnl_usdt",
    "unrealized_pnl_pct", "max_unrealized_pnl_pct_since_entry",
    "min_unrealized_pnl_pct_since_entry", "hold_minutes",
]


class PositionSnapshotStore:
    def __init__(
        self,
        data_dir: Path,
        position_report: PositionReportService,
        trade_db: TradeRecordDB,
    ) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = data_dir / "positions_snapshot.json"
        self.history_path = data_dir / "positions_history.csv"
        self.position_report = position_report
        self.trade_db = trade_db
        self._hold_extremes: dict[str, dict] = {}

    def _entry_map(self) -> dict[str, dict]:
        return {r["symbol"]: r for r in self.trade_db.open_trades()}

    @staticmethod
    def _hold_minutes(entry_time: str | None) -> int:
        if not entry_time:
            return 0
        try:
            t0 = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(now_kst(), "%Y-%m-%d %H:%M:%S")
            return max(0, int((t1 - t0).total_seconds() / 60))
        except ValueError:
            return 0

    def _update_extremes(self, symbol: str, pnl_pct: float) -> dict:
        ex = self._hold_extremes.setdefault(symbol, {"max": pnl_pct, "min": pnl_pct})
        ex["max"] = max(ex["max"], pnl_pct)
        ex["min"] = min(ex["min"], pnl_pct)
        return ex

    def pop_hold_stats(self, symbol: str) -> dict:
        return self._hold_extremes.pop(symbol.upper(), {"max": 0.0, "min": 0.0})

    def capture(self, paper_positions: list[dict] | None = None) -> list[dict]:
        ts = now_kst()
        entries = self._entry_map()
        rows: list[dict] = []

        live = self.position_report.fetch_open_positions()
        if live:
            source = live
        elif paper_positions:
            source = []
            for p in paper_positions:
                entry = float(p.get("entry_price") or 0)
                mark = float(p.get("current_price") or entry)
                upnl_pct = float(p.get("unrealized_pnl_pct") or 0)
                qty = entries.get(p["symbol"], {}).get("quantity") or 0
                source.append({
                    "symbol": p["symbol"],
                    "side": p.get("side", "LONG"),
                    "entry_price": entry,
                    "mark_price": mark,
                    "quantity": qty,
                    "leverage": entries.get(p["symbol"], {}).get("leverage", 1),
                    "unrealized_pnl_usdt": round(
                        (entries.get(p["symbol"], {}).get("trade_size_usdt") or 0) * upnl_pct / 100, 4
                    ),
                    "unrealized_pnl_pct": upnl_pct,
                })
        else:
            source = []

        for p in source:
            sym = p["symbol"]
            ent = entries.get(sym, {})
            entry_time = ent.get("entry_time")
            upnl_pct = float(p.get("unrealized_pnl_pct") or 0)
            ex = self._update_extremes(sym, upnl_pct)
            lev = int(p.get("leverage") or 1)
            mark = float(p.get("mark_price") or 0)
            qty = float(p.get("quantity") or 0)
            margin = (mark * qty / lev) if lev > 0 else mark * qty

            rec = {
                "snapshot_time_kst": ts,
                "symbol": sym,
                "side": p.get("side", "LONG"),
                "entry_time": entry_time or "",
                "entry_price": p.get("entry_price", 0),
                "mark_price": mark,
                "quantity": qty,
                "leverage": lev,
                "margin_used_usdt": round(margin, 4),
                "unrealized_pnl_usdt": p.get("unrealized_pnl_usdt", 0),
                "unrealized_pnl_pct": upnl_pct,
                "max_unrealized_pnl_pct_since_entry": round(ex["max"], 4),
                "min_unrealized_pnl_pct_since_entry": round(ex["min"], 4),
                "hold_minutes": self._hold_minutes(entry_time),
            }
            rows.append(rec)

        payload = {"snapshot_time_kst": ts, "positions": rows, "count": len(rows)}
        self.snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        write_header = not self.history_path.exists()
        with self.history_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            if write_header:
                w.writeheader()
            for rec in rows:
                w.writerow(rec)

        print(f"[REVIEW] position snapshot saved count={len(rows)}")
        return rows
