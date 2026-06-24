"""Review & learning data layer facade (V1.2)."""

from __future__ import annotations

import json
from pathlib import Path

from scout_auto_os.engine.review.missed_winners import MissedWinnersStore
from scout_auto_os.engine.review.parse_reason import parse_reason_fields
from scout_auto_os.engine.review.position_snapshot import PositionSnapshotStore
from scout_auto_os.engine.review.review_snapshot import ReviewSnapshotBuilder
from scout_auto_os.engine.review.safe import review_safe
from scout_auto_os.engine.review.scan_history import ScanHistoryStore
from scout_auto_os.engine.position_report import PositionReportService
from scout_auto_os.engine.trade_record import TradeRecordService
from scout_auto_os.storage.db import now_kst


class ReviewLayer:
    def __init__(
        self,
        data_dir: Path,
        trade_recorder: TradeRecordService,
        position_report: PositionReportService,
        price_fn,
        config: dict,
    ) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trade_recorder = trade_recorder
        self.config = config
        self.scan_store = ScanHistoryStore(data_dir)
        self.snapshot_store = PositionSnapshotStore(
            data_dir, position_report, trade_recorder.db,
        )
        self.missed_store = MissedWinnersStore(data_dir, price_fn)
        self.since_start = now_kst()
        self.scan_count = 0
        self._last_snapshot_id: str | None = None
        self.review_snapshot = ReviewSnapshotBuilder(data_dir, trade_recorder.db, self.since_start)
        trade_recorder.set_hold_stats_provider(self.snapshot_store.pop_hold_stats)

    def _build_context(self, row: dict, snapshot_id: str) -> dict:
        parsed = parse_reason_fields(row.get("reason", ""))
        return {
            "scan_rank": row.get("rank"),
            "score": row.get("a6_score"),
            "expected_ev": row.get("expected_ev"),
            "reason_1h": parsed["reason_1h"],
            "reason_2h": parsed["reason_2h"],
            "range_pct": parsed["range_pct"],
            "top5_snapshot_id": snapshot_id,
            "volume_state": row.get("volume_state"),
            "trend_alive": row.get("trend_alive"),
            "acceleration": row.get("acceleration"),
            "entry_price": row.get("entry_price"),
            "top5": [
                {"rank": r.get("rank"), "symbol": r.get("symbol"), "score": r.get("a6_score")}
                for r in []  # filled by caller if needed
            ],
        }

    @review_safe("prepare_scan")
    def prepare_scan(
        self,
        top5: list[dict],
        occupied: set[str],
        locked: set[str],
        slots_available: int,
        kill_switch: bool,
    ) -> None:
        temp_id = "pending"
        slots_left = slots_available
        for row in top5:
            sym = row["symbol"]
            if sym in occupied or sym in locked or kill_switch or slots_left <= 0:
                continue
            ctx = self._build_context(row, temp_id)
            self.trade_recorder.set_entry_context(sym, ctx)
            slots_left -= 1

    @review_safe("complete_scan")
    def complete_scan(
        self,
        scan_time_kst: str,
        top5: list[dict],
        entered_symbols: set[str],
        occupied_before: set[str],
        locked: set[str],
        slots_available: int,
        kill_switch: bool,
    ) -> None:
        self.scan_count += 1
        snapshot_id = self.scan_store.save_scan(
            scan_time_kst, top5, entered_symbols,
            occupied_before, locked, slots_available, kill_switch,
        )
        self._last_snapshot_id = snapshot_id
        top5_by_sym = {r["symbol"]: r for r in top5}
        block_map: dict[str, str] = {}
        slots_left = slots_available
        for row in top5:
            sym = row["symbol"]
            if sym in entered_symbols:
                block_map[sym] = ""
            elif sym in occupied_before:
                block_map[sym] = "already_occupied"
            elif sym in locked:
                block_map[sym] = "manual_lock"
            elif kill_switch:
                block_map[sym] = "kill_switch"
            elif slots_left <= 0:
                block_map[sym] = "no_slot_available"
            else:
                block_map[sym] = "not_selected_this_cycle"
            if sym not in occupied_before and sym not in locked and not kill_switch:
                slots_left = max(0, slots_left - 1)

        for sym in entered_symbols:
            row = top5_by_sym.get(sym)
            if row:
                ctx = self._build_context(row, snapshot_id)
                self.trade_recorder.enrich_entry_context(sym, ctx)

        self.missed_store.schedule(scan_time_kst, top5, entered_symbols, block_map)

    @review_safe("position_snapshot")
    def capture_positions(self, paper_positions: list[dict] | None = None) -> list[dict]:
        return self.snapshot_store.capture(paper_positions)

    @review_safe("missed_winners")
    def process_missed_winners(self) -> int:
        return self.missed_store.process_due()

    @review_safe("review_snapshot")
    def update_snapshot(self) -> None:
        open_pos: list[dict] = []
        if self.snapshot_store.snapshot_path.exists():
            data = json.loads(self.snapshot_store.snapshot_path.read_text(encoding="utf-8"))
            open_pos = data.get("positions", [])
        unrealized = sum(p.get("unrealized_pnl_usdt", 0) for p in open_pos)
        missed = self.missed_store.recent(10)
        payload = self.review_snapshot.build(open_pos, unrealized, missed, self.scan_count)
        self.review_snapshot.write(payload)

    def get_snapshot(self) -> dict | None:
        path = self.data_dir / "review_snapshot.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
