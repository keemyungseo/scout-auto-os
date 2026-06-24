"""TOP5 scan history — data/scans.csv + scans.jsonl (V1.2)."""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from scout_auto_os.engine.review.parse_reason import parse_reason_fields
from scout_auto_os.storage.db import now_kst

SCAN_FIELDS = [
    "scan_time_kst", "top5_snapshot_id", "rank", "symbol", "score", "expected_ev",
    "reason_1h", "reason_2h", "range_pct", "selected_for_entry", "entry_block_reason",
    "current_price", "volume_info",
]


class ScanHistoryStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = data_dir / "scans.csv"
        self.jsonl_path = data_dir / "scans.jsonl"

    def save_scan(
        self,
        scan_time_kst: str,
        top5: list[dict],
        entered_symbols: set[str],
        occupied: set[str],
        locked: set[str],
        slots_available: int,
        kill_switch: bool,
    ) -> str:
        snapshot_id = f"scan_{uuid.uuid4().hex[:12]}"
        rows: list[dict] = []
        slots_left = slots_available

        for row in top5:
            sym = row["symbol"]
            parsed = parse_reason_fields(row.get("reason", ""))
            if sym in entered_symbols:
                selected = True
                block = ""
                slots_left = max(0, slots_left - 1)
            elif sym in occupied:
                selected = False
                block = "already_occupied"
            elif sym in locked:
                selected = False
                block = "manual_lock"
            elif kill_switch:
                selected = False
                block = "kill_switch"
            elif slots_left <= 0:
                selected = False
                block = "no_slot_available"
            else:
                selected = False
                block = "not_selected_this_cycle"

            rec = {
                "scan_time_kst": scan_time_kst,
                "top5_snapshot_id": snapshot_id,
                "rank": row.get("rank", 0),
                "symbol": sym,
                "score": round(float(row.get("a6_score", 0)), 4),
                "expected_ev": round(float(row.get("expected_ev", 0)), 4),
                "reason_1h": parsed["reason_1h"],
                "reason_2h": parsed["reason_2h"],
                "range_pct": parsed["range_pct"],
                "selected_for_entry": selected,
                "entry_block_reason": block,
                "current_price": round(float(row.get("entry_price", 0)), 8),
                "volume_info": str(row.get("volume_state", "")),
            }
            rows.append(rec)

        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SCAN_FIELDS)
            if write_header:
                w.writeheader()
            for rec in rows:
                w.writerow({**rec, "selected_for_entry": str(rec["selected_for_entry"]).lower()})

        with self.jsonl_path.open("a", encoding="utf-8") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[REVIEW] scan history saved snapshot={snapshot_id} rows={len(rows)}")
        return snapshot_id
