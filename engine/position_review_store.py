"""Position review CSV store (V1.4) — research-compatible audit trail."""

from __future__ import annotations

import csv
from pathlib import Path

from season2_p37_scout_decision_hierarchy import write_csv

REVIEW_FIELDS = [
    "review_time_kst",
    "position_id",
    "symbol",
    "entry_time_kst",
    "hold_minutes",
    "entry_alive_score",
    "current_alive_score",
    "alive_delta",
    "trend_alive_entry",
    "trend_alive_current",
    "momentum_alive_entry",
    "momentum_alive_current",
    "volume_alive_entry",
    "volume_alive_current",
    "expansion_alive_current",
    "exhaustion_current",
    "hold_recommendation",
    "review_reason",
    "exit_reason",
    "unrealized_pnl_pct",
]


class PositionReviewStore:
    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "position_review.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=REVIEW_FIELDS).writeheader()

    def append(self, row: dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
            w.writerow({k: row.get(k, "") for k in REVIEW_FIELDS})

    def latest_by_symbol(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        out: dict[str, dict] = {}
        for r in rows:
            out[r["symbol"]] = r
        return out
