"""Position evaluation audit logs."""

from __future__ import annotations

import csv
from pathlib import Path

from scout_auto_os.storage.db import now_kst

REVIEW_FIELDS = [
    "timestamp",
    "thesis_id",
    "position_id",
    "symbol",
    "side",
    "source",
    "auto_manage",
    "entry_time",
    "entry_price",
    "current_price",
    "roi",
    "elapsed_minutes",
    "expected_horizon",
    "expected_return",
    "success_probability",
    "mfe",
    "mae",
    "peak_roi",
    "drawdown_from_peak",
    "thesis_validity_score",
    "exit_pressure_score",
    "hold_confidence",
    "action",
    "action_reason",
    "thesis_update_reason",
    "should_exit",
    "expected_roi",
    "expected_progress",
    "progress_ratio",
    "progress_delta",
    "expectation_score",
    "thesis_state",
    "thesis_version",
    "curve_version",
    "thesis_transition_reason",
]

DECISION_FIELDS = [
    "timestamp",
    "thesis_id",
    "position_id",
    "symbol",
    "action",
    "should_exit",
    "action_reason",
]

GUARD_FIELDS = [
    "timestamp",
    "symbol",
    "position_id",
    "source",
    "auto_manage",
    "manual_lock",
    "event",
    "detail",
    "blocked",
]


class PositionEvaluationLogger:
    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir / "position_evaluation"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.review_path = self.dir / "position_review.csv"
        self.decision_path = self.dir / "position_decision_log.csv"
        self.guard_path = self.dir / "manual_position_guard.csv"
        self._ensure(self.review_path, REVIEW_FIELDS)
        self._ensure(self.decision_path, DECISION_FIELDS)
        self._ensure(self.guard_path, GUARD_FIELDS)

    @staticmethod
    def _ensure(path: Path, fields: list[str]) -> None:
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log_review(self, row: dict) -> None:
        row = {"timestamp": now_kst(), **row}
        with self.review_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
            w.writerow({k: row.get(k, "") for k in REVIEW_FIELDS})

    def log_decision(self, row: dict) -> None:
        row = {"timestamp": now_kst(), **row}
        with self.decision_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=DECISION_FIELDS, extrasaction="ignore")
            w.writerow({k: row.get(k, "") for k in DECISION_FIELDS})

    def log_guard(self, row: dict) -> None:
        row = {"timestamp": now_kst(), **row}
        with self.guard_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=GUARD_FIELDS, extrasaction="ignore")
            w.writerow({k: row.get(k, "") for k in GUARD_FIELDS})
