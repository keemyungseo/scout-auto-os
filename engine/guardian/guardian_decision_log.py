"""Append-only Guardian decision CSV + log."""

from __future__ import annotations

import csv
from pathlib import Path

from scout_auto_os.storage.db import now_kst

DECISION_FIELDS = (
    "timestamp",
    "symbol",
    "action",
    "reason",
    "progress_ratio",
    "time_progress",
    "drawdown_pressure",
    "contract_id",
    "current_roi",
    "elapsed_minutes",
    "peak_roi",
    "drawdown_from_peak",
    "overperformance",
    "expected_horizon",
)

SUMMARY_FIELDS = (
    "symbol",
    "action",
    "reason",
    "progress_ratio",
    "time_progress",
    "drawdown_pressure",
    "contract_id",
)


class GuardianDecisionLog:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.decision_path = self.out_dir / "guardian_decision.csv"
        self.log_path = self.out_dir / "guardian_decision_log.csv"
        self._ensure_headers()

    def _ensure_headers(self) -> None:
        for path, fields in (
            (self.decision_path, DECISION_FIELDS),
            (self.log_path, DECISION_FIELDS),
        ):
            if not path.exists():
                with path.open("w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=fields).writeheader()

    def write_decisions(self, rows: list[dict], *, timestamp: str | None = None) -> Path:
        """Overwrite guardian_decision.csv with latest batch."""
        ts = timestamp or now_kst()
        stamped = []
        for row in rows:
            stamped.append({**row, "timestamp": ts})
        _write_csv(self.decision_path, DECISION_FIELDS, stamped)
        self.append_log(stamped)
        return self.decision_path

    def append_log(self, rows: list[dict]) -> None:
        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=DECISION_FIELDS, extrasaction="ignore")
            for row in rows:
                w.writerow({k: row.get(k, "") for k in DECISION_FIELDS})

    def reset_log(self) -> None:
        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=DECISION_FIELDS).writeheader()


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
