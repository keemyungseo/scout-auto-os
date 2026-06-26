"""Guardian log with Trade Thesis context — does not alter decisions."""

from __future__ import annotations

import csv
from pathlib import Path

from scout_auto_os.engine.guardian.progress_engine import GuardianProgressResult
from scout_auto_os.engine.guardian.trade_thesis import GuardianTradeThesis
from scout_auto_os.storage.db import now_kst

THESIS_LOG_FIELDS = (
    "timestamp",
    "thesis_id",
    "contract_id",
    "prediction_key",
    "symbol",
    "action",
    "reason",
    "entry_reason",
    "predicted_dna",
    "expected_roi",
    "expected_horizon",
    "confidence",
    "guardian_state",
    "progress_ratio",
    "time_progress",
    "drawdown_pressure",
    "guardian_score",
    "formula_name",
    "predator_version",
)

THESIS_LOG_CSV = "guardian_thesis_log.csv"
THESIS_SUMMARY_JSON = "guardian_thesis_summary.json"


def merge_thesis_progress(
    thesis: GuardianTradeThesis,
    progress: GuardianProgressResult,
) -> dict:
    """Attach thesis to progress result — action/reason unchanged."""
    return {
        "thesis_id": thesis.thesis_id,
        "contract_id": thesis.contract_id,
        "prediction_key": thesis.prediction_key,
        "symbol": thesis.symbol,
        "action": progress.recommendation,
        "reason": progress.reason,
        "entry_reason": thesis.entry_reason,
        "predicted_dna": thesis.predicted_dna,
        "expected_roi": round(thesis.expected_roi, 4),
        "expected_horizon": thesis.expected_horizon,
        "confidence": thesis.confidence,
        "guardian_state": progress.guardian_state,
        "progress_ratio": round(progress.progress_ratio, 4),
        "time_progress": round(progress.time_progress, 4),
        "drawdown_pressure": round(progress.drawdown_pressure, 4),
        "guardian_score": round(progress.guardian_score, 2),
        "formula_name": thesis.formula_name,
        "predator_version": thesis.predator_version,
    }


class GuardianThesisLog:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / THESIS_LOG_CSV

    def write_log(self, rows: list[dict], *, timestamp: str | None = None) -> Path:
        ts = timestamp or now_kst()
        stamped = [{**row, "timestamp": ts} for row in rows]
        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=THESIS_LOG_FIELDS, extrasaction="ignore")
            w.writeheader()
            for row in stamped:
                w.writerow({k: row.get(k, "") for k in THESIS_LOG_FIELDS})
        return self.log_path
