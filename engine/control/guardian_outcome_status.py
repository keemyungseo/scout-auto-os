"""Guardian outcome summary for Command Center Performance Card (read-only)."""

from __future__ import annotations

import json
from pathlib import Path

SUMMARY_JSON = "guardian_outcome_summary.json"
OUTCOME_CSV = "guardian_outcome.csv"


def build_guardian_outcome_status(data_dir: Path) -> dict:
    guardian_dir = data_dir / "guardian"
    summary_path = guardian_dir / SUMMARY_JSON

    summary: dict = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    return {
        "ok": True,
        "dry_run": True,
        "mode": "OUTCOME_ANALYSIS",
        "summary": {
            "last_update": summary.get("last_update", ""),
            "trade_count": summary.get("trade_count", 0),
            "avg_guardian_score": summary.get("avg_guardian_score", 0),
            "avg_final_roi": summary.get("avg_final_roi", 0),
            "avg_peak_roi": summary.get("avg_peak_roi", 0),
            "avg_max_drawdown": summary.get("avg_max_drawdown", 0),
            "grade_distribution": summary.get("grade_distribution", {}),
            "subscore_averages": summary.get("subscore_averages", {}),
            "action_avg_roi": summary.get("action_avg_roi", {}),
            "correlation": summary.get("correlation", {}),
        },
        "best_10": summary.get("best_10", []),
        "worst_10": summary.get("worst_10", []),
        "has_data": bool(summary),
        "data_sources": {
            "outcome_summary_json": summary_path.exists(),
            "outcome_csv": (guardian_dir / OUTCOME_CSV).exists(),
        },
    }
