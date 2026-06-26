"""Guardian progress status for Command Center (read-only)."""

from __future__ import annotations

import json
from pathlib import Path

from scout_auto_os.engine.guardian.guardian_thesis_log import THESIS_LOG_CSV, THESIS_SUMMARY_JSON
from scout_auto_os.engine.guardian.progress_output import PROGRESS_CSV, SUMMARY_JSON


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    import csv
    try:
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def build_guardian_progress_status(data_dir: Path) -> dict:
    guardian_dir = data_dir / "guardian"
    summary_path = guardian_dir / SUMMARY_JSON
    csv_path = guardian_dir / PROGRESS_CSV
    thesis_summary_path = guardian_dir / THESIS_SUMMARY_JSON
    thesis_log_path = guardian_dir / THESIS_LOG_CSV

    summary: dict = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    thesis_summary: dict = {}
    if thesis_summary_path.exists():
        try:
            thesis_summary = json.loads(thesis_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            thesis_summary = {}

    rows = _read_csv(csv_path)
    thesis_rows = _read_csv(thesis_log_path)
    display_rows = thesis_rows if thesis_rows else rows
    recent = display_rows[:20]

    return {
        "ok": True,
        "dry_run": True,
        "mode": "RULE_BASED",
        "summary": {
            "last_update": thesis_summary.get("last_update") or summary.get("last_update", ""),
            "total_rows": thesis_summary.get("total_rows") or summary.get("total_rows", len(rows)),
            "avg_guardian_score": summary.get("avg_guardian_score", 0),
            "state_counts": summary.get("state_counts", {}),
            "recommendation_counts": summary.get("recommendation_counts", {}),
            "met_scenario": summary.get("scenarios", {}).get("met", {}),
            "thesis_count": thesis_summary.get("thesis_count", 0),
            "analysis_by_action": thesis_summary.get("analysis_by_action", {}),
        },
        "rows": display_rows,
        "recent": recent,
        "thesis": {
            "has_data": bool(thesis_rows or thesis_summary),
            "recent": recent,
            "analysis_by_action": thesis_summary.get("analysis_by_action", {}),
        },
        "data_sources": {
            "progress_csv": csv_path.exists(),
            "summary_json": summary_path.exists(),
            "thesis_log_csv": thesis_log_path.exists(),
            "thesis_summary_json": thesis_summary_path.exists(),
        },
        "has_data": bool(rows or summary or thesis_rows or thesis_summary),
    }
