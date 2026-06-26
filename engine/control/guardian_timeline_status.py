"""Guardian timeline summary for Command Center (read-only)."""

from __future__ import annotations

import json
from pathlib import Path

SUMMARY_JSON = "guardian_timeline_summary.json"
TIMELINE_CSV = "guardian_timeline.csv"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    import csv
    try:
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def build_guardian_timeline_status(data_dir: Path) -> dict:
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
        "mode": "TIMELINE_REPLAY",
        "summary": {
            "last_update": summary.get("last_update", ""),
            "trade_count": summary.get("trade_count", 0),
            "total_timeline_points": summary.get("total_timeline_points", 0),
            "bar_interval_minutes": summary.get("bar_interval_minutes", 15),
            "avg_hold_minutes_until_exit_rec": summary.get("avg_hold_minutes_until_exit_rec", 0),
            "avg_trail_start_minutes": summary.get("avg_trail_start_minutes", 0),
            "avg_exit_rec_minutes": summary.get("avg_exit_rec_minutes", 0),
            "avg_recommendation_changes": summary.get("avg_recommendation_changes", 0),
            "transition_frequency": summary.get("transition_frequency", {}),
            "state_point_frequency": summary.get("state_point_frequency", {}),
            "featured": summary.get("featured", {}),
        },
        "trades_index": summary.get("trades_index", []),
        "has_data": bool(summary),
        "data_sources": {
            "timeline_summary_json": summary_path.exists(),
            "timeline_csv": (guardian_dir / TIMELINE_CSV).exists(),
        },
    }


def timeline_for_trade(data_dir: Path, trade_id: str) -> list[dict]:
    """Load timeline points for one trade — for future UI drill-down."""
    rows = _read_csv(data_dir / "guardian" / TIMELINE_CSV)
    return [r for r in rows if r.get("trade_id") == trade_id]
