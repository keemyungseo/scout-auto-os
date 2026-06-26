"""Portfolio decision summary for Command Center (read-only)."""

from __future__ import annotations

import json
from pathlib import Path

SUMMARY_JSON = "portfolio_decision_summary.json"
DECISION_CSV = "portfolio_decision.csv"


def build_portfolio_decision_status(data_dir: Path) -> dict:
    portfolio_dir = data_dir / "portfolio"
    summary_path = portfolio_dir / SUMMARY_JSON

    summary: dict = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    return {
        "ok": True,
        "dry_run": True,
        "mode": "PORTFOLIO_DECISION",
        "summary": {
            "last_update": summary.get("last_update", ""),
            "candidate_events": summary.get("candidate_events", 0),
            "avg_slot_utilization": summary.get("avg_slot_utilization", 0),
            "replacement_count": summary.get("replacement_count", 0),
            "missed_trades": summary.get("missed_trades", 0),
            "replacement_success_rate": summary.get("replacement_success_rate", 0),
            "decision_counts": summary.get("decision_counts", {}),
            "admitted_avg_roi": summary.get("admitted_avg_roi", 0),
        },
        "slot_status": summary.get("slot_status", {}),
        "replacement_queue": summary.get("replacement_queue", []),
        "waiting_candidates": summary.get("waiting_candidates", []),
        "has_data": bool(summary),
        "data_sources": {
            "decision_summary_json": summary_path.exists(),
            "decision_csv": (portfolio_dir / DECISION_CSV).exists(),
        },
    }
