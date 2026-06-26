"""Guardian progress CSV + JSON summary for Command Center."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from scout_auto_os.engine.guardian.progress_engine import PROGRESS_FIELDS, GuardianProgressResult
from scout_auto_os.storage.db import now_kst

PROGRESS_CSV = "guardian_progress.csv"
SUMMARY_JSON = "guardian_progress_summary.json"
REPORT_MD = "guardian_progress_report.md"


def write_progress_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in PROGRESS_FIELDS})


def build_summary(
    results: list[GuardianProgressResult],
    *,
    scenarios: dict | None = None,
) -> dict:
    states = Counter(r.guardian_state for r in results)
    recs = Counter(r.recommendation for r in results)
    scores = [r.guardian_score for r in results]
    avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0

    return {
        "last_update": now_kst(),
        "total_rows": len(results),
        "avg_guardian_score": avg_score,
        "state_counts": dict(states),
        "recommendation_counts": dict(recs),
        "scenarios": scenarios or {},
        "dry_run": True,
        "mode": "RULE_BASED",
        "recent": [r.to_row() for r in results[:20]],
    }


def write_summary_json(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_progress_report(
    path: Path,
    results: list[GuardianProgressResult],
    summary: dict,
) -> None:
    scenarios = summary.get("scenarios", {})
    met = scenarios.get("met", {})
    lines = [
        "# Guardian Progress Engine V1 — Report",
        "",
        f"**Generated:** {summary.get('last_update', '')}",
        f"**Trades evaluated:** {summary.get('total_rows', 0)}",
        f"**Avg guardian score:** {summary.get('avg_guardian_score', 0)}",
        "",
        "## State distribution",
        "",
    ]
    for state, count in sorted(summary.get("state_counts", {}).items()):
        lines.append(f"- {state}: {count}")

    lines.extend([
        "",
        "## Recommendation distribution",
        "",
    ])
    for rec, count in sorted(summary.get("recommendation_counts", {}).items()):
        lines.append(f"- {rec}: {count}")

    lines.extend([
        "",
        "## MET scenario",
        "",
        f"- State: **{met.get('guardian_state', '—')}**",
        f"- THESIS_FAILED: **{met.get('is_thesis_failed', False)}**",
        f"- Reason: {met.get('reason', '')}",
        "",
        "## Design",
        "",
        "- Rule-based progress only — no ML",
        "- Every row includes human-readable reason",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
