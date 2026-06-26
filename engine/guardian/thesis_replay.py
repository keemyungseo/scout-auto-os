"""Trade Thesis replay + action distribution analysis."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from scout_auto_os.engine.guardian.decision_engine import (
    contract_from_replay_row,
    position_from_replay_outcome,
)
from scout_auto_os.engine.guardian.guardian_thesis_log import (
    THESIS_LOG_CSV,
    THESIS_SUMMARY_JSON,
    GuardianThesisLog,
    merge_thesis_progress,
)
from scout_auto_os.engine.guardian.progress_engine import evaluate_progress
from scout_auto_os.engine.guardian.thesis_store import GuardianThesisStore
from scout_auto_os.engine.guardian.trade_thesis import (
    DEFAULT_FORMULA,
    build_thesis_from_replay_row,
)
from scout_auto_os.engine.predator.inference import load_replay_bundle
from scout_auto_os.storage.db import now_kst

DEFAULT_ELAPSED_MIN = 240
ANALYSIS_JSON = "guardian_thesis_analysis.json"
REPORT_MD = "guardian_thesis_report.md"


def analyze_by_action(rows: list[dict]) -> dict:
    by_action: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("action", "UNKNOWN")].append(row)

    for action, items in grouped.items():
        by_action[action] = {
            "count": len(items),
            "entry_reason_top": _top_counts((r.get("entry_reason", "") for r in items), n=5),
            "predicted_dna": dict(Counter(r.get("predicted_dna", "") for r in items)),
            "confidence": _confidence_stats(items),
        }
    return by_action


def _top_counts(values, n: int = 5) -> list[dict]:
    counts = Counter(values)
    return [{"value": k, "count": v} for k, v in counts.most_common(n)]


def _confidence_stats(items: list[dict]) -> dict:
    vals = [float(r.get("confidence", 0)) for r in items]
    if not vals:
        return {"min": 0, "max": 0, "avg": 0}
    return {
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "avg": round(sum(vals) / len(vals), 2),
    }


def build_thesis_summary(
    rows: list[dict],
    *,
    analysis: dict,
) -> dict:
    return {
        "last_update": now_kst(),
        "total_rows": len(rows),
        "thesis_count": len(rows),
        "dry_run": True,
        "mode": "THESIS_CONTEXT_ONLY",
        "formula_name": DEFAULT_FORMULA,
        "analysis_by_action": analysis,
        "recent": rows[:20],
    }


def run_thesis_replay(
    data_dir: Path,
    *,
    elapsed_minutes: int = DEFAULT_ELAPSED_MIN,
    config: dict | None = None,
) -> dict:
    """157-trade replay with thesis — progress/decision logic unchanged."""
    bundle = load_replay_bundle(data_dir / "trade_dna")
    theses = [build_thesis_from_replay_row(row) for row in bundle]

    log_rows: list[dict] = []
    for row, thesis in zip(bundle, theses):
        contract = contract_from_replay_row(row)
        contract["expected_horizon"] = thesis.expected_horizon
        contract["contract_id"] = thesis.contract_id
        position = position_from_replay_outcome(row, elapsed_minutes=elapsed_minutes)
        progress = evaluate_progress(
            contract,
            position,
            contract_id=thesis.contract_id,
            config=config,
        )
        log_rows.append(merge_thesis_progress(thesis, progress))

    out_dir = data_dir / "guardian"
    store = GuardianThesisStore(data_dir)
    store.save_batch(theses, replace=True)

    logger = GuardianThesisLog(out_dir)
    log_path = logger.write_log(log_rows)

    analysis = analyze_by_action(log_rows)
    summary = build_thesis_summary(log_rows, analysis=analysis)

    analysis_path = out_dir / ANALYSIS_JSON
    summary_path = out_dir / THESIS_SUMMARY_JSON
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = _write_report(out_dir / REPORT_MD, log_rows, analysis, summary)

    return {
        "ok": True,
        "trade_count": len(log_rows),
        "thesis_jsonl": str(store.path),
        "thesis_log_csv": str(log_path),
        "analysis_json": str(analysis_path),
        "summary_json": str(summary_path),
        "report_md": str(report_path),
        "summary": summary,
        "analysis": analysis,
    }


def _write_report(path: Path, rows: list[dict], analysis: dict, summary: dict) -> Path:
    lines = [
        "# Guardian Trade Thesis Engine V1 — Report",
        "",
        f"**Generated:** {summary.get('last_update', '')}",
        f"**Theses:** {summary.get('thesis_count', 0)}",
        "",
        "## By action",
        "",
    ]
    for action, stats in sorted(analysis.items()):
        lines.append(f"### {action} (n={stats.get('count', 0)})")
        lines.append(f"- DNA: {stats.get('predicted_dna', {})}")
        lines.append(f"- Confidence avg: {stats.get('confidence', {}).get('avg', 0)}")
        lines.append("")

    lines.extend([
        "## Note",
        "",
        "- Decision logic unchanged — thesis attached for context only",
        "- Every log row includes entry_reason for human review",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
