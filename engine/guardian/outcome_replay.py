"""Guardian Outcome Analyzer V1 — post-trade evaluation (read-only)."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

from scout_auto_os.engine.guardian.outcome_loader import load_outcome_inputs
from scout_auto_os.engine.guardian.outcome_metrics import extract_trade_facts
from scout_auto_os.engine.guardian.outcome_scores import GuardianOutcomeEvaluation, evaluate_trade_outcome
from scout_auto_os.storage.db import now_kst

OUTCOME_CSV = "guardian_outcome.csv"
SUMMARY_JSON = "guardian_outcome_summary.json"
REPORT_MD = "guardian_outcome_report.md"

OUTCOME_FIELDS = (
    "trade_id", "symbol", "side", "entry_time", "exit_time", "hold_minutes",
    "final_roi", "peak_roi", "max_drawdown",
    "hold_count", "trail_start_minutes", "reduce_count", "exit_minutes", "emergency",
    "final_recommendation", "final_state",
    "exit_timing_score", "trail_timing_score", "hold_quality_score",
    "drawdown_control_score", "contract_adherence_score", "overall_guardian_score",
    "outcome_grade", "explanation",
    "entry_reason", "predicted_dna", "confidence", "expected_roi", "expected_horizon",
)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return round(num / (den_x * den_y), 4)


def analyze_outcomes(evaluations: list[GuardianOutcomeEvaluation]) -> dict:
    if not evaluations:
        return {
            "trade_count": 0,
            "avg_guardian_score": 0.0,
            "grade_distribution": {},
            "action_avg_roi": {},
            "correlation": {},
            "avg_final_roi": 0.0,
            "avg_peak_roi": 0.0,
            "avg_max_drawdown": 0.0,
        }

    scores = [e.overall_guardian_score for e in evaluations]
    rois = [e.facts.final_roi for e in evaluations]
    dds = [e.facts.max_drawdown for e in evaluations]

    grade_dist = dict(Counter(e.outcome_grade for e in evaluations))

    action_rois: dict[str, list[float]] = {}
    for e in evaluations:
        act = e.facts.final_recommendation or "UNKNOWN"
        action_rois.setdefault(act, []).append(e.facts.final_roi)
    action_avg = {
        act: round(sum(v) / len(v), 4) for act, v in sorted(action_rois.items())
    }

    subscore_avgs = {
        "exit_timing": round(sum(e.exit_timing_score for e in evaluations) / len(evaluations), 2),
        "trail_timing": round(sum(e.trail_timing_score for e in evaluations) / len(evaluations), 2),
        "hold_quality": round(sum(e.hold_quality_score for e in evaluations) / len(evaluations), 2),
        "drawdown_control": round(sum(e.drawdown_control_score for e in evaluations) / len(evaluations), 2),
        "contract_adherence": round(sum(e.contract_adherence_score for e in evaluations) / len(evaluations), 2),
    }

    sorted_evals = sorted(evaluations, key=lambda e: e.overall_guardian_score, reverse=True)
    best_10 = [_featured_entry(e) for e in sorted_evals[:10]]
    worst_10 = [_featured_entry(e) for e in sorted_evals[-10:]][::-1]

    return {
        "trade_count": len(evaluations),
        "avg_guardian_score": round(sum(scores) / len(scores), 2),
        "avg_final_roi": round(sum(rois) / len(rois), 4),
        "avg_peak_roi": round(sum(e.facts.peak_roi for e in evaluations) / len(evaluations), 4),
        "avg_max_drawdown": round(sum(dds) / len(dds), 4),
        "grade_distribution": grade_dist,
        "subscore_averages": subscore_avgs,
        "action_avg_roi": action_avg,
        "correlation": {
            "guardian_score_vs_roi": pearson(scores, rois),
            "guardian_score_vs_drawdown": pearson(scores, dds),
        },
        "best_10": best_10,
        "worst_10": worst_10,
    }


def _featured_entry(e: GuardianOutcomeEvaluation) -> dict:
    f = e.facts
    return {
        "trade_id": f.trade_id,
        "symbol": f.symbol,
        "side": f.side,
        "overall_guardian_score": e.overall_guardian_score,
        "outcome_grade": e.outcome_grade,
        "final_roi": f.final_roi,
        "peak_roi": f.peak_roi,
        "max_drawdown": f.max_drawdown,
        "final_recommendation": f.final_recommendation,
        "explanation": e.explanation,
    }


def run_outcome_analysis(data_dir: Path) -> dict:
    """Evaluate all trades from timeline + thesis — no decision changes."""
    inputs = load_outcome_inputs(data_dir)
    by_trade = inputs["timeline_by_trade"]
    theses = inputs["theses"]

    evaluations: list[GuardianOutcomeEvaluation] = []
    for trade_id, points in sorted(by_trade.items()):
        thesis = theses.get(trade_id)
        facts = extract_trade_facts(trade_id, points, thesis)
        if facts is None:
            continue
        evaluations.append(evaluate_trade_outcome(facts, points))

    analysis = analyze_outcomes(evaluations)
    out_dir = data_dir / "guardian"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [e.to_row() for e in evaluations]
    _write_csv(out_dir / OUTCOME_CSV, OUTCOME_FIELDS, rows)

    summary = {
        "last_update": now_kst(),
        "dry_run": True,
        "mode": "OUTCOME_ANALYSIS",
        **analysis,
        "data_sources": {k: v.exists() for k, v in inputs["paths"].items()},
    }
    (out_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = _write_report(out_dir / REPORT_MD, analysis, evaluations)

    return {
        "trade_count": len(evaluations),
        "avg_guardian_score": analysis["avg_guardian_score"],
        "grade_distribution": analysis["grade_distribution"],
        "outcome_csv": str(out_dir / OUTCOME_CSV),
        "summary_json": str(out_dir / SUMMARY_JSON),
        "report_md": str(report_path),
        "analysis": analysis,
    }


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_report(
    path: Path,
    analysis: dict,
    evaluations: list[GuardianOutcomeEvaluation],
) -> Path:
    lines = [
        "# Guardian Outcome Analyzer V1",
        "",
        "Read-only post-trade evaluation. No decision rules changed.",
        "",
        f"- Trades: **{analysis['trade_count']}**",
        f"- Avg Guardian Score: **{analysis['avg_guardian_score']}**",
        f"- Avg final ROI: **{analysis['avg_final_roi']}%**",
        f"- Avg peak ROI: **{analysis['avg_peak_roi']}%**",
        f"- Avg max drawdown: **{analysis['avg_max_drawdown']}%**",
        "",
        "## Grade distribution",
        "",
    ]
    for grade, count in sorted(analysis.get("grade_distribution", {}).items()):
        lines.append(f"- {grade}: {count}")

    lines.extend(["", "## Sub-score averages", ""])
    for k, v in analysis.get("subscore_averages", {}).items():
        lines.append(f"- {k}: {v}")

    lines.extend(["", "## Action별 평균 ROI", ""])
    for act, roi in analysis.get("action_avg_roi", {}).items():
        lines.append(f"- {act}: {roi}%")

    corr = analysis.get("correlation", {})
    lines.extend([
        "",
        "## Correlation",
        "",
        f"- Guardian Score vs ROI: {corr.get('guardian_score_vs_roi')}",
        f"- Guardian Score vs Drawdown: {corr.get('guardian_score_vs_drawdown')}",
        "",
        "## Best 10",
        "",
    ])
    for item in analysis.get("best_10", []):
        lines.append(f"### {item['symbol']} ({item['outcome_grade']} — {item['overall_guardian_score']})")
        lines.append(f"- trade_id: `{item['trade_id']}`")
        lines.append(f"- final_roi={item['final_roi']}% peak={item['peak_roi']}%")
        lines.append(f"- {item['explanation']}")
        lines.append("")

    lines.extend(["## Worst 10", ""])
    for item in analysis.get("worst_10", []):
        lines.append(f"### {item['symbol']} ({item['outcome_grade']} — {item['overall_guardian_score']})")
        lines.append(f"- trade_id: `{item['trade_id']}`")
        lines.append(f"- final_roi={item['final_roi']}% peak={item['peak_roi']}%")
        lines.append(f"- {item['explanation']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
