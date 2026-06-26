"""Guardian Timeline Replay V1 — 157 trades, bar-by-bar observation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scout_auto_os.engine.guardian.timeline_analysis import analyze_timelines
from scout_auto_os.engine.guardian.timeline_curve import build_trade_snapshots
from scout_auto_os.engine.guardian.timeline_engine import TIMELINE_FIELDS, TradeTimeline, evaluate_trade_timeline
from scout_auto_os.engine.predator.inference import load_replay_bundle
from scout_auto_os.engine.predator.labeler_diagnostics import ReplaySources
from scout_auto_os.storage.db import now_kst

TIMELINE_CSV = "guardian_timeline.csv"
TRANSITION_STATS_CSV = "guardian_transition_statistics.csv"
SUMMARY_JSON = "guardian_timeline_summary.json"
REPORT_MD = "guardian_timeline_report.md"
FEATURED_DIR = "timeline_featured"

HEI_TRADE_ID = "2026-06-06 22:00:00|HEIUSDT|long"
MET_SCENARIO_ID = "met_scenario_extended_hold"


def run_timeline_replay(
    data_dir: Path,
    pkg_root: Path,
    *,
    max_minutes: int = 240,
    config: dict | None = None,
) -> dict:
    bundle = load_replay_bundle(data_dir / "trade_dna")
    sources = ReplaySources.discover(pkg_root)

    timelines: list[TradeTimeline] = []
    all_points: list[dict] = []
    all_transitions: list[dict] = []

    for row in bundle:
        trade_key = row["trade_key"]
        klines = sources.forward.get((row.get("scan_kst", ""), row["symbol"]), [])
        cluster = sources.cluster.get(trade_key, {})
        snaps = build_trade_snapshots(
            row, klines=klines, cluster_row=cluster, max_minutes=max_minutes,
        )
        tl = evaluate_trade_timeline(row, snaps, config=config)
        timelines.append(tl)
        all_points.extend(tl.points)
        all_transitions.extend(t.to_row() for t in tl.transitions)

    analysis = analyze_timelines(timelines)

    # MET synthetic extended-hold timeline (representative)
    met_tl = _build_met_scenario_timeline(config)
    hei_tl = _find_timeline(timelines, HEI_TRADE_ID)

    out_dir = data_dir / "guardian"
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(out_dir / TIMELINE_CSV, TIMELINE_FIELDS, all_points)
    _write_csv(
        out_dir / TRANSITION_STATS_CSV,
        ("transition", "count", "avg_roi_at_transition"),
        analysis["transition_statistics"],
    )

    summary = _build_summary(timelines, analysis, met_tl, hei_tl)
    (out_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    featured_dir = out_dir / FEATURED_DIR
    featured_dir.mkdir(exist_ok=True)
    if hei_tl:
        _write_featured_md(featured_dir / "HEI_timeline.md", hei_tl, "HEI — target exceeded hold")
    if met_tl:
        _write_featured_md(featured_dir / "MET_timeline.md", met_tl, "MET — extended hold failure (synthetic)")

    report_path = _write_report(out_dir / REPORT_MD, analysis, summary)

    return {
        "ok": True,
        "trade_count": len(timelines),
        "timeline_points": len(all_points),
        "transition_rows": len(all_transitions),
        "timeline_csv": str(out_dir / TIMELINE_CSV),
        "transition_csv": str(out_dir / TRANSITION_STATS_CSV),
        "summary_json": str(out_dir / SUMMARY_JSON),
        "report_md": str(report_path),
        "analysis": analysis,
        "summary": summary,
    }


def _build_met_scenario_timeline(config: dict | None) -> TradeTimeline | None:
    """Synthetic MET: 2-day hold with weak progress — observe state evolution."""
    from scout_auto_os.engine.guardian.decision_engine import contract_from_replay_row
    from scout_auto_os.engine.guardian.progress_engine import evaluate_progress

    trade_row = {
        "trade_key": MET_SCENARIO_ID,
        "scan_kst": "2026-06-01 00:00:00",
        "symbol": "METUSDT",
        "direction": "long",
        "predicted_roi": 3.0,
        "predicted_peak_roi": 5.0,
        "predicted_drawdown": 8.0,
        "predicted_win_prob": 0.6,
        "value_score": 55.0,
        "predicted_dna_type": "TYPE_1",
        "runner_probability": 0.3,
        "actual_roi": 1.0,
        "actual_peak_roi": 7.0,
    }
    contract = contract_from_replay_row(trade_row)
    contract["expected_horizon"] = 90

    from datetime import datetime, timedelta

    base = datetime.strptime("2026-06-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    checkpoints = [
        (60, 2.5), (120, 3.0), (240, 2.8), (480, 2.0), (720, 1.5),
        (1440, 1.2), (2880, 1.0),
    ]
    snaps = []
    for elapsed, roi in checkpoints:
        snaps.append({
            "timestamp": (base + timedelta(minutes=elapsed)).strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_minutes": elapsed,
            "current_roi": roi,
            "peak_roi": 7.0,
            "drawdown_from_peak": round(7.0 - roi, 4),
        })

    tl = TradeTimeline(trade_id=MET_SCENARIO_ID, symbol="METUSDT", side="long")
    prev_state = None
    prev_rec = None
    for snap in snaps:
        progress = evaluate_progress(contract, snap, contract_id=MET_SCENARIO_ID, config=config)
        point = {
            "trade_id": MET_SCENARIO_ID,
            "timestamp": snap["timestamp"],
            "elapsed_minutes": snap["elapsed_minutes"],
            "current_roi": snap["current_roi"],
            "progress_ratio": round(progress.progress_ratio, 4),
            "guardian_score": round(progress.guardian_score, 2),
            "guardian_state": progress.guardian_state,
            "recommendation": progress.recommendation,
            "reason": progress.reason,
        }
        tl.points.append(point)
        if prev_state and progress.guardian_state != prev_state:
            from scout_auto_os.engine.guardian.timeline_engine import StateTransition
            tl.transitions.append(StateTransition(
                trade_id=MET_SCENARIO_ID,
                from_state=prev_state,
                to_state=progress.guardian_state,
                elapsed_minutes=snap["elapsed_minutes"],
                timestamp=snap["timestamp"],
                current_roi=snap["current_roi"],
                progress_ratio=progress.progress_ratio,
            ))
        if prev_rec and progress.recommendation != prev_rec:
            tl.recommendation_changes += 1
        prev_state = progress.guardian_state
        prev_rec = progress.recommendation
    return tl


def _find_timeline(timelines: list[TradeTimeline], trade_id: str) -> TradeTimeline | None:
    for tl in timelines:
        if tl.trade_id == trade_id:
            return tl
    return None


def _build_summary(
    timelines: list[TradeTimeline],
    analysis: dict,
    met_tl: TradeTimeline | None,
    hei_tl: TradeTimeline | None,
) -> dict:
    trades_index = [
        {
            "trade_id": tl.trade_id,
            "symbol": tl.symbol,
            "side": tl.side,
            "point_count": len(tl.points),
            "transition_count": tl.transition_count,
            "recommendation_changes": tl.recommendation_changes,
            "final_state": tl.points[-1].get("guardian_state") if tl.points else "",
            "final_recommendation": tl.points[-1].get("recommendation") if tl.points else "",
        }
        for tl in timelines
    ]
    return {
        "last_update": now_kst(),
        "dry_run": True,
        "mode": "TIMELINE_REPLAY",
        "bar_interval_minutes": 15,
        **analysis,
        "trades_index": trades_index,
        "featured": {
            "hei_trade_id": HEI_TRADE_ID if hei_tl else "",
            "hei_transition_count": hei_tl.transition_count if hei_tl else 0,
            "met_scenario_id": MET_SCENARIO_ID if met_tl else "",
            "met_final_state": met_tl.points[-1].get("guardian_state") if met_tl and met_tl.points else "",
            "met_thesis_failed": (
                met_tl.points[-1].get("guardian_state") == "THESIS_FAILED"
                if met_tl and met_tl.points else False
            ),
        },
    }


def _write_featured_md(path: Path, tl: TradeTimeline, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        f"**Trade ID:** `{tl.trade_id}`",
        f"**Transitions:** {tl.transition_count}",
        f"**Recommendation changes:** {tl.recommendation_changes}",
        "",
        "## State transitions",
        "",
    ]
    for tr in tl.transitions:
        lines.append(
            f"- `{tr.elapsed_minutes}m` {tr.from_state} → **{tr.to_state}** "
            f"(ROI {tr.current_roi:.2f}%, progress {tr.progress_ratio:.2f})"
        )
    lines.extend(["", "## Full timeline", "", "| Elapsed | ROI | Progress | Score | State | Rec |", "|---:|---:|---:|---:|---|---|"])
    for p in tl.points:
        lines.append(
            f"| {p['elapsed_minutes']}m | {p['current_roi']} | {p['progress_ratio']} "
            f"| {p['guardian_score']} | {p['guardian_state']} | {p['recommendation']} |"
        )
    lines.extend(["", "## Sample reasons", ""])
    for p in tl.points[:: max(1, len(tl.points) // 5)]:
        lines.append(f"**{p['elapsed_minutes']}m — {p['guardian_state']} / {p['recommendation']}**")
        lines.append(f"> {p['reason'][:200]}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(path: Path, analysis: dict, summary: dict) -> Path:
    lines = [
        "# Guardian Timeline Replay V1 — Report",
        "",
        f"**Generated:** {summary.get('last_update', '')}",
        f"**Trades:** {analysis.get('trade_count', 0)}",
        f"**Timeline points:** {analysis.get('total_timeline_points', 0)}",
        "",
        "## Timing",
        "",
        f"- Avg minutes until EXIT rec: {analysis.get('avg_hold_minutes_until_exit_rec', 0)}",
        f"- Avg TRAIL start minutes: {analysis.get('avg_trail_start_minutes', 0)}",
        f"- Avg EXIT rec minutes: {analysis.get('avg_exit_rec_minutes', 0)}",
        f"- Avg recommendation changes: {analysis.get('avg_recommendation_changes', 0)}",
        "",
        "## Top transitions",
        "",
    ]
    for row in analysis.get("transition_statistics", [])[:15]:
        lines.append(
            f"- {row['transition']}: {row['count']} "
            f"(avg ROI {row['avg_roi_at_transition']}%)"
        )
    feat = summary.get("featured", {})
    lines.extend([
        "",
        "## Featured",
        "",
        f"- HEI: `{feat.get('hei_trade_id', '')}` transitions={feat.get('hei_transition_count', 0)}",
        f"- MET scenario: `{feat.get('met_scenario_id', '')}` final={feat.get('met_final_state', '')}",
        "",
        "## Note",
        "",
        "- 15m bar interval from forward klines",
        "- Decision rules unchanged — observation only",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
