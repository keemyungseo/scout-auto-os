"""Portfolio Decision replay on 157-trade bundle."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from scout_auto_os.engine.guardian.outcome_loader import load_theses
from scout_auto_os.engine.guardian.trade_thesis import build_thesis_from_replay_row
from scout_auto_os.engine.portfolio.decision.decision_engine import evaluate_candidate
from scout_auto_os.engine.portfolio.decision.models import (
    DECISION_IGNORE,
    DECISION_REPLACE,
    DECISION_WAIT,
    PortfolioPosition,
    PortfolioSlotBook,
    PredatorCandidate,
)
from scout_auto_os.engine.portfolio.decision.position_score import rescore_book_positions
from scout_auto_os.engine.predator.inference import load_replay_bundle
from scout_auto_os.engine.predator.policies import policy_b_soft_50s
from scout_auto_os.storage.db import now_kst

DECISION_CSV = "portfolio_decision.csv"
SUMMARY_JSON = "portfolio_decision_summary.json"
REPORT_MD = "portfolio_decision_report.md"

DECISION_FIELDS = (
    "timestamp", "slot", "side", "current_symbol", "candidate_symbol",
    "decision", "reason", "current_score", "candidate_score",
)

_TIME_FMT = "%Y-%m-%d %H:%M:%S"


def _parse_time(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, _TIME_FMT)
    except ValueError:
        return None


def _load_timeline(data_dir: Path) -> dict[str, list[dict]]:
    path = data_dir / "guardian" / "guardian_timeline.csv"
    if not path.exists():
        return {}
    by_trade: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_trade[row.get("trade_id", "")].append(row)
    for tid in by_trade:
        by_trade[tid].sort(key=lambda r: int(float(r.get("elapsed_minutes", 0))))
    return dict(by_trade)


def _load_progress(data_dir: Path) -> dict[str, dict]:
    path = data_dir / "guardian" / "guardian_progress.csv"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("contract_id", "")
            if cid:
                out[cid] = row
    return out


def _timeline_point_at(timeline: dict[str, list[dict]], trade_id: str, elapsed_min: int) -> dict | None:
    points = timeline.get(trade_id, [])
    if not points:
        return None
    best = None
    for p in points:
        em = int(float(p.get("elapsed_minutes", 0)))
        if em <= elapsed_min:
            best = p
        else:
            break
    return best or points[0]


def _progress_fallback(progress: dict[str, dict], trade_id: str) -> dict:
    return progress.get(trade_id, {})


def _refresh_position_metrics(
    pos: PortfolioPosition,
    as_of: datetime,
    timeline: dict[str, list[dict]],
    progress: dict[str, dict],
    thesis_map: dict,
) -> None:
    entry_dt = _parse_time(pos.entry_time)
    if entry_dt:
        elapsed = max(0, int((as_of - entry_dt).total_seconds() / 60))
    else:
        elapsed = pos.elapsed_minutes

    pos.elapsed_minutes = elapsed
    pt = _timeline_point_at(timeline, pos.trade_id, elapsed)
    if pt:
        pos.guardian_score = float(pt.get("guardian_score", pos.guardian_score))
        pos.guardian_state = pt.get("guardian_state", pos.guardian_state)
        pos.recommendation = pt.get("recommendation", pos.recommendation)
        pos.current_roi = float(pt.get("current_roi", 0))
    else:
        fb = _progress_fallback(progress, pos.trade_id)
        if fb:
            pos.guardian_score = float(fb.get("guardian_score", pos.guardian_score))
            pos.guardian_state = fb.get("guardian_state", pos.guardian_state)
            pos.recommendation = fb.get("recommendation", pos.recommendation)

    thesis = thesis_map.get(pos.trade_id)
    if thesis:
        pos.expected_horizon = int(thesis.expected_horizon)
        pos.expected_roi = float(thesis.expected_roi)
        pos.confidence = float(thesis.confidence)
        pos.value_score = float(thesis.value_score)


def _expire_positions(
    book: PortfolioSlotBook,
    as_of: datetime,
    thesis_map: dict,
) -> None:
    for side in ("long", "short"):
        kept: list[PortfolioPosition] = []
        for pos in book.slots_for(side):
            entry_dt = _parse_time(pos.entry_time)
            horizon = pos.expected_horizon
            thesis = thesis_map.get(pos.trade_id)
            if thesis:
                horizon = int(thesis.expected_horizon)
            if entry_dt and as_of >= entry_dt + timedelta(minutes=horizon):
                continue
            kept.append(pos)
        book.set_slots(side, kept)


def _reindex_slots(book: PortfolioSlotBook, side: str) -> None:
    slots = book.slots_for(side)
    reindexed = []
    for i, p in enumerate(slots, start=1):
        p.slot_id = f"{side}_{i}"
        reindexed.append(p)
    book.set_slots(side, reindexed)


def _build_candidate(row: dict, gate: dict) -> PredatorCandidate:
    thesis = build_thesis_from_replay_row(row, formula_name="policy_b_soft_50s")
    return PredatorCandidate(
        trade_id=row["trade_key"],
        symbol=row["symbol"],
        side=str(row.get("side", row.get("direction", "long"))).lower(),
        timestamp=row.get("scan_kst", row["trade_key"].split("|")[0]),
        value_score=float(row.get("value_score", 0)),
        expected_roi=float(row.get("predicted_roi", 0)),
        expected_win_prob=float(row.get("predicted_win_prob", 0)),
        confidence=thesis.confidence,
        gate_action=gate.get("action", "SKIP"),
        gate_reason=gate.get("reason", ""),
        contract_id=thesis.contract_id,
        thesis_id=thesis.thesis_id,
        actual_roi=float(row.get("actual_roi", 0)),
    )


def run_portfolio_decision_replay(
    data_dir: Path,
    *,
    config: dict | None = None,
) -> dict:
    bundle = load_replay_bundle(data_dir / "trade_dna")
    bundle.sort(key=lambda r: r.get("scan_kst", r.get("trade_key", "")))

    timeline = _load_timeline(data_dir)
    progress = _load_progress(data_dir)
    thesis_map = load_theses(data_dir / "guardian" / "trade_thesis.jsonl")

    book = PortfolioSlotBook()
    all_records: list = []
    replacements: list[dict] = []
    utilization_samples: list[float] = []
    waiting_queue: list[dict] = []

    for row in bundle:
        ts = row.get("scan_kst", row["trade_key"].split("|")[0])
        as_of = _parse_time(ts)
        if as_of is None:
            continue

        _expire_positions(book, as_of, thesis_map)

        for side in ("long", "short"):
            slots = book.slots_for(side)
            for pos in slots:
                _refresh_position_metrics(pos, as_of, timeline, progress, thesis_map)
            rescore_book_positions(slots, config)
            book.set_slots(side, slots)

        gate = policy_b_soft_50s({
            "value_score": row.get("value_score", 0),
            "predicted_dna_type": row.get("predicted_dna_type", ""),
            "runner_probability": row.get("runner_probability", 0),
            "predicted_drawdown": row.get("predicted_drawdown", 0),
            "predicted_win_prob": row.get("predicted_win_prob", 0),
        })
        row["gate_reason"] = gate.get("reason", "")

        candidate = _build_candidate(row, gate)
        if candidate.side not in ("long", "short"):
            continue

        book, records, admitted, replaced_out = evaluate_candidate(book, candidate, config=config)
        all_records.extend(records)

        thesis = thesis_map.get(candidate.trade_id)
        if admitted and thesis:
            admitted.expected_horizon = int(thesis.expected_horizon)

        for rec in records:
            if rec.decision == DECISION_WAIT:
                waiting_queue.append({
                    "timestamp": rec.timestamp,
                    "side": rec.side,
                    "symbol": rec.candidate_symbol,
                    "candidate_score": rec.candidate_score,
                    "blocked_by": rec.current_symbol,
                })

        if replaced_out is not None and admitted is not None:
            replacements.append({
                "timestamp": candidate.timestamp,
                "side": candidate.side,
                "out_symbol": replaced_out.symbol,
                "out_trade_id": replaced_out.trade_id,
                "in_symbol": candidate.symbol,
                "in_trade_id": candidate.trade_id,
                "out_score": replaced_out.portfolio_value_score,
                "in_score": admitted.portfolio_value_score,
                "out_actual_roi": replaced_out.actual_roi,
                "in_actual_roi": candidate.actual_roi,
            })

        utilization_samples.append(
            (len(book.long_slots) + len(book.short_slots)) / 6.0
        )

        _reindex_slots(book, candidate.side)

    analysis = _analyze_replay(all_records, replacements, utilization_samples, bundle)
    out_dir = data_dir / "portfolio"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [r.to_row() for r in all_records]
    _write_csv(out_dir / DECISION_CSV, DECISION_FIELDS, rows)

    summary = {
        "last_update": now_kst(),
        "dry_run": True,
        "mode": "PORTFOLIO_DECISION_REPLAY",
        **analysis,
        "slot_status": book.slot_snapshot(),
        "replacement_queue": waiting_queue[-20:],
        "waiting_candidates": [w for w in waiting_queue if w],
    }
    (out_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(out_dir / REPORT_MD, analysis, all_records)

    return {
        "trade_count": len(bundle),
        "decision_rows": len(all_records),
        "analysis": analysis,
        "decision_csv": str(out_dir / DECISION_CSV),
        "summary_json": str(out_dir / SUMMARY_JSON),
        "report_md": str(out_dir / REPORT_MD),
    }


def _analyze_replay(
    records: list,
    replacements: list[dict],
    utilization_samples: list[float],
    bundle: list[dict],
) -> dict:
    decision_counts = dict(Counter(r.decision for r in records))
    enter_candidates = sum(1 for r in records if r.candidate_symbol and r.decision != DECISION_IGNORE)

    missed = decision_counts.get(DECISION_WAIT, 0) + sum(
        1 for r in records if r.decision == DECISION_IGNORE and "gate=" not in r.reason
    )

    repl_success = 0
    repl_total = len(replacements)
    for rep in replacements:
        if rep.get("in_actual_roi", 0) > rep.get("out_actual_roi", 0):
            repl_success += 1

    avg_util = round(sum(utilization_samples) / len(utilization_samples), 4) if utilization_samples else 0.0

    by_action_roi: dict[str, list[float]] = defaultdict(list)
    admitted_keys = {
        (r.candidate_symbol, r.side)
        for r in records
        if r.decision == DECISION_REPLACE and r.candidate_symbol
    }
    key_to_roi = {
        (row["symbol"], str(row.get("side", row.get("direction", ""))).lower()): float(row.get("actual_roi", 0))
        for row in bundle
    }
    for key in admitted_keys:
        roi = key_to_roi.get(key, 0.0)
        by_action_roi["admitted"].append(roi)

    return {
        "candidate_events": len([r for r in records if r.candidate_symbol]),
        "decision_counts": decision_counts,
        "replacement_count": repl_total,
        "missed_trades": missed,
        "replacement_success_rate": round(repl_success / repl_total, 4) if repl_total else 0.0,
        "replacement_success_count": repl_success,
        "avg_slot_utilization": avg_util,
        "avg_long_slots": round(avg_util * 3, 2),
        "avg_short_slots": round(avg_util * 3, 2),
        "enter_candidate_count": enter_candidates,
        "admitted_avg_roi": round(
            sum(by_action_roi["admitted"]) / len(by_action_roi["admitted"]), 4
        ) if by_action_roi["admitted"] else 0.0,
    }


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_report(path: Path, analysis: dict, records: list) -> None:
    lines = [
        "# Portfolio Decision Engine V1",
        "",
        "Rule-based Long3/Short3 slot decisions. No live orders.",
        "",
        f"- Candidate events: **{analysis.get('candidate_events', 0)}**",
        f"- Avg slot utilization: **{analysis.get('avg_slot_utilization', 0)}**",
        f"- Replacements: **{analysis.get('replacement_count', 0)}**",
        f"- Missed trades (WAIT/low-score): **{analysis.get('missed_trades', 0)}**",
        f"- Replacement success rate: **{analysis.get('replacement_success_rate', 0)}**",
        "",
        "## Decision distribution",
        "",
    ]
    for dec, cnt in sorted(analysis.get("decision_counts", {}).items()):
        lines.append(f"- {dec}: {cnt}")

    lines.extend(["", "## Recent REPLACE decisions", ""])
    repl = [r for r in records if r.decision == DECISION_REPLACE][-15:]
    for r in repl:
        lines.append(f"- {r.timestamp} {r.side} {r.slot}: {r.reason}")

    path.write_text("\n".join(lines), encoding="utf-8")
