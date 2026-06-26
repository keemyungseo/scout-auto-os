"""Multi-round survivor system — generalization over time, vol, regime."""

from __future__ import annotations

import statistics
from collections import defaultdict

from scout_auto_os.engine.research.formula_league_v2.constants import (
    BASELINE_FORMULA_ID,
    SURVIVOR_MIN_ROUNDS_WIN,
    SURVIVOR_ROUND_TOP_PCT,
)
from scout_auto_os.engine.research.formula_league_v2.evaluator import evaluate_formula_on_scans
from scout_auto_os.engine.research.formula_league_v2.formula import SearchFormula
from scout_auto_os.engine.research.zero_base.validation import classify_regime


def _week_key(scan: str) -> str:
    return scan[:10]


def _month_key(scan: str) -> str:
    return scan[:7]


def scan_regime_map(annotated: dict[str, list[dict]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for scan, rows in annotated.items():
        out[scan] = classify_regime(rows)
    return out


def scan_volatility_band(annotated: dict[str, list[dict]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for scan, rows in annotated.items():
        ranges = [float(r["features"].get("1h_current_range_pct", 0)) for r in rows]
        if not ranges:
            out[scan] = "unknown"
            continue
        med = statistics.median(ranges)
        if med >= 20.0:
            out[scan] = "high_volatility"
        elif med <= 10.0:
            out[scan] = "low_volatility"
        else:
            out[scan] = "mid_volatility"
    return out


def build_survivor_rounds(
    all_scans: list[str],
    train_ratio: float = 0.7,
) -> list[dict]:
    scans = sorted(all_scans)
    cut = max(1, int(len(scans) * train_ratio))
    blind = scans[cut:]

    rounds: list[dict] = [
        {"round_id": "temporal_blind", "scans": blind, "split_type": "temporal"},
    ]

    by_week: dict[str, list[str]] = defaultdict(list)
    for s in scans:
        by_week[_week_key(s)].append(s)
    for wk, wscans in sorted(by_week.items()):
        rounds.append({"round_id": f"week_{wk}", "scans": wscans, "split_type": "weekly"})

    by_month: dict[str, list[str]] = defaultdict(list)
    for s in scans:
        by_month[_month_key(s)].append(s)
    for mo, mscans in sorted(by_month.items()):
        rounds.append({"round_id": f"month_{mo}", "scans": mscans, "split_type": "monthly"})

    return rounds


def run_survivor_league(
    formulas: list[SearchFormula],
    annotated: dict[str, list[dict]],
    all_scans: list[str],
    fwd: dict,
    th,
    stats: dict,
    regime_map: dict[str, str],
    vol_map: dict[str, str],
    train_ratio: float = 0.7,
) -> tuple[list[dict], list[dict], list[SearchFormula]]:
    rounds = build_survivor_rounds(all_scans, train_ratio)
    baseline_id = BASELINE_FORMULA_ID

    round_results: list[dict] = []
    formula_round_scores: dict[str, list[dict]] = defaultdict(list)

    for rnd in rounds:
        scans = rnd["scans"]
        round_metrics: list[dict] = []
        for formula in formulas:
            _, agg = evaluate_formula_on_scans(formula, annotated, scans, fwd, th, stats)
            agg["round_id"] = rnd["round_id"]
            agg["split_type"] = rnd["split_type"]
            round_metrics.append(agg)
            formula_round_scores[formula.formula_id].append(agg)

        baseline_avg = next(
            (float(m["avg_return_2h"]) for m in round_metrics if m["formula_id"] == baseline_id),
            0.0,
        )
        for m in round_metrics:
            m["baseline_avg_2h"] = baseline_avg
            m["beats_baseline"] = float(m.get("avg_return_2h", 0)) >= baseline_avg
        round_metrics.sort(key=lambda x: -float(x.get("generalization_score", 0)))
        keep_n = max(3, int(len(round_metrics) * SURVIVOR_ROUND_TOP_PCT))
        survivors_ids = {m["formula_id"] for m in round_metrics[:keep_n]}
        survivors_ids.add(baseline_id)
        for m in round_metrics:
            m["round_survivor"] = m["formula_id"] in survivors_ids
        round_results.extend(round_metrics)

    # Regime rounds
    for regime in sorted(set(regime_map.values())):
        scans = [s for s in all_scans if regime_map.get(s) == regime]
        if len(scans) < 2:
            continue
        for formula in formulas:
            _, agg = evaluate_formula_on_scans(formula, annotated, scans, fwd, th, stats)
            agg["round_id"] = f"regime_{regime}"
            agg["split_type"] = "regime"
            agg["beats_baseline"] = False
            formula_round_scores[formula.formula_id].append(agg)
            round_results.append(agg)

    for vol in sorted(set(vol_map.values())):
        scans = [s for s in all_scans if vol_map.get(s) == vol]
        if len(scans) < 2:
            continue
        for formula in formulas:
            _, agg = evaluate_formula_on_scans(formula, annotated, scans, fwd, th, stats)
            agg["round_id"] = f"vol_{vol}"
            agg["split_type"] = "volatility"
            formula_round_scores[formula.formula_id].append(agg)
            round_results.append(agg)

    survivor_rows: list[dict] = []
    surviving_formulas: list[SearchFormula] = []
    formula_by_id = {f.formula_id: f for f in formulas}

    baseline_gen = next(
        (float(r.get("generalization_score", 0)) for r in round_results if r.get("formula_id") == baseline_id and r.get("round_id") == "temporal_blind"),
        0.0,
    )

    for fid, rounds_data in formula_round_scores.items():
        blind_rounds = [r for r in rounds_data if r.get("split_type") in ("temporal", "weekly", "monthly")]
        wins = sum(1 for r in blind_rounds if r.get("beats_baseline"))
        win_rate = wins / len(blind_rounds) if blind_rounds else 0.0
        stab = statistics.pstdev([float(r.get("avg_return_2h", 0)) for r in blind_rounds]) if len(blind_rounds) > 1 else 0.0
        avg_gen = statistics.mean([float(r.get("generalization_score", 0)) for r in blind_rounds]) if blind_rounds else 0.0
        temporal = next((r for r in blind_rounds if r.get("round_id") == "temporal_blind"), None)
        survived = (
            fid == baseline_id
            or (
                win_rate >= SURVIVOR_MIN_ROUNDS_WIN
                and avg_gen > baseline_gen
                and temporal is not None
                and temporal.get("beats_baseline")
                and float(temporal.get("trade_count", 0)) >= 8
            )
        )
        row = {
            "formula_id": fid,
            "rounds_total": len(blind_rounds),
            "rounds_beat_baseline": wins,
            "round_win_rate": round(win_rate, 4),
            "stability_std": round(stab, 4),
            "avg_generalization_score": round(avg_gen, 4),
            "survived": survived,
        }
        survivor_rows.append(row)
        if survived and fid in formula_by_id:
            surviving_formulas.append(formula_by_id[fid])

    survivor_rows.sort(key=lambda x: (-x["round_win_rate"], -x["avg_generalization_score"]))
    return round_results, survivor_rows, surviving_formulas
