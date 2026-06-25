"""Blind validation of cluster formulas vs baselines."""

from __future__ import annotations

import statistics
from collections import defaultdict

from scout_auto_os.engine.research.directional.dna.formulas import ClusterFormula, rank_by_formula
from scout_auto_os.engine.research.directional.engines import rank_long, rank_short
from scout_auto_os.engine.research.directional.evaluation import aggregate_directional, to_long_metrics, to_short_metrics
from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics
from scout_auto_os.engine.research.zero_base.random_baseline import generate_random_draws


def split_scans(scans: list[str], train_ratio: float = 0.7) -> tuple[list[str], list[str]]:
    scans = sorted(scans)
    cut = max(1, int(len(scans) * train_ratio))
    return scans[:cut], scans[cut:]


def evaluate_picks_on_scans(
    by_scan: dict,
    fwd: dict,
    scans: list[str],
    pick_fn,
    direction: str,
) -> list[dict]:
    samples: list[dict] = []
    for scan_kst in scans:
        rows = by_scan[scan_kst]
        syms = pick_fn(rows)
        for sym in syms:
            klines = fwd.get((scan_kst, sym))
            if not klines:
                continue
            raw = compute_forward_metrics(klines)
            if not raw:
                continue
            m = to_long_metrics(raw) if direction == "long" else to_short_metrics(raw)
            samples.append(m)
    return samples


def blind_validate_engine(
    engine: str,
    direction: str,
    formulas: list[ClusterFormula],
    by_scan: dict,
    fwd: dict,
    train_scans: list[str],
    blind_scans: list[str],
    baseline_agg: dict,
) -> list[dict]:
    results: list[dict] = []

    def random_pick(rows):
        syms = [r["symbol"] for r in rows]
        return generate_random_draws(syms, 5, 1, 42)[0]

    def engine_pick(rows):
        if direction == "long":
            return rank_long(rows, engine, 5)
        return rank_short(rows, engine, 5)

    for label, pick_fn in (
        ("RANDOM", random_pick),
        ("PATTERN_CHAMPION", engine_pick),
    ):
        blind_samples = evaluate_picks_on_scans(by_scan, fwd, blind_scans, pick_fn, direction)
        agg = aggregate_directional(blind_samples, direction)
        results.append({
            "engine": engine,
            "formula_name": label,
            "split": "blind",
            "sample_count": agg.get("sample_count", 0),
            "avg_return_2h": agg.get("avg_return_2h", 0),
            "win_rate": agg.get("win_rate", 0),
            "profit_factor": agg.get("profit_factor", 0),
            "trap_rate": agg.get("trap_rate", 0),
            "max_drawdown_avg": agg.get("max_drawdown_avg", 0),
            "delta_vs_random": round(float(agg.get("avg_return_2h", 0)) - float(baseline_agg.get("avg_return_2h", 0)), 4),
        })

    for formula in formulas:
        def pick_fn(rows, f=formula):
            return rank_by_formula(rows, f, 5)

        blind_samples = evaluate_picks_on_scans(by_scan, fwd, blind_scans, pick_fn, direction)
        agg = aggregate_directional(blind_samples, direction)
        results.append({
            "engine": engine,
            "formula_name": formula.name,
            "split": "blind",
            "sample_count": agg.get("sample_count", 0),
            "avg_return_2h": agg.get("avg_return_2h", 0),
            "win_rate": agg.get("win_rate", 0),
            "profit_factor": agg.get("profit_factor", 0),
            "trap_rate": agg.get("trap_rate", 0),
            "max_drawdown_avg": agg.get("max_drawdown_avg", 0),
            "delta_vs_random": round(float(agg.get("avg_return_2h", 0)) - float(baseline_agg.get("avg_return_2h", 0)), 4),
            "delta_vs_champion": 0.0,
        })

    champ = next((r for r in results if r["formula_name"] == "PATTERN_CHAMPION"), {})
    for r in results:
        if r["formula_name"] not in ("RANDOM", "PATTERN_CHAMPION"):
            r["delta_vs_champion"] = round(
                float(r.get("avg_return_2h", 0)) - float(champ.get("avg_return_2h", 0)), 4,
            )
    return results


def cluster_performance(samples: list[dict], direction: str) -> dict:
    if not samples:
        return {"sample_count": 0}
    if direction == "short":
        rets = [float(s["metrics"].get("short_return_2h", -float(s["metrics"].get("return_2h", 0)))) for s in samples]
    else:
        rets = [float(s["metrics"].get("return_2h", 0)) for s in samples]
    traps = sum(1 for s in samples if s["metrics"].get("label_trap"))
    wins = sum(1 for r in rets if r >= 3.0)
    return {
        "sample_count": len(samples),
        "avg_return_2h": round(statistics.mean(rets), 4) if rets else 0,
        "return_4h_avg": round(statistics.mean([float(s["metrics"].get("return_4h", 0)) for s in samples]), 4) if samples else 0,
        "win_rate": round(wins / len(rets) * 100, 2) if rets else 0,
        "trap_rate": round(traps / len(samples) * 100, 2) if samples else 0,
        "profit_factor": _pf(rets),
    }


def _pf(rets: list[float]) -> float:
    wins = sum(r for r in rets if r > 0)
    losses = abs(sum(r for r in rets if r < 0))
    if losses <= 0:
        return round(wins, 2) if wins else 0.0
    return round(wins / losses, 2)
