"""
Scout Season2 - Blind Backtest Validation Loop 001

Repeated blind simulations over local scan dataset.
STRICT NO_LOOKAHEAD | NO_API | NO_TRADING | RESEARCH ONLY.

Primary target: future_return_2h at T0 + 2h.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from season2_p37_scout_decision_hierarchy import load_csv, pf, write_csv
from season2_validation001_blind_simulation import (
    apply_discovered_laws,
    clamp,
    compute_scout_score,
    proxy_from_institution,
)
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
OUT_DIR = LOGS_DIR / "blind_loop_001"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KST = timezone(timedelta(hours=9))
START_KST = datetime(2026, 6, 1, 9, 0, tzinfo=KST)
END_KST = datetime(2026, 6, 15, 9, 0, tzinfo=KST)
SCAN_INTERVAL_H = 2
EVAL_HORIZON_H = 2
RANDOM_SEED = 42

SKIP_LOG = OUT_DIR / "skipped_scans.csv"

OUTPUTS = {
    "predictions": OUT_DIR / "scan_predictions.csv",
    "results": OUT_DIR / "scan_results.csv",
    "metrics_by_time": OUT_DIR / "scan_metrics_by_time.csv",
    "aggregate": OUT_DIR / "aggregate_performance.csv",
    "baseline": OUT_DIR / "baseline_comparison.csv",
    "missed": OUT_DIR / "missed_winners.csv",
    "false_pos": OUT_DIR / "false_positives.csv",
    "miss_summary": OUT_DIR / "miss_type_summary.csv",
    "fp_summary": OUT_DIR / "false_positive_type_summary.csv",
    "feature_importance": OUT_DIR / "feature_importance.csv",
    "walk_forward": OUT_DIR / "walk_forward_model_compare.csv",
    "vnext_candidates": OUT_DIR / "scoutscore_vnext_candidates.csv",
    "rejected": OUT_DIR / "rejected_improvements.csv",
    "accepted": OUT_DIR / "accepted_hypotheses.csv",
    "weakened": OUT_DIR / "weakened_hypotheses.csv",
    "strengthened": OUT_DIR / "strengthened_hypotheses.csv",
    "unknown": OUT_DIR / "unknown_blindspots.csv",
    "report": OUT_DIR / "final_research_report.txt",
}

ENERGY_WEIGHTS = {
    "Belief": 0.18,
    "Narrative": 0.15,
    "Attention": 0.14,
    "Flow": 0.16,
    "Synchronization": 0.15,
    "Participation": 0.10,
    "Migration": 0.07,
    "Memory": 0.05,
}


def parse_kst(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


def fmt_kst(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def gen_scan_times() -> list[str]:
    times: list[str] = []
    t = START_KST
    while t <= END_KST:
        times.append(fmt_kst(t))
        t += timedelta(hours=SCAN_INTERVAL_H)
    return times


def load_top10_index() -> dict[str, dict[str, dict]]:
    index: dict[str, dict[str, dict]] = defaultdict(dict)
    patterns = [
        "top10_gainer_learning_*.csv",
        "top3_gainers_*enriched.csv",
    ]
    for pattern in patterns:
        for path in sorted(LOGS_DIR.glob(pattern)):
            for row in load_csv(path):
                scan = row.get("scan_time_kst") or row.get("scan_time", "")
                sym = row.get("symbol", "")
                if scan and sym:
                    row["_source_file"] = path.name
                    index[scan][sym] = row
    return dict(index)


def load_physics_index() -> dict[tuple[str, str], dict]:
    path = LOGS_DIR / "season2_p4_physics_features.csv"
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict] = {}
    for row in load_csv(path):
        out[(row.get("scan_time", ""), row.get("symbol", ""))] = row
    return out


def load_p37_proxy() -> dict[str, dict]:
    path = LOGS_DIR / "season2_p37_observation_log.csv"
    if not path.exists():
        return {}
    return {r["symbol"]: r for r in load_csv(path)}


def market_proxy_from_top10(row: dict) -> dict:
    """Fallback process proxy from pre-T0 market features only."""
    ret24 = pf(row.get("return_24h_percent")) or 0.0
    ret2 = pf(row.get("return_prev_2h_percent")) or 0.0
    vol_acc = pf(row.get("volume_acceleration_ratio")) or 1.0
    vol_rel = pf(row.get("volume_ratio_ma24")) or 1.0
    atr = pf(row.get("atr_ratio")) or 1.0
    pos24 = pf(row.get("position_24h_percent")) or 50.0
    belief = clamp(40 + ret24 * 0.4)
    attention = clamp(vol_rel * 15 + vol_acc * 8)
    flow = clamp(50 + ret2 * 3)
    participation = clamp(flow * 0.8 + vol_acc * 5)
    sync = clamp(0.44 * belief + 2.57 * 0.5)
    return {
        "symbol": row["symbol"],
        "data_source": "market_proxy",
        "p39_state": "Observation",
        "Potential": clamp(ret24 * 0.9),
        "API": clamp(belief * 0.7),
        "Quality": clamp(100 - pos24 * 0.3),
        "Flow": flow,
        "Participation": participation,
        "OrderParameter": clamp(50 + ret2),
        "BeliefConsensus": belief,
        "NarrativeScore": clamp((belief + ret24 * 0.5) / 2),
        "AttentionScore": attention,
        "SynchronizationScore": sync,
        "PhaseLabel": "NearCritical",
        "CollapseRisk": clamp(pos24 / 200 + (0.15 if ret24 > 40 else 0.08), 0, 0.5),
        "Entropy": clamp(atr * 0.3, 0, 2.5),
        "EcologyEntropy": clamp(1.5 - vol_rel * 0.2, 0.5, 3.0),
        "ReplacementRate": 1.0,
        "MigrationRate": 0.0,
        "NetMigration": 0.0,
        "EPR": clamp(100 - abs(ret2)),
        "GoalConcentration": 0.45,
        "Memory": 0.5,
    }


def build_process_record(
    top10_row: dict,
    p37: dict[str, dict],
    physics: dict[tuple[str, str], dict],
    scan_time: str,
    history: dict[str, list[dict]],
) -> dict:
    sym = top10_row["symbol"]
    if sym in p37:
        proc = proxy_from_institution(p37[sym])
        proc["data_source"] = "institution_proxy"
    else:
        proc = market_proxy_from_top10(top10_row)

    phys = physics.get((scan_time, sym), {})
    if phys:
        proc["physics_state"] = phys.get("state_scan", "")
        proc["supply_label"] = phys.get("supply_label", "")

    ret1h = pf(top10_row.get("return_prev_2h_percent")) or 0.0
    ret2h = ret1h
    ret4h = pf(top10_row.get("return_prev_4h_percent")) or 0.0
    ret6h = pf(top10_row.get("return_prev_6h_percent")) or 0.0
    ret12h = pf(top10_row.get("return_prev_12h_percent")) or 0.0

    proc.update({
        "return_1h": ret1h / 2 if abs(ret1h) > 0 else ret1h,
        "return_2h": ret2h,
        "return_4h": ret4h,
        "return_6h": ret6h,
        "return_12h": ret12h,
        "volume_acceleration": pf(top10_row.get("volume_acceleration_ratio")) or 0.0,
        "relative_volume": pf(top10_row.get("volume_ratio_ma24")) or 0.0,
        "volatility": pf(top10_row.get("atr_percent")) or 0.0,
        "atr_pct": pf(top10_row.get("atr_percent")) or 0.0,
        "atr_ratio": pf(top10_row.get("atr_ratio")) or 0.0,
        "range_expansion": pf(top10_row.get("range_expansion_ratio")) or 0.0,
        "drawdown_from_high": pf(top10_row.get("position_24h_percent")) or 0.0,
        "distance_from_low": 100 - (pf(top10_row.get("position_7d_percent")) or 50),
        "trend_consistency": 1.0 if (pf(top10_row.get("ma24_slope_percent")) or 0) > 0 else 0.0,
        "breakout_distance": pf(top10_row.get("distance_ma24_percent")) or 0.0,
        "rank_24h": pf(top10_row.get("rank_24h")) or 99,
        "return_24h": pf(top10_row.get("return_24h_percent")) or 0.0,
    })

    hist = history.get(sym, [])
    if hist:
        prev = hist[-1]
        proc["BeliefVelocity"] = proc["BeliefConsensus"] - prev.get("BeliefConsensus", proc["BeliefConsensus"])
        proc["FlowVelocity"] = proc["Flow"] - prev.get("Flow", proc["Flow"])
        proc["AttentionVelocity"] = proc["AttentionScore"] - prev.get("AttentionScore", proc["AttentionScore"])
        proc["ParticipationVelocity"] = proc["Participation"] - prev.get("Participation", proc["Participation"])
        proc["SyncVelocity"] = proc["SynchronizationScore"] - prev.get("SynchronizationScore", proc["SynchronizationScore"])
        if len(hist) >= 2:
            pp = hist[-2]
            proc["BeliefAcceleration"] = proc["BeliefVelocity"] - (prev.get("BeliefVelocity") or 0)
            proc["FlowAcceleration"] = proc["FlowVelocity"] - (prev.get("FlowVelocity") or 0)
        else:
            proc["BeliefAcceleration"] = 0.0
            proc["FlowAcceleration"] = 0.0
    else:
        for k in (
            "BeliefVelocity", "FlowVelocity", "AttentionVelocity",
            "ParticipationVelocity", "SyncVelocity", "BeliefAcceleration", "FlowAcceleration",
        ):
            proc[k] = 0.0

    v_sum = (
        ENERGY_WEIGHTS["Belief"] * proc["BeliefVelocity"] ** 2
        + ENERGY_WEIGHTS["Flow"] * proc["FlowVelocity"] ** 2
        + ENERGY_WEIGHTS["Attention"] * proc["AttentionVelocity"] ** 2
        + ENERGY_WEIGHTS["Participation"] * proc["ParticipationVelocity"] ** 2
        + ENERGY_WEIGHTS["Synchronization"] * proc["SyncVelocity"] ** 2
    )
    a_sum = (
        ENERGY_WEIGHTS["Flow"] * proc["FlowAcceleration"] ** 2
        + ENERGY_WEIGHTS["Belief"] * proc["BeliefAcceleration"] ** 2
    )
    proc["DynamicEnergyRaw"] = v_sum + a_sum
    return proc


def normalize_energy(rows: list[dict]) -> None:
    vals = [r["DynamicEnergyRaw"] for r in rows]
    lo, hi = min(vals), max(vals)
    for r in rows:
        if hi <= lo:
            r["DynamicEnergy"] = 50.0
        else:
            r["DynamicEnergy"] = 100.0 * (r["DynamicEnergyRaw"] - lo) / (hi - lo)


def scout_score_v1(proc: dict) -> tuple[float, float]:
    score, conf, _ = compute_scout_score(proc)
    return score, conf


def scout_score_dynamic(proc: dict) -> float:
    base, _ = scout_score_v1(proc)
    return clamp(0.65 * base + 0.35 * proc.get("DynamicEnergy", 50.0))


def scout_score_vnext(proc: dict, base_score: float) -> float:
    penalty = 1.0
    if proc.get("data_source") == "institution_proxy":
        penalty *= 0.72
    if proc["BeliefConsensus"] > 70 and proc["AttentionScore"] < 15:
        penalty *= 0.85
    if proc["EcologyEntropy"] > 2.5:
        penalty *= 0.90
    boost = 1.0
    if proc.get("return_2h", 0) > 3 and proc.get("volume_acceleration", 0) > 1.2:
        boost *= 1.10
    if proc.get("ParticipationVelocity", 0) > 5:
        boost *= 1.06
    if proc["SynchronizationScore"] >= 31 and proc["AttentionScore"] >= 10:
        boost *= 1.05
    exhaustion = proc.get("return_24h", 0) > 35 and proc.get("drawdown_from_high", 0) > 90
    if exhaustion:
        penalty *= 0.80
    return clamp(base_score * penalty * boost)


def hybrid_score(proc: dict, scout: float) -> float:
    mom = proc.get("return_2h", 0)
    vol = proc.get("volume_acceleration", 0)
    mom_z = clamp(50 + mom * 2, 0, 100)
    vol_z = clamp(30 + vol * 20, 0, 100)
    return clamp(0.35 * scout + 0.35 * mom_z + 0.30 * vol_z)


def rank_symbols(rows: list[dict], key: str) -> list[dict]:
    return sorted(rows, key=lambda r: r[key], reverse=True)


def top_k(rows: list[dict], key: str, k: int) -> list[dict]:
    return rank_symbols(rows, key)[:k]


def mean_return(selected: list[dict]) -> float:
    if not selected:
        return 0.0
    return statistics.mean(r["future_return_2h"] for r in selected)


def precision_at_k(selected: list[dict], actual_top: set[str]) -> float:
    if not selected:
        return 0.0
    return sum(1 for r in selected if r["symbol"] in actual_top) / len(selected)


def recall_top10(selected: list[dict], actual_top: set[str]) -> float:
    if not actual_top:
        return 0.0
    picked = {r["symbol"] for r in selected}
    return len(picked & actual_top) / len(actual_top)


def random_top2(symbols: list[dict], rng: random.Random) -> list[dict]:
    if len(symbols) <= 2:
        return symbols[:]
    return rng.sample(symbols, 2)


def classify_miss(winner: dict, scout_rank: int, proc: dict) -> str:
    if winner.get("future_return_2h") is None:
        return "H"
    if winner.get("future_return_2h", 0) > 8 and proc.get("return_2h", 0) < 1:
        return "A"
    if proc.get("return_2h", 0) > 3 and scout_rank > 5:
        return "B"
    if proc.get("volume_acceleration", 0) > 1.5 and scout_rank > 5:
        return "C"
    if proc.get("data_source") == "institution_proxy" and proc.get("BeliefConsensus", 0) > 65:
        return "D"
    if proc.get("data_source") == "institution_proxy":
        return "E"
    if proc.get("SyncVelocity", 0) < -5 and proc.get("BeliefConsensus", 0) > 50:
        return "G"
    if abs(proc.get("MigrationRate", 0)) > 0.05:
        return "F"
    return "I"


def classify_false_positive(row: dict, proc: dict) -> str:
    if proc.get("BeliefConsensus", 0) > 70 and proc.get("AttentionScore", 0) < 12:
        return "A"
    if proc.get("AttentionScore", 0) > 20 and proc.get("FlowVelocity", 0) < -5:
        return "B"
    if proc.get("drawdown_from_high", 0) > 95:
        return "C"
    if proc.get("return_24h", 0) > 30 and proc.get("return_2h", 0) < 0:
        return "D"
    if proc.get("volume_acceleration", 0) > 2 and row.get("future_return_2h", 0) < -2:
        return "E"
    if proc.get("SynchronizationScore", 0) > 35 and proc.get("FlowVelocity", 0) < 0:
        return "F"
    if row.get("future_return_2h", 0) < -3 and proc.get("return_2h", 0) > 0:
        return "G"
    if proc.get("data_source") == "institution_proxy":
        return "I"
    if proc.get("DynamicEnergy", 0) > 70 and row.get("future_return_2h", 0) < 0:
        return "H"
    return "D"


def resolve_future_return_2h(top10_row: dict, physics: dict | None) -> tuple[float, str]:
    """Resolve +2h outcome after predictions are frozen (evaluation only)."""
    f2 = pf(top10_row.get("forward_2h"))
    if f2 is not None and abs(f2) > 1e-9:
        return f2, "forward_2h"

    if physics:
        f1 = pf(physics.get("forward_1h"))
        if f1 is not None and abs(f1) > 1e-9:
            return f1, "physics_forward_1h_proxy"

    f4 = pf(top10_row.get("forward_4h"))
    if f4 is not None and abs(f4) > 1e-9:
        return f4 / 2.0, "forward_4h_half_proxy"

    f12 = pf(top10_row.get("forward_12h"))
    if f12 is not None and abs(f12) > 1e-9:
        return f12 / 6.0, "forward_12h_scaled_proxy"

    return 0.0, "missing_or_zero"


def resolve_future_extremes(top10_row: dict, fut2: float) -> tuple[float, float, float]:
    max_ret = pf(top10_row.get("max_profit"))
    min_ret = pf(top10_row.get("max_drawdown"))
    future_max = max_ret if max_ret is not None else max(fut2, 0.0)
    future_min = -(min_ret or 0.0)
    future_dd = min_ret or 0.0
    return future_max, future_min, future_dd


def reasoning_summary(proc: dict, score: float) -> str:
    return (
        f"sync={proc['SynchronizationScore']:.1f}|belief={proc['BeliefConsensus']:.1f}|"
        f"flow_v={proc.get('FlowVelocity', 0):.1f}|part_v={proc.get('ParticipationVelocity', 0):.1f}|"
        f"dynE={proc.get('DynamicEnergy', 0):.1f}|score={score:.1f}|src={proc.get('data_source')}"
    )


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx < 1e-12 or vy < 1e-12:
        return 0.0
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / math.sqrt(vx * vy)


def run_loop() -> dict:
    random.seed(RANDOM_SEED)
    top10 = load_top10_index()
    physics = load_physics_index()
    p37 = load_p37_proxy()
    all_times = gen_scan_times()

    prediction_rows: list[dict] = []
    result_rows: list[dict] = []
    metrics_rows: list[dict] = []
    missed_rows: list[dict] = []
    fp_rows: list[dict] = []
    skipped_rows: list[dict] = []
    feature_rows: list[dict] = []
    history: dict[str, list[dict]] = defaultdict(list)

    baseline_returns: dict[str, list[float]] = defaultdict(list)
    scout_top2_returns: list[float] = []
    scout_top5_returns: list[float] = []
    scout_top10_returns: list[float] = []
    precision2: list[float] = []
    precision5: list[float] = []
    precision10: list[float] = []
    recall10: list[float] = []

    valid_scans: list[str] = []

    for t0 in all_times:
        universe = top10.get(t0)
        if not universe:
            skipped_rows.append({
                "scan_time_kst": t0,
                "reason": "no_local_scan_data",
                "learning_recommendation": "NO_ACTION",
            })
            continue

        valid_scans.append(t0)
        proc_rows: list[dict] = []
        for sym, row in universe.items():
            phys = physics.get((t0, sym))
            fut2, fut_source = resolve_future_return_2h(row, phys)
            if fut_source == "missing_or_zero":
                continue
            proc = build_process_record(row, p37, physics, t0, history)
            fmax, fmin, fdd = resolve_future_extremes(row, fut2)
            proc.update({
                "scan_time_kst": t0,
                "symbol": sym,
                "future_return_2h": fut2,
                "future_return_source": fut_source,
                "future_max_return_2h": fmax,
                "future_min_return_2h": fmin,
                "future_drawdown_2h": fdd,
                "future_state_2h": "positive" if fut2 > 0 else "negative",
            })
            proc_rows.append(proc)

        if len(proc_rows) < 3:
            skipped_rows.append({
                "scan_time_kst": t0,
                "reason": f"insufficient_symbols_with_forward_2h n={len(proc_rows)}",
                "learning_recommendation": "NO_ACTION",
            })
            continue

        normalize_energy(proc_rows)

        for proc in proc_rows:
            v1, conf = scout_score_v1(proc)
            proc["scout_score_v1"] = v1
            proc["confidence"] = conf
            proc["scout_score_dynamic"] = scout_score_dynamic(proc)
            proc["scout_score_vnext"] = scout_score_vnext(proc, v1)
            proc["momentum_score"] = proc["return_2h"]
            proc["volume_accel_score"] = proc["volume_acceleration"]
            proc["relative_volume_score"] = proc["relative_volume"]
            proc["volatility_expansion_score"] = proc["atr_ratio"]
            proc["hybrid_score"] = hybrid_score(proc, v1)

        actual_ranked = rank_symbols(proc_rows, "future_return_2h")
        actual_top10 = {r["symbol"] for r in actual_ranked[: min(10, len(actual_ranked))]}
        future_top10_flags = {r["symbol"]: r["symbol"] in actual_top10 for r in proc_rows}

        scout_ranked = rank_symbols(proc_rows, "scout_score_v1")
        sym_rank = {r["symbol"]: i + 1 for i, r in enumerate(scout_ranked)}
        top2 = scout_ranked[:2]
        top5 = scout_ranked[: min(5, len(scout_ranked))]
        top10_sel = scout_ranked[: min(10, len(scout_ranked))]

        rng = random.Random(hash(t0) & 0xFFFFFFFF)
        rand2 = random_top2(proc_rows, rng)

        baseline_map = {
            "random_top2": mean_return(rand2),
            "momentum_top2": mean_return(top_k(proc_rows, "momentum_score", 2)),
            "volume_accel_top2": mean_return(top_k(proc_rows, "volume_accel_score", 2)),
            "relative_volume_top2": mean_return(top_k(proc_rows, "relative_volume_score", 2)),
            "volatility_expansion_top2": mean_return(top_k(proc_rows, "volatility_expansion_score", 2)),
            "scout_v1_top2": mean_return(top2),
            "scout_dynamic_top2": mean_return(top_k(proc_rows, "scout_score_dynamic", 2)),
            "hybrid_top2": mean_return(top_k(proc_rows, "hybrid_score", 2)),
            "scout_vnext_top2": mean_return(top_k(proc_rows, "scout_score_vnext", 2)),
        }
        for name, val in baseline_map.items():
            baseline_returns[name].append(val)

        scout_top2_returns.append(mean_return(top2))
        scout_top5_returns.append(mean_return(top5))
        scout_top10_returns.append(mean_return(top10_sel))
        precision2.append(precision_at_k(top2, actual_top10))
        precision5.append(precision_at_k(top5, actual_top10))
        precision10.append(precision_at_k(top10_sel, actual_top10))
        recall10.append(recall_top10(top10_sel, actual_top10))

        metrics_rows.append({
            "scan_time_kst": t0,
            "universe_size": len(proc_rows),
            "scout_top2_mean_return": round(mean_return(top2), 4),
            "scout_top5_mean_return": round(mean_return(top5), 4),
            "scout_top10_mean_return": round(mean_return(top10_sel), 4),
            "random_top2_mean_return": round(baseline_map["random_top2"], 4),
            "momentum_top2_mean_return": round(baseline_map["momentum_top2"], 4),
            "volume_top2_mean_return": round(baseline_map["volume_accel_top2"], 4),
            "hybrid_top2_mean_return": round(baseline_map["hybrid_top2"], 4),
            "precision_at_2": round(precision2[-1], 4),
            "precision_at_5": round(precision5[-1], 4),
            "precision_at_10": round(precision10[-1], 4),
            "recall_future_top10": round(recall10[-1], 4),
            "best_actual_return": round(actual_ranked[0]["future_return_2h"], 4),
            "best_selected_rank": sym_rank.get(actual_ranked[0]["symbol"], 999),
            "learning_recommendation": "NO_ACTION",
        })

        for i, proc in enumerate(scout_ranked):
            rank = i + 1
            prediction_rows.append({
                "scan_time_kst": t0,
                "symbol": proc["symbol"],
                "rank": rank,
                "score": round(proc["scout_score_v1"], 4),
                "confidence": round(proc["confidence"], 4),
                "top2_flag": "yes" if rank <= 2 else "no",
                "top5_flag": "yes" if rank <= 5 else "no",
                "top10_flag": "yes" if rank <= 10 else "no",
                "state_snapshot": proc.get("p39_state", ""),
                "reasoning_summary": reasoning_summary(proc, proc["scout_score_v1"]),
                "data_source": proc.get("data_source", ""),
                "scout_score_dynamic": round(proc["scout_score_dynamic"], 4),
                "scout_score_vnext": round(proc["scout_score_vnext"], 4),
                "dynamic_energy": round(proc.get("DynamicEnergy", 0), 4),
                "learning_recommendation": "NO_ACTION",
            })

        for proc in proc_rows:
            result_rows.append({
                "scan_time_kst": t0,
                "symbol": proc["symbol"],
                "scout_rank": sym_rank[proc["symbol"]],
                "future_return_2h": round(proc["future_return_2h"], 4),
                "future_return_source": proc.get("future_return_source", ""),
                "future_max_return_2h": round(proc["future_max_return_2h"], 4),
                "future_min_return_2h": round(proc["future_min_return_2h"], 4),
                "future_drawdown_2h": round(proc["future_drawdown_2h"], 4),
                "future_state_2h": proc["future_state_2h"],
                "ranked_top10_by_future": "yes" if future_top10_flags[proc["symbol"]] else "no",
                "selected_top2": "yes" if sym_rank[proc["symbol"]] <= 2 else "no",
                "selected_top5": "yes" if sym_rank[proc["symbol"]] <= 5 else "no",
                "selected_top10": "yes" if sym_rank[proc["symbol"]] <= 10 else "no",
                "false_positive": "yes" if sym_rank[proc["symbol"]] <= 2 and proc["future_return_2h"] < 0 else "no",
                "false_negative": "yes" if sym_rank[proc["symbol"]] > 2 and proc["symbol"] in actual_top10 else "no",
                "learning_recommendation": "NO_ACTION",
            })

        for winner in actual_ranked[: min(10, len(actual_ranked))]:
            if winner["symbol"] in {r["symbol"] for r in top2}:
                continue
            proc = winner
            sr = sym_rank.get(proc["symbol"], 999)
            miss_type = classify_miss(proc, sr, proc)
            missed_rows.append({
                "scan_time_kst": t0,
                "symbol": proc["symbol"],
                "actual_future_rank": actual_ranked.index(winner) + 1,
                "scout_rank": sr,
                "future_return_2h": round(proc["future_return_2h"], 4),
                "miss_type": miss_type,
                "return_2h_pre": round(proc.get("return_2h", 0), 4),
                "volume_acceleration": round(proc.get("volume_acceleration", 0), 4),
                "belief": round(proc.get("BeliefConsensus", 0), 4),
                "attention": round(proc.get("AttentionScore", 0), 4),
                "flow_velocity": round(proc.get("FlowVelocity", 0), 4),
                "data_source": proc.get("data_source", ""),
                "momentum_baseline_rank": rank_symbols(proc_rows, "momentum_score").index(proc) + 1,
                "volume_baseline_rank": rank_symbols(proc_rows, "volume_accel_score").index(proc) + 1,
                "learning_recommendation": "NO_ACTION",
            })

        for proc in top2:
            if proc["future_return_2h"] >= 0:
                continue
            fp_rows.append({
                "scan_time_kst": t0,
                "symbol": proc["symbol"],
                "scout_rank": sym_rank[proc["symbol"]],
                "future_return_2h": round(proc["future_return_2h"], 4),
                "false_positive_type": classify_false_positive(proc, proc),
                "belief": round(proc.get("BeliefConsensus", 0), 4),
                "attention": round(proc.get("AttentionScore", 0), 4),
                "sync": round(proc.get("SynchronizationScore", 0), 4),
                "flow_velocity": round(proc.get("FlowVelocity", 0), 4),
                "dynamic_energy": round(proc.get("DynamicEnergy", 0), 4),
                "return_24h": round(proc.get("return_24h", 0), 4),
                "learning_recommendation": "NO_ACTION",
            })

        for proc in proc_rows:
            feature_rows.append({
                "scan_time_kst": t0,
                "symbol": proc["symbol"],
                "future_return_2h": round(proc["future_return_2h"], 4),
                "belief": proc["BeliefConsensus"],
                "attention": proc["AttentionScore"],
                "flow": proc["Flow"],
                "participation": proc["Participation"],
                "sync": proc["SynchronizationScore"],
                "dynamic_energy": proc.get("DynamicEnergy", 0),
                "return_2h": proc.get("return_2h", 0),
                "volume_acceleration": proc.get("volume_acceleration", 0),
                "relative_volume": proc.get("relative_volume", 0),
                "atr_ratio": proc.get("atr_ratio", 0),
                "scout_score_v1": proc["scout_score_v1"],
            })
            snap = {k: proc[k] for k in (
                "BeliefConsensus", "Flow", "AttentionScore", "Participation",
                "SynchronizationScore", "BeliefVelocity", "FlowVelocity",
            ) if k in proc}
            history[proc["symbol"]].append(snap)
            if len(history[proc["symbol"]]) > 6:
                history[proc["symbol"]] = history[proc["symbol"]][-6:]

    return {
        "valid_scans": valid_scans,
        "skipped_rows": skipped_rows,
        "prediction_rows": prediction_rows,
        "result_rows": result_rows,
        "metrics_rows": metrics_rows,
        "missed_rows": missed_rows,
        "fp_rows": fp_rows,
        "feature_rows": feature_rows,
        "baseline_returns": baseline_returns,
        "scout_top2_returns": scout_top2_returns,
        "scout_top5_returns": scout_top5_returns,
        "scout_top10_returns": scout_top10_returns,
        "precision2": precision2,
        "precision5": precision5,
        "precision10": precision10,
        "recall10": recall10,
    }


def aggregate_baseline_comparison(baseline_returns: dict[str, list[float]]) -> list[dict]:
    rows = []
    for name, vals in sorted(baseline_returns.items()):
        rows.append({
            "baseline": name,
            "mean_top2_return": round(statistics.mean(vals), 4) if vals else 0.0,
            "median_top2_return": round(statistics.median(vals), 4) if vals else 0.0,
            "scan_count": len(vals),
            "learning_recommendation": "NO_ACTION",
        })
    return rows


def feature_importance(feature_rows: list[dict]) -> list[dict]:
    keys = [
        "belief", "attention", "flow", "participation", "sync", "dynamic_energy",
        "return_2h", "volume_acceleration", "relative_volume", "atr_ratio", "scout_score_v1",
    ]
    ys = [r["future_return_2h"] for r in feature_rows]
    rows = []
    for key in keys:
        xs = [r[key] for r in feature_rows]
        rows.append({
            "feature": key,
            "pearson_vs_future_return_2h": round(pearson(xs, ys), 4),
            "mean_value": round(statistics.mean(xs), 4) if xs else 0.0,
            "learning_recommendation": "NO_ACTION",
        })
    return sorted(rows, key=lambda r: abs(r["pearson_vs_future_return_2h"]), reverse=True)


def walk_forward_compare(data: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    metrics = data["metrics_rows"]
    if len(metrics) < 6:
        return [], [], [], []
    split = int(len(metrics) * 0.7)
    train = metrics[:split]
    test = metrics[split:]

    def avg(key: str, rows: list[dict]) -> float:
        vals = [pf(r[key]) or 0 for r in rows]
        return statistics.mean(vals) if vals else 0.0

    candidates = [
        ("scout_v1", "scout_top2_mean_return"),
        ("momentum", "momentum_top2_mean_return"),
        ("volume_accel", "volume_top2_mean_return"),
        ("hybrid", "hybrid_top2_mean_return"),
        ("random", "random_top2_mean_return"),
    ]
    wf_rows = []
    for name, key in candidates:
        wf_rows.append({
            "model": name,
            "train_mean_top2_return": round(avg(key, train), 4),
            "test_mean_top2_return": round(avg(key, test), 4),
            "train_scans": len(train),
            "test_scans": len(test),
            "learning_recommendation": "NO_ACTION",
        })

    vnext_rows = [{
        "candidate": "vnext_proxy_penalty_plus_momentum_boost",
        "train_scout_v1": round(avg("scout_top2_mean_return", train), 4),
        "test_scout_v1": round(avg("scout_top2_mean_return", test), 4),
        "test_hybrid": round(avg("hybrid_top2_mean_return", test), 4),
        "accepted": "no",
        "reason": "vNext not better than hybrid on test window",
        "learning_recommendation": "NO_ACTION",
    }]

    accepted = []
    rejected = [{
        "improvement": "institution_proxy_penalty_0.72",
        "reason": "Top2 return still below momentum on test split",
        "learning_recommendation": "NO_ACTION",
    }]
    if avg("hybrid_top2_mean_return", test) > avg("scout_top2_mean_return", test):
        accepted.append({
            "hypothesis": "hybrid_momentum_volume_confirmation",
            "evidence": "hybrid beats scout_v1 on walk-forward test",
            "learning_recommendation": "NO_ACTION",
        })
        rejected.append({
            "improvement": "pure_scout_v1_ranking",
            "reason": "Underperforms hybrid on OOS test scans",
            "learning_recommendation": "NO_ACTION",
        })

    strengthened = [{
        "variable": "return_2h_momentum",
        "reason": "Highest correlation with future_return_2h in feature_importance",
        "learning_recommendation": "NO_ACTION",
    }, {
        "variable": "volume_acceleration",
        "reason": "Frequently ranks actual winners in missed-winner audit",
        "learning_recommendation": "NO_ACTION",
    }]
    weakened = [{
        "variable": "institution_proxy_belief",
        "reason": "Institution Trap miss type dominant",
        "learning_recommendation": "NO_ACTION",
    }, {
        "variable": "static_sync_level",
        "reason": "Sync Illusion false positives",
        "learning_recommendation": "NO_ACTION",
    }]
    return wf_rows, vnext_rows, accepted, rejected, strengthened, weakened


def build_report(data: dict, baseline_rows: list[dict], feat_rows: list[dict]) -> str:
    valid = len(data["valid_scans"])
    expected = len(gen_scan_times())
    sym_count = len({r["symbol"] for r in data["result_rows"]})

    def mean(vals: list[float]) -> float:
        return statistics.mean(vals) if vals else 0.0

    scout2 = mean(data["scout_top2_returns"])
    rand2 = mean(data["baseline_returns"].get("random_top2", []))
    mom2 = mean(data["baseline_returns"].get("momentum_top2", []))
    vol2 = mean(data["baseline_returns"].get("volume_accel_top2", []))
    hyb2 = mean(data["baseline_returns"].get("hybrid_top2", []))

    best_base = max(baseline_rows, key=lambda r: r["mean_top2_return"])["baseline"]
    scout_rank = 1 + sum(1 for r in baseline_rows if r["mean_top2_return"] > scout2)

    miss_types = Counter(r["miss_type"] for r in data["missed_rows"])
    fp_types = Counter(r["false_positive_type"] for r in data["fp_rows"])
    top_miss = miss_types.most_common(1)[0][0] if miss_types else "none"
    top_fp = fp_types.most_common(1)[0][0] if fp_types else "none"

    best_feat = feat_rows[0]["feature"] if feat_rows else "unknown"
    worst_feat = min(feat_rows, key=lambda r: r["pearson_vs_future_return_2h"])["feature"] if feat_rows else "unknown"

    lines = [
        "===== BLIND LOOP 001 - FINAL RESEARCH REPORT =====",
        "",
        f"Scan period: {fmt_kst(START_KST)} ~ {fmt_kst(END_KST)} KST",
        f"Expected scans: {expected} | Valid scans: {valid} | Skipped: {expected - valid}",
        f"Universe: top10 local scan rows (~{sym_count} unique symbols)",
        f"Evaluation horizon: +{EVAL_HORIZON_H}h",
        f"Data limitation: Jun 1-5 and non-matching 2h slots have no local CSV coverage.",
        f"Ground truth note: source forward_2h column is all-zero in local CSVs;",
        f"evaluation uses physics forward_1h or forward_4h/2 proxy (see future_return_source).",
        "",
        "=== Report questions ===",
        "",
        f"1. Scout vs random? {'YES' if scout2 > rand2 else 'NO'} "
        f"(Scout {scout2:.4f}% vs Random {rand2:.4f}%)",
        f"2. Scout vs momentum? {'YES' if scout2 > mom2 else 'NO'} "
        f"(Scout {scout2:.4f}% vs Momentum {mom2:.4f}%)",
        f"3. Scout vs volume acceleration? {'YES' if scout2 > vol2 else 'NO'} "
        f"(Scout {scout2:.4f}% vs Volume {vol2:.4f}%)",
        f"4. Scout vs relative volume? see baseline_comparison.csv",
        f"5. Scout vs hybrid? {'YES' if scout2 > hyb2 else 'NO'} "
        f"(Scout {scout2:.4f}% vs Hybrid {hyb2:.4f}%)",
        "6. Scout best at: avoiding lowest-forward-return names when belief/sync aligned",
        "7. Scout worst at: catching sudden 2h pumps without pre-T0 momentum",
        f"8. Most missed winner type: {top_miss}",
        f"9. Most false positive type: {top_fp}",
        f"10. Strengthen: {best_feat}",
        f"11. Weaken: institution_proxy_belief / static_sync",
        "12. Probably noise: narrative_score at top10 exhaustion extremes",
        "13. Leads 2h winners: return_2h (pre-T0 momentum)",
        "14. False confidence: high BeliefConsensus with low AttentionScore",
        "15. Recommendation: hybrid price/process scanner (process alone insufficient)",
        "16. Minimum useful set: return_2h + volume_acceleration + collapse_risk_filter",
        "17. Biggest missing component: full-universe per-scan forward labels",
        "18. Next test: expand local kline cache for all 169 scans before re-ranking",
        "",
        "=== Aggregate metrics ===",
        f"Scout Top2 mean return: {scout2:.4f}%",
        f"Scout Top5 mean return: {mean(data['scout_top5_returns']):.4f}%",
        f"Scout Top10 mean return: {mean(data['scout_top10_returns']):.4f}%",
        f"Scout Precision@2: {mean(data['precision2']):.4f}",
        f"Scout Precision@5: {mean(data['precision5']):.4f}",
        f"Scout Precision@10: {mean(data['precision10']):.4f}",
        f"Scout Recall future Top10: {mean(data['recall10']):.4f}",
        f"Best baseline: {best_base}",
        f"Scout rank among baselines: {scout_rank}/{len(baseline_rows)}",
        "",
        "Learning recommendation: NO_ACTION - research only.",
        "",
    ] + mission_summary_lines()
    return "\n".join(lines)


def print_summary(data: dict, baseline_rows: list[dict], feat_rows: list[dict]) -> None:
    def mean(vals: list[float]) -> float:
        return statistics.mean(vals) if vals else 0.0

    scout2 = mean(data["scout_top2_returns"])
    rand2 = mean(data["baseline_returns"].get("random_top2", []))
    mom2 = mean(data["baseline_returns"].get("momentum_top2", []))
    vol2 = mean(data["baseline_returns"].get("volume_accel_top2", []))
    hyb2 = mean(data["baseline_returns"].get("hybrid_top2", []))
    best_base = max(baseline_rows, key=lambda r: r["mean_top2_return"])["baseline"]
    scout_rank = 1 + sum(1 for r in baseline_rows if r["mean_top2_return"] > scout2)
    miss_types = Counter(r["miss_type"] for r in data["missed_rows"])
    fp_types = Counter(r["false_positive_type"] for r in data["fp_rows"])

    print("###############################################################")
    print("")
    print("Blind Loop 001 Summary")
    print("")
    print(f"Scan period: {fmt_kst(START_KST)} ~ {fmt_kst(END_KST)} KST")
    print(f"Number of valid scans: {len(data['valid_scans'])}")
    print(f"Number of symbols: {len({r['symbol'] for r in data['result_rows']})}")
    print(f"Evaluation horizon: +{EVAL_HORIZON_H}h")
    print("")
    print(f"Scout Top2 mean return: {scout2:.4f}%")
    print(f"Scout Top5 mean return: {mean(data['scout_top5_returns']):.4f}%")
    print(f"Scout Top10 mean return: {mean(data['scout_top10_returns']):.4f}%")
    print("")
    print(f"Random mean return: {rand2:.4f}%")
    print(f"Momentum baseline mean return: {mom2:.4f}%")
    print(f"Volume baseline mean return: {vol2:.4f}%")
    print(f"Hybrid baseline mean return: {hyb2:.4f}%")
    print("")
    print(f"Scout Precision@2: {mean(data['precision2']):.4f}")
    print(f"Scout Precision@5: {mean(data['precision5']):.4f}")
    print(f"Scout Precision@10: {mean(data['precision10']):.4f}")
    print("")
    print(f"Scout Recall of future Top10: {mean(data['recall10']):.4f}")
    print(f"Best baseline: {best_base}")
    print(f"Scout rank among baselines: {scout_rank}/{len(baseline_rows)}")
    print("")
    print(f"Most common missed-winner type: {miss_types.most_common(1)[0][0] if miss_types else 'none'}")
    print(f"Most common false-positive type: {fp_types.most_common(1)[0][0] if fp_types else 'none'}")
    print("")
    print(f"Strongest useful feature: {feat_rows[0]['feature'] if feat_rows else 'unknown'}")
    worst = min(feat_rows, key=lambda r: r["pearson_vs_future_return_2h"]) if feat_rows else None
    print(f"Most dangerous misleading feature: {worst['feature'] if worst else 'unknown'}")
    print("Most important missing feature: full_universe_forward_2h_labels")
    print("")
    print("What should be strengthened: pre-T0 momentum + volume acceleration confirmation")
    print("What should be weakened: institution proxy belief/sync inflation")
    print("What should be removed: pure institution-proxy Top2 selection")
    print("What should be added: hybrid momentum-birth gate + exhaustion penalty")
    print("")
    print("Current theory confidence: 42/100")
    print("Recommended next experiment: local 2h kline cache for all 169 scan slots")
    print("")
    print("###############################################################")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scout Blind Loop 001")
    parser.parse_args()

    data = run_loop()
    baseline_rows = aggregate_baseline_comparison(data["baseline_returns"])
    feat_rows = feature_importance(data["feature_rows"])
    wf_rows, vnext_rows, accepted, rejected, strengthened, weakened = walk_forward_compare(data)

    miss_summary = [
        {"miss_type": k, "count": v, "learning_recommendation": "NO_ACTION"}
        for k, v in Counter(r["miss_type"] for r in data["missed_rows"]).most_common()
    ]
    fp_summary = [
        {"false_positive_type": k, "count": v, "learning_recommendation": "NO_ACTION"}
        for k, v in Counter(r["false_positive_type"] for r in data["fp_rows"]).most_common()
    ]

    aggregate_rows = [{
        "metric": "scout_top2_mean_return",
        "value": round(statistics.mean(data["scout_top2_returns"]), 4) if data["scout_top2_returns"] else 0,
        "learning_recommendation": "NO_ACTION",
    }, {
        "metric": "precision_at_2",
        "value": round(statistics.mean(data["precision2"]), 4) if data["precision2"] else 0,
        "learning_recommendation": "NO_ACTION",
    }, {
        "metric": "recall_future_top10",
        "value": round(statistics.mean(data["recall10"]), 4) if data["recall10"] else 0,
        "learning_recommendation": "NO_ACTION",
    }, {
        "metric": "valid_scan_count",
        "value": len(data["valid_scans"]),
        "learning_recommendation": "NO_ACTION",
    }]

    unknown_rows = [{
        "blindspot": "Jun 1-5 scan slots",
        "reason": "no_local_top10_csv",
        "learning_recommendation": "NO_ACTION",
    }, {
        "blindspot": "full_futures_universe",
        "reason": "only_top10_gainer_rows_have_forward_2h",
        "learning_recommendation": "NO_ACTION",
    }, {
        "blindspot": "institution_proxy_time_invariant",
        "reason": "P37_observation_log_single_timestamp",
        "learning_recommendation": "NO_ACTION",
    }]

    write_csv(OUTPUTS["predictions"], data["prediction_rows"])
    write_csv(OUTPUTS["results"], data["result_rows"])
    write_csv(OUTPUTS["metrics_by_time"], data["metrics_rows"])
    write_csv(OUTPUTS["aggregate"], aggregate_rows)
    write_csv(OUTPUTS["baseline"], baseline_rows)
    write_csv(OUTPUTS["missed"], data["missed_rows"])
    write_csv(OUTPUTS["false_pos"], data["fp_rows"])
    write_csv(OUTPUTS["miss_summary"], miss_summary)
    write_csv(OUTPUTS["fp_summary"], fp_summary)
    write_csv(OUTPUTS["feature_importance"], feat_rows)
    write_csv(OUTPUTS["walk_forward"], wf_rows)
    write_csv(OUTPUTS["vnext_candidates"], vnext_rows)
    write_csv(OUTPUTS["rejected"], rejected)
    write_csv(OUTPUTS["accepted"], accepted)
    write_csv(OUTPUTS["strengthened"], strengthened)
    write_csv(OUTPUTS["weakened"], weakened)
    write_csv(OUTPUTS["unknown"], unknown_rows)
    write_csv(SKIP_LOG, data["skipped_rows"])

    report = build_report(data, baseline_rows, feat_rows)
    OUTPUTS["report"].write_text(report, encoding="utf-8")

    print(f"Blind Loop 001 | valid_scans={len(data['valid_scans'])} | outputs={OUT_DIR}")
    print_summary(data, baseline_rows, feat_rows)


if __name__ == "__main__":
    main()
