"""
Scout Learning Season2 - P8 Participant State Engine & Market Psychology Discovery

Research only. Estimate collective participant psychology — not universal trading rules.
"""

import argparse
import csv
import itertools
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p5_historical_expansion import (
    MIN_BUCKET,
    STEP1_DATES,
    build_unified_records,
    collect_missing,
    compression_ctx,
    global_baseline_mae,
    late_leader,
    loo_mae,
    spread_f6,
    vol_zone,
)
from season2_p6_market_memory import attach_forward_targets, symbol_archetype
from season2_p7_behaviour_grammar import assign_cluster, build_sentence, enrich_physics

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

STATE_CSV = LOGS_DIR / "season2_p8_participant_states.csv"
MEMORY_CSV = LOGS_DIR / "season2_p8_memory_horizons.csv"
INTERACTION_CSV = LOGS_DIR / "season2_p8_context_interactions.csv"
FAMILY_CSV = LOGS_DIR / "season2_p8_participant_families.csv"
TRANSITION_CSV = LOGS_DIR / "season2_p8_state_transitions.csv"
REGISTRY_CSV = LOGS_DIR / "season2_p8_hypothesis_registry.csv"
ENGINE_CSV = LOGS_DIR / "season2_p8_participant_engine_output.csv"
REPORT_TXT = LOGS_DIR / "season2_p8_research_report.txt"

FAMILY_CSV_P7 = LOGS_DIR / "season2_p7_symbol_grammar_families.csv"
HORIZON_CSV_P6 = LOGS_DIR / "season2_p6_memory_horizon_summary.csv"

MIN_STATE = 6
MIN_INTERACT = 8

# Candidate participant states — scored empirically, not assumed
PARTICIPANT_STATES = [
    "inventory_recovery",
    "trapped_long",
    "trapped_short",
    "profit_taking",
    "fresh_accumulation",
    "late_chasing",
    "panic_exit",
    "forced_liquidation",
    "exhaustion",
    "rotation",
    "distribution",
    "hidden_accumulation",
    "violent_markup",
    "steady_markup",
    "late_markup",
    "neutral_observation",
]

# Auto-generated from scan labels
AUTO_STATES = [
    "scan_compression",
    "scan_exhaustion",
    "scan_expansion",
    "scan_warmup",
    "scan_choppy",
    "scan_transition",
]

# Memory horizons mapped to available lookback fields (adaptive, not fixed MA)
MEMORY_HORIZONS = [
    ("4h", "distance_ma6_percent", "return_prev_2h_percent"),
    ("8h", "distance_ma6_percent", "return_prev_4h_percent"),
    ("12h", "distance_ma12_percent", "return_prev_6h_percent"),
    ("24h", "distance_ma24_percent", "return_prev_12h_percent"),
    ("3d", "distance_ma48_percent", "return_prev_24h_percent"),
    ("7d", "distance_ma84_percent", "return_prev_7d_percent"),
    ("14d", "position_7d_percent", "return_prev_7d_percent"),
    ("30d", "position_7d_percent", "return_prev_7d_percent"),
    ("60d", "position_7d_percent", "return_prev_7d_percent"),
]

# Phase buckets for transition model
PHASE_MAP = {
    "fresh_accumulation": "accumulation",
    "hidden_accumulation": "accumulation",
    "inventory_recovery": "recovery",
    "steady_markup": "markup",
    "violent_markup": "markup",
    "late_chasing": "late_markup",
    "late_markup": "late_markup",
    "profit_taking": "distribution",
    "distribution": "distribution",
    "trapped_long": "collapse",
    "panic_exit": "collapse",
    "forced_liquidation": "collapse",
    "exhaustion": "distribution",
    "rotation": "rotation",
    "trapped_short": "recovery",
    "neutral_observation": "rotation",
}


def _pf(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def enrich_panel_fields(records: list[dict]) -> None:
    from season2_p5_historical_expansion import discover_dataset_paths, load_rows

    extra: dict[tuple[str, str], dict] = {}
    for path in discover_dataset_paths():
        for row in load_rows(path):
            scan = row.get("scan_time_kst", "")
            sym = row.get("symbol", "")
            if not scan or not sym:
                continue
            extra[(scan, sym)] = {
                k: _pf(row.get(k))
                for k in (
                    "distance_ma6_percent", "distance_ma12_percent", "distance_ma24_percent",
                    "distance_ma48_percent", "distance_ma84_percent",
                    "return_prev_2h_percent", "return_prev_4h_percent", "return_prev_6h_percent",
                    "return_prev_12h_percent", "return_prev_48h_percent", "return_prev_7d_percent",
                    "position_48h_percent", "max_drawdown",
                )
            }
            extra[(scan, sym)]["break_48h"] = row.get("break_48h_highest_close", "") in ("YES", "True", "true", "1")
            extra[(scan, sym)]["break_7d"] = row.get("break_7d_highest_close", "") in ("YES", "True", "true", "1")
    for record in records:
        data = extra.get((record["scan_time"], record["symbol"]), {})
        record.update({k: v for k, v in data.items() if v is not None})


def load_grammar_families() -> dict[str, str]:
    if not FAMILY_CSV_P7.exists():
        return {}
    sym_to_family: dict[str, str] = {}
    with FAMILY_CSV_P7.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for sym in (row.get("symbols") or "").split("|"):
                if sym:
                    sym_to_family[sym] = row["family_id"]
    return sym_to_family


def state_scores(record: dict) -> dict[str, float]:
    """Score each participant state as behavioural proxy — no fixed TA meaning."""
    s: dict[str, float] = {k: 0.0 for k in PARTICIPANT_STATES + AUTO_STATES}

    pos7 = record.get("position_7d_percent") or 0
    pos24 = record.get("position_24h_percent") or 0
    ret24 = record.get("return_24h_at_scan") or 0
    rank = record.get("rank") or 99
    dd = record.get("max_drawdown") or 0
    lw = record.get("lower_wick_pct") or 0
    uw = record.get("upper_wick_pct") or 0
    body = record.get("body_pct") or 50
    vol_r = record.get("vol_ratio_6") or record.get("volume_ratio_ma24") or 1.0
    compressed = compression_ctx(record)
    cong = record.get("congestion_bars_12") or 0

    if lw >= 40 and record.get("long_tail_recovery"):
        s["inventory_recovery"] += 3
    if record.get("near_recent_low") or (record.get("dist_from_12h_low_pct") or 99) < 5:
        s["inventory_recovery"] += 1.5
    if record.get("vol_recovery"):
        s["inventory_recovery"] += 2

    if record.get("break_24h") and uw >= 40:
        s["trapped_long"] += 3
    if record.get("fast_rejection") and pos24 >= 90:
        s["trapped_long"] += 2
    if record.get("break_24h") and ret24 >= 25 and uw >= 35:
        s["trapped_long"] += 2

    if lw >= 45 and not record.get("bull", True) and pos7 < 40:
        s["trapped_short"] += 2
    if record.get("gap_up") and lw >= 30:
        s["trapped_short"] += 1.5

    if (record.get("near_recent_high") or pos7 >= 90) and record.get("vol_exhaustion"):
        s["profit_taking"] += 3
    if uw >= 45 and vol_r >= 1.5:
        s["profit_taking"] += 2

    if compressed and vol_r < 1.0 and record.get("slow_absorption"):
        s["fresh_accumulation"] += 3
    if compressed and vol_r < 0.9:
        s["fresh_accumulation"] += 2
    if cong >= 6 and vol_r < 1.1:
        s["hidden_accumulation"] += 2.5
    if compressed and not record.get("break_24h"):
        s["hidden_accumulation"] += 1.5

    if rank <= 3 and ret24 >= 30 and vol_r >= 1.3:
        s["late_chasing"] += 4
    if late_leader(record) and body >= 60:
        s["late_markup"] += 3
    if late_leader(record):
        s["late_chasing"] += 2

    if record.get("collapse_label") == "YES" and vol_r >= 2:
        s["panic_exit"] += 3
    if (record.get("state_scan") or "") == "Exhaustion" and uw >= 40:
        s["panic_exit"] += 2

    if dd >= 15 and record.get("collapse_label") == "YES":
        s["forced_liquidation"] += 3
    if record.get("vol_shock") and (record.get("target_f6") or 0) < -10:
        s["forced_liquidation"] += 2

    if record.get("vol_exhaustion") or (record.get("state_scan") or "") == "Exhaustion":
        s["exhaustion"] += 2.5
    if pos7 >= 95:
        s["exhaustion"] += 2

    if cong >= 5 or (record.get("reversal_count_12") or 0) >= 5:
        s["rotation"] += 2.5

    if uw >= 40 and record.get("vol_exhaustion"):
        s["distribution"] += 3
    if record.get("break_24h") and uw >= 35:
        s["distribution"] += 2

    if record.get("range_expand") and vol_r >= 2 and body >= 65:
        s["violent_markup"] += 3
    if (record.get("consec_candles") or 0) >= 2 and vol_r >= 1.2 and body >= 45:
        s["steady_markup"] += 2.5
    if (record.get("state_4h") or "") in ("Continuation", "Acceleration", "Expansion"):
        s["steady_markup"] += 1.5

    scan = record.get("state_scan") or "Choppy"
    s[f"scan_{scan.lower()}"] = 5.0

    if max(s.values()) < 1:
        s["neutral_observation"] = 1.0

    return s


def assign_participant_state(record: dict) -> tuple[str, float, str]:
    scores = state_scores(record)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    primary, score = ranked[0]
    secondary = ranked[1][0] if len(ranked) > 1 and ranked[1][1] >= score * 0.7 else ""
    return primary, round(score, 2), secondary


def memory_bucket(value: float | None, horizon: str) -> str:
    if value is None:
        return f"{horizon}_na"
    cuts = [5, 15, 30] if "position" not in horizon else [50, 75, 90]
    if "7d" in horizon or "14d" in horizon or "30d" in horizon:
        cuts = [60, 80, 95]
    bounds = [-1e9] + cuts + [1e9]
    labels = [f"lt{cuts[0]}"] + [f"{cuts[i]}_{cuts[i+1]}" for i in range(len(cuts) - 1)] + [f"gt{cuts[-1]}"]
    for i in range(len(bounds) - 1):
        if bounds[i] <= float(value) < bounds[i + 1]:
            return f"{horizon}_{labels[i]}"
    return f"{horizon}_na"


def test_memory_horizons(records: list[dict]) -> list[dict]:
    rows = []
    for label, dist_field, ret_field in MEMORY_HORIZONS:
        groups: dict[str, list] = defaultdict(list)
        for r in records:
            val = r.get(dist_field) if r.get(dist_field) is not None else r.get(ret_field)
            if val is None or r.get("target_f6") is None:
                continue
            groups[memory_bucket(val, label)].append(r["target_f6"])
        if sum(len(v) for v in groups.values()) < MIN_STATE * 2:
            continue
        meds = [statistics.median(v) for v in groups.values() if len(v) >= MIN_STATE]
        spread = max(meds) - min(meds) if len(meds) >= 2 else 0
        mae = loo_mae(
            [r for r in records if (r.get(dist_field) is not None or r.get(ret_field) is not None) and r.get("target_f6") is not None],
            lambda r, f=dist_field, rf=ret_field, lb=label: memory_bucket(
                r.get(f) if r.get(f) is not None else r.get(rf), lb
            ),
        )[0]
        rows.append(
            {
                "horizon": label,
                "field_used": dist_field if any(r.get(dist_field) is not None for r in records) else ret_field,
                "n": sum(len(v) for v in groups.values()),
                "spread_f6": round(spread, 2),
                "loo_mae": round(mae, 2),
                "confidence": "high" if spread >= 8 else "medium" if spread >= 5 else "hypothesis",
            }
        )
    return rows


def best_memory_per_symbol(records: list[dict]) -> dict[str, str]:
    sym_best: dict[str, str] = {}
    sym_groups: dict[str, list] = defaultdict(list)
    for r in records:
        sym_groups[r["symbol"]].append(r)
    for sym, group in sym_groups.items():
        if len(group) < MIN_STATE:
            continue
        horizon_rows = test_memory_horizons(group)
        if horizon_rows:
            best = max(horizon_rows, key=lambda x: x["spread_f6"])
            sym_best[sym] = best["horizon"]
    return sym_best


def state_outcome_stats(records: list[dict]) -> dict[str, dict]:
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r["participant_state"]].append(r)
    stats = {}
    for state, group in groups.items():
        f6 = [x.get("target_f6") for x in group if x.get("target_f6") is not None]
        collapses = sum(1 for x in group if x.get("collapse_label") == "YES")
        stats[state] = {
            "n": len(group),
            "median_f6": round(statistics.median(f6), 2) if f6 else 0,
            "collapse_pct": round(collapses / len(group) * 100, 1) if group else 0,
            "persist_pct": round(sum(1 for v in f6 if v >= 0) / len(f6) * 100, 1) if f6 else 50,
            "spread_f6": round(max(f6) - min(f6), 2) if len(f6) >= 2 else 0,
        }
    return stats


def hypothesis_status(state: str, stats: dict, baseline: float, cond_count: int) -> tuple[str, str]:
    st = stats.get(state, {})
    spread = st.get("spread_f6", 0)
    n = st.get("n", 0)
    if cond_count >= 1:
        return "CONDITIONAL", f"revived in {cond_count} context(s)"
    if n >= 20 and spread >= 8:
        return "ACTIVE", "strong outcome separation across participant state"
    if n >= 12 and spread >= 5:
        return "RETIRED", "moderate edge; monitor on expansion"
    if n >= MIN_STATE:
        return "DEAD_FOR_NOW", "weak global separation; search conditional revival"
    return "OPEN", "insufficient sample"


def search_context_interactions(records: list[dict], baseline: float) -> list[dict]:
    """Task 3: participant_state × grammar × family × memory × trend_cluster."""
    rows: list[dict] = []
    dims = [
        ("participant_state", lambda r: r.get("participant_state", "unknown")),
        ("grammar", lambda r: r.get("primary_trigram", "unknown")[:40]),
        ("symbol_family", lambda r: r.get("grammar_family", "unknown")),
        ("memory_horizon", lambda r: r.get("adaptive_memory", "unknown")),
        ("trend_cluster", lambda r: r.get("behaviour_cluster", "unknown")),
        ("vol_zone", lambda r: vol_zone(r)),
    ]

    for state in {r["participant_state"] for r in records}:
        state_subset = [r for r in records if r["participant_state"] == state]
        if len(state_subset) < MIN_STATE:
            continue
        global_mae = loo_mae(state_subset, lambda r, s=state: s)[0]
        global_spread = spread_f6(state_subset, lambda r, s=state: s)

        for dim_name, dim_fn in dims[1:]:
            ctx_values = {dim_fn(r) for r in state_subset}
            for ctx_val in ctx_values:
                if ctx_val in ("unknown", ""):
                    continue
                ctx_subset = [r for r in state_subset if dim_fn(r) == ctx_val]
                if len(ctx_subset) < MIN_INTERACT:
                    continue
                ctx_mae = loo_mae(ctx_subset, lambda r, s=state: s)[0]
                ctx_spread = spread_f6(ctx_subset, lambda r, s=state: s)
                gain = global_mae - ctx_mae
                f6 = [r.get("target_f6") for r in ctx_subset if r.get("target_f6") is not None]
                med = round(statistics.median(f6), 2) if f6 else None
                if gain >= 0.25 or (ctx_spread >= 5 and ctx_mae < baseline * 0.98):
                    rows.append(
                        {
                            "participant_state": state,
                            "interaction_dim": dim_name,
                            "context_value": str(ctx_val)[:50],
                            "sample_size": len(ctx_subset),
                            "global_mae": round(global_mae, 2),
                            "context_mae": round(ctx_mae, 2),
                            "gain": round(gain, 2),
                            "context_spread_f6": round(ctx_spread, 2),
                            "median_forward_6h": med,
                            "confidence": "high" if gain >= 0.5 else "medium" if gain >= 0.35 else "hypothesis",
                            "status": "CONDITIONAL" if gain >= 0.35 else "OPEN",
                        }
                    )

        for (d1, fn1), (d2, fn2) in itertools.combinations(dims[1:4], 2):
            combos: dict[tuple, list] = defaultdict(list)
            for r in state_subset:
                combos[(fn1(r), fn2(r))].append(r)
            for (v1, v2), grp in combos.items():
                if v1 in ("unknown", "") or v2 in ("unknown", "") or len(grp) < MIN_INTERACT:
                    continue
                ctx_mae = loo_mae(grp, lambda r, s=state: s)[0]
                ctx_spread = spread_f6(grp, lambda r, s=state: s)
                gain = global_mae - ctx_mae
                if gain >= 0.35 and ctx_spread >= 4:
                    rows.append(
                        {
                            "participant_state": state,
                            "interaction_dim": f"{d1}+{d2}",
                            "context_value": f"{v1}|{v2}"[:50],
                            "sample_size": len(grp),
                            "global_mae": round(global_mae, 2),
                            "context_mae": round(ctx_mae, 2),
                            "gain": round(gain, 2),
                            "context_spread_f6": round(ctx_spread, 2),
                            "median_forward_6h": round(statistics.median([x["target_f6"] for x in grp if x.get("target_f6") is not None]), 2) if grp else None,
                            "confidence": "high" if gain >= 0.5 else "medium",
                            "status": "CONDITIONAL",
                        }
                    )
    return rows


def participant_families(records: list[dict]) -> list[dict]:
    sym_vecs: dict[str, dict] = defaultdict(lambda: Counter())
    sym_meta: dict[str, list] = defaultdict(list)
    for r in records:
        sym = r["symbol"]
        sym_vecs[sym][r["participant_state"]] += 1
        sym_vecs[sym][f"mem:{r.get('adaptive_memory', '?')}"] += 1
        sym_vecs[sym][f"gram:{r.get('primary_trigram', '?')[:20]}"] += 1
        sym_meta[sym].append(r)

    def sym_similarity(a: str, b: str) -> float:
        keys = set(sym_vecs[a]) | set(sym_vecs[b])
        if not keys:
            return 0.0
        dot = sum(sym_vecs[a].get(k, 0) * sym_vecs[b].get(k, 0) for k in keys)
        na = math.sqrt(sum(v * v for v in sym_vecs[a].values()))
        nb = math.sqrt(sum(v * v for v in sym_vecs[b].values()))
        return dot / (na * nb) if na and nb else 0.0

    symbols = sorted(sym_vecs.keys(), key=lambda s: -len(sym_meta[s]))
    assigned: set[str] = set()
    families: dict[str, list[str]] = {}
    for sym in symbols:
        if sym in assigned:
            continue
        fid = f"participant_family_{len(families) + 1}"
        families[fid] = [sym]
        assigned.add(sym)
        for other in symbols:
            if other in assigned:
                continue
            if sym_similarity(sym, other) >= 0.55:
                families[fid].append(other)
                assigned.add(other)

    rows = []
    for fid, members in sorted(families.items(), key=lambda x: -len(x[1])):
        all_recs = [r for sym in members for r in sym_meta[sym]]
        f6 = [r.get("target_f6") for r in all_recs if r.get("target_f6") is not None]
        collapses = sum(1 for r in all_recs if r.get("collapse_label") == "YES")
        states = Counter(r["participant_state"] for r in all_recs)
        rows.append(
            {
                "family_id": fid,
                "symbols": "|".join(members[:15]),
                "symbol_count": len(members),
                "appearances": len(all_recs),
                "dominant_psychology": states.most_common(1)[0][0] if states else "",
                "psychology_mix": "|".join(f"{s}({c})" for s, c in states.most_common(3)),
                "median_forward_6h": round(statistics.median(f6), 2) if f6 else "",
                "collapse_rate_pct": round(collapses / len(all_recs) * 100, 1) if all_recs else 0,
                "trend_persistence_pct": round(sum(1 for v in f6 if v >= 0) / len(f6) * 100, 1) if f6 else "",
                "memory_dominant": Counter(r.get("adaptive_memory") for r in all_recs).most_common(1)[0][0] if all_recs else "",
            }
        )
    return rows


def build_transition_matrix(records: list[dict]) -> list[dict]:
    """Task 5: phase transitions from consecutive scans per symbol."""
    by_sym: dict[str, list] = defaultdict(list)
    for r in records:
        by_sym[r["symbol"]].append(r)
    transitions: Counter[tuple[str, str]] = Counter()
    transition_f6: dict[tuple[str, str], list] = defaultdict(list)
    transition_collapse: Counter[tuple[str, str]] = Counter()

    for sym, group in by_sym.items():
        sorted_g = sorted(group, key=lambda x: x["scan_time"])
        for i in range(len(sorted_g) - 1):
            cur = sorted_g[i]
            nxt = sorted_g[i + 1]
            if cur["date"] == nxt["date"]:
                continue
            p_from = PHASE_MAP.get(cur["participant_state"], "rotation")
            p_to = PHASE_MAP.get(nxt["participant_state"], "rotation")
            transitions[(p_from, p_to)] += 1
            if cur.get("target_f6") is not None:
                transition_f6[(p_from, p_to)].append(cur["target_f6"])
            if cur.get("collapse_label") == "YES":
                transition_collapse[(p_from, p_to)] += 1

    rows = []
    for (p_from, p_to), count in transitions.most_common():
        f6 = transition_f6.get((p_from, p_to), [])
        total_from = sum(c for (a, _), c in transitions.items() if a == p_from)
        prob = round(count / total_from * 100, 1) if total_from else 0
        med = round(statistics.median(f6), 2) if f6 else ""
        collapse = round(transition_collapse.get((p_from, p_to), 0) / count * 100, 1) if count else 0
        stable = "stable" if prob >= 25 and collapse < 20 else "fragile" if collapse >= 30 else "mixed"
        rows.append(
            {
                "from_phase": p_from,
                "to_phase": p_to,
                "count": count,
                "transition_probability_pct": prob,
                "median_forward_6h": med,
                "collapse_rate_pct": collapse,
                "stability": stable,
                "confidence": "high" if count >= 15 else "medium" if count >= 8 else "hypothesis",
            }
        )
    return rows


def recommend_action(state: str, stats: dict, record: dict) -> tuple[str, str]:
    st = stats.get(state, {})
    med = st.get("median_f6", 0)
    collapse = st.get("collapse_pct", 0)
    persist = st.get("persist_pct", 50)

    if collapse >= 35 or state in ("panic_exit", "forced_liquidation", "trapped_long"):
        return "Avoid", f"participant {state}: {collapse}% collapse in cohort"
    if med >= 8 and collapse < 10 and persist >= 55:
        return "Strong Buy", f"{state} cohort med_f6={med}% persist={persist}%"
    if med >= 4 and collapse < 15:
        return "Buy", f"{state} shows positive forward bias med={med}%"
    if med >= 0 and collapse < 20:
        return "Hold", f"{state} neutral-positive; monitor persistence"
    if state in ("late_chasing", "late_markup", "exhaustion"):
        return "Watch", f"late-cycle psychology {state}; elevated fade risk"
    return "Watch", f"{state} insufficient edge or mixed outcomes"


def participant_engine_row(record: dict, stats: dict, cond_for_state: list[dict]) -> dict:
    state = record["participant_state"]
    st = stats.get(state, {})
    action, reason = recommend_action(state, stats, record)
    risks = []
    if st.get("collapse_pct", 0) >= 25:
        risks.append("high_collapse_cohort")
    if record.get("behaviour_cluster") in ("distribution", "persistent_trend"):
        risks.append(record["behaviour_cluster"])
    if cond_for_state:
        risks.append("conditional_only")

    return {
        "date": record["date"],
        "symbol": record["symbol"],
        "scan_time": record["scan_time"],
        "market_regime": record.get("market_regime"),
        "participant_state": state,
        "secondary_state": record.get("secondary_state"),
        "state_confidence_score": record.get("state_score"),
        "adaptive_memory_horizon": record.get("adaptive_memory"),
        "behaviour_sentence": record.get("sentence"),
        "grammar_family": record.get("grammar_family"),
        "expected_persistence_pct": st.get("persist_pct", 50),
        "expected_return_6h_pct": st.get("median_f6", 0),
        "expected_drawdown_pct": record.get("max_drawdown"),
        "collapse_probability_pct": st.get("collapse_pct", 0),
        "confidence": "high" if st.get("n", 0) >= 30 else "medium" if st.get("n", 0) >= 15 else "hypothesis",
        "hypothesis_status": record.get("state_status", "OPEN"),
        "recommended_action": action,
        "reason": reason,
        "risk_factors": "|".join(risks) if risks else "",
        "recommendation_score": round(max(0, 5 + (st.get("median_f6") or 0) * 0.2 - st.get("collapse_pct", 0) * 0.05), 1),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-limit", type=int, default=1)
    parser.add_argument("--skip-collect", action="store_true")
    args = parser.parse_args()

    collected = []
    if not args.skip_collect:
        missing = [d for d in STEP1_DATES if not (LOGS_DIR / f"top10_gainer_learning_{d.replace('-', '')}.csv").exists()]
        collected = collect_missing(missing, args.collect_limit)

    records = build_unified_records()
    enrich_physics(records)
    enrich_panel_fields(records)
    attach_forward_targets(records)
    baseline = global_baseline_mae(records)

    grammar_families = load_grammar_families()
    sym_memory = best_memory_per_symbol(records)
    global_memory_rows = test_memory_horizons(records)
    best_global_memory = max(global_memory_rows, key=lambda x: x["spread_f6"])["horizon"] if global_memory_rows else "24h"

    sym_groups: dict[str, list] = defaultdict(list)
    for r in records:
        sym_groups[r["symbol"]].append(r)
    sym_arch = {sym: symbol_archetype(grp) for sym, grp in sym_groups.items()}

    for record in records:
        primary, score, secondary = assign_participant_state(record)
        record["participant_state"] = primary
        record["state_score"] = score
        record["secondary_state"] = secondary
        sent = build_sentence(record)
        record.update(sent)
        record["grammar_family"] = grammar_families.get(record["symbol"], "unknown")
        record["adaptive_memory"] = sym_memory.get(record["symbol"], best_global_memory)
        record["symbol_archetype"] = sym_arch.get(record["symbol"], {}).get("archetype", "unknown")
        record["behaviour_cluster"] = assign_cluster(record.get("tokens", []))

    state_stats = state_outcome_stats(records)
    interaction_rows = search_context_interactions(records, baseline)
    cond_by_state: dict[str, list] = defaultdict(list)
    for row in interaction_rows:
        if row.get("status") == "CONDITIONAL":
            cond_by_state[row["participant_state"]].append(row)

    state_rows = []
    registry_rows = []
    for state in sorted(set(r["participant_state"] for r in records)):
        st = state_stats.get(state, {})
        cond_n = len(cond_by_state.get(state, []))
        status, reason = hypothesis_status(state, state_stats, baseline, cond_n)
        for r in records:
            if r["participant_state"] == state:
                r["state_status"] = status
        state_rows.append(
            {
                "participant_state": state,
                "frequency": st.get("n", 0),
                "median_forward_6h": st.get("median_f6"),
                "collapse_probability_pct": st.get("collapse_pct"),
                "persistence_pct": st.get("persist_pct"),
                "outcome_spread_f6": st.get("spread_f6"),
                "conditional_contexts": cond_n,
                "status": status,
                "confidence": "high" if st.get("n", 0) >= 30 else "medium" if st.get("n", 0) >= 15 else "hypothesis",
            }
        )
        registry_rows.append(
            {
                "hypothesis_name": state,
                "current_status": status,
                "sample_size": st.get("n", 0),
                "median_forward_6h": st.get("median_f6"),
                "collapse_pct": st.get("collapse_pct"),
                "conditional_revivals": cond_n,
                "decision_reason": reason,
            }
        )

    family_rows = participant_families(records)
    transition_rows = build_transition_matrix(records)

    engine_rows = []
    for record in records[-40:]:
        cond = cond_by_state.get(record["participant_state"], [])
        engine_rows.append(participant_engine_row(record, state_stats, cond))

    write_csv(STATE_CSV, state_rows)
    write_csv(MEMORY_CSV, global_memory_rows)
    write_csv(INTERACTION_CSV, interaction_rows)
    write_csv(FAMILY_CSV, family_rows)
    write_csv(TRANSITION_CSV, transition_rows)
    write_csv(REGISTRY_CSV, registry_rows)
    write_csv(ENGINE_CSV, engine_rows)

    dates = sorted({r["date"] for r in records})
    top_states = sorted(state_rows, key=lambda x: -x["frequency"])[:8]
    lines = [
        "===== SCOUT SEASON2 P8 - PARTICIPANT STATE ENGINE =====",
        "",
        f"Sample: {len(records)} records | {dates[0]}..{dates[-1]} ({len(dates)} days)",
        f"Collected this run: {collected or '(none)'}",
        f"Baseline MAE: {baseline:.2f}% | Best global memory: {best_global_memory}",
        "",
        "--- Task 1: Participant state proxies ---",
    ]
    for row in top_states:
        lines.append(
            f"  {row['participant_state']}: n={row['frequency']} med_f6={row['median_forward_6h']}% "
            f"collapse={row['collapse_probability_pct']}% [{row['status']}]"
        )

    lines.extend(["", "--- Task 2: Adaptive memory horizons ---"])
    for row in sorted(global_memory_rows, key=lambda x: -x["spread_f6"])[:6]:
        lines.append(
            f"  {row['horizon']}: spread={row['spread_f6']} mae={row['loo_mae']} field={row['field_used']}"
        )

    lines.extend(["", "--- Task 3: Context interactions (top gain) ---"])
    if interaction_rows:
        for row in sorted(interaction_rows, key=lambda x: -(x.get("gain") or 0))[:6]:
            lines.append(
                f"  {row['participant_state']} × {row['interaction_dim']}={row['context_value']}: "
                f"gain={row['gain']} n={row['sample_size']} [{row['status']}]"
            )
    else:
        lines.append("  (no interaction cleared threshold)")

    lines.extend(["", "--- Task 5: Phase transitions (top) ---"])
    for row in transition_rows[:8]:
        lines.append(
            f"  {row['from_phase']} -> {row['to_phase']}: p={row['transition_probability_pct']}% "
            f"n={row['count']} collapse={row['collapse_rate_pct']}% [{row['stability']}]"
        )

    active = sum(1 for r in registry_rows if r["current_status"] == "ACTIVE")
    conditional = sum(1 for r in registry_rows if r["current_status"] == "CONDITIONAL")
    lines.extend([
        "",
        f"--- Registry: ACTIVE={active} CONDITIONAL={conditional} ---",
        "",
        "--- Philosophy ---",
        " Features are proxies for participant behaviour, not mathematics",
        " Never permanently discard — DEAD_FOR_NOW may REVIVE in context",
        "",
        f"Engine: {ENGINE_CSV}",
        "=" * 58,
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P8 PARTICIPANT STATE ENGINE =====")
    print(f"Records: {len(records)} | States: {len(state_rows)} | Memory best: {best_global_memory}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
