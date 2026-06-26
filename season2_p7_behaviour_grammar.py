"""
Scout Learning Season2 - P7 Behaviour Grammar & Sequence Intelligence

Research only. Discover recurring behavioural sequences, not isolated indicators.
"""

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p5_historical_expansion import (
    CONTEXTS,
    MIN_BUCKET,
    MIN_CONTEXT,
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

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_CSV = LOGS_DIR / "season2_p7_behaviour_sequences.csv"
CLUSTER_CSV = LOGS_DIR / "season2_p7_behaviour_clusters.csv"
CONDITIONAL_CSV = LOGS_DIR / "season2_p7_conditional_behaviour.csv"
FAMILY_CSV = LOGS_DIR / "season2_p7_symbol_grammar_families.csv"
PERSIST_CSV = LOGS_DIR / "season2_p7_behaviour_persistence.csv"
REGISTRY_CSV = LOGS_DIR / "season2_p7_grammar_registry.csv"
ENGINE_CSV = LOGS_DIR / "season2_p7_behaviour_engine_output.csv"
REPORT_TXT = LOGS_DIR / "season2_p7_research_report.txt"
PHYSICS_CSV = LOGS_DIR / "season2_p4_physics_features.csv"
SYMBOL_MEM_CSV = LOGS_DIR / "season2_p6_symbol_memory.csv"

HORIZONS = ["30m", "1h", "2h", "4h", "6h", "12h", "24h"]
MIN_SEQ = 6
MIN_COND = 8

# Empirical token vocabulary — meanings learned from outcomes, not assumed
TOKEN_ORDER = [
    "small_body", "mid_body", "large_body",
    "upper_wick", "lower_wick",
    "inside_bar", "engulf", "fast_rejection",
    "vol_increase", "vol_decrease", "vol_exhaustion",
    "congestion", "compression", "breakout", "breakout_fail",
    "vol_expansion", "vol_contraction",
    "trend_cont", "trend_cont_bear", "gap_momentum",
    "absorption", "distribution_candle",
]

CLUSTER_RULES = [
    ("accumulation", lambda t: "compression" in t and ("vol_decrease" in t or "congestion" in t)),
    ("rotation", lambda t: "congestion" in t and "small_body" in t),
    ("explosive", lambda t: "large_body" in t and "vol_increase" in t and "trend_cont" in t),
    ("distribution", lambda t: "upper_wick" in t and ("vol_exhaustion" in t or "fast_rejection" in t)),
    ("manipulation", lambda t: "breakout_fail" in t or ("breakout" in t and "fast_rejection" in t)),
    ("persistent_trend", lambda t: "trend_cont" in t and "vol_increase" in t and "compression" not in t),
    ("late_trend", lambda t: "vol_exhaustion" in t and "large_body" in t),
    ("failed_breakout", lambda t: "breakout" in t and ("fast_rejection" in t or "upper_wick" in t)),
    ("mean_reverting", lambda t: "lower_wick" in t and "vol_decrease" in t),
]


def _pf(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _pb(val):
    if val in (True, "True", "true", "1", 1):
        return True
    if val in (False, "False", "false", "0", 0, "", None):
        return False
    return bool(val)


def enrich_physics(records: list[dict]) -> None:
    if not PHYSICS_CSV.exists():
        return
    index: dict[tuple[str, str], dict] = {}
    with PHYSICS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row.get("scan_time", ""), row.get("symbol", ""))
            index[key] = row
    float_keys = (
        "body_pct", "upper_wick_pct", "lower_wick_pct", "congestion_bars_12",
        "consec_candles", "vol_ratio_6", "vol_ratio_12", "reversal_count_12",
        "dist_from_12h_high_pct", "dist_from_12h_low_pct", "range_ratio_3v5",
    )
    bool_keys = (
        "inside_bar", "engulf", "fast_rejection", "vol_exhaustion", "vol_persist_up",
        "vol_persist_down", "vol_shock", "vol_recovery", "range_expand", "range_contract",
        "break_24h", "gap_up", "gap_down", "near_recent_high", "near_recent_low",
        "long_tail_recovery", "long_tail_rejection", "slow_absorption", "three_push_up",
    )
    for record in records:
        row = index.get((record["scan_time"], record["symbol"]))
        if not row:
            continue
        for key in float_keys:
            v = _pf(row.get(key))
            if v is not None:
                record[key] = v
        for key in bool_keys:
            if row.get(key) != "":
                record[key] = _pb(row.get(key))
        record["has_physics"] = True


def load_symbol_archetypes() -> dict[str, dict]:
    if not SYMBOL_MEM_CSV.exists():
        return {}
    out = {}
    with SYMBOL_MEM_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["symbol"]] = row
    return out


def scan_tokens(record: dict) -> list[str]:
    tokens: list[str] = []
    body = record.get("body_pct")
    if body is not None:
        if body < 30:
            tokens.append("small_body")
        elif body >= 60:
            tokens.append("large_body")
        else:
            tokens.append("mid_body")

    if (record.get("upper_wick_pct") or 0) >= 40:
        tokens.append("upper_wick")
    if (record.get("lower_wick_pct") or 0) >= 40:
        tokens.append("lower_wick")
    if record.get("inside_bar"):
        tokens.append("inside_bar")
    if record.get("engulf"):
        tokens.append("engulf")
    if record.get("fast_rejection") or record.get("long_tail_rejection"):
        tokens.append("fast_rejection")

    vr = record.get("vol_ratio_6") or record.get("volume_ratio_ma24")
    if vr is not None:
        if vr >= 1.4:
            tokens.append("vol_increase")
        elif vr <= 0.85:
            tokens.append("vol_decrease")
    if record.get("vol_exhaustion"):
        tokens.append("vol_exhaustion")
    if record.get("vol_persist_up"):
        tokens.append("vol_increase")
    if record.get("vol_persist_down"):
        tokens.append("vol_decrease")

    cong = record.get("congestion_bars_12")
    if cong is not None and cong >= 5:
        tokens.append("congestion")
    if compression_ctx(record) or record.get("pre6_tight_range"):
        tokens.append("compression")
    if record.get("break_24h") or record.get("near_recent_high"):
        tokens.append("breakout")
    if record.get("break_24h") and (record.get("return_24h_at_scan") or 0) >= 20:
        if record.get("fast_rejection") or (record.get("upper_wick_pct") or 0) > 45:
            tokens.append("breakout_fail")

    if record.get("range_expand"):
        tokens.append("vol_expansion")
    if record.get("range_contract") or record.get("pre6_volatility_compression"):
        tokens.append("vol_contraction")

    cc = record.get("consec_candles")
    if cc is not None:
        if cc >= 2:
            tokens.append("trend_cont")
        elif cc <= -2:
            tokens.append("trend_cont_bear")
    if record.get("gap_up") or record.get("gap_down"):
        tokens.append("gap_momentum")
    if record.get("slow_absorption") or record.get("long_tail_recovery"):
        tokens.append("absorption")
    if record.get("vol_exhaustion") and record.get("upper_wick_pct", 0) >= 35:
        tokens.append("distribution_candle")

    return tokens


def prior_tokens(record: dict) -> list[str]:
    """Pre-scan phrase inferred from memory fields — not assumed TA."""
    tokens: list[str] = []
    rev = record.get("reversal_count_12")
    if rev is not None and rev >= 5:
        tokens.append("congestion")
    cong = record.get("congestion_bars_12")
    if cong is not None and cong >= 7:
        tokens.append("compression")
    state = record.get("state_4h") or record.get("state_scan") or ""
    if state in ("Compression", "Choppy"):
        tokens.append("compression")
    if state in ("Acceleration", "Continuation"):
        tokens.append("trend_cont")
    if state in ("Exhaustion", "Collapse"):
        tokens.append("vol_exhaustion")
    if (record.get("return_prev_24h_percent") or record.get("return_24h_at_scan") or 0) >= 30:
        tokens.append("gap_momentum")
    return tokens


def build_sentence(record: dict) -> dict:
    prior = prior_tokens(record)
    scan = scan_tokens(record)
    ordered = []
    seen = set()
    for t in prior + scan:
        if t not in seen:
            ordered.append(t)
            seen.add(t)
    if not ordered:
        ordered = ["mid_body"]

    phrase = " > ".join(ordered[:5])
    bigrams = ["|".join(ordered[i : i + 2]) for i in range(len(ordered) - 1)]
    trigrams = ["|".join(ordered[i : i + 3]) for i in range(len(ordered) - 2)]
    primary = trigrams[0] if trigrams else (bigrams[0] if bigrams else ordered[0])

    return {
        "tokens": ordered,
        "sentence": phrase,
        "primary_trigram": primary,
        "bigrams": bigrams,
        "trigrams": trigrams,
    }


def assign_cluster(tokens: list[str]) -> str:
    for name, rule in CLUSTER_RULES:
        if rule(tokens):
            return name
    if "trend_cont" in tokens:
        return "persistent_trend"
    if "congestion" in tokens:
        return "rotation"
    return "unclassified"


def sequence_confidence(n: int, spread: float) -> str:
    if n >= 40 and spread >= 8:
        return "high"
    if n >= 20 and spread >= 5:
        return "medium"
    if n >= MIN_SEQ:
        return "hypothesis"
    return "insufficient"


def grammar_status(global_spread: float, global_mae: float, baseline: float, cond_rows: list[dict]) -> tuple[str, str]:
    best_gain = max((r.get("gain") or 0) for r in cond_rows) if cond_rows else 0
    if global_spread >= 8 and global_mae <= baseline * 0.98:
        return "ACTIVE", "strong global sequence spread with MAE beat"
    if best_gain >= 0.35 or any(r.get("status") == "CONDITIONAL" for r in cond_rows):
        return "CONDITIONAL", f"weak globally; best context gain {best_gain:.2f}"
    if global_spread >= 5:
        return "RETIRED", "moderate spread; monitor on expansion"
    if global_spread >= 3:
        return "DEAD_FOR_NOW", "below threshold; search revival contexts"
    return "DEAD_FOR_NOW", "no measurable sequence edge on current sample"


def group_return_spread(group: list[dict]) -> float:
    f6 = [r.get("target_f6") for r in group if r.get("target_f6") is not None]
    return round(max(f6) - min(f6), 2) if len(f6) >= 2 else 0.0


def evaluate_sequence_global(records: list[dict], seq_key: str) -> dict:
    subset = [r for r in records if r.get("primary_trigram") == seq_key]
    if len(subset) < MIN_SEQ:
        return {"n": len(subset), "mae": None, "spread": None}
    key_fn = lambda r, s=seq_key: s
    mae = loo_mae(subset, key_fn)[0]
    spread = group_return_spread(subset)
    return {"n": len(subset), "mae": round(mae, 2), "spread": spread}


def evaluate_sequence_conditional(records: list[dict], seq_key: str, baseline: float) -> list[dict]:
    subset = [r for r in records if r.get("primary_trigram") == seq_key]
    if len(subset) < MIN_SEQ:
        return []
    global_mae = loo_mae(subset, lambda r, s=seq_key: s)[0]
    rows: list[dict] = []
    extra_contexts = CONTEXTS + [
        ("symbol_archetype", "arch", lambda r: r.get("symbol_archetype", "unknown")),
        ("memory_length", "mem", lambda r: r.get("symbol_memory_length", "unknown")),
        ("physics_available", "phys", lambda r: bool(r.get("has_physics"))),
    ]
    for ctx_name, _tag, ctx_fn in extra_contexts:
        ctx_subset = [r for r in subset if ctx_fn(r)]
        if len(ctx_subset) < MIN_COND:
            continue
        ctx_mae = loo_mae(ctx_subset, lambda r, s=seq_key: s)[0]
        ctx_spread = spread_f6(ctx_subset, lambda r, s=seq_key: s)
        gain = global_mae - ctx_mae
        f6_vals = [r.get("target_f6") for r in ctx_subset if r.get("target_f6") is not None]
        med_ret = round(statistics.median(f6_vals), 2) if f6_vals else None
        collapses = sum(1 for r in ctx_subset if r.get("collapse_label") == "YES")
        collapse_pct = round(collapses / len(ctx_subset) * 100, 1)
        status = "CONDITIONAL" if gain >= 0.25 or (ctx_spread >= 5 and ctx_mae < baseline) else "OPEN"
        if gain >= 0.35 or (ctx_mae < baseline * 0.97 and ctx_spread >= 4):
            rows.append(
                {
                    "sequence": seq_key,
                    "context": ctx_name,
                    "context_value": str(ctx_fn(ctx_subset[0]))[:40],
                    "sample_size": len(ctx_subset),
                    "global_mae": round(global_mae, 2),
                    "context_mae": round(ctx_mae, 2),
                    "gain": round(gain, 2),
                    "context_spread_f6": round(ctx_spread, 2),
                    "median_forward_6h": med_ret,
                    "collapse_pct": collapse_pct,
                    "confidence": sequence_confidence(len(ctx_subset), ctx_spread),
                    "status": status,
                }
            )
    return rows


def persistence_curve(records: list[dict], seq_key: str) -> list[dict]:
    subset = [r for r in records if r.get("primary_trigram") == seq_key]
    rows = []
    peak_spread = 0.0
    peak_h = ""
    for horizon in HORIZONS:
        vals = []
        for r in subset:
            v = r.get("forwards", {}).get(horizon)
            if v is not None:
                vals.append(v)
        if len(vals) < MIN_SEQ:
            continue
        med = statistics.median(vals)
        spread = max(vals) - min(vals) if len(vals) >= 2 else 0
        persist = sum(1 for v in vals if v >= 0) / len(vals) * 100
        peak_spread = max(peak_spread, spread)
        if spread == peak_spread:
            peak_h = horizon
        rows.append(
            {
                "sequence": seq_key,
                "horizon": horizon,
                "n": len(vals),
                "median_return": round(med, 2),
                "spread": round(spread, 2),
                "persistence_pct": round(persist, 1),
            }
        )
    half_life = ">24h"
    for row in rows:
        if peak_spread > 0 and row["spread"] <= peak_spread * 0.5:
            half_life = row["horizon"]
            break
    for row in rows:
        row["peak_horizon"] = peak_h
        row["half_life"] = half_life
    return rows


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def symbol_grammar_families(records: list[dict]) -> list[dict]:
    sym_trigrams: dict[str, set[str]] = defaultdict(set)
    sym_meta: dict[str, list] = defaultdict(list)
    for r in records:
        sym = r["symbol"]
        sym_trigrams[sym].add(r.get("primary_trigram", ""))
        sym_meta[sym].append(r)

    symbols = sorted(sym_trigrams.keys(), key=lambda s: -len(sym_meta[s]))
    families: dict[str, list[str]] = defaultdict(list)
    assigned: set[str] = set()

    for i, sym in enumerate(symbols):
        if sym in assigned:
            continue
        family_id = f"grammar_family_{len(families) + 1}"
        families[family_id].append(sym)
        assigned.add(sym)
        for other in symbols[i + 1 :]:
            if other in assigned:
                continue
            sim = jaccard(sym_trigrams[sym], sym_trigrams[other])
            if sim >= 0.35:
                families[family_id].append(other)
                assigned.add(other)

    rows = []
    for family_id, members in sorted(families.items(), key=lambda x: -len(x[1])):
        all_tri = Counter()
        collapse_n = 0
        total_n = 0
        f6_all = []
        mem_lengths = Counter()
        for sym in members:
            for r in sym_meta[sym]:
                all_tri[r.get("primary_trigram", "")] += 1
                total_n += 1
                if r.get("collapse_label") == "YES":
                    collapse_n += 1
                if r.get("target_f6") is not None:
                    f6_all.append(r["target_f6"])
                mem_lengths[r.get("symbol_memory_length", "?")] += 1
        top_seq = all_tri.most_common(3)
        rows.append(
            {
                "family_id": family_id,
                "symbols": "|".join(members[:12]),
                "symbol_count": len(members),
                "appearances": total_n,
                "top_sequences": "|".join(f"{s}({c})" for s, c in top_seq),
                "pattern_recurrence": round(max(all_tri.values()) / total_n, 2) if total_n else 0,
                "transition_similarity": round(len(all_tri) / max(total_n, 1), 2),
                "median_forward_6h": round(statistics.median(f6_all), 2) if f6_all else "",
                "collapse_rate_pct": round(collapse_n / total_n * 100, 1) if total_n else 0,
                "dominant_memory": mem_lengths.most_common(1)[0][0] if mem_lengths else "",
            }
        )
    return rows


def next_transition_hint(records: list[dict], sentence: str) -> str:
    """Empirical: what cluster follows similar sentences in sample."""
    tokens = sentence.split(" > ")
    cluster = assign_cluster(tokens)
    followers = Counter()
    for r in records:
        if r.get("behaviour_cluster") == cluster:
            st = r.get("state_6h") or r.get("state_4h") or "unknown"
            followers[st] += 1
    if not followers:
        return "unknown"
    return followers.most_common(1)[0][0]


def behaviour_engine_row(record: dict, seq_stats: dict, sym_arch: dict, baseline: float) -> dict:
    sentence = record.get("sentence", "")
    cluster = record.get("behaviour_cluster", "unclassified")
    seq_key = record.get("primary_trigram", "")
    stats = seq_stats.get(seq_key, {})
    f6 = record.get("target_f6")
    collapse_pct = stats.get("collapse_pct", 0)
    med_ret = stats.get("median_forward_6h", 0)
    persist_4 = record.get("forwards", {}).get("4h")
    persist_prob = stats.get("persistence_pct", 50)

    action = "Watch"
    if collapse_pct >= 35:
        action = "Avoid"
    elif med_ret >= 5 and collapse_pct < 15:
        action = "Buy"
    elif med_ret >= 2:
        action = "Watch"

    risks = []
    if cluster in ("manipulation", "failed_breakout", "late_trend"):
        risks.append(cluster)
    if sym_arch.get("archetype") == "manipulation_prone":
        risks.append("manipulation_prone_symbol")
    if record.get("behaviour_status") == "DEAD_FOR_NOW":
        risks.append("weak_grammar")

    return {
        "date": record["date"],
        "symbol": record["symbol"],
        "scan_time": record["scan_time"],
        "market_regime": record.get("market_regime"),
        "symbol_archetype": sym_arch.get("archetype", record.get("symbol_archetype")),
        "memory_horizon": stats.get("peak_horizon", "6h"),
        "current_sentence": sentence,
        "behaviour_cluster": cluster,
        "primary_sequence": seq_key,
        "likely_next_transition": stats.get("likely_transition", "unknown"),
        "persistence_probability_pct": persist_prob,
        "collapse_probability_pct": collapse_pct,
        "expected_return_6h_pct": med_ret,
        "expected_drawdown_pct": record.get("max_drawdown"),
        "confidence": stats.get("confidence", "hypothesis"),
        "grammar_status": record.get("behaviour_status", "OPEN"),
        "recommended_action": action,
        "reason": f"cluster={cluster}; seq n={stats.get('n', 0)} med_f6={med_ret}",
        "risk_factors": "|".join(risks) if risks else "",
        "recommendation_score": round(max(0, 10 - (stats.get("global_mae") or baseline) + med_ret * 0.1), 1),
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
    attach_forward_targets(records)
    baseline = global_baseline_mae(records)

    sym_arch_index = load_symbol_archetypes()
    sym_groups: dict[str, list] = defaultdict(list)
    for r in records:
        sym_groups[r["symbol"]].append(r)
    sym_computed = {sym: symbol_archetype(group) for sym, group in sym_groups.items()}

    for record in records:
        record["has_physics"] = record.get("has_physics", False)
        sent = build_sentence(record)
        record.update(sent)
        record["behaviour_cluster"] = assign_cluster(record["tokens"])
        arch = sym_arch_index.get(record["symbol"]) or sym_computed.get(record["symbol"], {})
        record["symbol_archetype"] = arch.get("archetype", "unknown")
        record["symbol_memory_length"] = arch.get("memory_length", "unknown")

    # Task 1: sequence discovery
    seq_groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        seq_groups[r["primary_trigram"]].append(r)

    sequence_rows = []
    seq_stats: dict[str, dict] = {}
    for seq_key, group in sorted(seq_groups.items(), key=lambda x: -len(x[1])):
        if not seq_key:
            continue
        f6 = [r.get("target_f6") for r in group if r.get("target_f6") is not None]
        collapses = sum(1 for r in group if r.get("collapse_label") == "YES")
        n = len(group)
        med_ret = round(statistics.median(f6), 2) if f6 else None
        collapse_pct = round(collapses / n * 100, 1) if n else 0
        perf = evaluate_sequence_global(records, seq_key)
        spread = perf.get("spread") or 0
        conf = sequence_confidence(n, spread)
        persist_pct = round(sum(1 for v in f6 if v >= 0) / len(f6) * 100, 1) if f6 else None
        clusters = Counter(r.get("behaviour_cluster") for r in group)
        sequence_rows.append(
            {
                "sequence": seq_key,
                "sentence_example": group[0].get("sentence", ""),
                "frequency": n,
                "median_forward_6h": med_ret,
                "mean_forward_6h": round(statistics.mean(f6), 2) if f6 else None,
                "collapse_probability_pct": collapse_pct,
                "global_mae": perf.get("mae"),
                "spread_f6": spread,
                "dominant_cluster": clusters.most_common(1)[0][0] if clusters else "",
                "confidence": conf,
            }
        )
        seq_stats[seq_key] = {
            "n": n,
            "median_forward_6h": med_ret or 0,
            "collapse_pct": collapse_pct,
            "persistence_pct": persist_pct or 50,
            "global_mae": perf.get("mae") or baseline,
            "confidence": conf,
            "likely_transition": next_transition_hint(group, group[0].get("sentence", "")),
        }

    # Task 2: behaviour clustering
    cluster_groups: dict[str, list] = defaultdict(list)
    for r in records:
        cluster_groups[r["behaviour_cluster"]].append(r)
    cluster_rows = []
    for cluster, group in sorted(cluster_groups.items(), key=lambda x: -len(x[1])):
        f6 = [r.get("target_f6") for r in group if r.get("target_f6") is not None]
        collapses = sum(1 for r in group if r.get("collapse_label") == "YES")
        top_seq = Counter(r.get("primary_trigram") for r in group).most_common(3)
        cluster_rows.append(
            {
                "cluster": cluster,
                "count": len(group),
                "median_forward_6h": round(statistics.median(f6), 2) if f6 else "",
                "collapse_rate_pct": round(collapses / len(group) * 100, 1),
                "top_sequences": "|".join(f"{s}({c})" for s, c in top_seq),
                "confidence": sequence_confidence(len(group), spread_f6(group, lambda r: r["behaviour_cluster"])),
            }
        )

    # Task 3 + registry: conditional behaviour + lifecycle
    conditional_rows: list[dict] = []
    registry_rows: list[dict] = []
    top_sequences = [r["sequence"] for r in sorted(sequence_rows, key=lambda x: -x["frequency"])[:40]]

    for seq_key in top_sequences:
        cond = evaluate_sequence_conditional(records, seq_key, baseline)
        conditional_rows.extend(cond)
        perf = evaluate_sequence_global(records, seq_key)
        spread = perf.get("spread") or 0
        mae = perf.get("mae") or baseline
        status, reason = grammar_status(spread, mae, baseline, cond)
        if any(c.get("status") == "CONDITIONAL" for c in cond) and status == "DEAD_FOR_NOW":
            status = "CONDITIONAL"
            reason = "revived in specific context"
        elif spread >= 5 and status == "DEAD_FOR_NOW":
            status = "RETIRED"
        persist = persistence_curve(records, seq_key)
        peak_h = persist[0]["peak_horizon"] if persist else "6h"
        if seq_key in seq_stats:
            seq_stats[seq_key]["peak_horizon"] = peak_h
        for r in records:
            if r.get("primary_trigram") == seq_key:
                r["behaviour_status"] = status
        registry_rows.append(
            {
                "grammar_sequence": seq_key,
                "current_status": status,
                "frequency": perf.get("n", 0),
                "global_mae": mae,
                "spread_f6": spread,
                "peak_persistence_horizon": peak_h,
                "conditional_contexts": sum(1 for c in cond if c.get("status") == "CONDITIONAL"),
                "decision_reason": reason,
            }
        )

    # Task 4: cross-symbol grammar families
    family_rows = symbol_grammar_families(records)

    # Task 5: behaviour persistence
    persist_rows: list[dict] = []
    for seq_key in top_sequences[:25]:
        persist_rows.extend(persistence_curve(records, seq_key))

    # Task 7: behaviour engine prototype
    engine_rows = []
    for record in records[-40:]:
        sk = record.get("primary_trigram", "")
        arch = sym_arch_index.get(record["symbol"]) or sym_computed.get(record["symbol"], {})
        engine_rows.append(behaviour_engine_row(record, seq_stats, arch, baseline))

    write_csv(SEQUENCE_CSV, sequence_rows)
    write_csv(CLUSTER_CSV, cluster_rows)
    write_csv(CONDITIONAL_CSV, conditional_rows)
    write_csv(FAMILY_CSV, family_rows)
    write_csv(PERSIST_CSV, persist_rows)
    write_csv(REGISTRY_CSV, registry_rows)
    write_csv(ENGINE_CSV, engine_rows)

    dates = sorted({r["date"] for r in records})
    physics_n = sum(1 for r in records if r.get("has_physics"))
    top5 = sorted(sequence_rows, key=lambda x: -x["frequency"])[:5]
    active = [r for r in registry_rows if r["current_status"] == "ACTIVE"]
    conditional = [r for r in registry_rows if r["current_status"] == "CONDITIONAL"]

    lines = [
        "===== SCOUT SEASON2 P7 - BEHAVIOUR GRAMMAR & SEQUENCE INTELLIGENCE =====",
        "",
        f"Sample: {len(records)} records | {dates[0]}..{dates[-1]} ({len(dates)} days)",
        f"Physics-enriched rows: {physics_n}/{len(records)}",
        f"Collected this run: {collected or '(none)'}",
        f"Unique trigram sequences: {len(sequence_rows)} | Baseline MAE: {baseline:.2f}%",
        "",
        "--- Task 1: Behaviour Sequence Discovery (top frequency) ---",
    ]
    for row in top5:
        lines.append(
            f"  {row['sequence']}: n={row['frequency']} med_f6={row['median_forward_6h']}% "
            f"collapse={row['collapse_probability_pct']}% spread={row['spread_f6']} [{row['confidence']}]"
        )

    lines.extend(["", "--- Task 2: Behaviour Clusters ---"])
    for row in cluster_rows[:8]:
        lines.append(
            f"  {row['cluster']}: n={row['count']} med_f6={row['median_forward_6h']}% "
            f"collapse={row['collapse_rate_pct']}%"
        )

    lines.extend(["", "--- Task 3: Conditional revival ---"])
    if conditional_rows:
        for row in sorted(conditional_rows, key=lambda x: -(x.get("gain") or 0))[:6]:
            lines.append(
                f"  {row['sequence'][:40]} @ {row['context']}={row['context_value']}: "
                f"gain={row['gain']} n={row['sample_size']} [{row['status']}]"
            )
    else:
        lines.append("  (no conditional gain above threshold on current sample)")

    lines.extend(["", "--- Task 4: Symbol grammar families (top) ---"])
    for row in family_rows[:5]:
        lines.append(
            f"  {row['family_id']}: {row['symbol_count']} symbols, n={row['appearances']} "
            f"collapse={row['collapse_rate_pct']}% top={row['top_sequences'][:60]}"
        )

    lines.extend(["", "--- Task 5: Grammar lifecycle ---"])
    lines.append(f"  ACTIVE: {len(active)} | CONDITIONAL: {len(conditional)} | RETIRED/DEAD_FOR_NOW: {len(registry_rows) - len(active) - len(conditional)}")
    for row in registry_rows[:8]:
        lines.append(f"  {row['grammar_sequence'][:45]}: {row['current_status']} | {row['decision_reason'][:50]}")

    lines.extend([
        "",
        "--- Research notes ---",
        " Markets as behavioural sentences: tokens -> phrases -> stories",
        " Sequence edge is conditional; weak globally may revive in symbol family/regime",
        " Physics tokens cover subset; panel inference used for expanded dates",
        " Never permanently discard grammars — use DEAD_FOR_NOW / RETIRED",
        "",
        f"Sequences: {SEQUENCE_CSV}",
        f"Engine: {ENGINE_CSV}",
        "=" * 62,
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("===== P7 BEHAVIOUR GRAMMAR =====")
    print(f"Records: {len(records)} | Sequences: {len(sequence_rows)} | Physics: {physics_n}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
