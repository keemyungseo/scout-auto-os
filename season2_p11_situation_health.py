"""
Scout Learning Season2 - P11 Situation Health Engine & Momentum of Context Discovery

Research only. Is a situation becoming healthier or weaker?
"""

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p5_historical_expansion import (
    SEEDS,
    STEP1_DATES,
    build_unified_records,
    collect_missing,
    global_baseline_mae,
)
from season2_p6_market_memory import attach_forward_targets
from season2_p7_behaviour_grammar import enrich_physics
from season2_p8_participant_state import enrich_panel_fields
from season2_p9_conditional_interaction_mining import prepare_records
from season2_p10_situation_evolution import (
    build_edges,
    label_records,
    load_interactions,
)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

HEALTH_SCORES_CSV = LOGS_DIR / "season2_p11_health_scores.csv"
HEALTH_TRANSITIONS_CSV = LOGS_DIR / "season2_p11_health_transitions.csv"
CONTEXT_MOMENTUM_CSV = LOGS_DIR / "season2_p11_context_momentum.csv"
PARTICIPANT_ENERGY_CSV = LOGS_DIR / "season2_p11_participant_energy.csv"
GRAMMAR_VITALITY_CSV = LOGS_DIR / "season2_p11_grammar_vitality.csv"
HEALTH_FAMILIES_CSV = LOGS_DIR / "season2_p11_health_families.csv"
CONDITIONAL_HEALTH_CSV = LOGS_DIR / "season2_p11_conditional_health.csv"
ENGINE_CSV = LOGS_DIR / "season2_p11_health_engine_output.csv"
REGISTRY_CSV = LOGS_DIR / "season2_p11_health_registry.csv"
REPORT_TXT = LOGS_DIR / "season2_p11_research_report.txt"

MIN_N = 6
HEALTH_CLASSES = ("IMPROVING", "STABLE", "WEAKENING", "FRAGILE", "RECOVERING")

# Participant energy weights (empirical proxy, not assumed TA)
ENERGY_MAP = {
    "scan_warm-up": 0.75,
    "scan_compression": 0.65,
    "fresh_accumulation": 0.8,
    "hidden_accumulation": 0.7,
    "scan_transition": 0.55,
    "steady_markup": 0.7,
    "violent_markup": 0.85,
    "late_chasing": 0.45,
    "late_markup": 0.35,
    "scan_exhaustion": 0.25,
    "exhaustion": 0.2,
    "profit_taking": 0.3,
    "distribution": 0.25,
    "panic_exit": 0.1,
    "forced_liquidation": 0.05,
    "inventory_recovery": 0.6,
    "rotation": 0.5,
    "trapped_long": 0.15,
    "scan_choppy": 0.45,
    "scan_expansion": 0.72,
}


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def cohort_metrics(group: list[dict]) -> dict:
    f6 = [r["target_f6"] for r in group if r.get("target_f6") is not None]
    if not f6:
        return {}
    collapses = sum(1 for r in group if r.get("collapse_label") == "YES")
    persist = sum(1 for v in f6 if v >= 0) / len(f6) * 100
    med = statistics.median(f6)
    return {
        "n": len(group),
        "median_f6": round(med, 2),
        "mean_f6": round(statistics.mean(f6), 2),
        "persistence_pct": round(persist, 1),
        "collapse_pct": round(collapses / len(group) * 100, 1),
        "spread_f6": round(max(f6) - min(f6), 2) if len(f6) > 1 else 0,
    }


def raw_health_score(m: dict) -> float:
    """0-100 health from forward outcomes."""
    if not m:
        return 50.0
    ret = m.get("median_f6") or 0
    persist = m.get("persistence_pct") or 50
    collapse = m.get("collapse_pct") or 0
    score = 50 + ret * 2.5 + (persist - 50) * 0.3 - collapse * 0.8
    return clamp(score)


def classify_health(m: dict, acceleration: float = 0.0, fragile_transition_pct: float = 0.0) -> str:
    med = m.get("median_f6") or 0
    collapse = m.get("collapse_pct") or 0
    persist = m.get("persistence_pct") or 50

    if collapse >= 25 or fragile_transition_pct >= 25:
        return "FRAGILE"
    if med < -3 and acceleration > 1.5:
        return "RECOVERING"
    if acceleration >= 1.0 and med >= 0:
        return "IMPROVING"
    if acceleration <= -1.0 or (med >= 2 and collapse >= 15):
        return "WEAKENING"
    if persist >= 55 and collapse < 12:
        return "STABLE"
    if med >= 3 and collapse < 10:
        return "IMPROVING"
    if med < 0:
        return "WEAKENING"
    return "STABLE"


def situation_acceleration(edges: list[dict], situation: str) -> float:
    """Median forward return delta along consecutive same-situation edges."""
    deltas = []
    for e in edges:
        cur, nxt = e["from_record"], e["to_record"]
        if cur.get("situation_archetype") != situation:
            continue
        f0, f1 = cur.get("target_f6"), nxt.get("target_f6")
        if f0 is not None and f1 is not None:
            deltas.append(f1 - f0)
    return round(statistics.median(deltas), 2) if deltas else 0.0


def fragile_transition_pct(situation: str, sit_transitions: list[dict]) -> float:
    fragile_targets = {"forced_liquidation_zone", "distribution", "late_chasing_zone"}
    total = sum(int(r["transition_count"]) for r in sit_transitions if r["from_state"] == situation)
    fragile = sum(
        int(r["transition_count"])
        for r in sit_transitions
        if r["from_state"] == situation and r["to_state"] in fragile_targets
    )
    return round(fragile / total * 100, 1) if total else 0.0


def task1_health_scores(records: list[dict], edges: list[dict], sit_transitions: list[dict]) -> list[dict]:
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r.get("situation_archetype", "unknown")].append(r)

    rows = []
    for situation, group in sorted(groups.items(), key=lambda x: -len(x[1])):
        m = cohort_metrics(group)
        if not m or m["n"] < MIN_N:
            continue
        accel = situation_acceleration(edges, situation)
        frag_pct = fragile_transition_pct(situation, sit_transitions)
        hscore = raw_health_score(m)
        hclass = classify_health(m, accel, frag_pct)
        decay = round(max(0, -accel) + frag_pct * 0.2, 2)
        rows.append(
            {
                "situation_archetype": situation,
                "health_score": round(hscore, 1),
                "health_class": hclass,
                "persistence_pct": m["persistence_pct"],
                "acceleration": accel,
                "decay": decay,
                "median_forward_6h": m["median_f6"],
                "collapse_pct": m["collapse_pct"],
                "fragile_transition_pct": frag_pct,
                "expected_forward_improvement": round(accel + m["median_f6"] * 0.1, 2),
                "sample_size": m["n"],
                "confidence": "high" if m["n"] >= 30 else "medium" if m["n"] >= MIN_N else "hypothesis",
            }
        )
    return rows


def record_health_class(record: dict, health_index: dict[str, dict]) -> str:
    sit = record.get("situation_archetype", "unknown")
    base = health_index.get(sit, {})
    accel = record.get("_health_accel", 0)
    m = {
        "median_f6": record.get("target_f6") or base.get("median_forward_6h", 0),
        "collapse_pct": 100 if record.get("collapse_label") == "YES" else base.get("collapse_pct", 0),
        "persistence_pct": base.get("persistence_pct", 50),
    }
    return classify_health(m, accel, base.get("fragile_transition_pct", 0))


def annotate_record_momentum(records: list[dict], edges: list[dict]) -> None:
    prior_f6: dict[str, float] = {}
    by_sym: dict[str, list] = defaultdict(list)
    for r in records:
        by_sym[r["symbol"]].append(r)
    for sym, group in by_sym.items():
        group.sort(key=lambda x: x["scan_time"])
        for i, r in enumerate(group):
            if i == 0:
                r["_health_accel"] = 0.0
            else:
                prev = group[i - 1].get("target_f6")
                cur = r.get("target_f6")
                r["_health_accel"] = round(cur - prev, 2) if prev is not None and cur is not None else 0.0


def task2_health_transitions(edges: list[dict], health_index: dict[str, dict]) -> list[dict]:
    for e in edges:
        e["from_health"] = record_health_class(e["from_record"], health_index)
        e["to_health"] = record_health_class(e["to_record"], health_index)

    counts: Counter[tuple[str, str]] = Counter()
    collapse_to: Counter[tuple[str, str]] = Counter()
    f6_to: dict[tuple[str, str], list] = defaultdict(list)

    for e in edges:
        key = (e["from_health"], e["to_health"])
        counts[key] += 1
        if e["to_record"].get("target_f6") is not None:
            f6_to[key].append(e["to_record"]["target_f6"])
        if e["to_record"].get("collapse_label") == "YES":
            collapse_to[key] += 1

    from_totals: Counter[str] = Counter()
    for (fr, _), c in counts.items():
        from_totals[fr] += c

    rows = []
    for (fr, to), count in counts.most_common():
        if count < MIN_N and from_totals[fr] < MIN_N * 2:
            continue
        prob = round(count / from_totals[fr] * 100, 1) if from_totals[fr] else 0
        f6 = f6_to.get((fr, to), [])
        rows.append(
            {
                "from_health": fr,
                "to_health": to,
                "transition_count": count,
                "transition_probability_pct": prob,
                "median_return_at_next": round(statistics.median(f6), 2) if f6 else "",
                "collapse_at_next_pct": round(collapse_to.get((fr, to), 0) / count * 100, 1),
                "stability": "stable" if fr == to and prob >= 35 else "evolving",
                "confidence": "high" if count >= 15 else "medium" if count >= MIN_N else "hypothesis",
            }
        )
    return rows


def task3_context_momentum(records: list[dict], health_index: dict[str, dict]) -> list[dict]:
    contexts = [
        ("supply", lambda r: r.get("supply_label", "unknown")),
        ("memory", lambda r: r.get("memory", "unknown")),
        ("participant_state", lambda r: r.get("participant_state", "unknown")),
        ("grammar", lambda r: (r.get("grammar") or "unknown")[:35]),
        ("market_physics", lambda r: r.get("market_physics", "unknown")),
    ]
    rows = []
    for situation, base in health_index.items():
        base_score = base["health_score"]
        sit_records = [r for r in records if r.get("situation_archetype") == situation]
        for ctx_name, ctx_fn in contexts:
            groups: dict[str, list] = defaultdict(list)
            for r in sit_records:
                groups[ctx_fn(r)].append(r)
            for ctx_val, grp in groups.items():
                if len(grp) < MIN_N or ctx_val in ("unknown", ""):
                    continue
                m = cohort_metrics(grp)
                ctx_score = raw_health_score(m)
                delta = round(ctx_score - base_score, 2)
                rows.append(
                    {
                        "situation": situation,
                        "context_dimension": ctx_name,
                        "context_value": ctx_val,
                        "base_health_score": base_score,
                        "context_health_score": round(ctx_score, 1),
                        "health_delta": delta,
                        "momentum": "improves" if delta >= 5 else "degrades" if delta <= -5 else "neutral",
                        "median_forward_6h": m.get("median_f6"),
                        "collapse_pct": m.get("collapse_pct"),
                        "sample_size": m["n"],
                        "confidence": "high" if m["n"] >= 20 else "medium" if m["n"] >= MIN_N else "hypothesis",
                    }
                )
    return sorted(rows, key=lambda x: -abs(x["health_delta"]))


def participant_energy_score(state: str) -> float:
    for key, val in ENERGY_MAP.items():
        if key in (state or ""):
            return val
    return 0.5


def task4_participant_energy(records: list[dict], edges: list[dict]) -> list[dict]:
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r.get("participant_state", "unknown")].append(r)

    energy_curve: dict[str, list] = defaultdict(list)
    for e in edges:
        ps = e["from_record"].get("participant_state", "unknown")
        nps = e["to_record"].get("participant_state", "unknown")
        energy_curve[ps].append(participant_energy_score(nps) - participant_energy_score(ps))

    rows = []
    for state, group in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(group) < MIN_N:
            continue
        m = cohort_metrics(group)
        base_e = participant_energy_score(state)
        curve = energy_curve.get(state, [])
        delta = round(statistics.mean(curve), 3) if curve else 0
        label = "accumulation" if "compression" in state or "accumulation" in state else "chasing" if "late" in state else "exhaustion" if "exhaustion" in state else "recovery" if "recovery" in state or "warm" in state else "confidence" if base_e >= 0.65 else "neutral"
        rows.append(
            {
                "participant_state": state,
                "energy_level": round(base_e * 100, 1),
                "energy_curve_delta": delta,
                "energy_trend": "increasing" if delta > 0.05 else "fading" if delta < -0.05 else "stable",
                "behaviour_label": label,
                "median_forward_6h": m.get("median_f6"),
                "collapse_pct": m.get("collapse_pct"),
                "sample_size": m["n"],
                "confidence": "high" if m["n"] >= 25 else "medium" if m["n"] >= MIN_N else "hypothesis",
            }
        )
    return rows


def task5_grammar_vitality(records: list[dict], gram_transitions: list[dict]) -> list[dict]:
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        g = (r.get("grammar") or "unknown")[:40]
        groups[g].append(r)

    self_prob: dict[str, float] = {}
    for row in gram_transitions:
        if row.get("from_state") == row.get("to_state"):
            self_prob[row["from_state"][:40]] = float(row.get("transition_probability_pct") or 0)

    rows = []
    for grammar, group in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(group) < MIN_N:
            continue
        m = cohort_metrics(group)
        sp = self_prob.get(grammar, 0)
        h = raw_health_score(m)
        if h >= 60 and sp >= 35:
            vitality = "strong"
        elif h >= 45 and sp >= 20:
            vitality = "stable"
        elif h < 40 or m.get("median_f6", 0) < -2:
            vitality = "fading"
        elif m.get("median_f6", 0) > 0 and sp < 20:
            vitality = "reviving"
        else:
            vitality = "stable"
        rows.append(
            {
                "grammar": grammar,
                "vitality": vitality,
                "health_score": round(h, 1),
                "self_persistence_pct": sp,
                "median_forward_6h": m.get("median_f6"),
                "collapse_pct": m.get("collapse_pct"),
                "sample_size": m["n"],
                "confidence": "high" if m["n"] >= 20 else "medium" if m["n"] >= MIN_N else "hypothesis",
            }
        )
    return rows


def task6_health_families(records: list[dict]) -> list[dict]:
    sym_health: dict[str, list[float]] = defaultdict(list)
    sym_class: dict[str, Counter] = defaultdict(Counter)
    sym_meta: dict[str, list] = defaultdict(list)

    for r in records:
        sym = r["symbol"]
        h = raw_health_score(cohort_metrics([r]) or {"median_f6": r.get("target_f6") or 0, "persistence_pct": 50, "collapse_pct": 0})
        sym_health[sym].append(h)
        sym_meta[sym].append(r)

    sym_avg = {sym: statistics.mean(vals) for sym, vals in sym_health.items() if vals}
    symbols = sorted(sym_avg.keys(), key=lambda s: -len(sym_health[s]))
    assigned: set[str] = set()
    families: dict[str, list[str]] = {}

    for sym in symbols:
        if sym in assigned:
            continue
        fid = f"health_family_{len(families) + 1}"
        families[fid] = [sym]
        assigned.add(sym)
        for other in symbols:
            if other in assigned:
                continue
            if abs(sym_avg[sym] - sym_avg[other]) <= 8 and len(sym_health[other]) >= 2:
                families[fid].append(other)
                assigned.add(other)

    rows = []
    for fid, members in sorted(families.items(), key=lambda x: -len(x[1])):
        recs = [r for sym in members for r in sym_meta[sym]]
        f6 = [r.get("target_f6") for r in recs if r.get("target_f6") is not None]
        collapses = sum(1 for r in recs if r.get("collapse_label") == "YES")
        avg_h = round(statistics.mean([sym_avg[s] for s in members]), 1)
        rows.append(
            {
                "family_id": fid,
                "symbols": "|".join(members[:12]),
                "symbol_count": len(members),
                "avg_health_score": avg_h,
                "median_forward_6h": round(statistics.median(f6), 2) if f6 else "",
                "collapse_rate_pct": round(collapses / len(recs) * 100, 1) if recs else 0,
                "health_behaviour": "durable" if avg_h >= 55 and collapses / max(len(recs), 1) < 0.1 else "fragile" if avg_h < 45 else "mixed",
            }
        )
    return rows


def load_registries() -> list[dict]:
    items = []
    for path, kind, col in [
        (LOGS_DIR / "season2_p9_hypothesis_registry.csv", "p9", "hypothesis_name"),
        (LOGS_DIR / "season2_p10_evolution_registry.csv", "p10", "item_name"),
        (LOGS_DIR / "season2_p7_grammar_registry.csv", "grammar", "grammar_sequence"),
        (LOGS_DIR / "season2_p8_hypothesis_registry.csv", "participant", "hypothesis_name"),
    ]:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                items.append(
                    {
                        "kind": kind,
                        "name": row.get(col, ""),
                        "prior_status": row.get("current_status", "OPEN"),
                    }
                )
    for seed in SEEDS:
        items.append({"kind": "seed", "name": seed.feature, "prior_status": seed.prior_status})
    return items


def task7_conditional_health(records: list[dict], context_momentum: list[dict]) -> tuple[list[dict], list[dict]]:
    ctx_boost = defaultdict(list)
    for row in context_momentum:
        if row["momentum"] == "improves" and row["health_delta"] >= 5:
            key = f"{row['situation']}+{row['context_dimension']}={row['context_value']}"
            ctx_boost[key].append(row)

    cond_rows = []
    for row in sorted(context_momentum, key=lambda x: -x["health_delta"])[:40]:
        status = "ACTIVE" if row["health_delta"] >= 10 else "CONDITIONAL" if row["health_delta"] >= 5 else "RETIRED" if row["health_delta"] >= 0 else "DEAD_FOR_NOW"
        cond_rows.append(
            {
                "situation": row["situation"],
                "context": f"{row['context_dimension']}={row['context_value']}",
                "health_delta": row["health_delta"],
                "context_health_score": row["context_health_score"],
                "sample_size": row["sample_size"],
                "momentum": row["momentum"],
                "status": status,
            }
        )

    registry = []
    for item in load_registries():
        name = item["name"]
        best_delta = 0
        best_ctx = ""
        for row in context_momentum:
            if name in row.get("context_value", "") or name in row.get("situation", ""):
                if row["health_delta"] > best_delta:
                    best_delta = row["health_delta"]
                    best_ctx = f"{row['situation']}|{row['context_dimension']}={row['context_value']}"
        status = item["prior_status"]
        if item["prior_status"] in ("RETIRED", "DEAD_FOR_NOW") and best_delta >= 8:
            status = "REVIVED"
        elif best_delta >= 5:
            status = "CONDITIONAL"
        elif best_delta >= 0:
            status = "RETIRED"
        registry.append(
            {
                "hypothesis_kind": item["kind"],
                "hypothesis_name": name[:60],
                "prior_status": item["prior_status"],
                "current_status": status,
                "best_health_delta": best_delta,
                "best_health_context": best_ctx[:80],
                "decision_reason": f"health retest delta={best_delta}",
            }
        )
    return cond_rows, registry


def health_action(
    hclass: str,
    direction: str,
    strengthen_prob: float,
    weaken_prob: float,
    collapse_prob: float,
) -> tuple[str, str]:
    if collapse_prob >= 30 or hclass == "FRAGILE":
        return "Avoid", f"fragile health collapse={collapse_prob}%"
    if hclass == "WEAKENING" and weaken_prob >= 40:
        return "Reduce", f"weakening momentum p={weaken_prob}%"
    if hclass == "IMPROVING" and strengthen_prob >= 35 and collapse_prob < 10:
        return "Strong Buy", "situation strengthening with low collapse"
    if hclass in ("IMPROVING", "RECOVERING") and collapse_prob < 15:
        return "Buy", f"health {hclass.lower()} direction={direction}"
    if hclass == "STABLE" and strengthen_prob >= 30:
        return "Hold", "stable health with strengthening bias"
    if hclass == "WEAKENING":
        return "Watch", "internal weakness despite surface label"
    return "Watch", f"health={hclass} strengthen={strengthen_prob}%"


def task8_engine(
    records: list[dict],
    health_index: dict[str, dict],
    health_transitions: list[dict],
    participant_energy: list[dict],
    grammar_vitality: list[dict],
) -> list[dict]:
    pe_index = {r["participant_state"]: r for r in participant_energy}
    gv_index = {r["grammar"]: r for r in grammar_vitality}

    def transition_probs(hclass: str) -> tuple[float, float, float]:
        subset = [r for r in health_transitions if r["from_health"] == hclass]
        if not subset:
            return 33.0, 33.0, 0.0
        strengthen = sum(r["transition_probability_pct"] for r in subset if r["to_health"] in ("IMPROVING", "RECOVERING", "STABLE"))
        weaken = sum(r["transition_probability_pct"] for r in subset if r["to_health"] in ("WEAKENING", "FRAGILE"))
        collapse = sum(r["transition_probability_pct"] for r in subset if r["to_health"] == "FRAGILE")
        return round(strengthen, 1), round(weaken, 1), round(collapse, 1)

    rows = []
    for record in records[-40:]:
        sit = record.get("situation_archetype", "unknown")
        base = health_index.get(sit, {})
        hscore = base.get("health_score", 50)
        accel = record.get("_health_accel", 0)
        hclass = record_health_class(record, health_index)
        direction = "strengthening" if accel > 0.5 else "weakening" if accel < -0.5 else "flat"

        ps = record.get("participant_state", "")
        gram = (record.get("grammar") or "")[:40]
        pe = pe_index.get(ps, {})
        gv = gv_index.get(gram, {})

        sp, wp, cp = transition_probs(hclass)
        action, reason = health_action(hclass, direction, sp, wp, cp)

        rows.append(
            {
                "date": record["date"],
                "symbol": record["symbol"],
                "scan_time": record["scan_time"],
                "current_situation": sit,
                "health_score": hscore,
                "health_class": hclass,
                "health_direction": direction,
                "health_acceleration": accel,
                "participant_state": ps,
                "participant_energy": pe.get("energy_level", ""),
                "participant_energy_trend": pe.get("energy_trend", ""),
                "grammar": gram,
                "grammar_vitality": gv.get("vitality", ""),
                "strengthen_probability_pct": sp,
                "weaken_probability_pct": wp,
                "collapse_probability_pct": cp,
                "confidence": base.get("confidence", "hypothesis"),
                "recommended_action": action,
                "reason": reason,
                "risk_factors": "|".join(
                    x for x in [
                        "fragile" if hclass == "FRAGILE" else "",
                        "fading_grammar" if gv.get("vitality") == "fading" else "",
                        "fading_energy" if pe.get("energy_trend") == "fading" else "",
                        "fake_trend_risk" if sit == "healthy_trend" and hclass == "WEAKENING" else "",
                    ] if x
                ),
            }
        )
    return rows


def load_p10_transitions() -> tuple[list[dict], list[dict]]:
    sit, gram = [], []
    if (LOGS_DIR / "season2_p10_situation_transitions.csv").exists():
        with (LOGS_DIR / "season2_p10_situation_transitions.csv").open(encoding="utf-8") as f:
            sit = list(csv.DictReader(f))
    if (LOGS_DIR / "season2_p10_grammar_transitions.csv").exists():
        with (LOGS_DIR / "season2_p10_grammar_transitions.csv").open(encoding="utf-8") as f:
            gram = list(csv.DictReader(f))
    return sit, gram


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


def strategic_insights(health_scores: list[dict], context_momentum: list[dict]) -> list[str]:
    lines = ["--- Task 9: Strategic insight ---"]
    improving = [r for r in health_scores if r["health_class"] == "IMPROVING"]
    weakening = [r for r in health_scores if r["health_class"] in ("WEAKENING", "FRAGILE")]
    lines.append(f"  Healthier situations: {', '.join(r['situation_archetype'] for r in improving[:5]) or '(none)'}")
    lines.append(f"  Decaying situations: {', '.join(r['situation_archetype'] for r in weakening[:5]) or '(none)'}")

    fake = [r for r in health_scores if r["situation_archetype"] == "healthy_trend" and r["health_class"] == "WEAKENING"]
    if fake:
        lines.append("  Fake trend signal: healthy_trend label with WEAKENING internal health")
    recover = [r for r in health_scores if r["health_class"] == "RECOVERING"]
    if recover:
        lines.append(f"  Recovery paths: {', '.join(r['situation_archetype'] for r in recover)}")

    durable = sorted(health_scores, key=lambda x: (-x["health_score"], -x["persistence_pct"]))[:3]
    for r in durable:
        lines.append(f"  Durable: {r['situation_archetype']} score={r['health_score']} persist={r['persistence_pct']}%")

    top_ctx = sorted(context_momentum, key=lambda x: -x["health_delta"])[:3]
    for r in top_ctx:
        lines.append(f"  Context boost: {r['situation']}+{r['context_dimension']}={r['context_value']} delta=+{r['health_delta']}")

    weak_ctx = sorted(context_momentum, key=lambda x: x["health_delta"])[:2]
    for r in weak_ctx:
        lines.append(f"  Context drag: {r['situation']}+{r['context_dimension']}={r['context_value']} delta={r['health_delta']}")
    return lines


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
    prepare_records(records)
    interactions = load_interactions()
    label_records(records, interactions)

    edges = build_edges(records)
    annotate_record_momentum(records, edges)
    sit_trans, gram_trans = load_p10_transitions()

    health_scores = task1_health_scores(records, edges, sit_trans)
    health_index = {r["situation_archetype"]: r for r in health_scores}
    health_transitions = task2_health_transitions(edges, health_index)
    context_momentum = task3_context_momentum(records, health_index)
    participant_energy = task4_participant_energy(records, edges)
    grammar_vitality = task5_grammar_vitality(records, gram_trans)
    health_families = task6_health_families(records)
    conditional_health, registry = task7_conditional_health(records, context_momentum)
    engine = task8_engine(records, health_index, health_transitions, participant_energy, grammar_vitality)

    write_csv(HEALTH_SCORES_CSV, health_scores)
    write_csv(HEALTH_TRANSITIONS_CSV, health_transitions)
    write_csv(CONTEXT_MOMENTUM_CSV, context_momentum)
    write_csv(PARTICIPANT_ENERGY_CSV, participant_energy)
    write_csv(GRAMMAR_VITALITY_CSV, grammar_vitality)
    write_csv(HEALTH_FAMILIES_CSV, health_families)
    write_csv(CONDITIONAL_HEALTH_CSV, conditional_health)
    write_csv(ENGINE_CSV, engine)
    write_csv(REGISTRY_CSV, registry)

    dates = sorted({r["date"] for r in records})
    lines = [
        "===== SCOUT SEASON2 P11 - SITUATION HEALTH ENGINE =====",
        "",
        f"Sample: {len(records)} records | {dates[0]}..{dates[-1]} ({len(dates)} days)",
        f"Collected: {collected or '(none)'} | Health archetypes: {len(health_scores)}",
        "",
        "--- Task 1: Situation health scores ---",
    ]
    for row in sorted(health_scores, key=lambda x: -x["health_score"])[:8]:
        lines.append(
            f"  {row['situation_archetype']}: score={row['health_score']} [{row['health_class']}] "
            f"accel={row['acceleration']} collapse={row['collapse_pct']}%"
        )

    lines.extend(["", "--- Task 2: Health transitions (top) ---"])
    for row in health_transitions[:8]:
        lines.append(
            f"  {row['from_health']} -> {row['to_health']}: p={row['transition_probability_pct']}% n={row['transition_count']}"
        )

    lines.extend(["", "--- Task 4: Participant energy ---"])
    for row in participant_energy[:6]:
        lines.append(
            f"  {row['participant_state']}: energy={row['energy_level']} trend={row['energy_trend']} [{row['behaviour_label']}]"
        )

    lines.extend(["", "--- Task 5: Grammar vitality ---"])
    for row in sorted(grammar_vitality, key=lambda x: -x["health_score"])[:5]:
        lines.append(f"  {row['grammar'][:40]}: {row['vitality']} health={row['health_score']}")

    revivals = [r for r in registry if r["current_status"] == "REVIVED"]
    lines.extend(["", f"--- Task 7: Registry revivals: {len(revivals)} ---"])
    for row in revivals[:6]:
        lines.append(f"  {row['hypothesis_name']}: {row['prior_status']} -> REVIVED delta={row['best_health_delta']}")

    lines.extend(strategic_insights(health_scores, context_momentum))
    lines.extend([
        "",
        "--- Philosophy ---",
        " Price is consequence; health determines durability",
        " healthy_trend+strengthening != healthy_trend+weakening",
        "",
        f"Engine: {ENGINE_CSV}",
        "=" * 58,
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P11 SITUATION HEALTH ENGINE =====")
    print(f"Records: {len(records)} | Health scores: {len(health_scores)} | Revivals: {len(revivals)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
