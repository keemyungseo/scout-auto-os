"""
Scout Learning Season2 - P13 Adaptive Market Ecology & Functional Role Discovery

Research only. Discover what repeating behaviours appear to accomplish inside
the market ecosystem — not who moves markets.

Governed by Scout Research Constitution: empirical behaviour only; probabilistic
roles; confidence on every output; unknown is valid; no price prediction.
"""

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p5_historical_expansion import STEP1_DATES, build_unified_records, collect_missing
from season2_p6_market_memory import attach_forward_targets
from season2_p7_behaviour_grammar import enrich_physics
from season2_p8_participant_state import enrich_panel_fields
from season2_p9_conditional_interaction_mining import prepare_records
from season2_p10_situation_evolution import build_edges, label_records, load_interactions
from season2_p11_situation_health import (
    annotate_record_momentum,
    cohort_metrics,
    load_registries,
    record_health_class,
    task1_health_scores,
)
from season2_p12_situation_pressure import (
    load_csv_index,
    load_p10_situation_transitions,
    record_pressure,
)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

FUNCTIONS_CSV = LOGS_DIR / "season2_p13_market_functions.csv"
SITUATION_ROLES_CSV = LOGS_DIR / "season2_p13_situation_roles.csv"
ROLE_TRANSITIONS_CSV = LOGS_DIR / "season2_p13_role_transitions.csv"
FUNCTIONAL_FAMILIES_CSV = LOGS_DIR / "season2_p13_functional_families.csv"
ROLE_PERSISTENCE_CSV = LOGS_DIR / "season2_p13_role_persistence.csv"
ROLE_CONFLICTS_CSV = LOGS_DIR / "season2_p13_role_conflicts.csv"
ENGINE_CSV = LOGS_DIR / "season2_p13_role_engine_output.csv"
REGISTRY_CSV = LOGS_DIR / "season2_p13_role_registry.csv"
REPORT_TXT = LOGS_DIR / "season2_p13_research_report.txt"

MIN_N = 6

SEED_FUNCTIONS = (
    "liquidity_transfer",
    "accumulation",
    "distribution",
    "exhaustion",
    "repair",
    "rebalancing",
    "rotation",
    "trend_extension",
    "volatility_absorption",
    "forced_repricing",
)

FUNCTION_DESCRIPTIONS = {
    "liquidity_transfer": "Volume and participation shift between holders without stable trend commitment",
    "accumulation": "Position building under compression with restrained volatility",
    "distribution": "Existing gains exchanged to new participants via exhaustion patterns",
    "exhaustion": "Participant energy depletion after extended markup or chase",
    "repair": "Post-stress normalization and inventory recovery",
    "rebalancing": "Cross-participant allocation reset without directional extension",
    "rotation": "Capital rotates across symbols or phases without single-trend dominance",
    "trend_extension": "Healthy continuation of established markup under supportive supply",
    "volatility_absorption": "High activity absorbed without proportional collapse",
    "forced_repricing": "Supply collapse or liquidation forces rapid price reset",
    "unknown": "Insufficient empirical cohesion to assign a dominant function",
}


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


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


def score_accumulation(record: dict) -> float:
    sit = record.get("situation_archetype", "")
    ps = record.get("participant_state") or ""
    gram = record.get("grammar") or ""
    supply = record.get("supply_label") or ""
    s = 0.0
    if sit == "early_accumulation":
        s += 35
    if any(x in ps for x in ("compression", "accumulation", "warm")):
        s += 25
    if any(x in gram for x in ("compression", "congestion", "vol_decrease")):
        s += 20
    if supply in ("LOW_SUPPLY", "MID_SUPPLY"):
        s += 15
    if record.get("pressure_score", 50) < 35:
        s += 10
    return clamp(s)


def score_distribution(record: dict) -> float:
    sit = record.get("situation_archetype", "")
    ps = record.get("participant_state") or ""
    gram = record.get("grammar") or ""
    cluster = record.get("behaviour_cluster") or record.get("trend_cluster") or ""
    s = 0.0
    if sit == "distribution":
        s += 40
    if any(x in ps for x in ("exhaustion", "profit_taking", "distribution")):
        s += 30
    if "vol_exhaustion" in gram or "upper_wick" in gram:
        s += 25
    if cluster == "distribution":
        s += 20
    return clamp(s)


def score_exhaustion(record: dict) -> float:
    ps = record.get("participant_state") or ""
    gram = record.get("grammar") or ""
    s = 0.0
    if "exhaustion" in ps or "late_markup" in ps:
        s += 40
    if "vol_exhaustion" in gram:
        s += 35
    if record.get("participant_stress", 0) >= 70:
        s += 25
    if record.get("situation_archetype") in ("late_chasing_zone", "distribution"):
        s += 15
    return clamp(s)


def score_repair(record: dict) -> float:
    ps = record.get("participant_state") or ""
    hclass = record.get("health_class") or ""
    s = 0.0
    if "recovery" in ps or "inventory_recovery" in ps:
        s += 40
    if hclass in ("IMPROVING", "RECOVERING"):
        s += 30
    if record.get("pressure_direction") == "releasing":
        s += 20
    if record.get("pressure_score", 50) < 30:
        s += 15
    return clamp(s)


def score_rebalancing(record: dict) -> float:
    sit = record.get("situation_archetype", "")
    ps = record.get("participant_state") or ""
    cluster = record.get("behaviour_cluster") or record.get("trend_cluster") or ""
    s = 0.0
    if sit in ("rotation_chop", "unclassified_situation"):
        s += 30
    if "rotation" in ps or "transition" in ps:
        s += 25
    if cluster == "rotation":
        s += 25
    if record.get("supply_label") == "MID_SUPPLY":
        s += 15
    return clamp(s)


def score_rotation(record: dict) -> float:
    cluster = record.get("behaviour_cluster") or record.get("trend_cluster") or ""
    ps = record.get("participant_state") or ""
    gram = record.get("grammar") or ""
    s = 0.0
    if cluster == "rotation":
        s += 35
    if "choppy" in ps or "rotation" in ps:
        s += 30
    if "inside_bar" in gram or "small_body" in gram:
        s += 20
    if record.get("situation_archetype") == "rotation_chop":
        s += 20
    return clamp(s)


def score_trend_extension(record: dict) -> float:
    sit = record.get("situation_archetype", "")
    ps = record.get("participant_state") or ""
    gram = record.get("grammar") or ""
    supply = record.get("supply_label") or ""
    s = 0.0
    if sit == "healthy_trend":
        s += 35
    if any(x in ps for x in ("markup", "accumulation", "expansion")):
        s += 25
    if "vol_increase" in gram or "trend_cont" in gram or "gap_momentum" in gram:
        s += 25
    if supply in ("MID_SUPPLY", "HIGH_SUPPLY"):
        s += 20
    if record.get("health_score", 0) and float(record.get("health_score") or 0) >= 60:
        s += 10
    return clamp(s)


def score_volatility_absorption(record: dict) -> float:
    supply = record.get("supply_label") or ""
    ps = record.get("participant_state") or ""
    s = 0.0
    if supply == "HIGH_SUPPLY":
        s += 35
    if "compression" in ps:
        s += 25
    if record.get("collapse_label") != "YES" and record.get("pressure_score", 50) >= 40:
        s += 20
    if record.get("situation_archetype") == "healthy_trend" and record.get("pressure_band") in ("medium", "high"):
        s += 15
    return clamp(s)


def score_forced_repricing(record: dict) -> float:
    supply = record.get("supply_label") or ""
    ps = record.get("participant_state") or ""
    sit = record.get("situation_archetype", "")
    s = 0.0
    if supply == "COLLAPSE":
        s += 50
    if record.get("collapse_label") == "YES":
        s += 40
    if any(x in ps for x in ("liquidation", "panic", "trapped")):
        s += 35
    if sit == "forced_liquidation_zone":
        s += 35
    return clamp(s)


def score_liquidity_transfer(record: dict) -> float:
    ps = record.get("participant_state") or ""
    gram = record.get("grammar") or ""
    sit = record.get("situation_archetype", "")
    s = 0.0
    if sit == "late_chasing_zone":
        s += 35
    if "chasing" in ps or "transition" in ps:
        s += 30
    if "gap_momentum" in gram:
        s += 25
    if record.get("pressure_band") in ("medium", "high"):
        s += 15
    return clamp(s)


ROLE_SCORERS = {
    "accumulation": score_accumulation,
    "distribution": score_distribution,
    "exhaustion": score_exhaustion,
    "repair": score_repair,
    "rebalancing": score_rebalancing,
    "rotation": score_rotation,
    "trend_extension": score_trend_extension,
    "volatility_absorption": score_volatility_absorption,
    "forced_repricing": score_forced_repricing,
    "liquidity_transfer": score_liquidity_transfer,
}


def infer_role_scores(record: dict) -> dict[str, float]:
    return {name: fn(record) for name, fn in ROLE_SCORERS.items()}


def role_probabilities(scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(v, 0.1) for v in scores.values())
    if total <= 0:
        return {k: 100.0 / len(scores) for k in scores}
    return {k: round(max(v, 0.1) / total * 100, 1) for k, v in scores.items()}


def annotate_roles(records: list[dict]) -> None:
    for record in records:
        scores = infer_role_scores(record)
        probs = role_probabilities(scores)
        ranked = sorted(probs.items(), key=lambda x: -x[1])
        primary, p1 = ranked[0]
        secondary, p2 = ranked[1] if len(ranked) > 1 else ("unknown", 0.0)
        if p1 < 18:
            primary = "unknown"
        record["_role_scores"] = scores
        record["_role_probs"] = probs
        record["primary_role"] = primary
        record["secondary_role"] = secondary if p2 >= 12 else "unknown"
        record["primary_role_confidence_pct"] = p1 if primary != "unknown" else max(0, p1 - 5)
        record["role_ambiguity_pct"] = round(p2 if primary != "unknown" else 100 - p1, 1)
        record["unknown_probability_pct"] = round(max(0, 100 - p1 - max(0, p2 - 10)), 1)


def supporting_evidence(record: dict, role: str) -> list[str]:
    bits = []
    sit = record.get("situation_archetype", "")
    ps = record.get("participant_state") or ""
    supply = record.get("supply_label") or ""
    if role == "accumulation" and sit == "early_accumulation":
        bits.append("situation=early_accumulation")
    if role == "trend_extension" and sit == "healthy_trend":
        bits.append("situation=healthy_trend")
    if role == "distribution" and sit == "distribution":
        bits.append("situation=distribution")
    if role == "forced_repricing" and supply == "COLLAPSE":
        bits.append("supply=COLLAPSE")
    if role == "exhaustion" and "exhaustion" in ps:
        bits.append(f"participant={ps}")
    if role == "repair" and "recovery" in ps:
        bits.append(f"participant={ps}")
    if role == "liquidity_transfer" and sit == "late_chasing_zone":
        bits.append("situation=late_chasing")
    if role == "volatility_absorption" and supply == "HIGH_SUPPLY":
        bits.append("supply=HIGH_SUPPLY")
    score = record.get("_role_scores", {}).get(role, 0)
    if score >= 50:
        bits.append(f"role_score={score:.0f}")
    return bits[:4]


def conflicting_evidence(record: dict, role: str) -> list[str]:
    bits = []
    scores = record.get("_role_scores", {})
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    for other, sc in ranked[:3]:
        if other != role and sc >= 30:
            bits.append(f"also_{other}={sc:.0f}")
    if record.get("primary_role") != role and record.get("secondary_role") == role:
        bits.append("secondary_not_primary")
    if record.get("evolution_conflict", 0) >= 35:
        bits.append("evolution_conflict")
    if record.get("pressure_score", 0) >= 55 and role in ("trend_extension", "accumulation"):
        bits.append("elevated_pressure")
    return bits[:4]


def confidence_tier(n: int, spread: float) -> str:
    if n >= 25 and spread >= 15:
        return "high"
    if n >= MIN_N:
        return "medium"
    return "hypothesis"


def task1_market_functions(records: list[dict]) -> list[dict]:
    rows = []
    for func in SEED_FUNCTIONS:
        grp = [r for r in records if r["primary_role"] == func or r.get("_role_scores", {}).get(func, 0) >= 35]
        if len(grp) < MIN_N:
            continue
        primary_grp = [r for r in records if r["primary_role"] == func]
        m = cohort_metrics(primary_grp if len(primary_grp) >= MIN_N else grp)
        persist = sum(1 for r in grp if r.get("primary_role") == func) / len(grp) * 100
        rows.append(
            {
                "function_name": func,
                "description": FUNCTION_DESCRIPTIONS[func],
                "discovery_type": "seed",
                "sample_size": len(grp),
                "primary_assignment_pct": round(len(primary_grp) / len(grp) * 100, 1) if grp else 0,
                "median_forward_6h": m.get("median_f6", ""),
                "collapse_rate_pct": m.get("collapse_pct", 0),
                "avg_role_score": round(statistics.mean(r["_role_scores"][func] for r in grp), 1),
                "persistence_as_primary_pct": round(persist, 1),
                "confidence": confidence_tier(len(grp), m.get("median_f6", 0) or 0),
            }
        )

    # Emergent functions from cohesive ambiguous cohorts
    sig_groups: dict[str, list] = defaultdict(list)
    for r in records:
        if r["primary_role"] != "unknown":
            continue
        key = "|".join(
            [
                r.get("situation_archetype", "?")[:20],
                (r.get("participant_state") or "?")[:20],
                r.get("supply_label", "?"),
            ]
        )
        sig_groups[key].append(r)

    for key, grp in sig_groups.items():
        if len(grp) < MIN_N:
            continue
        top_role = Counter(
            max(r["_role_scores"].items(), key=lambda x: x[1])[0] for r in grp
        ).most_common(1)[0]
        if top_role[1] / len(grp) < 0.55:
            name = f"emergent_{key.replace('|', '_')[:40]}"
            m = cohort_metrics(grp)
            rows.append(
                {
                    "function_name": name,
                    "description": f"Ambiguous cohort: {key}",
                    "discovery_type": "emergent",
                    "sample_size": len(grp),
                    "primary_assignment_pct": 0,
                    "median_forward_6h": m.get("median_f6", ""),
                    "collapse_rate_pct": m.get("collapse_pct", 0),
                    "avg_role_score": round(statistics.mean(max(r["_role_scores"].values()) for r in grp), 1),
                    "persistence_as_primary_pct": 0,
                    "confidence": "hypothesis",
                }
            )
    return sorted(rows, key=lambda x: -x["sample_size"])


def task2_situation_roles(records: list[dict]) -> list[dict]:
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r.get("situation_archetype", "unknown")].append(r)

    rows = []
    for situation, grp in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(grp) < MIN_N:
            continue
        role_counts = Counter(r["primary_role"] for r in grp)
        total = len(grp)
        top3 = role_counts.most_common(3)
        for rank, (role, count) in enumerate(top3, 1):
            prob = round(count / total * 100, 1)
            rows.append(
                {
                    "situation": situation,
                    "possible_function": role,
                    "function_probability_pct": prob,
                    "rank": rank,
                    "sample_size": total,
                    "median_forward_6h": cohort_metrics([r for r in grp if r["primary_role"] == role]).get("median_f6", ""),
                    "confidence": confidence_tier(count, prob),
                }
            )
    return rows


def task3_role_transitions(edges: list[dict]) -> list[dict]:
    counts: Counter[tuple[str, str]] = Counter()
    f6: dict[tuple[str, str], list] = defaultdict(list)
    collapse: Counter[tuple[str, str]] = Counter()

    for e in edges:
        fr = e["from_record"].get("primary_role", "unknown")
        to = e["to_record"].get("primary_role", "unknown")
        key = (fr, to)
        counts[key] += 1
        if e["to_record"].get("target_f6") is not None:
            f6[key].append(e["to_record"]["target_f6"])
        if e["to_record"].get("collapse_label") == "YES":
            collapse[key] += 1

    from_totals: Counter[str] = Counter()
    for (fr, _), c in counts.items():
        from_totals[fr] += c

    rows = []
    for (fr, to), count in counts.most_common():
        if count < MIN_N and from_totals[fr] < MIN_N * 2:
            continue
        prob = round(count / from_totals[fr] * 100, 1) if from_totals[fr] else 0
        rows.append(
            {
                "from_function": fr,
                "to_function": to,
                "transition_count": count,
                "transition_probability_pct": prob,
                "persistence": "stable" if fr == to and prob >= 35 else "transitioning",
                "median_return_at_next": round(statistics.median(f6[(fr, to)]), 2) if f6.get((fr, to)) else "",
                "collapse_at_next_pct": round(collapse[(fr, to)] / count * 100, 1),
                "confidence": "high" if count >= 15 else "medium" if count >= MIN_N else "hypothesis",
            }
        )
    return rows


def task4_functional_families(records: list[dict]) -> list[dict]:
    sym_profiles: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    sym_n: dict[str, int] = defaultdict(int)

    for r in records:
        sym = r["symbol"]
        sym_n[sym] += 1
        for role, prob in r["_role_probs"].items():
            sym_profiles[sym][role] += prob

    sym_avg = {}
    for sym, prof in sym_profiles.items():
        n = sym_n[sym]
        sym_avg[sym] = {role: v / n for role, v in prof.items()}

    symbols = sorted(sym_avg.keys(), key=lambda s: -sym_n[s])
    assigned: set[str] = set()
    families: dict[str, list[str]] = {}

    def profile_distance(a: dict, b: dict) -> float:
        roles = set(a) | set(b)
        return statistics.mean(abs(a.get(r, 0) - b.get(r, 0)) for r in roles)

    for sym in symbols:
        if sym in assigned or sym_n[sym] < 2:
            continue
        fid = f"ecology_family_{len(families) + 1}"
        families[fid] = [sym]
        assigned.add(sym)
        for other in symbols:
            if other in assigned or sym_n[other] < 2:
                continue
            if profile_distance(sym_avg[sym], sym_avg[other]) <= 12:
                families[fid].append(other)
                assigned.add(other)

    rows = []
    for fid, members in sorted(families.items(), key=lambda x: -len(x[1])):
        dom_roles = Counter()
        total_n = 0
        for sym in members:
            dom = max(sym_avg[sym].items(), key=lambda x: x[1])[0]
            dom_roles[dom] += sym_n[sym]
            total_n += sym_n[sym]
        dominant = dom_roles.most_common(1)[0][0] if dom_roles else "unknown"
        avg_vec = {}
        for role in SEED_FUNCTIONS:
            avg_vec[role] = round(
                statistics.mean(sym_avg[s][role] for s in members if role in sym_avg[s]), 1
            )
        rows.append(
            {
                "family_id": fid,
                "symbols": "|".join(members[:15]),
                "symbol_count": len(members),
                "dominant_function": dominant,
                "avg_trend_extension_pct": avg_vec.get("trend_extension", 0),
                "avg_exhaustion_pct": avg_vec.get("exhaustion", 0),
                "avg_repair_pct": avg_vec.get("repair", 0),
                "appearances": total_n,
                "confidence": "high" if total_n >= 30 else "medium" if total_n >= MIN_N else "hypothesis",
            }
        )
    return rows


def task5_role_persistence(edges: list[dict], records: list[dict]) -> list[dict]:
    streaks: dict[str, list[int]] = defaultdict(list)
    by_sym: dict[str, list] = defaultdict(list)
    for r in records:
        by_sym[r["symbol"]].append(r)
    for sym, group in by_sym.items():
        group.sort(key=lambda x: x["scan_time"])
        streak = 1
        for i in range(1, len(group)):
            if group[i]["primary_role"] == group[i - 1]["primary_role"]:
                streak += 1
            else:
                if streak > 1:
                    streaks[group[i - 1]["primary_role"]].append(streak)
                streak = 1
        if streak > 1:
            streaks[group[-1]["primary_role"]].append(streak)

    strengthen: dict[str, Counter] = defaultdict(Counter)
    weaken: dict[str, Counter] = defaultdict(Counter)
    for e in edges:
        role = e["from_record"].get("primary_role", "unknown")
        if e["from_record"].get("primary_role") == e["to_record"].get("primary_role"):
            if float(e["from_record"].get("health_score") or 0) >= 55:
                strengthen[role]["supportive_health"] += 1
            if e["from_record"].get("supply_label") in ("MID_SUPPLY", "HIGH_SUPPLY"):
                strengthen[role]["supportive_supply"] += 1
            if e["from_record"].get("pressure_direction") == "releasing":
                strengthen[role]["pressure_release"] += 1
        else:
            if e["from_record"].get("pressure_direction") == "building":
                weaken[role]["pressure_build"] += 1
            if float(e["from_record"].get("evolution_conflict") or 0) >= 35:
                weaken[role]["evolution_conflict"] += 1
            if e["from_record"].get("supply_label") == "COLLAPSE":
                weaken[role]["collapse_supply"] += 1

    rows = []
    for role in SEED_FUNCTIONS + ("unknown",):
        grp = [r for r in records if r["primary_role"] == role]
        if len(grp) < MIN_N and not streaks.get(role):
            continue
        med_streak = round(statistics.median(streaks[role]), 1) if streaks.get(role) else 1.0
        same_next = sum(1 for e in edges if e["from_record"].get("primary_role") == role and e["to_record"].get("primary_role") == role)
        total_from = sum(1 for e in edges if e["from_record"].get("primary_role") == role)
        persist_pct = round(same_next / total_from * 100, 1) if total_from else 0
        str_top = strengthen[role].most_common(2)
        weak_top = weaken[role].most_common(2)
        rows.append(
            {
                "function": role,
                "median_streak_scans": med_streak,
                "persistence_next_scan_pct": persist_pct,
                "sample_size": len(grp),
                "strengthened_by": "|".join(f"{k}:{v}" for k, v in str_top) if str_top else "",
                "weakened_by": "|".join(f"{k}:{v}" for k, v in weak_top) if weak_top else "",
                "confidence": confidence_tier(len(grp), persist_pct),
            }
        )
    return rows


def task6_role_conflicts(records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        scores = r["_role_scores"]
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        if len(ranked) < 2:
            continue
        top, s1 = ranked[0]
        second, s2 = ranked[1]
        gap = s1 - s2
        if gap > 20 and s1 >= 35:
            continue
        ambiguity = round(100 - gap, 1) if s1 >= 20 else r["unknown_probability_pct"]
        if ambiguity < 45 and gap >= 15:
            continue
        rows.append(
            {
                "date": r["date"],
                "symbol": r["symbol"],
                "scan_time": r["scan_time"],
                "situation": r.get("situation_archetype"),
                "primary_role": top,
                "conflicting_role": second,
                "role_gap": round(gap, 1),
                "ambiguity_score": ambiguity,
                "unknown_probability_pct": r["unknown_probability_pct"],
                "supporting_roles": f"{top}:{s1:.0f}|{second}:{s2:.0f}",
                "confidence": "medium" if ambiguity >= 55 else "hypothesis",
            }
        )
    return sorted(rows, key=lambda x: -x["ambiguity_score"])[:80]


def ecological_stance(record: dict) -> tuple[str, str]:
    role = record.get("primary_role", "unknown")
    conf = float(record.get("primary_role_confidence_pct") or 0)
    unknown = float(record.get("unknown_probability_pct") or 0)
    amb = float(record.get("role_ambiguity_pct") or 0)
    supply = record.get("supply_label") or ""
    pressure = float(record.get("pressure_score") or 50)

    if role == "forced_repricing" and conf >= 40:
        return "Avoid", f"dominant forced_repricing conf={conf:.0f}%"
    if unknown >= 45 or amb >= 50:
        return "Wait", f"high unknown={unknown:.0f}% ambiguity={amb:.0f}%"
    if role in ("repair", "rebalancing") and conf >= 35 and pressure < 40:
        return "Strong Buy", f"{role} with releasing ecology conf={conf:.0f}%"
    if role == "accumulation" and conf >= 30 and supply != "COLLAPSE":
        return "Buy", f"accumulation conf={conf:.0f}%"
    if role in ("exhaustion", "distribution") and conf >= 35:
        return "Reduce", f"{role} function dominant conf={conf:.0f}%"
    if role == "trend_extension" and conf >= 35 and pressure < 55:
        return "Buy", f"trend_extension conf={conf:.0f}%"
    if role == "volatility_absorption" and conf >= 30:
        return "Watch", f"absorption ecology conf={conf:.0f}%"
    if pressure >= 55:
        return "Reduce", f"elevated pressure={pressure:.0f} role={role}"
    return "Watch", f"role={role} conf={conf:.0f}% unknown={unknown:.0f}%"


def task7_role_engine(records: list[dict]) -> list[dict]:
    rows = []
    for record in records[-40:]:
        probs = record["_role_probs"]
        ranked = sorted(probs.items(), key=lambda x: -x[1])[:3]
        candidates = "|".join(f"{r}:{p}%" for r, p in ranked)
        primary = record["primary_role"]
        stance, reason = ecological_stance(record)
        rows.append(
            {
                "date": record["date"],
                "symbol": record["symbol"],
                "scan_time": record["scan_time"],
                "situation": record.get("situation_archetype"),
                "role_candidates": candidates,
                "primary_function": primary,
                "secondary_function": record.get("secondary_role"),
                "confidence_pct": record.get("primary_role_confidence_pct"),
                "unknown_probability_pct": record.get("unknown_probability_pct"),
                "ambiguity_pct": record.get("role_ambiguity_pct"),
                "supporting_evidence": "|".join(supporting_evidence(record, primary)),
                "conflicting_evidence": "|".join(conflicting_evidence(record, primary)),
                "health_class": record.get("health_class"),
                "pressure_score": record.get("pressure_score"),
                "supply_label": record.get("supply_label"),
                "recommended_stance": stance,
                "reason": reason,
                "confidence_tier": record.get("situation_confidence", "hypothesis"),
            }
        )
    return rows


def role_registry(functions: list[dict], transitions: list[dict]) -> list[dict]:
    rows = []
    for item in load_registries():
        rows.append(
            {
                "hypothesis_kind": item["kind"],
                "hypothesis_name": item["name"][:60],
                "prior_status": item["prior_status"],
                "current_status": item["prior_status"],
                "ecology_note": "retest under functional role framework",
            }
        )
    for fn in functions:
        if fn["discovery_type"] == "emergent":
            rows.append(
                {
                    "hypothesis_kind": "emergent_function",
                    "hypothesis_name": fn["function_name"],
                    "prior_status": "OPEN",
                    "current_status": "CONDITIONAL",
                    "ecology_note": fn["description"][:80],
                }
            )
    for tr in transitions[:5]:
        if tr["from_function"] in SEED_FUNCTIONS and tr["to_function"] in SEED_FUNCTIONS:
            rows.append(
                {
                    "hypothesis_kind": "role_transition",
                    "hypothesis_name": f"{tr['from_function']}->{tr['to_function']}",
                    "prior_status": "OPEN",
                    "current_status": "ACTIVE" if tr["transition_count"] >= 15 else "CONDITIONAL",
                    "ecology_note": f"p={tr['transition_probability_pct']}% n={tr['transition_count']}",
                }
            )
    return rows


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
    sit_trans = load_p10_situation_transitions()

    health_scores = task1_health_scores(records, edges, sit_trans)
    health_index = {r["situation_archetype"]: r for r in health_scores}
    for r in records:
        r["health_class"] = record_health_class(r, health_index)
        h = health_index.get(r.get("situation_archetype", ""), {})
        r["health_score"] = h.get("health_score", "")

    grammar_vitality = load_csv_index(LOGS_DIR / "season2_p11_grammar_vitality.csv", "grammar")
    for r in records:
        pr = record_pressure(r, health_index, grammar_vitality, sit_trans)
        r.update(pr)
        if r.get("_health_accel") is None:
            r["_health_accel"] = 0

    # pressure direction for repair scoring
    bands = {"low": 0, "medium": 1, "high": 2, "explosive": 3}
    for e in edges:
        e["to_record"]["pressure_direction"] = "flat"
    for e in edges:
        fr = bands.get(e["from_record"].get("pressure_band"), 1)
        to = bands.get(e["to_record"].get("pressure_band"), 1)
        if to > fr:
            e["to_record"]["pressure_direction"] = "building"
        elif to < fr:
            e["to_record"]["pressure_direction"] = "releasing"

    annotate_roles(records)

    functions = task1_market_functions(records)
    situation_roles = task2_situation_roles(records)
    role_transitions = task3_role_transitions(edges)
    functional_families = task4_functional_families(records)
    role_persistence = task5_role_persistence(edges, records)
    role_conflicts = task6_role_conflicts(records)
    engine = task7_role_engine(records)
    registry = role_registry(functions, role_transitions)

    write_csv(FUNCTIONS_CSV, functions)
    write_csv(SITUATION_ROLES_CSV, situation_roles)
    write_csv(ROLE_TRANSITIONS_CSV, role_transitions)
    write_csv(FUNCTIONAL_FAMILIES_CSV, functional_families)
    write_csv(ROLE_PERSISTENCE_CSV, role_persistence)
    write_csv(ROLE_CONFLICTS_CSV, role_conflicts)
    write_csv(ENGINE_CSV, engine)
    write_csv(REGISTRY_CSV, registry)

    dates = sorted({r["date"] for r in records})
    lines = [
        "===== SCOUT SEASON2 P13 - ADAPTIVE MARKET ECOLOGY =====",
        "",
        f"Sample: {len(records)} records | {dates[0]}..{dates[-1]} ({len(dates)} days)",
        f"Collected: {collected or '(none)'} | Functions mined: {len(functions)}",
        "",
        "--- Task 1: Recurring market functions ---",
    ]
    for row in functions[:10]:
        lines.append(
            f"  {row['function_name']}: n={row['sample_size']} collapse={row['collapse_rate_pct']}% "
            f"primary={row['primary_assignment_pct']}% [{row['confidence']}]"
        )

    lines.extend(["", "--- Task 2: Situation -> function ---"])
    seen = set()
    for row in situation_roles:
        if row["situation"] in seen:
            continue
        top = [x for x in situation_roles if x["situation"] == row["situation"]][:2]
        parts = ", ".join(f"{x['possible_function']}({x['function_probability_pct']}%)" for x in top)
        lines.append(f"  {row['situation']}: {parts}")
        seen.add(row["situation"])

    lines.extend(["", "--- Task 3: Role transitions ---"])
    for row in role_transitions[:8]:
        lines.append(
            f"  {row['from_function']} -> {row['to_function']}: p={row['transition_probability_pct']}% "
            f"n={row['transition_count']}"
        )

    lines.extend(["", "--- Task 5: Role persistence ---"])
    for row in sorted(role_persistence, key=lambda x: -x["persistence_next_scan_pct"])[:6]:
        lines.append(
            f"  {row['function']}: persist={row['persistence_next_scan_pct']}% "
            f"streak={row['median_streak_scans']}"
        )

    lines.extend(["", "--- Task 6: Conflicting roles (top ambiguity) ---"])
    for row in role_conflicts[:5]:
        lines.append(
            f"  {row['symbol']} {row['situation']}: {row['primary_role']} vs {row['conflicting_role']} "
            f"ambiguity={row['ambiguity_score']}"
        )

    lines.extend([
        "",
        "--- Scout Research Constitution ---",
        " Functional roles inferred from behaviour — no hidden actors",
        " Roles are probabilistic; unknown is valid; no price prediction",
        "",
        "--- Operating principle ---",
        " Markets are adaptive ecosystems; scout studies recurring functions",
        "",
        f"Engine: {ENGINE_CSV}",
        "=" * 58,
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P13 ADAPTIVE MARKET ECOLOGY =====")
    print(f"Records: {len(records)} | Functions: {len(functions)} | Families: {len(functional_families)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
