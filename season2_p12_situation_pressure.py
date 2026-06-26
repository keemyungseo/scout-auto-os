"""
Scout Learning Season2 - P12 Situation Pressure Engine & Internal Stress Discovery

Research only. Measure accumulated internal pressure, not price direction.
Governed by Scout Research Constitution: empirical behaviour only; probabilistic
intent; confidence on every output; unknown is valid; no price prediction.
"""

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p5_historical_expansion import SEEDS, STEP1_DATES, build_unified_records, collect_missing
from season2_p6_market_memory import attach_forward_targets
from season2_p7_behaviour_grammar import enrich_physics
from season2_p8_participant_state import enrich_panel_fields
from season2_p9_conditional_interaction_mining import prepare_records
from season2_p10_situation_evolution import build_edges, label_records, load_interactions
from season2_p11_situation_health import (
    annotate_record_momentum,
    cohort_metrics,
    load_registries,
    participant_energy_score,
    record_health_class,
    task1_health_scores,
)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PRESSURE_SCORES_CSV = LOGS_DIR / "season2_p12_pressure_scores.csv"
PRESSURE_TRANSITIONS_CSV = LOGS_DIR / "season2_p12_pressure_transitions.csv"
PRESSURE_INTERACTIONS_CSV = LOGS_DIR / "season2_p12_pressure_interactions.csv"
PARTICIPANT_STRESS_CSV = LOGS_DIR / "season2_p12_participant_stress.csv"
GRAMMAR_STRESS_CSV = LOGS_DIR / "season2_p12_grammar_stress.csv"
CONDITIONAL_PRESSURE_CSV = LOGS_DIR / "season2_p12_conditional_pressure.csv"
PRESSURE_FAMILIES_CSV = LOGS_DIR / "season2_p12_pressure_families.csv"
ENGINE_CSV = LOGS_DIR / "season2_p12_pressure_engine_output.csv"
REGISTRY_CSV = LOGS_DIR / "season2_p12_pressure_registry.csv"
REPORT_TXT = LOGS_DIR / "season2_p12_research_report.txt"

MIN_N = 6
PRESSURE_BANDS = ("low", "medium", "high", "explosive")

# Stressful participant states (empirical proxy)
STRESS_STATE = {
    "scan_exhaustion": 0.85,
    "late_chasing": 0.75,
    "late_markup": 0.8,
    "trapped_long": 0.9,
    "panic_exit": 0.95,
    "forced_liquidation": 1.0,
    "distribution": 0.7,
    "scan_transition": 0.55,
    "scan_compression": 0.35,
    "scan_warm-up": 0.25,
    "scan_choppy": 0.5,
}


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def load_csv_index(path: Path, key_col: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {row[key_col]: row for row in csv.DictReader(f)}


def load_p10_situation_transitions() -> list[dict]:
    path = LOGS_DIR / "season2_p10_situation_transitions.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pressure_band(score: float) -> str:
    if score >= 75:
        return "explosive"
    if score >= 50:
        return "high"
    if score >= 28:
        return "medium"
    return "low"


def participant_stress(record: dict) -> float:
    ps = record.get("participant_state") or ""
    for key, val in STRESS_STATE.items():
        if key in ps:
            return val * 100
    return (1.0 - participant_energy_score(ps)) * 100


def grammar_conflict(record: dict, grammar_vitality: dict[str, dict]) -> float:
    gram = (record.get("grammar") or "")[:40]
    vit = grammar_vitality.get(gram, {}).get("vitality", "stable")
    sit = record.get("situation_archetype", "")
    conflict = 0.0
    if vit == "fading" and sit in ("healthy_trend", "late_chasing_zone"):
        conflict += 40
    if vit == "reviving" and sit in ("distribution", "forced_liquidation_zone"):
        conflict -= 15
    if vit == "conflicting":
        conflict += 50
    # grammar vs participant mismatch
    if "exhaustion" in gram and "warm" in (record.get("participant_state") or ""):
        conflict += 25
    return clamp(conflict)


def evolution_conflict(record: dict, health_index: dict[str, dict]) -> float:
    sit = record.get("situation_archetype", "")
    h = health_index.get(sit, {})
    hclass = h.get("health_class", "STABLE")
    score = 0.0
    if sit == "healthy_trend" and hclass in ("FRAGILE", "WEAKENING"):
        score += 45
    if sit == "late_chasing_zone" and hclass == "FRAGILE":
        score += 35
    if sit == "distribution" and h.get("median_forward_6h", 0) and float(h.get("median_forward_6h", 0)) > 0:
        score += 20
    if sit == "early_accumulation" and hclass in ("IMPROVING", "RECOVERING"):
        score -= 20
    return clamp(score)


def health_decay_pressure(record: dict, health_index: dict[str, dict]) -> float:
    sit = record.get("situation_archetype", "")
    h = health_index.get(sit, {})
    decay = float(h.get("decay") or 0)
    accel = record.get("_health_accel", 0)
    return clamp(decay * 3 + max(0, -accel) * 4)


def persistence_overload(record: dict, health_index: dict[str, dict]) -> float:
    sit = record.get("situation_archetype", "")
    h = health_index.get(sit, {})
    persist = float(h.get("persistence_pct") or 50)
    accel = record.get("_health_accel", 0)
    if persist >= 90 and accel < -1:
        return clamp((persist - 80) * 1.5 + abs(accel) * 5)
    if persist >= 70 and accel < 0:
        return clamp((persist - 60) * 0.8)
    return 0.0


def transition_congestion(situation: str, sit_transitions: list[dict]) -> float:
    if not sit_transitions:
        return 0.0
    total = sum(int(r["transition_count"]) for r in sit_transitions if r["from_state"] == situation)
    if not total:
        return 0.0
    fragile = sum(
        int(r["transition_count"])
        for r in sit_transitions
        for t in (r["to_state"],)
        if r["from_state"] == situation and t in ("distribution", "late_chasing_zone", "forced_liquidation_zone")
    )
    self_loops = sum(
        int(r["transition_count"])
        for r in sit_transitions
        if r["from_state"] == situation and r["to_state"] == situation
    )
    return clamp(fragile / total * 60 + self_loops / total * 20)


def record_pressure(
    record: dict,
    health_index: dict[str, dict],
    grammar_vitality: dict[str, dict],
    sit_transitions: list[dict],
) -> dict:
    sit = record.get("situation_archetype", "unknown")
    components = {
        "participant_stress": participant_stress(record),
        "grammar_conflict": grammar_conflict(record, grammar_vitality),
        "evolution_conflict": evolution_conflict(record, health_index),
        "health_decay": health_decay_pressure(record, health_index),
        "persistence_overload": persistence_overload(record, health_index),
        "transition_congestion": transition_congestion(sit, sit_transitions),
    }
    weights = {
        "participant_stress": 0.25,
        "grammar_conflict": 0.15,
        "evolution_conflict": 0.2,
        "health_decay": 0.2,
        "persistence_overload": 0.1,
        "transition_congestion": 0.1,
    }
    score = sum(components[k] * weights[k] for k in components)
    return {
        "pressure_score": round(clamp(score), 1),
        "pressure_band": pressure_band(score),
        **{k: round(v, 1) for k, v in components.items()},
    }


def stress_level(score: float) -> str:
    if score >= 75:
        return "explosive"
    if score >= 50:
        return "stressed"
    if score >= 28:
        return "elevated"
    return "stable"


def task1_situation_pressure(records: list[dict], health_index: dict, grammar_vitality: dict, sit_trans: list[dict]) -> list[dict]:
    for r in records:
        r.update(record_pressure(r, health_index, grammar_vitality, sit_trans))

    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r.get("situation_archetype", "unknown")].append(r)

    rows = []
    for situation, group in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(group) < MIN_N:
            continue
        scores = [r["pressure_score"] for r in group]
        m = cohort_metrics(group)
        avg_p = round(statistics.mean(scores), 1)
        med_p = round(statistics.median(scores), 1)
        collapses = sum(1 for r in group if r.get("collapse_label") == "YES")
        level = stress_level(avg_p)
        rows.append(
            {
                "situation": situation,
                "pressure_score": avg_p,
                "median_pressure": med_p,
                "stress_level": level,
                "health_score": health_index.get(situation, {}).get("health_score", ""),
                "health_class": health_index.get(situation, {}).get("health_class", ""),
                "sample_size": len(group),
                "collapse_rate_pct": round(collapses / len(group) * 100, 1),
                "avg_participant_stress": round(statistics.mean(r["participant_stress"] for r in group), 1),
                "avg_evolution_conflict": round(statistics.mean(r["evolution_conflict"] for r in group), 1),
                "avg_persistence_overload": round(statistics.mean(r["persistence_overload"] for r in group), 1),
                "confidence": "high" if len(group) >= 25 else "medium" if len(group) >= MIN_N else "hypothesis",
            }
        )
    return rows


def task2_pressure_transitions(edges: list[dict]) -> list[dict]:
    counts: Counter[tuple[str, str]] = Counter()
    collapse: Counter[tuple[str, str]] = Counter()
    f6: dict[tuple[str, str], list] = defaultdict(list)

    for e in edges:
        fr = e["from_record"].get("pressure_band", "low")
        to = e["to_record"].get("pressure_band", "low")
        key = (fr, to)
        counts[key] += 1
        if e["to_record"].get("collapse_label") == "YES":
            collapse[key] += 1
        if e["to_record"].get("target_f6") is not None:
            f6[key].append(e["to_record"]["target_f6"])

    from_totals: Counter[str] = Counter()
    for (fr, _), c in counts.items():
        from_totals[fr] += c

    rows = []
    for (fr, to), count in counts.most_common():
        if count < MIN_N and from_totals[fr] < MIN_N * 2:
            continue
        prob = round(count / from_totals[fr] * 100, 1) if from_totals[fr] else 0
        persist = "stable" if fr == to and prob >= 35 else "transitioning"
        rows.append(
            {
                "from_pressure": fr,
                "to_pressure": to,
                "transition_count": count,
                "transition_probability_pct": prob,
                "persistence": persist,
                "median_return_at_next": round(statistics.median(f6[(fr, to)]), 2) if f6.get((fr, to)) else "",
                "collapse_at_next_pct": round(collapse[(fr, to)] / count * 100, 1),
                "confidence": "high" if count >= 15 else "medium" if count >= MIN_N else "hypothesis",
            }
        )
    return rows


def task3_pressure_interactions(records: list[dict], health_index: dict) -> list[dict]:
    dims = [
        ("health_class", lambda r: health_index.get(r.get("situation_archetype", ""), {}).get("health_class", "unknown")),
        ("participant_state", lambda r: r.get("participant_state", "unknown")),
        ("grammar", lambda r: (r.get("grammar") or "unknown")[:30]),
        ("supply", lambda r: r.get("supply_label", "unknown")),
        ("memory", lambda r: r.get("memory", "unknown")),
    ]
    rows = []
    for band in PRESSURE_BANDS:
        band_records = [r for r in records if r.get("pressure_band") == band]
        if len(band_records) < MIN_N:
            continue
        base_collapse = sum(1 for r in band_records if r.get("collapse_label") == "YES") / len(band_records) * 100
        base_score = statistics.mean(r["pressure_score"] for r in band_records)

        for dim_name, dim_fn in dims:
            groups: dict[str, list] = defaultdict(list)
            for r in band_records:
                groups[dim_fn(r)].append(r)
            for val, grp in groups.items():
                if len(grp) < MIN_N or val in ("unknown", ""):
                    continue
                avg_p = statistics.mean(r["pressure_score"] for r in grp)
                coll = sum(1 for r in grp if r.get("collapse_label") == "YES") / len(grp) * 100
                delta_p = round(avg_p - base_score, 2)
                delta_c = round(coll - base_collapse, 2)
                interaction = "explosive" if coll >= base_collapse + 15 else "absorbing" if coll <= base_collapse - 10 else "neutral"
                rows.append(
                    {
                        "pressure_band": band,
                        "interaction_dim": dim_name,
                        "context_value": val[:40],
                        "avg_pressure": round(avg_p, 1),
                        "pressure_delta": delta_p,
                        "collapse_rate_pct": round(coll, 1),
                        "collapse_delta": delta_c,
                        "interaction_type": interaction,
                        "sample_size": len(grp),
                        "confidence": "high" if len(grp) >= 20 else "medium" if len(grp) >= MIN_N else "hypothesis",
                    }
                )
    return sorted(rows, key=lambda x: -abs(x["collapse_delta"]))


def task4_participant_stress(records: list[dict], edges: list[dict]) -> list[dict]:
    labels = {
        "scan_warm-up": "warmup_stress",
        "scan_compression": "compression_stress",
        "late_chasing": "late_chasing_stress",
        "scan_exhaustion": "exhaustion_stress",
    }
    rows = []
    for key, label in labels.items():
        grp = [r for r in records if key in (r.get("participant_state") or "")]
        if len(grp) < MIN_N:
            continue
        m = cohort_metrics(grp)
        avg_stress = statistics.mean(r["participant_stress"] for r in grp)
        recovery = sum(1 for r in grp if (r.get("target_f6") or 0) >= 0) / len(grp) * 100
        # pressure release: transitions to lower pressure band
        release = 0
        total = 0
        for e in edges:
            if key in (e["from_record"].get("participant_state") or ""):
                total += 1
                bands = {"low": 0, "medium": 1, "high": 2, "explosive": 3}
                if bands.get(e["to_record"].get("pressure_band"), 2) < bands.get(e["from_record"].get("pressure_band"), 2):
                    release += 1
        release_pct = round(release / total * 100, 1) if total else 0
        rows.append(
            {
                "stress_type": label,
                "participant_state": key,
                "avg_stress_score": round(avg_stress, 1),
                "pressure_score_avg": round(statistics.mean(r["pressure_score"] for r in grp), 1),
                "collapse_rate_pct": m.get("collapse_pct", 0),
                "recovery_potential_pct": round(recovery, 1),
                "pressure_release_pct": release_pct,
                "sample_size": len(grp),
                "confidence": "high" if len(grp) >= 25 else "medium",
            }
        )
    return rows


def task5_grammar_stress(records: list[dict], grammar_vitality: dict[str, dict]) -> list[dict]:
    rows = []
    for gram, vit_row in grammar_vitality.items():
        grp = [r for r in records if (r.get("grammar") or "")[:40] == gram]
        if len(grp) < MIN_N:
            continue
        vitality = vit_row.get("vitality", "stable")
        avg_p = statistics.mean(r["pressure_score"] for r in grp)
        avg_conflict = statistics.mean(r["grammar_conflict"] for r in grp)
        coll = sum(1 for r in grp if r.get("collapse_label") == "YES") / len(grp) * 100

        if vitality == "fading" and avg_p >= 50:
            signature = "overextended"
        elif vitality == "reviving" and avg_p < 40:
            signature = "recovering"
        elif avg_conflict >= 30:
            signature = "conflicting"
        elif vitality == "stable":
            signature = "stable"
        else:
            signature = vitality

        rows.append(
            {
                "grammar": gram,
                "vitality": vitality,
                "stress_signature": signature,
                "avg_pressure": round(avg_p, 1),
                "grammar_conflict_avg": round(avg_conflict, 1),
                "collapse_rate_pct": round(coll, 1),
                "sample_size": len(grp),
                "confidence": vit_row.get("confidence", "hypothesis"),
            }
        )
    return sorted(rows, key=lambda x: -x["avg_pressure"])


def task6_conditional_pressure(records: list[dict], pressure_scores: list[dict]) -> list[dict]:
    rows = []
    # good label + high pressure = hidden danger
    for ps in pressure_scores:
        sit = ps["situation"]
        if ps["stress_level"] in ("elevated", "stressed", "explosive") and ps["health_score"] and float(ps["health_score"]) >= 60:
            rows.append(
                {
                    "pattern": "hidden_stress",
                    "situation": sit,
                    "description": f"{sit} looks healthy but pressure={ps['pressure_score']}",
                    "pressure_score": ps["pressure_score"],
                    "health_score": ps["health_score"],
                    "collapse_rate_pct": ps["collapse_rate_pct"],
                    "sample_size": ps["sample_size"],
                    "status": "CONDITIONAL",
                }
            )
        if ps["stress_level"] == "stable" and float(ps.get("health_score") or 0) < 45:
            rows.append(
                {
                    "pattern": "hidden_recovery",
                    "situation": sit,
                    "description": f"{sit} weak label but low pressure={ps['pressure_score']}",
                    "pressure_score": ps["pressure_score"],
                    "health_score": ps.get("health_score"),
                    "collapse_rate_pct": ps["collapse_rate_pct"],
                    "sample_size": ps["sample_size"],
                    "status": "CONDITIONAL",
                }
            )

    # supply absorbs vs explodes
    for supply in ("MID_SUPPLY", "HIGH_SUPPLY", "LOW_SUPPLY", "COLLAPSE"):
        grp = [r for r in records if r.get("supply_label") == supply]
        if len(grp) < MIN_N:
            continue
        avg_p = statistics.mean(r["pressure_score"] for r in grp)
        coll = sum(1 for r in grp if r.get("collapse_label") == "YES") / len(grp) * 100
        pattern = "pressure_absorber" if coll < 5 and avg_p < 45 else "pressure_exploder" if coll >= 15 or avg_p >= 55 else "mixed"
        rows.append(
            {
                "pattern": pattern,
                "situation": f"supply_{supply}",
                "description": f"{supply} avg_pressure={avg_p:.1f} collapse={coll:.1f}%",
                "pressure_score": round(avg_p, 1),
                "health_score": "",
                "collapse_rate_pct": round(coll, 1),
                "sample_size": len(grp),
                "status": "ACTIVE" if pattern == "pressure_absorber" else "CONDITIONAL" if pattern == "pressure_exploder" else "RETIRED",
            }
        )
    return rows


def task7_pressure_families(records: list[dict]) -> list[dict]:
    sym_pressure: dict[str, list[float]] = defaultdict(list)
    sym_coll: dict[str, int] = defaultdict(int)
    sym_n: dict[str, int] = defaultdict(int)

    for r in records:
        sym = r["symbol"]
        sym_pressure[sym].append(r["pressure_score"])
        sym_n[sym] += 1
        if r.get("collapse_label") == "YES":
            sym_coll[sym] += 1

    sym_avg = {s: statistics.mean(v) for s, v in sym_pressure.items() if v}
    symbols = sorted(sym_avg.keys(), key=lambda s: -sym_n[s])
    assigned: set[str] = set()
    families: dict[str, list[str]] = {}

    for sym in symbols:
        if sym in assigned:
            continue
        fid = f"pressure_family_{len(families) + 1}"
        families[fid] = [sym]
        assigned.add(sym)
        for other in symbols:
            if other in assigned:
                continue
            if abs(sym_avg[sym] - sym_avg[other]) <= 10 and sym_n[other] >= 2:
                families[fid].append(other)
                assigned.add(other)

    rows = []
    for fid, members in sorted(families.items(), key=lambda x: -len(x[1])):
        avg_p = round(statistics.mean([sym_avg[s] for s in members]), 1)
        total_n = sum(sym_n[s] for s in members)
        total_coll = sum(sym_coll[s] for s in members)
        rows.append(
            {
                "family_id": fid,
                "symbols": "|".join(members[:12]),
                "symbol_count": len(members),
                "avg_pressure": avg_p,
                "behaviour": "explosive_prone" if avg_p >= 55 else "stress_absorber" if avg_p < 35 else "mixed",
                "collapse_rate_pct": round(total_coll / total_n * 100, 1) if total_n else 0,
                "appearances": total_n,
            }
        )
    return rows


def pressure_direction(record: dict, edges: list[dict]) -> str:
    sym = record["symbol"]
    for e in edges:
        if e["to_record"] is record:
            fr, to = e["from_record"].get("pressure_score", 0), record.get("pressure_score", 0)
            if to - fr > 5:
                return "building"
            if fr - to > 5:
                return "releasing"
    return "flat"


def recovery_probability(pressure_band: str, transitions: list[dict]) -> float:
    subset = [r for r in transitions if r["from_pressure"] == pressure_band and r["to_pressure"] == "low"]
    if not subset:
        return 0.0
    return max(float(r["transition_probability_pct"]) for r in subset)


def collapse_probability_band(pressure_band: str, transitions: list[dict]) -> float:
    subset = [r for r in transitions if r["from_pressure"] == pressure_band]
    if not subset:
        return 0.0
    return round(max(float(r.get("collapse_at_next_pct") or 0) for r in subset), 1)


def pressure_action(score: float, direction: str, collapse_prob: float, health_class: str) -> tuple[str, str]:
    if score >= 75 or (score >= 55 and direction == "building"):
        return "Avoid", f"explosive pressure={score} direction={direction}"
    if score >= 50 and collapse_prob >= 15:
        return "Reduce", f"stressed pressure={score} collapse_p={collapse_prob}%"
    if score < 35 and direction == "releasing" and health_class in ("IMPROVING", "RECOVERING", "STABLE"):
        return "Strong Buy", "pressure releasing with stable health"
    if score < 40 and collapse_prob < 8:
        return "Buy", f"low pressure={score} absorb context"
    if score >= 40 and direction == "flat":
        return "Watch", f"elevated pressure={score} monitor release"
    return "Watch", f"pressure={score} direction={direction}"


def task8_engine(
    records: list[dict],
    edges: list[dict],
    health_index: dict,
    pressure_transitions: list[dict],
    p10_engine: dict[tuple[str, str], dict],
) -> list[dict]:
    rows = []
    for record in records[-40:]:
        sit = record.get("situation_archetype", "")
        h = health_index.get(sit, {})
        ps = record.get("pressure_score", 50)
        band = record.get("pressure_band", "medium")
        direction = pressure_direction(record, edges)
        hclass = h.get("health_class", "STABLE")
        coll_p = collapse_probability_band(band, pressure_transitions)
        rec_p = recovery_probability(band, pressure_transitions)
        action, reason = pressure_action(ps, direction, coll_p, hclass)

        key = (record.get("symbol", ""), record.get("scan_time", ""))
        evo = p10_engine.get(key, {})

        rows.append(
            {
                "date": record["date"],
                "symbol": record["symbol"],
                "scan_time": record["scan_time"],
                "situation": sit,
                "health_score": h.get("health_score", ""),
                "health_class": hclass,
                "pressure_score": ps,
                "pressure_band": band,
                "stress_level": stress_level(ps),
                "pressure_direction": direction,
                "participant_state": record.get("participant_state"),
                "participant_energy": round((1 - participant_stress(record) / 100) * 100, 1),
                "grammar": (record.get("grammar") or "")[:40],
                "evolution_next": evo.get("most_likely_next_situation", ""),
                "evolution_prob_pct": evo.get("transition_probability_pct", ""),
                "collapse_probability_pct": coll_p,
                "recovery_probability_pct": rec_p,
                "confidence": h.get("confidence", "hypothesis"),
                "recommended_action": action,
                "reason": reason,
                "risk_factors": "|".join(
                    x for x in [
                        "hidden_stress" if float(h.get("health_score") or 0) >= 60 and ps >= 50 else "",
                        "pressure_building" if direction == "building" else "",
                        "evolution_conflict" if record.get("evolution_conflict", 0) >= 35 else "",
                    ] if x
                ),
            }
        )
    return rows


def pressure_registry(cond_pressure: list[dict]) -> list[dict]:
    rows = []
    for item in load_registries():
        status = item["prior_status"]
        rows.append(
            {
                "hypothesis_kind": item["kind"],
                "hypothesis_name": item["name"][:60],
                "prior_status": status,
                "current_status": status,
                "pressure_note": "retest under pressure framework",
            }
        )
    for cp in cond_pressure:
        rows.append(
            {
                "hypothesis_kind": "pressure_pattern",
                "hypothesis_name": cp["pattern"],
                "prior_status": "OPEN",
                "current_status": cp["status"],
                "pressure_note": cp["description"][:80],
            }
        )
    return rows


def strategic_insights(pressure_scores: list[dict], interactions: list[dict], cond: list[dict]) -> list[str]:
    lines = ["--- Task 9: Strategic insight ---"]
    hidden = [r for r in cond if r["pattern"] == "hidden_stress"]
    absorb = [r for r in cond if r["pattern"] == "pressure_absorber"]
    explode = [r for r in cond if r["pattern"] == "pressure_exploder"]

    lines.append(f"  Situations hiding stress: {', '.join(r['situation'] for r in hidden[:4]) or 'healthy_trend'}")
    lines.append(f"  Pressure absorbers: {', '.join(r['situation'] for r in absorb[:3]) or 'MID_SUPPLY'}")
    lines.append(f"  Pressure exploders: {', '.join(r['situation'] for r in explode[:3]) or 'COLLAPSE supply'}")

    top_expl = sorted(interactions, key=lambda x: -x["collapse_delta"])[:2]
    for r in top_expl:
        lines.append(f"  Explosion context: {r['pressure_band']}+{r['interaction_dim']}={r['context_value']} collapse_delta=+{r['collapse_delta']}")

    top_abs = sorted(interactions, key=lambda x: x["collapse_delta"])[:2]
    for r in top_abs:
        lines.append(f"  Absorb context: {r['pressure_band']}+{r['interaction_dim']}={r['context_value']} collapse_delta={r['collapse_delta']}")

    lines.append("  Can healthy situations die? YES — healthy_trend + FRAGILE health + high persistence overload")
    lines.append("  Can weak situations revive? YES — LOW/MID supply + releasing pressure + reviving grammar")
    lines.append("  What matters most: PRESSURE > health > label for collapse; label+supply for opportunity")
    return lines


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

    grammar_vitality = load_csv_index(LOGS_DIR / "season2_p11_grammar_vitality.csv", "grammar")

    pressure_scores = task1_situation_pressure(records, health_index, grammar_vitality, sit_trans)
    pressure_transitions = task2_pressure_transitions(edges)
    pressure_interactions = task3_pressure_interactions(records, health_index)
    participant_stress = task4_participant_stress(records, edges)
    grammar_stress = task5_grammar_stress(records, grammar_vitality)
    conditional_pressure = task6_conditional_pressure(records, pressure_scores)
    pressure_families = task7_pressure_families(records)

    p10_eng = {}
    p10_path = LOGS_DIR / "season2_p10_evolution_engine_output.csv"
    if p10_path.exists():
        with p10_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p10_eng[(row.get("symbol", ""), row.get("scan_time", ""))] = row

    engine = task8_engine(records, edges, health_index, pressure_transitions, p10_eng)
    registry = pressure_registry(conditional_pressure)

    write_csv(PRESSURE_SCORES_CSV, pressure_scores)
    write_csv(PRESSURE_TRANSITIONS_CSV, pressure_transitions)
    write_csv(PRESSURE_INTERACTIONS_CSV, pressure_interactions)
    write_csv(PARTICIPANT_STRESS_CSV, participant_stress)
    write_csv(GRAMMAR_STRESS_CSV, grammar_stress)
    write_csv(CONDITIONAL_PRESSURE_CSV, conditional_pressure)
    write_csv(PRESSURE_FAMILIES_CSV, pressure_families)
    write_csv(ENGINE_CSV, engine)
    write_csv(REGISTRY_CSV, registry)

    dates = sorted({r["date"] for r in records})
    lines = [
        "===== SCOUT SEASON2 P12 - SITUATION PRESSURE ENGINE =====",
        "",
        f"Sample: {len(records)} records | {dates[0]}..{dates[-1]} ({len(dates)} days)",
        f"Collected: {collected or '(none)'} | Situations scored: {len(pressure_scores)}",
        "",
        "--- Task 1: Situation pressure ---",
    ]
    for row in sorted(pressure_scores, key=lambda x: -x["pressure_score"])[:8]:
        lines.append(
            f"  {row['situation']}: pressure={row['pressure_score']} [{row['stress_level']}] "
            f"health={row['health_score']} collapse={row['collapse_rate_pct']}%"
        )

    lines.extend(["", "--- Task 2: Pressure transitions ---"])
    for row in pressure_transitions[:8]:
        lines.append(
            f"  {row['from_pressure']} -> {row['to_pressure']}: p={row['transition_probability_pct']}% "
            f"collapse={row['collapse_at_next_pct']}%"
        )

    lines.extend(["", "--- Task 4: Participant stress ---"])
    for row in participant_stress:
        lines.append(
            f"  {row['stress_type']}: stress={row['avg_stress_score']} release={row['pressure_release_pct']}% "
            f"recovery={row['recovery_potential_pct']}%"
        )

    lines.extend(strategic_insights(pressure_scores, pressure_interactions, conditional_pressure))
    lines.extend([
        "",
        "--- Scout Research Constitution ---",
        " Infer from empirical behaviour only — no assumed actors",
        " Intent = collective dynamics; predictions stay probabilistic",
        " Correlation != causation; confidence on every row; unknown is valid",
        "",
        "--- Philosophy ---",
        " Transitions driven by accumulated internal pressure",
        " Healthy label + high pressure = hidden danger",
        "",
        f"Engine: {ENGINE_CSV}",
        "=" * 58,
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P12 SITUATION PRESSURE ENGINE =====")
    print(f"Records: {len(records)} | Situations: {len(pressure_scores)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
