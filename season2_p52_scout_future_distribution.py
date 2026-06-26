"""
Scout Learning Season2 - P52 Process Future Distribution Engine

Discovers future probability distributions instead of point predictions.
Read-only on P39-P51. STRICT NO_ACTION. Pure Python.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p37_scout_decision_hierarchy import load_csv, pf, pi
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

FUTURE_DISTRIBUTION_CSV = LOGS_DIR / "season2_p52_future_distribution.csv"
TRANSITION_MATRIX_CSV = LOGS_DIR / "season2_p52_transition_matrix.csv"
ENTROPY_CSV = LOGS_DIR / "season2_p52_entropy.csv"
DECISION_REGIONS_CSV = LOGS_DIR / "season2_p52_decision_regions.csv"
CONFIDENCE_CSV = LOGS_DIR / "season2_p52_confidence.csv"
UNCERTAINTY_REDUCTION_CSV = LOGS_DIR / "season2_p52_uncertainty_reduction.csv"
FUTURE_ARCHETYPES_CSV = LOGS_DIR / "season2_p52_future_archetypes.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p52_process_report.txt"

OUTCOMES = ("Recovery", "Stable", "Expansion", "Oscillation", "Transition", "Collapse")
P39_STATES = ("Observation", "Potential", "Trend Start", "Trend Expansion", "Failure")

STATE_RANK = {
    "Failure": 0,
    "Observation": 1,
    "Potential": 2,
    "Trend Start": 3,
    "Trend Expansion": 4,
}

UNCERTAINTY_VARS = (
    "Memory",
    "Potential",
    "API",
    "Flow",
    "Attractor",
    "OrderParameter",
    "Quality",
    "Energy",
    "Persistence",
    "Resilience",
)

PROCESS_VARS = {
    "Potential": "var_Potential",
    "API": "var_API",
    "Flow": "var_FlowVelocity",
    "Attractor": "var_AttractorBias",
    "OrderParameter": "order_parameter_score",
    "Quality": "var_Quality",
    "Energy": "var_Energy",
    "Persistence": "var_Persistence",
    "Resilience": "var_Resilience",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        import csv
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def softmax(scores: dict[str, float], temp: float = 1.0) -> dict[str, float]:
    keys = list(scores.keys())
    exps = [math.exp(scores[k] / max(temp, 1e-6)) for k in keys]
    total = sum(exps) or 1.0
    return {k: exps[i] / total for i, k in enumerate(keys)}


def entropy(probs: dict[str, float]) -> float:
    h = 0.0
    for p in probs.values():
        if p > 1e-12:
            h -= p * math.log2(p)
    return h


def kl_div(p: dict[str, float], q: dict[str, float]) -> float:
    d = 0.0
    for k in OUTCOMES:
        pv = max(p.get(k, 0.0), 1e-12)
        qv = max(q.get(k, 0.0), 1e-12)
        d += pv * math.log2(pv / qv)
    return d


def state_vec(row: dict) -> list[float]:
    return [pf(row.get(PROCESS_VARS[v])) for v in PROCESS_VARS if v != "Memory"]


def l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def classify_outcome(from_state: str, to_state: str, delta_op: float) -> str:
    if to_state == "Failure" or delta_op < -25:
        return "Collapse"
    if to_state == "Trend Expansion":
        return "Expansion"
    if STATE_RANK.get(to_state, 1) > STATE_RANK.get(from_state, 1):
        return "Recovery"
    if from_state == to_state and abs(delta_op) < 8:
        return "Stable"
    if from_state == to_state and abs(delta_op) >= 8:
        return "Oscillation"
    if from_state in {"Trend Start", "Trend Expansion"} and to_state in {"Trend Start", "Trend Expansion"}:
        return "Oscillation"
    if from_state != to_state:
        return "Transition"
    return "Stable"


def build_transition_counts(transitions: list[dict]) -> tuple[dict[str, Counter], dict[str, Counter]]:
    state_counts: dict[str, Counter] = defaultdict(Counter)
    outcome_counts: dict[str, Counter] = defaultdict(Counter)
    for tr in transitions:
        fs = tr["from_state"]
        ts = tr["to_state"]
        state_counts[fs][ts] += 1
        delta_op = pf(tr.get("delta_op"))
        outcome_counts[fs][classify_outcome(fs, ts, delta_op)] += 1
    return state_counts, outcome_counts


def empirical_outcome_probs(from_state: str, outcome_counts: dict[str, Counter]) -> dict[str, float]:
    counts = outcome_counts.get(from_state, Counter())
    if not counts:
        return {o: 1.0 / len(OUTCOMES) for o in OUTCOMES}
    total = sum(counts.values())
    return {o: counts.get(o, 0) / total for o in OUTCOMES}


def score_future_distribution(
    row: dict,
    pidx: dict,
    memory_arch: str,
    op_hist: list[float],
) -> dict[str, float]:
    """Hypothesis scores -> softmax probabilities. Read-only process signals."""
    state = row["p39_state"]
    op = pf(row["order_parameter_score"])
    api = pf(row["var_API"])
    quality = pf(row["var_Quality"])
    energy = pf(row["var_Energy"])
    potential = pf(row["var_Potential"])
    flow = pf(row["var_FlowVelocity"])
    attractor = pf(row["var_AttractorBias"])
    persistence = pf(row["var_Persistence"])
    resilience = pf(row["var_Resilience"])
    stability = pf(pidx.get("state_stability"))
    collapse_att = pf(pidx.get("collapse_attractor_score"))
    healthy_att = pf(pidx.get("healthy_attractor_score"))
    confidence_p44 = pf(pidx.get("confidence"))

    op_momentum = op - statistics.mean(op_hist) if op_hist else 0.0
    op_vol = statistics.pstdev(op_hist) if len(op_hist) > 1 else 10.0

    scores = {
        "Recovery": 0.0,
        "Stable": 0.0,
        "Expansion": 0.0,
        "Oscillation": 0.0,
        "Transition": 0.0,
        "Collapse": 0.0,
    }

    if state in {"Observation", "Potential"}:
        scores["Recovery"] += 1.5 + api / 40 + quality / 50
        scores["Stable"] += 0.8 + stability / 50
    if state == "Trend Start":
        scores["Expansion"] += 1.2 + flow / 40 + op / 50
        scores["Collapse"] += collapse_att / 35 + max(0, -op_momentum) / 15
        scores["Oscillation"] += op_vol / 20
    if state == "Trend Expansion":
        scores["Expansion"] += 1.0 + potential / 45
        scores["Oscillation"] += 1.0
        scores["Collapse"] += collapse_att / 40
    if state == "Failure":
        scores["Recovery"] += resilience / 40
        scores["Collapse"] += 1.5
        scores["Transition"] += 0.8

    scores["Collapse"] += collapse_att / 30 + max(0, (50 - op)) / 25
    scores["Recovery"] += healthy_att / 35 + max(0, op_momentum) / 20
    scores["Stable"] += persistence / 60 + (1.0 if abs(op_momentum) < 5 else 0)
    scores["Expansion"] += max(0, flow - 50) / 25 + attractor / 50
    scores["Oscillation"] += op_vol / 15 + (1.0 if state == "Trend Start" else 0.3)
    scores["Transition"] += 0.5 + (100 - confidence_p44) / 80

    if memory_arch == "Collapse Memory":
        scores["Collapse"] += 0.8
    elif memory_arch == "Momentum Memory":
        scores["Expansion"] += 0.6
        scores["Recovery"] += 0.4
    elif memory_arch == "Stable Memory":
        scores["Stable"] += 0.7

    if energy < 20:
        scores["Collapse"] += 0.5
    if quality < 15 and state in {"Trend Start", "Potential"}:
        scores["Collapse"] += 0.6

    return softmax(scores, temp=1.2)


def distribution_archetype(probs: dict[str, float]) -> str:
    top = max(probs, key=probs.get)
    collapse = probs["Collapse"]
    expansion = probs["Expansion"]
    stable = probs["Stable"]
    recovery = probs["Recovery"]
    oscillation = probs["Oscillation"]

    if collapse >= 0.35:
        return "Collapse Drift"
    if expansion >= 0.30 and recovery >= 0.15:
        return "Constructive Expansion"
    if stable >= 0.35:
        return "Stable Persistence"
    if recovery >= 0.30 and oscillation >= 0.15:
        return "Recovery Loop"
    if oscillation >= 0.28:
        return "Oscillation"
    if max(probs.values()) < 0.25:
        return "Unknown"
    mapping = {
        "Recovery": "Recovery Loop",
        "Stable": "Stable Persistence",
        "Expansion": "Constructive Expansion",
        "Oscillation": "Oscillation",
        "Transition": "Unknown",
        "Collapse": "Collapse Drift",
    }
    return mapping.get(top, "Unknown")


def confidence_score(probs: dict[str, float], pidx: dict, ent: float) -> float:
    max_p = max(probs.values())
    max_ent = math.log2(len(OUTCOMES))
    entropy_conf = (1.0 - ent / max_ent) * 100
    stability = pf(pidx.get("state_stability"))
    persistence_h = pf(pidx.get("persistence_health"))
    blend = 0.45 * max_p * 100 + 0.35 * entropy_conf + 0.10 * stability + 0.10 * persistence_h
    return round(min(100.0, max(0.0, blend)), 2)


def variable_entropy_adjustment(
    base_scores: dict[str, float],
    var: str,
    row: dict,
    memory_signal: float,
) -> dict[str, float]:
    """Sharpen distribution using one variable's signal (hypothesis ablation inverse)."""
    adjusted = dict(base_scores)
    if var == "Memory":
        if memory_signal > 0.5:
            adjusted["Collapse"] += memory_signal
            adjusted["Stable"] -= 0.2
        else:
            adjusted["Stable"] += 0.3
        return adjusted
    field = PROCESS_VARS.get(var)
    if not field:
        return adjusted
    val = pf(row.get(field))
    center = 50.0
    strength = abs(val - center) / 50.0
    if var in ("API", "Quality", "Potential", "OrderParameter"):
        if val > center:
            adjusted["Recovery"] += strength
            adjusted["Expansion"] += strength * 0.5
            adjusted["Collapse"] -= strength * 0.4
        else:
            adjusted["Collapse"] += strength * 0.5
            adjusted["Transition"] += strength * 0.3
    elif var in ("Flow", "Attractor"):
        if val > center:
            adjusted["Expansion"] += strength
        else:
            adjusted["Oscillation"] += strength * 0.5
    elif var in ("Energy", "Persistence", "Resilience"):
        if val > center:
            adjusted["Stable"] += strength * 0.6
        else:
            adjusted["Collapse"] += strength * 0.4
    return adjusted


def load_checkpoints() -> tuple[str, dict[str, list[dict]], dict, dict]:
    order = load_csv(LOGS_DIR / "season2_p49_order_parameter.csv")
    pidx_rows = load_csv(LOGS_DIR / "season2_p44_process_index.csv")
    memory_arch = load_csv(LOGS_DIR / "season2_p51_memory_archetypes.csv")
    if not order:
        raise SystemExit("P49 order parameter required.")

    obs_id = order[0]["observation_id"]
    pidx_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in pidx_rows}
    arch_by = {
        r["symbol"]: r["memory_archetype"]
        for r in memory_arch
        if r.get("symbol") not in ("(aggregate)", "")
    }

    by_sym: dict[str, list[dict]] = defaultdict(list)
    for row in order:
        by_sym[row["symbol"]].append(dict(row))
    for sym in by_sym:
        by_sym[sym].sort(key=lambda r: pi(r["checkpoint_hour"]))

    return obs_id, dict(by_sym), pidx_by, arch_by


def run() -> None:
    obs_id, by_sym, pidx_by, _arch_by = load_checkpoints()
    load_csv(LOGS_DIR / "season2_p39_state_transition.csv")  # P39 input reference
    p50_ops = {
        (r["symbol"], r["from_checkpoint"]): r["operator_type_inferred"]
        for r in load_csv(LOGS_DIR / "season2_p50_evolution_operators.csv")
        if r.get("from_checkpoint")
    }
    p51_arch = {
        r["symbol"]: r["memory_archetype"]
        for r in load_csv(LOGS_DIR / "season2_p51_memory_archetypes.csv")
        if r.get("symbol") not in ("(aggregate)", "")
    }

    transitions: list[dict] = []
    for sym, rows in by_sym.items():
        for i in range(len(rows) - 1):
            transitions.append({
                "symbol": sym,
                "from_state": rows[i]["p39_state"],
                "to_state": rows[i + 1]["p39_state"],
                "from_checkpoint": rows[i]["checkpoint"],
                "to_checkpoint": rows[i + 1]["checkpoint"],
                "delta_op": pf(rows[i + 1]["order_parameter_score"]) - pf(rows[i]["order_parameter_score"]),
            })

    state_counts, outcome_counts = build_transition_counts(transitions)
    print(f"P52 Future Distribution Engine | {obs_id} | checkpoints={sum(len(v) for v in by_sym.values())}")

    # --- STEP 1 & 3 & 5 & 7: per-checkpoint distributions ---
    dist_rows: list[dict] = []
    entropy_rows: list[dict] = []
    confidence_rows: list[dict] = []
    archetype_rows: list[dict] = []
    checkpoint_records: list[dict] = []

    for sym, rows in by_sym.items():
        memory_arch = p51_arch.get(sym, "Unknown")
        op_hist: list[float] = []
        for row in rows:
            hour = pi(row["checkpoint_hour"])
            pidx = pidx_by.get((sym, hour), {})
            op_hist.append(pf(row["order_parameter_score"]))

        for idx, row in enumerate(rows):
            hour = pi(row["checkpoint_hour"])
            pidx = pidx_by.get((sym, hour), {})
            hist = op_hist[: idx + 1]
            emp = empirical_outcome_probs(row["p39_state"], outcome_counts)
            scored = score_future_distribution(row, pidx, memory_arch, hist)
            if p50_ops.get((sym, row["checkpoint"])) == "Collapse Evolution":
                scored = dict(scored)
                scored["Collapse"] = scored.get("Collapse", 0) + 0.15
                total_s = sum(scored.values()) or 1.0
                scored = {k: v / total_s for k, v in scored.items()}

            probs = {o: 0.5 * emp.get(o, 0) + 0.5 * scored.get(o, 0) for o in OUTCOMES}
            total = sum(probs.values()) or 1.0
            probs = {o: probs[o] / total for o in OUTCOMES}
            ent = entropy(probs)
            conf = confidence_score(probs, pidx, ent)
            arch = distribution_archetype(probs)

            actual_outcome = ""
            if idx < len(rows) - 1:
                actual_outcome = classify_outcome(
                    row["p39_state"],
                    rows[idx + 1]["p39_state"],
                    pf(rows[idx + 1]["order_parameter_score"]) - pf(row["order_parameter_score"]),
                )

            dist_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "p39_state": row["p39_state"],
                "prob_recovery": round(probs["Recovery"], 4),
                "prob_stable": round(probs["Stable"], 4),
                "prob_expansion": round(probs["Expansion"], 4),
                "prob_oscillation": round(probs["Oscillation"], 4),
                "prob_transition": round(probs["Transition"], 4),
                "prob_collapse": round(probs["Collapse"], 4),
                "future_entropy": round(ent, 4),
                "future_archetype": arch,
                "actual_next_outcome": actual_outcome,
                "learning_recommendation": "NO_ACTION",
            })

            entropy_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "p39_state": row["p39_state"],
                "future_entropy": round(ent, 4),
                "entropy_regime": "high" if ent > 2.2 else "medium" if ent > 1.8 else "low",
                "dominant_outcome": max(probs, key=probs.get),
                "dominant_probability": round(max(probs.values()), 4),
                "learning_recommendation": "NO_ACTION",
            })

            confidence_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "p39_state": row["p39_state"],
                "confidence_score": conf,
                "future_entropy": round(ent, 4),
                "dominant_outcome": max(probs, key=probs.get),
                "price_used": "no",
                "learning_recommendation": "NO_ACTION",
            })

            archetype_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "future_archetype": arch,
                "prob_recovery": round(probs["Recovery"], 4),
                "prob_stable": round(probs["Stable"], 4),
                "prob_expansion": round(probs["Expansion"], 4),
                "prob_oscillation": round(probs["Oscillation"], 4),
                "prob_transition": round(probs["Transition"], 4),
                "prob_collapse": round(probs["Collapse"], 4),
                "cluster_method": "distribution_signature",
                "learning_recommendation": "NO_ACTION",
            })

            base_scores = {
                o: math.log(max(probs[o], 1e-9)) for o in OUTCOMES
            }
            checkpoint_records.append({
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "hour": hour,
                "state": row["p39_state"],
                "vec": state_vec(row),
                "probs": probs,
                "entropy": ent,
                "row": row,
                "pidx": pidx,
                "memory_arch": memory_arch,
                "hist": hist,
                "base_scores": base_scores,
            })

    # --- STEP 2: Transition matrix ---
    matrix_rows: list[dict] = []
    for from_state in P39_STATES:
        counts = state_counts.get(from_state, Counter())
        total = sum(counts.values()) or 1
        for to_state in P39_STATES:
            matrix_rows.append({
                "observation_id": obs_id,
                "from_state": from_state,
                "to_state": to_state,
                "transition_count": counts.get(to_state, 0),
                "transition_probability": round(counts.get(to_state, 0) / total, 4),
                "sample_size": total,
                "learning_recommendation": "NO_ACTION",
            })
        oc = outcome_counts.get(from_state, Counter())
        ot = sum(oc.values()) or 1
        for outcome in OUTCOMES:
            matrix_rows.append({
                "observation_id": obs_id,
                "from_state": from_state,
                "to_state": f"(outcome:{outcome})",
                "transition_count": oc.get(outcome, 0),
                "transition_probability": round(oc.get(outcome, 0) / ot, 4),
                "sample_size": ot,
                "matrix_type": "outcome",
                "learning_recommendation": "NO_ACTION",
            })

    # --- STEP 4: Decision regions ---
    decision_rows: list[dict] = []
    for i in range(len(checkpoint_records)):
        for j in range(i + 1, len(checkpoint_records)):
            a, b = checkpoint_records[i], checkpoint_records[j]
            dist = l2(a["vec"], b["vec"])
            if dist > 35:
                continue
            kl = kl_div(a["probs"], b["probs"])
            if kl < 0.15:
                continue
            decision_rows.append({
                "observation_id": obs_id,
                "symbol_a": a["symbol"],
                "checkpoint_a": a["checkpoint"],
                "symbol_b": b["symbol"],
                "checkpoint_b": b["checkpoint"],
                "state_distance": round(dist, 2),
                "probability_kl_divergence": round(kl, 4),
                "entropy_a": round(a["entropy"], 4),
                "entropy_b": round(b["entropy"], 4),
                "decision_region": "yes" if dist < 30 and kl > 0.2 else "partial",
                "learning_recommendation": "NO_ACTION",
            })

    # --- STEP 6: Uncertainty reduction ---
    uncertainty_rows: list[dict] = []
    memory_signal = 1.0 if any(r.get("memory_archetype") == "Collapse Memory" for r in archetype_rows) else 0.0
    baseline_ents = [r["entropy"] for r in checkpoint_records]
    baseline_mean = statistics.mean(baseline_ents) if baseline_ents else 0.0

    var_reductions: list[tuple[str, float]] = []
    for var in UNCERTAINTY_VARS:
        ents: list[float] = []
        for rec in checkpoint_records:
            adj = variable_entropy_adjustment(
                rec["base_scores"], var, rec["row"], memory_signal if var == "Memory" else 0.0
            )
            adj_probs = softmax(adj, temp=1.0)
            ents.append(entropy(adj_probs))
        mean_ent = statistics.mean(ents) if ents else baseline_mean
        reduction = max(0.0, (baseline_mean - mean_ent) / (baseline_mean + 1e-9) * 100)
        var_reductions.append((var, reduction))

    total_red = sum(r for _, r in var_reductions) or 1.0
    var_reductions.sort(key=lambda x: -x[1])
    for rank, (var, reduction) in enumerate(var_reductions, 1):
        uncertainty_rows.append({
            "observation_id": obs_id,
            "variable": var,
            "uncertainty_reduction_pct": round(reduction / total_red * 100, 2),
            "rank": rank,
            "baseline_entropy_mean": round(baseline_mean, 4),
            "adjusted_entropy_mean": round(baseline_mean * (1 - reduction / 100), 4),
            "learning_recommendation": "NO_ACTION",
        })

    # Archetype aggregates
    arch_counts = Counter(r["future_archetype"] for r in archetype_rows)
    for arch, cnt in arch_counts.items():
        archetype_rows.append({
            "observation_id": obs_id,
            "symbol": "(aggregate)",
            "checkpoint": "",
            "checkpoint_hour": "",
            "future_archetype": arch,
            "observation_count": cnt,
            "cluster_method": "aggregate",
            "learning_recommendation": "NO_ACTION",
        })

    # Collapse pre-estimation stats
    collapse_precheck: list[dict] = []
    for rec in checkpoint_records:
        if rec["probs"]["Collapse"] >= 0.20:
            collapse_precheck.append(rec)

    report = build_report(
        obs_id,
        dist_rows,
        entropy_rows,
        confidence_rows,
        decision_rows,
        uncertainty_rows,
        arch_counts,
        collapse_precheck,
        transitions,
    )

    write_csv(FUTURE_DISTRIBUTION_CSV, dist_rows)
    write_csv(TRANSITION_MATRIX_CSV, matrix_rows)
    write_csv(ENTROPY_CSV, entropy_rows)
    write_csv(DECISION_REGIONS_CSV, decision_rows)
    write_csv(CONFIDENCE_CSV, confidence_rows)
    write_csv(UNCERTAINTY_REDUCTION_CSV, uncertainty_rows)
    write_csv(FUTURE_ARCHETYPES_CSV, archetype_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P52 outputs | distributions={len(dist_rows)} matrix={len(matrix_rows)} "
        f"decision_regions={len(decision_rows)} archetypes={len(archetype_rows)}"
    )


def build_report(
    obs_id: str,
    dist_rows: list[dict],
    entropy_rows: list[dict],
    confidence_rows: list[dict],
    decision_rows: list[dict],
    uncertainty_rows: list[dict],
    arch_counts: Counter,
    collapse_precheck: list[dict],
    transitions: list[dict],
) -> str:
    high_ent = max(entropy_rows, key=lambda r: r["future_entropy"])
    low_ent = min(entropy_rows, key=lambda r: r["future_entropy"])
    high_conf = max(confidence_rows, key=lambda r: r["confidence_score"])
    low_conf = min(confidence_rows, key=lambda r: r["confidence_score"])

    multi_future = sum(1 for r in dist_rows if r["prob_collapse"] > 0.15 and max(
        r["prob_recovery"], r["prob_expansion"], r["prob_stable"]
    ) > 0.15)

    collapse_trans = [t for t in transitions if classify_outcome(t["from_state"], t["to_state"], t["delta_op"]) == "Collapse"]
    collapse_estimated = 0
    for t in collapse_trans:
        match = next(
            (r for r in dist_rows if r["symbol"] == t["symbol"] and r["checkpoint"] == t["from_checkpoint"]),
            None,
        )
        if match and pf(match["prob_collapse"]) >= 0.20:
            collapse_estimated += 1

    memory_row = next((r for r in uncertainty_rows if r["variable"] == "Memory"), None)
    memory_reduces = memory_row and pf(memory_row["uncertainty_reduction_pct"]) > 5

    lines = [
        "===== SCOUT SEASON2 P52 - PROCESS FUTURE DISTRIBUTION =====",
        "",
        f"Observation ID: {obs_id}",
        "Future probability physics discovery - STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. Does every Process have one future or multiple futures?",
        f"   Multiple futures (hypothesis). {multi_future}/{len(dist_rows)} checkpoints show competing outcomes (>15% each).",
        "   Process space is probabilistic; single-path determinism rejected for this observation.",
        "",
        "2. Which State has highest uncertainty?",
        f"   Highest entropy: {high_ent['symbol']} {high_ent['checkpoint']} ({high_ent['p39_state']}) H={high_ent['future_entropy']}.",
        f"   Lowest entropy: {low_ent['symbol']} {low_ent['checkpoint']} ({low_ent['p39_state']}) H={low_ent['future_entropy']}.",
        "",
        "3. Which State has highest confidence?",
        f"   Highest: {high_conf['symbol']} {high_conf['checkpoint']} ({high_conf['p39_state']}) score={high_conf['confidence_score']}.",
        f"   Lowest: {low_conf['symbol']} {low_conf['checkpoint']} ({low_conf['p39_state']}) score={low_conf['confidence_score']}.",
        "",
        "4. Can Memory reduce future entropy?",
        (
            f"   {'Yes (hypothesis)' if memory_reduces else 'Weak / inconclusive'}. "
            f"Memory uncertainty reduction rank: {memory_row['rank'] if memory_row else 'n/a'} "
            f"({memory_row['uncertainty_reduction_pct'] if memory_row else 0}%)."
        ),
        "",
        "5. Can Collapse probability be estimated before transition?",
        f"   Partially. {collapse_estimated}/{len(collapse_trans)} collapse transitions had P(Collapse)>=0.20 at prior checkpoint.",
        f"   {len(collapse_precheck)} checkpoints flagged elevated collapse probability.",
        "",
        "6. Is Process evolution better described as probability than deterministic path?",
        "   Yes (hypothesis). Entropy mean="
        f"{round(statistics.mean(r['future_entropy'] for r in entropy_rows), 3)}; "
        f"{len(decision_rows)} decision regions where small state change shifts outcome probabilities.",
        "",
        "Future distribution archetypes:",
    ]
    for arch, cnt in arch_counts.most_common():
        lines.append(f"   {arch}: {cnt}")

    lines.extend([
        "",
        "Top uncertainty reducers:",
    ])
    for row in uncertainty_rows[:5]:
        lines.append(f"   #{row['rank']} {row['variable']}: {row['uncertainty_reduction_pct']}%")

    lines.extend([
        "",
        "Learning recommendation: NO_ACTION - Future Distribution stored as hypothesis only.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P52 Process Future Distribution Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
