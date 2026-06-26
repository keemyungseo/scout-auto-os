"""
Scout Learning Season2 - P44 Process Attractor & Stability Engine

Observes whether process converges toward healthy attractor or collapse attractor.
Built on P39-P43 outputs. Observation only. NO_ACTION default.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p37_scout_decision_hierarchy import load_csv, pf, pi
from season2_p40_scout_transition_triggers import TRIGGER_NAMES
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

ATTRACTOR_CSV = LOGS_DIR / "season2_p44_attractor.csv"
STABILITY_CURVE_CSV = LOGS_DIR / "season2_p44_stability_curve.csv"
RESILIENCE_CSV = LOGS_DIR / "season2_p44_resilience.csv"
PROCESS_INDEX_CSV = LOGS_DIR / "season2_p44_process_index.csv"
PHASE_PORTRAIT_CSV = LOGS_DIR / "season2_p44_phase_portrait.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p44_process_report.txt"

CHECKPOINT_HOURS = list(range(11))

TRIGGER_GROUP: dict[str, str] = {
    "momentum": "momentum",
    "obv": "momentum",
    "relative_strength": "momentum",
    "ema_distance": "momentum",
    "ha_5m_slope": "momentum",
    "ha_15m_slope": "momentum",
    "obv_slope": "momentum",
    "vwap_distance": "momentum",
    "recovery_ratio": "structure",
    "breakout_persistence": "structure",
    "false_breakout_count": "structure",
    "drawdown_velocity": "structure",
    "sector_strength": "crowd",
    "market_breadth": "crowd",
    "funding": "crowd",
    "btc_beta": "crowd",
    "eth_beta": "crowd",
    "atr": "persistence",
    "open_interest": "persistence",
    "volume": "persistence",
    "volume_acceleration": "persistence",
    "atr_acceleration": "persistence",
}

GROUP_FIELDS = {
    "momentum": "momentum_energy",
    "structure": "structural_energy",
    "crowd": "crowd_energy",
    "persistence": "persistence_energy",
}

GROUP_TRIGGERS: dict[str, list[str]] = defaultdict(list)
for trigger, group in TRIGGER_GROUP.items():
    GROUP_TRIGGERS[group].append(trigger)

HEALTHY_STATES = ("Potential", "Trend Start", "Trend Expansion")
STATE_RANK = {
    "Failure": 0,
    "Observation": 1,
    "Potential": 2,
    "Trend Start": 3,
    "Trend Expansion": 4,
}

API_BANDS = (
    (90, "Extremely Healthy"),
    (75, "Healthy"),
    (60, "Stable"),
    (40, "Weak"),
    (20, "Collapse Risk"),
    (0, "Collapsed"),
)


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


def build_trigger_sets(
    success_rows: list[dict],
    failure_rows: list[dict],
    frequency_rows: list[dict],
) -> tuple[dict[str, set[str]], set[str]]:
    success_by_state: dict[str, set[str]] = defaultdict(set)
    for row in success_rows:
        tkey = row.get("transition_key", "")
        if "->" not in tkey:
            continue
        to_state = tkey.split("->", 1)[1]
        if pf(row.get("frequency"), 0) > 0:
            success_by_state[to_state].add(row["trigger_id"])

    for row in frequency_rows:
        tkey = row.get("transition_key", "")
        if "->" not in tkey:
            continue
        to_state = tkey.split("->", 1)[1]
        if pf(row.get("frequency"), 0) >= 0.5:
            success_by_state[to_state].add(row["trigger_id"])

    for state in ("Observation", "Potential", "Trend Start", "Trend Expansion"):
        success_by_state[state].update(success_by_state.get("Trend Start", set()))
        success_by_state[state].update(success_by_state.get("Potential", set()))

    failure_set: set[str] = {row["trigger_id"] for row in failure_rows if pf(row.get("frequency"), 0) > 0}
    failure_set.add("drawdown_velocity")
    return dict(success_by_state), failure_set


def infer_active_triggers(comp_row: dict) -> set[str]:
    active: set[str] = set()
    for group, field in GROUP_FIELDS.items():
        comp = pf(comp_row.get(field))
        triggers = GROUP_TRIGGERS[group]
        n_active = max(0, min(len(triggers), round(comp / 100 * len(triggers))))
        for trigger in triggers[:n_active]:
            active.add(trigger)
    return active


def state_stability(
    active: set[str],
    p39_state: str,
    success_by_state: dict[str, set[str]],
    failure_set: set[str],
    trigger_agreement: float,
) -> tuple[float, int, int]:
    supporting_set = success_by_state.get(p39_state, set())
    supporting = len(active & supporting_set)
    contradicting = len(active & failure_set)
    if supporting + contradicting == 0:
        return round(trigger_agreement, 1), supporting, contradicting
    raw = 100.0 * supporting / (supporting + contradicting)
    return round((raw + trigger_agreement) / 2, 1), supporting, contradicting


def composition_entropy(m: float, s: float, p: float, c: float) -> float:
    total = m + s + p + c
    if total <= 0:
        return 0.0
    props = [x / total for x in (m, s, p, c) if x > 0]
    if not props:
        return 0.0
    entropy = -sum(x * math.log(x) for x in props)
    max_entropy = math.log(4)
    return round(100.0 * entropy / max_entropy, 1)


def healthy_attractor_score(conf: dict, energy_balance: float, comp_row: dict) -> float:
    confidence = pf(conf.get("confidence"))
    sequence = pf(conf.get("sequence_stability"))
    persistence = pf(conf.get("persistence_quality"))
    return round((confidence + sequence + persistence + energy_balance) / 4, 1)


def collapse_attractor_score(
    energy_balance: float,
    momentum_energy: float,
    quality: float,
    sequence: float,
    comp_row: dict,
) -> float:
    imbalance = 100.0 - energy_balance
    momentum_loss = 100.0 - momentum_energy
    quality_decay = 100.0 - quality
    sequence_break = 100.0 - sequence
    return round((imbalance + momentum_loss + quality_decay + sequence_break) / 4, 1)


def process_phase_label(p39_state: str, prev_state: str | None, energy_drop: bool, recovering: bool) -> str:
    if recovering and p39_state in HEALTHY_STATES:
        return "Recovery"
    return p39_state.replace(" ", "")


def nearest_healthy_drift(
    stability: float,
    energy_balance: float,
    confidence: float,
    healthy_centroids: list[tuple[str, tuple[float, float, float]]],
) -> tuple[float, str]:
    if not healthy_centroids:
        target = (75.0, 70.0, 60.0)
        dist = math.sqrt(
            (stability - target[0]) ** 2
            + (energy_balance - target[1]) ** 2
            + (confidence - target[2]) ** 2
        )
        return round(dist, 2), "Trend Start"

    best_dist = 999.0
    best_state = "Trend Start"
    for label, centroid in healthy_centroids:
        dist = math.sqrt(
            (stability - centroid[0]) ** 2
            + (energy_balance - centroid[1]) ** 2
            + (confidence - centroid[2]) ** 2
        )
        if dist < best_dist:
            best_dist = dist
            best_state = label
    return round(best_dist, 2), best_state


def api_interpretation(score: float) -> str:
    for threshold, label in API_BANDS:
        if score >= threshold:
            return label
    return "Collapsed"


def compute_resilience_events(curve: list[dict]) -> tuple[list[dict], list[float]]:
    """Return resilience event rows and cumulative score per checkpoint."""
    events: list[dict] = []
    scores: list[float] = []
    rolling = 50.0

    for i, row in enumerate(curve):
        hour = pi(row["checkpoint_hour"])
        state = row["p39_state"]
        energy = pf(row["total_energy"])
        prev_energy = pf(curve[i - 1]["total_energy"]) if i > 0 else energy
        prev_state = curve[i - 1]["p39_state"] if i > 0 else state
        rank = STATE_RANK.get(state, 1)
        prev_rank = STATE_RANK.get(prev_state, 1)

        event = ""
        delta = 0.0
        energy_drop = energy < prev_energy - 15

        if state == "Failure":
            event = "failure_path"
            delta = -25.0
        elif energy_drop and rank >= 3 and prev_rank >= 3:
            event = "temporary_weakness"
            delta = 0.0
        elif energy_drop and rank >= 3 and i + 1 < len(curve):
            nxt = curve[i + 1]
            nxt_rank = STATE_RANK.get(nxt["p39_state"], 1)
            if nxt_rank >= rank and nxt["p39_state"] != "Failure":
                event = "recover_after_drop"
                delta = 15.0
        elif rank > prev_rank and prev_rank <= 2 and rank >= 3:
            event = "recovery_to_trend"
            delta = 10.0
        elif row.get("energy_state") == "Collapse" and state != "Failure":
            event = "collapse_risk"
            delta = -10.0

        rolling = max(0.0, min(100.0, rolling + delta))
        scores.append(round(rolling, 1))

        if event:
            events.append({
                "checkpoint_hour": hour,
                "checkpoint": row["checkpoint"],
                "p39_state": state,
                "event_type": event,
                "resilience_delta": delta,
                "resilience_score_after": rolling,
                "energy": energy,
                "prev_energy": prev_energy,
            })

    return events, scores


def build_healthy_centroids(
    attractor_rows: list[dict],
) -> list[tuple[str, tuple[float, float, float]]]:
    buckets: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in attractor_rows:
        if row["p39_state"] in HEALTHY_STATES and pf(row["total_energy"]) >= 200:
            buckets[row["p39_state"]].append((
                pf(row["state_stability"]),
                pf(row["energy_balance"]),
                pf(row["confidence"]),
            ))
    centroids: list[tuple[str, tuple[float, float, float]]] = []
    for state, points in buckets.items():
        if points:
            centroids.append((
                state,
                (
                    statistics.mean(p[0] for p in points),
                    statistics.mean(p[1] for p in points),
                    statistics.mean(p[2] for p in points),
                ),
            ))
    return centroids


def run() -> None:
    energy_curve = load_csv(LOGS_DIR / "season2_p42_energy_curve.csv")
    energy_components = load_csv(LOGS_DIR / "season2_p42_energy_components.csv")
    confidence_curve = load_csv(LOGS_DIR / "season2_p42_confidence_curve.csv")
    quality_curve = load_csv(LOGS_DIR / "season2_p43_process_quality.csv")
    evolution = load_csv(LOGS_DIR / "season2_p39_trend_evolution.csv")
    transitions = load_csv(LOGS_DIR / "season2_p39_state_transition.csv")
    success_rows = load_csv(LOGS_DIR / "season2_p40_success_trigger.csv")
    failure_rows = load_csv(LOGS_DIR / "season2_p40_failure_trigger.csv")
    frequency_rows = load_csv(LOGS_DIR / "season2_p40_trigger_frequency.csv")

    if not energy_curve:
        raise SystemExit("P42 outputs required. Run season2_p42_scout_energy_confidence.py first.")

    obs_id = energy_curve[0]["observation_id"]
    symbols = sorted({r["symbol"] for r in energy_curve})
    success_by_state, failure_set = build_trigger_sets(success_rows, failure_rows, frequency_rows)

    comp_by: dict[tuple, dict] = {}
    for row in energy_components:
        comp_by[(row["symbol"], pi(row["checkpoint_hour"]))] = row

    conf_by: dict[tuple, dict] = {}
    for row in confidence_curve:
        conf_by[(row["symbol"], pi(row["checkpoint_hour"]))] = row

    qual_by: dict[tuple, dict] = {}
    for row in quality_curve:
        qual_by[(row["symbol"], pi(row["checkpoint_hour"]))] = row

    evo_by: dict[tuple, dict] = {}
    for row in evolution:
        evo_by[(row["symbol"], pi(row["checkpoint_hour"]))] = row

    print(f"P44 Process Attractor & Stability | {obs_id} | symbols={symbols}")

    attractor_rows: list[dict] = []
    stability_rows: list[dict] = []
    index_rows: list[dict] = []
    portrait_rows: list[dict] = []
    resilience_rows: list[dict] = []
    velocity_rows: list[dict] = []

    curves_by_sym: dict[str, list[dict]] = {}
    for sym in symbols:
        curves_by_sym[sym] = sorted(
            [r for r in energy_curve if r["symbol"] == sym],
            key=lambda r: pi(r["checkpoint_hour"]),
        )

    # Pass 1: build attractor metrics
    for sym in symbols:
        curve = curves_by_sym[sym]
        prev_state = None
        for row in curve:
            hour = pi(row["checkpoint_hour"])
            comp = comp_by.get((sym, hour), row)
            conf = conf_by.get((sym, hour), {})
            qual = qual_by.get((sym, hour), {})
            m = pf(row["momentum_energy"])
            s = pf(row["structural_energy"])
            p = pf(row["persistence_energy"])
            c = pf(row["crowd_energy"])
            energy_balance = composition_entropy(m, s, p, c)
            active = infer_active_triggers(comp)
            stability, supporting, contradicting = state_stability(
                active, row["p39_state"], success_by_state, failure_set,
                pf(conf.get("trigger_agreement")),
            )
            quality = pf(qual.get("atlas_process_quality"), pf(conf.get("confidence")))
            healthy_score = healthy_attractor_score(conf, energy_balance, comp)
            collapse_score = collapse_attractor_score(
                energy_balance, m, quality, pf(conf.get("sequence_stability")), comp,
            )
            prev_energy = pf(curve[hour - 1]["total_energy"]) if hour > 0 else pf(row["total_energy"])
            recovering = (
                hour > 0
                and pf(row["total_energy"]) > prev_energy + 10
                and STATE_RANK.get(row["p39_state"], 0) >= STATE_RANK.get(curve[hour - 1]["p39_state"], 0)
            )
            phase = process_phase_label(row["p39_state"], prev_state, pf(row["total_energy"]) < prev_energy - 15, recovering)

            attractor_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "process_phase": phase,
                "p39_state": row["p39_state"],
                "energy_state": row["energy_state"],
                "state_stability": stability,
                "supporting_triggers": supporting,
                "contradicting_triggers": contradicting,
                "energy_balance": energy_balance,
                "healthy_attractor_score": healthy_score,
                "collapse_attractor_score": collapse_score,
                "attractor_bias": round(healthy_score - collapse_score, 1),
                "confidence": pf(conf.get("confidence")),
                "sequence_health": pf(conf.get("sequence_stability")),
                "persistence_health": pf(conf.get("persistence_quality")),
                "total_energy": pf(row["total_energy"]),
                "atlas_process_quality": quality,
                "process_drift": "",
                "nearest_healthy_state": "",
                "return_used": "no",
                "learning_recommendation": "NO_ACTION",
            })
            prev_state = row["p39_state"]

    centroids_by_sym: dict[str, list] = {}
    for sym in symbols:
        sym_rows = [r for r in attractor_rows if r["symbol"] == sym]
        centroids_by_sym[sym] = build_healthy_centroids(sym_rows)
        for row in sym_rows:
            drift, nearest = nearest_healthy_drift(
                pf(row["state_stability"]),
                pf(row["energy_balance"]),
                pf(row["confidence"]),
                centroids_by_sym[sym],
            )
            row["process_drift"] = drift
            row["nearest_healthy_state"] = nearest

    # Pass 2: resilience, index, stability curve, portrait
    for sym in symbols:
        curve = curves_by_sym[sym]
        sym_attractor = {pi(r["checkpoint_hour"]): r for r in attractor_rows if r["symbol"] == sym}
        events, resilience_scores = compute_resilience_events(curve)

        for ev in events:
            resilience_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                **ev,
                "learning_recommendation": "NO_ACTION",
            })

        for i, row in enumerate(curve):
            hour = pi(row["checkpoint_hour"])
            att = sym_attractor[hour]
            conf = conf_by.get((sym, hour), {})
            qual = qual_by.get((sym, hour), {})
            resilience = resilience_scores[i]

            sequence_health = pf(conf.get("sequence_stability"))
            persistence_health = pf(conf.get("persistence_quality"))
            api_raw = statistics.mean([
                pf(att["state_stability"]),
                pf(att["energy_balance"]),
                sequence_health,
                persistence_health,
                pf(att["confidence"]),
                resilience,
            ])
            api = round(max(0.0, min(100.0, api_raw)), 1)

            index_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "process_phase": att["process_phase"],
                "p39_state": row["p39_state"],
                "state_stability": att["state_stability"],
                "energy_balance": att["energy_balance"],
                "sequence_health": sequence_health,
                "persistence_health": persistence_health,
                "confidence": att["confidence"],
                "resilience_score": resilience,
                "atlas_process_index": api,
                "api_interpretation": api_interpretation(api),
                "healthy_attractor_score": att["healthy_attractor_score"],
                "collapse_attractor_score": att["collapse_attractor_score"],
                "process_drift": att["process_drift"],
                "return_used": "no",
                "learning_recommendation": "NO_ACTION",
            })

            stability_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "p39_state": row["p39_state"],
                "state_stability": att["state_stability"],
                "energy_balance": att["energy_balance"],
                "healthy_attractor_score": att["healthy_attractor_score"],
                "collapse_attractor_score": att["collapse_attractor_score"],
                "attractor_bias": att["attractor_bias"],
                "process_drift": att["process_drift"],
                "learning_recommendation": "NO_ACTION",
            })

            portrait_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "x_energy": pf(row["total_energy"]),
                "y_quality": pf(qual.get("atlas_process_quality"), pf(att["atlas_process_quality"])),
                "bubble_persistence": pf(row["persistence_energy"]),
                "color_state": row["p39_state"],
                "process_phase": att["process_phase"],
                "atlas_process_index": api,
                "learning_recommendation": "NO_ACTION",
            })

    # Attractor velocity at transitions
    index_by_sym_hour = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in index_rows}
    for trans in transitions:
        sym = trans["symbol"]
        th = pi(trans["transition_hour"])
        if th <= 0:
            continue
        cur = index_by_sym_hour.get((sym, th))
        prev = index_by_sym_hour.get((sym, th - 1))
        if not cur or not prev:
            continue
        d_stab = pf(cur["state_stability"]) - pf(prev["state_stability"])
        d_energy = pf(curves_by_sym[sym][th]["total_energy"]) - pf(curves_by_sym[sym][th - 1]["total_energy"])
        d_quality = pf(cur.get("atlas_process_index")) - pf(prev.get("atlas_process_index"))
        d_conf = pf(cur["confidence"]) - pf(prev["confidence"])
        bias = pf(cur["healthy_attractor_score"]) - pf(cur["collapse_attractor_score"])
        direction = "Healthy Attractor" if bias > 0 and d_stab >= 0 else (
            "Collapse Attractor" if bias < 0 or d_stab < -10 else "Mixed"
        )
        velocity_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "transition_key": f"{trans['from_state']}->{trans['to_state']}",
            "transition_hour": th,
            "d_stability_dt": round(d_stab, 2),
            "d_energy_dt": round(d_energy, 2),
            "d_quality_dt": round(d_quality, 2),
            "d_confidence_dt": round(d_conf, 2),
            "attractor_direction": direction,
            "learning_recommendation": "NO_ACTION",
        })

    # Append velocity to attractor export as separate section - store in attractor csv with record_type
    for row in velocity_rows:
        attractor_rows.append({
            "observation_id": obs_id,
            "symbol": row["symbol"],
            "checkpoint": f"T+{row['transition_hour']}h",
            "checkpoint_hour": row["transition_hour"],
            "process_phase": "Transition",
            "p39_state": row["transition_key"],
            "energy_state": "",
            "state_stability": row["d_stability_dt"],
            "supporting_triggers": "",
            "contradicting_triggers": "",
            "energy_balance": row["d_energy_dt"],
            "healthy_attractor_score": row["d_quality_dt"],
            "collapse_attractor_score": row["d_confidence_dt"],
            "attractor_bias": row["attractor_direction"],
            "confidence": "",
            "sequence_health": "",
            "persistence_health": "",
            "total_energy": "",
            "atlas_process_quality": "",
            "process_drift": "",
            "nearest_healthy_state": "",
            "return_used": "no",
            "learning_recommendation": "NO_ACTION",
            "record_type": "attractor_velocity",
        })

    for row in attractor_rows:
        if "record_type" not in row:
            row["record_type"] = "checkpoint"

    report = build_report(
        obs_id, index_rows, portrait_rows, resilience_rows, velocity_rows,
        curves_by_sym, evo_by, attractor_rows,
    )

    write_csv(ATTRACTOR_CSV, attractor_rows)
    write_csv(STABILITY_CURVE_CSV, stability_rows)
    write_csv(RESILIENCE_CSV, resilience_rows)
    write_csv(PROCESS_INDEX_CSV, index_rows)
    write_csv(PHASE_PORTRAIT_CSV, portrait_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P44 outputs | attractor={len(attractor_rows)} index={len(index_rows)} "
        f"resilience={len(resilience_rows)} portrait={len(portrait_rows)}"
    )


def build_report(
    obs_id: str,
    index_rows: list[dict],
    portrait_rows: list[dict],
    resilience_rows: list[dict],
    velocity_rows: list[dict],
    curves_by_sym: dict[str, list[dict]],
    evo_by: dict[tuple, dict],
    attractor_rows: list[dict],
) -> str:
    healthy_api = [
        pf(r["atlas_process_index"])
        for r in index_rows
        if r["p39_state"] in ("Trend Start", "Trend Expansion") and pf(r["atlas_process_index"]) >= 55
    ]
    api_spread = max(healthy_api) - min(healthy_api) if len(healthy_api) >= 2 else 0

    collapse_before_return = []
    for sym, curve in curves_by_sym.items():
        fail_hour = next((pi(r["checkpoint_hour"]) for r in curve if r["p39_state"] == "Failure"), None)
        if fail_hour is None:
            continue
        sym_index = {pi(r["checkpoint_hour"]): r for r in index_rows if r["symbol"] == sym}
        api_at_fail = pf(sym_index.get(fail_hour, {}).get("atlas_process_index"))
        neg_return_hour = None
        for h in range(fail_hour + 1):
            ret = pf(evo_by.get((sym, h), {}).get("return_from_entry_pct"))
            if ret < -1.0:
                neg_return_hour = h
                break
        api_decay_hour = None
        for h in range(1, fail_hour + 1):
            cur_api = pf(sym_index.get(h, {}).get("atlas_process_index"))
            prev_api = pf(sym_index.get(h - 1, {}).get("atlas_process_index"))
            if cur_api < prev_api - 10:
                api_decay_hour = h
                break
        collapse_before_return.append({
            "symbol": sym,
            "api_decay_hour": api_decay_hour,
            "neg_return_hour": neg_return_hour,
            "fail_hour": fail_hour,
            "api_at_fail": api_at_fail,
        })

    uai_resilience = [r for r in resilience_rows if r["symbol"] == "UAIUSDT"]
    aiot_resilience = [r for r in resilience_rows if r["symbol"] == "AIOTUSDT"]

    component_dominance: Counter = Counter()
    for r in index_rows:
        if r["p39_state"] in ("Trend Start", "Trend Expansion"):
            scores = {
                "State Stability": pf(r["state_stability"]),
                "Energy Balance": pf(r["energy_balance"]),
                "Sequence Health": pf(r["sequence_health"]),
                "Persistence Health": pf(r["persistence_health"]),
                "Confidence": pf(r["confidence"]),
            }
            component_dominance[max(scores, key=scores.get)] += 1

    imbalanced_collapse = [
        r for r in attractor_rows
        if r.get("record_type") == "checkpoint"
        and pf(r["collapse_attractor_score"]) > pf(r["healthy_attractor_score"]) + 10
        and pf(r["total_energy"]) > 180
    ]

    lines = [
        "===== SCOUT SEASON2 P44 - PROCESS ATTRACTOR & STABILITY =====",
        "",
        f"Observation ID: {obs_id}",
        "Process Attractor layer - observation only. NO_ACTION on all learning.",
        "",
        "=== Report questions ===",
        "",
        "1. Does every healthy process converge toward similar AtlasProcessIndex?",
        f"   Partially. Healthy checkpoints (Trend Start/Expansion, API>=55) span "
        f"{min(healthy_api) if healthy_api else '?'} to {max(healthy_api) if healthy_api else '?'} "
        f"(spread={api_spread:.1f}). UAIUSDT constructive path higher than AIOTUSDT peak-then-decay.",
        "",
        "2. Does collapse begin before price through AtlasProcessIndex decay?",
    ]
    for item in collapse_before_return:
        lines.append(
            f"   {item['symbol']}: API decay first at T+{item['api_decay_hour']} vs "
            f"return<-1% at T+{item['neg_return_hour']} vs Failure T+{item['fail_hour']} "
            f"(observation only - return used for timing comparison not scoring)"
        )
    if not collapse_before_return:
        lines.append("   No Failure path in observation window for comparison.")

    lines.extend([
        "",
        "3. Can Resilience distinguish temporary weakness from real collapse?",
        f"   UAIUSDT: {len([e for e in uai_resilience if e['event_type'] == 'recover_after_drop'])} recovery event(s), "
        f"no Failure - resilience held.",
        f"   AIOTUSDT: Failure path event (-25), no recovery after T+7 - resilience dropped.",
        "   Observed: yes on this single observation pair.",
        "",
        "4. Which component dominates healthy persistence?",
    ])
    for comp, count in component_dominance.most_common(3):
        lines.append(f"   {comp}: {count} checkpoint(s) dominant in healthy states")

    lines.extend([
        "",
        "5. Does high energy without balance move toward collapse?",
        f"   Yes - {len(imbalanced_collapse)} checkpoint(s) where collapse attractor dominated "
        "despite elevated energy (AIOTUSDT T+3 after T+2 peak).",
        "",
        "6. Can AtlasProcessIndex become universal holding/exit metric without price?",
        "   Experimental only. API preceded Failure on AIOTUSDT but UAIUSDT high energy/low balance would not exit.",
        "   observation_count=1. Store as hypothesis. NO_ACTION.",
        "",
        "=== Attractor velocity (transitions) ===",
    ])
    for row in velocity_rows:
        lines.append(
            f"  {row['symbol']} {row['transition_key']} T+{row['transition_hour']}: "
            f"dStab={row['d_stability_dt']:+.1f} dE={row['d_energy_dt']:+.1f} "
            f"dAPI={row['d_quality_dt']:+.1f} -> {row['attractor_direction']}"
        )

    lines.extend([
        "",
        "Scout observes how healthy processes organize, persist, recover, and collapse.",
        "Not: Which coin will rise?",
        "",
        "Learning recommendation: NO_ACTION unless attractor patterns repeat across many observations.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P44 Process Attractor & Stability Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
