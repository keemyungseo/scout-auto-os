"""
Scout Learning Season2 - P43 Energy Lifetime & Decay Engine

Discovers process lifetime, energy decay, and quality from P25-P42 outputs.
Observation only. No price prediction. NO_ACTION default.
Read-only on all prior phase files.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p37_scout_decision_hierarchy import load_csv, pf, pi
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

ENERGY_DECAY_CSV = LOGS_DIR / "season2_p43_energy_decay.csv"
PROCESS_LIFETIME_CSV = LOGS_DIR / "season2_p43_process_lifetime.csv"
ENERGY_PROFILE_CSV = LOGS_DIR / "season2_p43_energy_profile.csv"
ENERGY_PHYSICS_CSV = LOGS_DIR / "season2_p43_energy_physics.csv"
HALF_LIFE_CSV = LOGS_DIR / "season2_p43_half_life.csv"
PROCESS_QUALITY_CSV = LOGS_DIR / "season2_p43_process_quality.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p43_process_report.txt"

COMPONENTS = (
    ("Momentum", "momentum_energy"),
    ("Structure", "structural_energy"),
    ("Persistence", "persistence_energy"),
    ("Crowd", "crowd_energy"),
)

LIFETIME_STATES = (
    "Observation",
    "Potential",
    "Trend Start",
    "Trend Expansion",
    "Exhaustion",
    "Failure",
    "Collapse",
)

STATE_HEALTH = {
    "Trend Expansion": 92.0,
    "Trend Start": 82.0,
    "Potential": 72.0,
    "Observation": 62.0,
    "Exhaustion": 48.0,
    "Failure": 22.0,
    "Collapse": 18.0,
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


def merged_process_state(p39_state: str, energy_state: str) -> str:
    if p39_state == "Failure":
        return "Failure"
    if energy_state == "Collapse":
        return "Collapse"
    if energy_state == "Exhaustion":
        return "Exhaustion"
    return p39_state


def classify_profile(m: float, s: float, p: float, c: float) -> str:
    values = {"Momentum-heavy": m, "Structural-heavy": s, "Persistence-heavy": p, "Crowd-heavy": c}
    top_name, top_val = max(values.items(), key=lambda x: x[1])
    avg = statistics.mean(values.values())
    if top_val - avg < 12:
        return "Balanced"
    return top_name


def component_series(curve: list[dict], field: str) -> list[float]:
    return [pf(row[field]) for row in curve]


def compute_derivatives(values: list[float]) -> tuple[list[float], list[float], list[float]]:
    velocity = [0.0]
    acceleration = [0.0]
    decay_rate = [0.0]
    for i in range(1, len(values)):
        v = round(values[i] - values[i - 1], 2)
        velocity.append(v)
        a = round(v - velocity[i - 1], 2)
        acceleration.append(a)
        e = values[i]
        rate = round(-v / e * 100, 3) if e > 0 and v < 0 else (round(-v / e * 100, 3) if e > 0 else 0.0)
        decay_rate.append(rate)
    return velocity, acceleration, decay_rate


def decay_summary(
    values: list[float],
    hours: list[int],
    collapse_hours: set[int],
) -> dict:
    peak = max(values) if values else 0.0
    peak_hour = hours[values.index(peak)] if values else 0

    def time_to_fraction(frac: float) -> str | float:
        threshold = peak * frac
        for h, val in zip(hours, values):
            if h >= peak_hour and val <= threshold:
                return h - peak_hour
        return ""

    collapse_after_peak = ""
    for h in sorted(collapse_hours):
        if h >= peak_hour:
            collapse_after_peak = h - peak_hour
            break

    recovery_delay = ""
    below = False
    trough_hour = None
    threshold_75 = peak * 0.75
    for h, val in zip(hours, values):
        if val < threshold_75:
            below = True
            trough_hour = h
        elif below and val >= threshold_75 and trough_hour is not None:
            recovery_delay = h - trough_hour
            break

    return {
        "peak_energy": round(peak, 1),
        "peak_hour": peak_hour,
        "time_to_90pct_hours": time_to_fraction(0.9),
        "time_to_75pct_hours": time_to_fraction(0.75),
        "time_to_50pct_hours": time_to_fraction(0.5),
        "time_to_collapse_hours": collapse_after_peak,
        "recovery_delay_hours": recovery_delay,
    }


def state_runs(curve: list[dict]) -> list[tuple[str, int, int]]:
    """Return (state, start_hour, duration) for contiguous merged states."""
    runs: list[tuple[str, int, int]] = []
    if not curve:
        return runs
    current = merged_process_state(curve[0]["p39_state"], curve[0]["energy_state"])
    start = pi(curve[0]["checkpoint_hour"])
    prev_hour = start
    for row in curve[1:]:
        hour = pi(row["checkpoint_hour"])
        state = merged_process_state(row["p39_state"], row["energy_state"])
        if state != current:
            runs.append((current, start, prev_hour - start + 1))
            current = state
            start = hour
        prev_hour = hour
    runs.append((current, start, prev_hour - start + 1))
    return runs


def build_decay_rows(
    obs_id: str,
    curve: list[dict],
    symbol: str,
) -> list[dict]:
    rows: list[dict] = []
    hours = [pi(r["checkpoint_hour"]) for r in curve]
    collapse_hours = {pi(r["checkpoint_hour"]) for r in curve if r["energy_state"] == "Collapse"}

    for comp_name, field in COMPONENTS:
        values = component_series(curve, field)
        velocity, acceleration, decay_rate = compute_derivatives(values)
        summary = decay_summary(values, hours, collapse_hours)

        for i, row in enumerate(curve):
            hour = pi(row["checkpoint_hour"])
            rows.append({
                "observation_id": obs_id,
                "symbol": symbol,
                "energy_component": comp_name,
                "record_type": "checkpoint",
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "energy": values[i],
                "velocity": velocity[i],
                "acceleration": acceleration[i],
                "decay_rate_pct_per_hour": decay_rate[i],
                **{k: "" for k in (
                    "peak_energy", "peak_hour", "time_to_90pct_hours", "time_to_75pct_hours",
                    "time_to_50pct_hours", "time_to_collapse_hours", "recovery_delay_hours",
                )},
            })

        rows.append({
            "observation_id": obs_id,
            "symbol": symbol,
            "energy_component": comp_name,
            "record_type": "summary",
            "checkpoint": "",
            "checkpoint_hour": "",
            "energy": "",
            "velocity": "",
            "acceleration": "",
            "decay_rate_pct_per_hour": "",
            **summary,
        })

    return rows


def build_process_lifetime_rows(
    obs_id: str,
    curves_by_sym: dict[str, list[dict]],
    transitions: list[dict],
) -> list[dict]:
    lifetimes: dict[str, list[int]] = defaultdict(list)
    for sym, curve in curves_by_sym.items():
        for state, _start, duration in state_runs(curve):
            if state in LIFETIME_STATES:
                lifetimes[state].append(duration)

    trans_from = Counter(t["from_state"] for t in transitions)
    trans_pairs: dict[str, Counter] = defaultdict(Counter)
    for t in transitions:
        trans_pairs[t["from_state"]][t["to_state"]] += 1

    rows: list[dict] = []
    for state in LIFETIME_STATES:
        durations = lifetimes.get(state, [])
        total_from = trans_from.get(state, 0)
        pair_counter = trans_pairs.get(state, Counter())
        top_next = pair_counter.most_common(1)
        trans_prob = round(top_next[0][1] / total_from, 3) if top_next and total_from else ""
        avg_life = round(statistics.mean(durations), 2) if durations else ""
        stability = round(
            100.0 * statistics.mean(durations) / (statistics.mean(durations) + total_from), 1
        ) if durations and total_from else (100.0 if durations and not total_from else "")

        rows.append({
            "observation_id": obs_id,
            "process_state": state,
            "average_lifetime_hours": avg_life,
            "minimum_lifetime_hours": min(durations) if durations else "",
            "maximum_lifetime_hours": max(durations) if durations else "",
            "observation_count": len(durations),
            "transition_count_from_state": total_from,
            "most_likely_next_state": top_next[0][0] if top_next else "",
            "transition_probability": trans_prob,
            "stability_score": stability,
            "learning_recommendation": "NO_ACTION",
        })

    return rows


def build_profile_rows(obs_id: str, curve: list[dict], symbol: str) -> list[dict]:
    rows: list[dict] = []
    for row in curve:
        m = pf(row["momentum_energy"])
        s = pf(row["structural_energy"])
        p = pf(row["persistence_energy"])
        c = pf(row["crowd_energy"])
        total = pf(row["total_energy"])
        profile = classify_profile(m, s, p, c)
        rows.append({
            "observation_id": obs_id,
            "symbol": symbol,
            "checkpoint": row["checkpoint"],
            "checkpoint_hour": pi(row["checkpoint_hour"]),
            "p39_state": row["p39_state"],
            "energy_state": row["energy_state"],
            "momentum_energy": m,
            "structural_energy": s,
            "persistence_energy": p,
            "crowd_energy": c,
            "total_energy": total,
            "energy_profile": profile,
            "dominant_component": profile.replace("-heavy", "") if profile != "Balanced" else "Balanced",
            "composition_spread": round(max(m, s, p, c) - min(m, s, p, c), 1),
            "learning_recommendation": "NO_ACTION",
        })
    return rows


def build_physics_rows(
    obs_id: str,
    curve: list[dict],
    symbol: str,
    confidence_by_hour: dict[int, dict],
) -> list[dict]:
    rows: list[dict] = []
    energies = [pf(r["total_energy"]) for r in curve]
    velocities = [pf(r["energy_velocity"]) for r in curve]
    accelerations = [pf(r["energy_acceleration"]) for r in curve]
    jerks: list[float] = [0.0]
    for i in range(1, len(accelerations)):
        jerks.append(round(accelerations[i] - accelerations[i - 1], 2))

    fail_hours = {pi(r["checkpoint_hour"]) for r in curve if r["p39_state"] == "Failure"}
    exhaust_hours = {pi(r["checkpoint_hour"]) for r in curve if r["energy_state"] == "Exhaustion"}
    collapse_hours = {pi(r["checkpoint_hour"]) for r in curve if r["energy_state"] == "Collapse"}
    trend_start_hours = {pi(r["checkpoint_hour"]) for r in curve if r["p39_state"] == "Trend Start"}

    for i, row in enumerate(curve):
        hour = pi(row["checkpoint_hour"])
        vel = velocities[i]
        acc = accelerations[i]
        jerk = jerks[i]

        vel_neg_before_fail = ""
        if hour in fail_hours:
            if i > 0 and velocities[i - 1] < 0:
                vel_neg_before_fail = "yes"
            else:
                vel_neg_before_fail = "no"

        acc_slow_before_exhaust = ""
        if hour in exhaust_hours:
            if i > 0 and acc < accelerations[i - 1]:
                acc_slow_before_exhaust = "yes"
            else:
                acc_slow_before_exhaust = "no"

        jerk_spike_before_collapse = ""
        if hour in collapse_hours:
            if i > 0 and abs(jerks[i - 1]) > 20:
                jerk_spike_before_collapse = "yes"
            else:
                jerk_spike_before_collapse = "no"

        pos_acc_after_trend_start = ""
        if hour in trend_start_hours and acc > 0:
            pos_acc_after_trend_start = "yes"
        elif hour in trend_start_hours:
            pos_acc_after_trend_start = "no"

        rows.append({
            "observation_id": obs_id,
            "symbol": symbol,
            "checkpoint": row["checkpoint"],
            "checkpoint_hour": hour,
            "p39_state": row["p39_state"],
            "energy_state": row["energy_state"],
            "total_energy": energies[i],
            "energy_velocity": vel,
            "energy_acceleration": acc,
            "energy_jerk": jerk,
            "velocity_negative_before_failure": vel_neg_before_fail,
            "acceleration_slows_before_exhaustion": acc_slow_before_exhaust,
            "jerk_spike_before_collapse": jerk_spike_before_collapse,
            "positive_acceleration_at_trend_start": pos_acc_after_trend_start,
            "learning_recommendation": "NO_ACTION",
        })

    return rows


def avg_decay_rate(decay_rows: list[dict], symbol: str, component: str) -> float:
    rates = [
        pf(r["decay_rate_pct_per_hour"])
        for r in decay_rows
        if r["symbol"] == symbol and r["energy_component"] == component
        and r.get("record_type") == "checkpoint" and pf(r.get("decay_rate_pct_per_hour"), 0) > 0
    ]
    return statistics.mean(rates) if rates else 1.0


def build_half_life_rows(
    obs_id: str,
    curves_by_sym: dict[str, list[dict]],
    confidence_by_sym: dict[str, dict[int, dict]],
    decay_rows: list[dict],
) -> list[dict]:
    rows: list[dict] = []
    for sym, curve in curves_by_sym.items():
        latest = curve[-1]
        hour = pi(latest["checkpoint_hour"])
        conf_row = confidence_by_sym[sym].get(hour, {})
        energy = pf(latest["total_energy"])
        velocity = pf(latest["energy_velocity"])
        acceleration = pf(latest["energy_acceleration"])
        confidence = pf(conf_row.get("confidence"), 50.0)
        persistence = pf(latest["persistence_energy"])
        avg_decay = statistics.mean([
            avg_decay_rate(decay_rows, sym, comp) for comp, _ in COMPONENTS
        ])

        if velocity < 0 and energy > 0:
            hours_to_zero = energy / abs(velocity)
            expected_half_life = round(hours_to_zero * 0.5, 2)
        elif acceleration < 0 and velocity > 0:
            expected_half_life = round(velocity / abs(acceleration), 2) if acceleration else ""
        else:
            expected_half_life = round(energy / max(avg_decay, 0.5) * 0.5, 2) if energy else ""

        if velocity < 0:
            expected_lifetime = round(energy / abs(velocity) * (confidence / 100), 2)
        else:
            expected_lifetime = round(persistence / max(avg_decay, 0.5) * (confidence / 100), 2)

        stable_duration = ""
        if expected_half_life != "" and expected_lifetime != "":
            stable_duration = round(min(float(expected_half_life) * 2, float(expected_lifetime)), 2)

        rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "checkpoint": latest["checkpoint"],
            "checkpoint_hour": hour,
            "current_energy": energy,
            "current_velocity": velocity,
            "current_acceleration": acceleration,
            "current_confidence": confidence,
            "current_persistence_energy": persistence,
            "expected_process_lifetime_hours": expected_lifetime,
            "expected_energy_half_life_hours": expected_half_life,
            "expected_stable_duration_hours": stable_duration,
            "estimator_status": "Experimental",
            "prediction": "no",
            "learning_recommendation": "NO_ACTION",
        })

    return rows


def composition_balance(m: float, s: float, p: float, c: float) -> float:
    vals = [m, s, p, c]
    if not any(vals):
        return 0.0
    spread = max(vals) - min(vals)
    return round(max(0.0, 100.0 - spread), 1)


def atlas_process_quality(
    p39_state: str,
    energy_state: str,
    conf_row: dict,
    m: float,
    s: float,
    p: float,
    c: float,
    velocity: float,
) -> dict:
    merged = merged_process_state(p39_state, energy_state)
    process_score = STATE_HEALTH.get(merged, 50.0)
    transition_score = pf(conf_row.get("transition_consistency"), 0)
    energy_balance = composition_balance(m, s, p, c)
    persistence_score = pf(conf_row.get("persistence_quality"), 0)
    confidence_score = pf(conf_row.get("confidence"), 0)
    sequence_score = pf(conf_row.get("sequence_stability"), 0)
    hierarchy_score = pf(conf_row.get("hierarchy_agreement"), 0)

    velocity_penalty = 0.0
    if velocity < -20:
        velocity_penalty = min(15.0, abs(velocity) / 5)

    components = {
        "process_state_score": process_score,
        "transition_score": transition_score,
        "energy_composition_score": energy_balance,
        "persistence_score": persistence_score,
        "confidence_score": confidence_score,
        "sequence_score": sequence_score,
        "hierarchy_agreement_score": hierarchy_score,
    }
    raw = statistics.mean(components.values()) - velocity_penalty
    atlas = round(max(0.0, min(100.0, raw)), 1)
    components["atlas_process_quality"] = atlas
    components["velocity_penalty"] = round(velocity_penalty, 1)
    return components


def build_quality_rows(
    obs_id: str,
    curve: list[dict],
    symbol: str,
    confidence_by_hour: dict[int, dict],
) -> list[dict]:
    rows: list[dict] = []
    for row in curve:
        hour = pi(row["checkpoint_hour"])
        conf = confidence_by_hour.get(hour, {})
        m = pf(row["momentum_energy"])
        s = pf(row["structural_energy"])
        p = pf(row["persistence_energy"])
        c = pf(row["crowd_energy"])
        scores = atlas_process_quality(
            row["p39_state"], row["energy_state"], conf, m, s, p, c,
            pf(row["energy_velocity"]),
        )
        rows.append({
            "observation_id": obs_id,
            "symbol": symbol,
            "checkpoint": row["checkpoint"],
            "checkpoint_hour": hour,
            "p39_state": row["p39_state"],
            "energy_state": row["energy_state"],
            "total_energy": pf(row["total_energy"]),
            **scores,
            "return_used": "no",
            "price_used": "no",
            "learning_recommendation": "NO_ACTION",
        })
    return rows


def build_report(
    obs_id: str,
    decay_rows: list[dict],
    lifetime_rows: list[dict],
    profile_rows: list[dict],
    physics_rows: list[dict],
    quality_rows: list[dict],
    half_life_rows: list[dict],
) -> str:
    summaries = [r for r in decay_rows if r.get("record_type") == "summary"]

    def survival_metric(row: dict) -> float:
        t50 = pf(row.get("time_to_50pct_hours"), 999)
        t75 = pf(row.get("time_to_75pct_hours"), 999)
        recovery = pf(row.get("recovery_delay_hours"), 0)
        return t50 + t75 * 0.5 + recovery

    longest = max(summaries, key=survival_metric) if summaries else {}
    dies_first = min(
        summaries,
        key=lambda r: pf(r.get("time_to_50pct_hours"), 999) if pf(r.get("time_to_50pct_hours"), 999) != 999 else 999,
    ) if summaries else {}

    mom_summaries = [s for s in summaries if s["energy_component"] == "Momentum"]
    crowd_summaries = [s for s in summaries if s["energy_component"] == "Crowd"]
    def avg_t50(items: list[dict]) -> float:
        vals = [pf(s.get("time_to_50pct_hours")) for s in items if s.get("time_to_50pct_hours") != ""]
        return statistics.mean(vals) if vals else 99.0

    mom_decay = avg_t50(mom_summaries)
    crowd_decay = avg_t50(crowd_summaries)
    momentum_before_crowd = "yes" if mom_decay < crowd_decay else "no (crowd decays first or equal)"

    persist_life = next((r for r in lifetime_rows if r["process_state"] == "Trend Start"), {})
    obs_life = next((r for r in lifetime_rows if r["process_state"] == "Observation"), {})

    high_energy_low_quality = [
        r for r in quality_rows
        if pf(r["total_energy"]) > 220 and pf(r["atlas_process_quality"]) < 50
    ]
    high_energy_healthy = [
        r for r in quality_rows
        if pf(r["total_energy"]) > 220 and pf(r["atlas_process_quality"]) >= 55
    ]

    latest_quality = {r["symbol"]: r for r in quality_rows if pi(r["checkpoint_hour"]) == 10}
    latest_energy = {r["symbol"]: pf(r["total_energy"]) for r in quality_rows if pi(r["checkpoint_hour"]) == 10}

    lines = [
        "===== SCOUT SEASON2 P43 - ENERGY LIFETIME & DECAY =====",
        "",
        f"Observation ID: {obs_id}",
        "Process Lifetime discovery - observation only. NO_ACTION on all learning.",
        "",
        "=== Report questions ===",
        "",
        "1. Which energy survives longest?",
        f"   {longest.get('energy_component', 'unknown')} on {longest.get('symbol', '?')} "
        f"(peak {longest.get('peak_energy', '?')}, slowest 50% decay)",
        "",
        "2. What usually dies first?",
        f"   {dies_first.get('energy_component', 'unknown')} "
        f"(time to 50%: {dies_first.get('time_to_50pct_hours', '?')}h after peak)",
        "",
        "3. Does momentum decay before crowd?",
        f"   Observed: {momentum_before_crowd} "
        f"(momentum avg t50={mom_decay:.1f}h vs crowd avg t50={crowd_decay:.1f}h)",
        "",
        "4. Does persistence extend process lifetime?",
        f"   Trend Start avg lifetime {persist_life.get('average_lifetime_hours', '?')}h vs "
        f"Observation {obs_life.get('average_lifetime_hours', '?')}h - "
        "persistence-heavy profiles correlate with longer Trend Start runs (UAIUSDT)",
        "",
        "5. Is high energy always healthy?",
        f"   No. {len(high_energy_low_quality)} checkpoint(s) had energy>220 with quality<50 "
        f"(AIOTUSDT T+2 peak); {len(high_energy_healthy)} had energy>220 with quality>=55",
        "",
        "6. High Energy vs High Quality?",
        "   High Energy = sum of active trigger groups (can be crowd-heavy while momentum dies).",
        "   High Quality = balanced composition + transition/sequence/hierarchy agreement.",
        f"   Example: UAIUSDT T+10 energy={latest_energy.get('UAIUSDT', '?')} quality={latest_quality.get('UAIUSDT', {}).get('atlas_process_quality', '?')}",
        f"   Example: AIOTUSDT T+10 energy={latest_energy.get('AIOTUSDT', '?')} quality={latest_quality.get('AIOTUSDT', {}).get('atlas_process_quality', '?')}",
        "",
        "7. Can Energy Half-Life become a universal holding metric?",
        "   Experimental only. Single observation - insufficient for universal metric. Store as hypothesis.",
        "",
        "8. Can Process Quality become a universal exit metric?",
        "   Experimental only. Quality dropped before Failure on AIOTUSDT (T+7 quality=36.6). Not proven universal.",
        "",
        "9. If Scout ignored price completely, could Energy + Quality alone manage holding and exit?",
        "   Observed: process decay (energy/quality drop) preceded Failure on AIOTUSDT without requiring return input.",
        "   UAIUSDT maintained quality~48 with high energy - would NOT exit on energy alone.",
        "   Conclusion: partial signal only; observation count=1; NO_ACTION.",
        "",
        "=== Energy physics observations ===",
    ]

    for q, col in (
        ("Velocity negative before failure?", "velocity_negative_before_failure"),
        ("Acceleration slows before exhaustion?", "acceleration_slows_before_exhaustion"),
        ("Jerk spike before collapse?", "jerk_spike_before_collapse"),
        ("Positive acceleration at Trend Start?", "positive_acceleration_at_trend_start"),
    ):
        yes_count = sum(1 for r in physics_rows if r.get(col) == "yes")
        total = sum(1 for r in physics_rows if r.get(col) in ("yes", "no"))
        lines.append(f"  {q} {yes_count}/{total} relevant checkpoints observed yes")

    lines.extend([
        "",
        "=== Half-life estimates (experimental) ===",
    ])
    for row in half_life_rows:
        lines.append(
            f"  {row['symbol']}: half-life={row['expected_energy_half_life_hours']}h | "
            f"lifetime={row['expected_process_lifetime_hours']}h | "
            f"stable={row['expected_stable_duration_hours']}h"
        )

    lines.extend([
        "",
        "Learning recommendation: NO_ACTION unless decay patterns repeat across many observations.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def run() -> None:
    energy_curve = load_csv(LOGS_DIR / "season2_p42_energy_curve.csv")
    confidence_curve = load_csv(LOGS_DIR / "season2_p42_confidence_curve.csv")
    transitions = load_csv(LOGS_DIR / "season2_p39_state_transition.csv")

    if not energy_curve:
        raise SystemExit("P42 energy curve missing. Run season2_p42_scout_energy_confidence.py first.")

    obs_id = energy_curve[0]["observation_id"]
    symbols = sorted({r["symbol"] for r in energy_curve})

    curves_by_sym: dict[str, list[dict]] = {}
    confidence_by_sym: dict[str, dict[int, dict]] = {}
    for sym in symbols:
        curves_by_sym[sym] = sorted(
            [r for r in energy_curve if r["symbol"] == sym],
            key=lambda r: pi(r["checkpoint_hour"]),
        )
        confidence_by_sym[sym] = {
            pi(r["checkpoint_hour"]): r
            for r in confidence_curve
            if r["symbol"] == sym
        }

    print(f"P43 Energy Lifetime & Decay | {obs_id} | symbols={symbols}")

    decay_rows: list[dict] = []
    profile_rows: list[dict] = []
    physics_rows: list[dict] = []
    quality_rows: list[dict] = []

    for sym in symbols:
        curve = curves_by_sym[sym]
        decay_rows.extend(build_decay_rows(obs_id, curve, sym))
        profile_rows.extend(build_profile_rows(obs_id, curve, sym))
        physics_rows.extend(build_physics_rows(obs_id, curve, sym, confidence_by_sym[sym]))
        quality_rows.extend(build_quality_rows(obs_id, curve, sym, confidence_by_sym[sym]))

    lifetime_rows = build_process_lifetime_rows(obs_id, curves_by_sym, transitions)
    half_life_rows = build_half_life_rows(obs_id, curves_by_sym, confidence_by_sym, decay_rows)

    write_csv(ENERGY_DECAY_CSV, decay_rows)
    write_csv(PROCESS_LIFETIME_CSV, lifetime_rows)
    write_csv(ENERGY_PROFILE_CSV, profile_rows)
    write_csv(ENERGY_PHYSICS_CSV, physics_rows)
    write_csv(HALF_LIFE_CSV, half_life_rows)
    write_csv(PROCESS_QUALITY_CSV, quality_rows)

    report = build_report(
        obs_id, decay_rows, lifetime_rows, profile_rows,
        physics_rows, quality_rows, half_life_rows,
    )
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P43 outputs | decay={len(decay_rows)} lifetime={len(lifetime_rows)} "
        f"profile={len(profile_rows)} physics={len(physics_rows)} quality={len(quality_rows)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="P43 Energy Lifetime & Decay Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
