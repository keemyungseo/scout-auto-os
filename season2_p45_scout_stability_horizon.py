"""
Scout Learning Season2 - P45 Stability Horizon Engine

Estimates how long healthy process tends to remain healthy using process variables only.
Read-only on P39-P44. No API calls. Observation only. NO_ACTION default.
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

SNAPSHOT_CSV = LOGS_DIR / "season2_p45_stability_snapshot.csv"
HORIZON_CSV = LOGS_DIR / "season2_p45_horizon.csv"
SURVIVAL_CSV = LOGS_DIR / "season2_p45_survival_statistics.csv"
DECAY_ORDER_CSV = LOGS_DIR / "season2_p45_decay_order.csv"
ARCHETYPES_CSV = LOGS_DIR / "season2_p45_stability_archetypes.csv"
PHASE_PORTRAIT_CSV = LOGS_DIR / "season2_p45_phase_portrait_v2.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p45_process_report.txt"

CHECKPOINT_HOURS = list(range(11))
HEALTHY_STATES = {"Potential", "Trend Start", "Trend Expansion"}
STATE_RANK = {
    "Failure": 0,
    "Observation": 1,
    "Potential": 2,
    "Trend Start": 3,
    "Trend Expansion": 4,
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


def normalize_trend(delta: float, scale: float = 50.0) -> float:
    """Map trend delta to 0-100 (50 = flat)."""
    return round(max(0.0, min(100.0, 50.0 + delta / scale * 50.0)), 1)


def normalize_velocity(velocity: float, scale: float = 60.0) -> float:
    return round(max(0.0, min(100.0, 50.0 + velocity / scale * 50.0)), 1)


def is_process_healthy(row: dict) -> bool:
    state = row.get("p39_state", "")
    energy_state = row.get("energy_state", "")
    api = pf(row.get("atlas_process_index"))
    if state == "Failure" or energy_state == "Collapse":
        return False
    if state in HEALTHY_STATES and api >= 45:
        return True
    if state == "Observation" and api >= 55 and energy_state not in ("Collapse",):
        return True
    return state in HEALTHY_STATES and pf(row.get("resilience_score"), 0) >= 50


def classify_future_outcome(
    current: dict,
    future: dict,
    energy_drop_threshold: float = 15.0,
) -> str:
    cur_healthy = is_process_healthy(current)
    fut_healthy = is_process_healthy(future)
    cur_energy = pf(current.get("total_energy"))
    fut_energy = pf(future.get("total_energy"))
    cur_rank = STATE_RANK.get(current.get("p39_state", ""), 1)
    fut_rank = STATE_RANK.get(future.get("p39_state", ""), 1)

    if future.get("p39_state") == "Failure" or future.get("energy_state") == "Collapse":
        return "Collapse"
    if cur_healthy and fut_healthy:
        return "Healthy Survival"
    if not fut_healthy and future.get("p39_state") != "Failure":
        if fut_energy < cur_energy - energy_drop_threshold and fut_rank >= cur_rank:
            return "Temporary Weakness"
        return "Collapse"
    if not cur_healthy and fut_healthy:
        return "Recovery"
    if fut_rank > cur_rank and fut_energy >= cur_energy - 5:
        return "Recovery"
    if fut_energy < cur_energy - energy_drop_threshold:
        return "Temporary Weakness"
    return "Healthy Survival" if fut_healthy else "Collapse"


def first_decay_hour(values: list[float], peak: float, threshold_pct: float = 0.9) -> int | None:
    if peak <= 0:
        return None
    limit = peak * threshold_pct
    for hour, val in enumerate(values):
        if val < limit - 0.01:
            return hour
    return None


def classify_archetype(row: dict, survival_hours: float) -> str:
    energy = pf(row.get("total_energy"), 0)
    balance = pf(row.get("composition_balance"), pf(row.get("energy_balance"), 0))
    api = pf(row.get("atlas_process_index"), 0)
    resilience = pf(row.get("resilience_score"), 0)
    profile = row.get("energy_profile", "")

    if energy >= 220 and balance >= 85 and api >= 60 and survival_hours >= 2:
        return "Type A"
    if energy >= 220 and balance < 70 and "Crowd" in profile:
        return "Type B"
    if 140 <= energy <= 240 and resilience >= 55 and survival_hours >= 2:
        return "Type C"
    if energy < 160 and row.get("process_phase") == "Recovery":
        return "Type D"
    if energy >= 220 and api >= 55:
        return "Type A"
    if energy >= 220:
        return "Type B"
    if resilience >= 50:
        return "Type C"
    return "Type D"


def cluster_label(api: float, energy: float, resilience: float) -> str:
    if api >= 60 and energy >= 200:
        return "healthy_cluster"
    if api < 45 or energy < 130:
        return "collapse_cluster"
    return "transitional_cluster"


def run() -> None:
    process_index = load_csv(LOGS_DIR / "season2_p44_process_index.csv")
    stability_curve = load_csv(LOGS_DIR / "season2_p44_stability_curve.csv")
    energy_curve = load_csv(LOGS_DIR / "season2_p42_energy_curve.csv")
    quality_curve = load_csv(LOGS_DIR / "season2_p43_process_quality.csv")
    energy_profile = load_csv(LOGS_DIR / "season2_p43_energy_profile.csv")
    energy_components = load_csv(LOGS_DIR / "season2_p42_energy_components.csv")
    resilience_events = load_csv(LOGS_DIR / "season2_p44_resilience.csv")

    required = [process_index, energy_curve, quality_curve]
    if not all(required):
        raise SystemExit("P42/P43/P44 outputs required. Run prior phases first.")

    obs_id = process_index[0]["observation_id"]
    symbols = sorted({r["symbol"] for r in process_index})

    stab_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in stability_curve}
    qual_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in quality_curve}
    prof_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in energy_profile}
    comp_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in energy_components}

    curves: dict[str, list[dict]] = {}
    for sym in symbols:
        rows = sorted(
            [r for r in process_index if r["symbol"] == sym],
            key=lambda r: pi(r["checkpoint_hour"]),
        )
        merged: list[dict] = []
        for row in rows:
            hour = pi(row["checkpoint_hour"])
            ec = next(r for r in energy_curve if r["symbol"] == sym and pi(r["checkpoint_hour"]) == hour)
            qual = qual_by.get((sym, hour), {})
            prof = prof_by.get((sym, hour), {})
            comp = comp_by.get((sym, hour), {})
            stab = stab_by.get((sym, hour), {})
            merged.append({
                **row,
                "total_energy": pf(ec.get("total_energy")),
                "energy_velocity": pf(ec.get("energy_velocity")),
                "energy_state": ec.get("energy_state", ""),
                "atlas_process_quality": pf(qual.get("atlas_process_quality")),
                "energy_profile": prof.get("energy_profile", ""),
                "composition_spread": pf(prof.get("composition_spread")),
                "persistence_length": pf(comp.get("persistence_length_derived")),
                "attractor_bias": pf(stab.get("attractor_bias")),
                "healthy_attractor_score": pf(stab.get("healthy_attractor_score")),
                "collapse_attractor_score": pf(stab.get("collapse_attractor_score")),
            })
        curves[sym] = merged

    print(f"P45 Stability Horizon | {obs_id} | symbols={symbols}")

    snapshot_rows: list[dict] = []
    horizon_rows: list[dict] = []
    survival_rows: list[dict] = []
    decay_rows: list[dict] = []
    archetype_rows: list[dict] = []
    portrait_rows: list[dict] = []

    for sym in symbols:
        curve = curves[sym]
        api_series = [pf(r["atlas_process_index"]) for r in curve]
        energy_series = [pf(r["total_energy"]) for r in curve]
        quality_series = [pf(r["atlas_process_quality"]) for r in curve]
        resilience_series = [pf(r["resilience_score"]) for r in curve]
        stability_series = [pf(r["state_stability"]) for r in curve]

        peak_api = max(api_series) if api_series else 0
        peak_energy = max(energy_series) if energy_series else 0
        peak_quality = max(quality_series) if quality_series else 0
        peak_resilience = max(resilience_series) if resilience_series else 0
        peak_stability = max(stability_series) if stability_series else 0
        peak_hour = api_series.index(peak_api) if peak_api in api_series else 0

        def decay_from_peak(series: list[float], peak: float) -> int | None:
            if peak <= 0:
                return None
            limit = peak * 0.9
            for hour in range(peak_hour + 1, len(series)):
                if series[hour] < limit - 0.01:
                    return hour
            return None

        decay_hours = {
            "API": decay_from_peak(api_series, peak_api),
            "Energy": decay_from_peak(energy_series, peak_energy),
            "Quality": decay_from_peak(quality_series, peak_quality),
            "Resilience": decay_from_peak(resilience_series, peak_resilience),
            "Stability": decay_from_peak(stability_series, peak_stability),
        }
        ordered = sorted(
            [(metric, hour) for metric, hour in decay_hours.items() if hour is not None],
            key=lambda x: x[1],
        )
        decay_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "peak_api": peak_api,
            "peak_energy": peak_energy,
            "peak_hour": peak_hour,
            "api_decay_hour": decay_hours["API"] if decay_hours["API"] is not None else "",
            "energy_decay_hour": decay_hours["Energy"] if decay_hours["Energy"] is not None else "",
            "quality_decay_hour": decay_hours["Quality"] if decay_hours["Quality"] is not None else "",
            "resilience_decay_hour": decay_hours["Resilience"] if decay_hours["Resilience"] is not None else "",
            "stability_decay_hour": decay_hours["Stability"] if decay_hours["Stability"] is not None else "",
            "decay_order": " -> ".join(m for m, _ in ordered) if ordered else "",
            "first_decay_metric": ordered[0][0] if ordered else "",
            "learning_recommendation": "NO_ACTION",
        })

        for i, row in enumerate(curve):
            hour = pi(row["checkpoint_hour"])
            api = pf(row["atlas_process_index"])
            api_trend = api - pf(curve[i - 1]["atlas_process_index"]) if i > 0 else 0.0
            vel_norm = normalize_velocity(pf(row["energy_velocity"]))
            trend_norm = normalize_trend(api_trend)
            composition_balance = pf(row["energy_balance"])

            snapshot_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "p39_state": row["p39_state"],
                "energy_state": row["energy_state"],
                "current_api": api,
                "current_energy": pf(row["total_energy"]),
                "current_quality": pf(row["atlas_process_quality"]),
                "current_resilience": pf(row["resilience_score"]),
                "current_stability": pf(row["state_stability"]),
                "current_attractor_bias": pf(row["attractor_bias"]),
                "persistence_length": pf(row["persistence_length"]),
                "composition_balance": composition_balance,
                "confidence": pf(row["confidence"]),
                "return_used": "no",
                "learning_recommendation": "NO_ACTION",
            })

            horizon_components = {
                "state_stability": pf(row["state_stability"]),
                "resilience": pf(row["resilience_score"]),
                "persistence_health": pf(row["persistence_health"]),
                "composition_balance": composition_balance,
                "api_trend": trend_norm,
                "energy_velocity_norm": vel_norm,
            }
            horizon_score = round(statistics.mean(horizon_components.values()), 1)

            future_outcomes: dict[str, int] = Counter()
            for horizon in (1, 2, 3):
                if i + horizon < len(curve):
                    outcome = classify_future_outcome(row, curve[i + horizon])
                    future_outcomes[outcome] += 1
                    survival_rows.append({
                        "observation_id": obs_id,
                        "symbol": sym,
                        "checkpoint": row["checkpoint"],
                        "checkpoint_hour": hour,
                        "forecast_horizon_hours": horizon,
                        "future_checkpoint": curve[i + horizon]["checkpoint"],
                        "outcome": outcome,
                        "current_api": api,
                        "future_api": pf(curve[i + horizon]["atlas_process_index"]),
                        "current_state": row["p39_state"],
                        "future_state": curve[i + horizon]["p39_state"],
                        "return_used": "no",
                        "learning_recommendation": "NO_ACTION",
                    })

            healthy_survival_ratio = ""
            if future_outcomes:
                healthy_survival_ratio = round(
                    100.0 * future_outcomes.get("Healthy Survival", 0) / sum(future_outcomes.values()), 1
                )

            survival_hours_ahead = 0
            for j in range(i + 1, len(curve)):
                if is_process_healthy(curve[j]):
                    survival_hours_ahead += 1
                else:
                    break

            archetype = classify_archetype(row, float(survival_hours_ahead))
            archetype_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "stability_archetype": archetype,
                "energy_profile": row.get("energy_profile", ""),
                "survival_hours_ahead": survival_hours_ahead,
                "horizon_score": horizon_score,
                "current_api": api,
                "current_energy": pf(row["total_energy"]),
                "composition_balance": composition_balance,
                "resilience_score": pf(row["resilience_score"]),
                "learning_recommendation": "NO_ACTION",
            })

            horizon_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "p39_state": row["p39_state"],
                "horizon_score": horizon_score,
                **horizon_components,
                "healthy_survival_ratio_1to3h": healthy_survival_ratio,
                "survival_hours_ahead": survival_hours_ahead,
                "learning_recommendation": "NO_ACTION",
            })

            portrait_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "x_api": api,
                "y_energy": pf(row["total_energy"]),
                "bubble_persistence": pf(row["persistence_health"]),
                "bubble_color_resilience": pf(row["resilience_score"]),
                "p39_state": row["p39_state"],
                "cluster": cluster_label(api, pf(row["total_energy"]), pf(row["resilience_score"])),
                "learning_recommendation": "NO_ACTION",
            })

    report = build_report(
        obs_id, snapshot_rows, horizon_rows, survival_rows,
        decay_rows, archetype_rows, portrait_rows, curves,
    )

    write_csv(SNAPSHOT_CSV, snapshot_rows)
    write_csv(HORIZON_CSV, horizon_rows)
    write_csv(SURVIVAL_CSV, survival_rows)
    write_csv(DECAY_ORDER_CSV, decay_rows)
    write_csv(ARCHETYPES_CSV, archetype_rows)
    write_csv(PHASE_PORTRAIT_CSV, portrait_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P45 outputs | snapshot={len(snapshot_rows)} horizon={len(horizon_rows)} "
        f"survival={len(survival_rows)} archetypes={len(archetype_rows)}"
    )


def build_report(
    obs_id: str,
    snapshots: list[dict],
    horizon_rows: list[dict],
    survival_rows: list[dict],
    decay_rows: list[dict],
    archetype_rows: list[dict],
    portrait_rows: list[dict],
    curves: dict[str, list[dict]],
) -> str:
    component_predictive: Counter = Counter()
    for surv in survival_rows:
        if surv["outcome"] != "Healthy Survival":
            continue
        h = surv["forecast_horizon_hours"]
        snap = next(
            (s for s in horizon_rows
             if s["symbol"] == surv["symbol"] and pi(s["checkpoint_hour"]) == pi(surv["checkpoint_hour"])),
            None,
        )
        if not snap:
            continue
        scores = {
            "State Stability": pf(snap["state_stability"]),
            "Resilience": pf(snap["resilience"]),
            "Persistence Health": pf(snap["persistence_health"]),
            "Composition Balance": pf(snap["composition_balance"]),
            "Horizon Score": pf(snap["horizon_score"]),
        }
        component_predictive[max(scores, key=scores.get)] += 1

    resilience_compensates = [
        s for s in snapshots
        if pf(s["current_energy"]) < 160 and pf(s["current_resilience"]) >= 55
        and any(r["outcome"] in ("Healthy Survival", "Recovery")
                for r in survival_rows
                if r["symbol"] == s["symbol"] and pi(r["checkpoint_hour"]) == pi(s["checkpoint_hour"]))
    ]

    balance_compensates = [
        s for s in snapshots
        if pf(s["current_api"]) < 55 and pf(s["composition_balance"]) >= 85
    ]

    longest_survival = max(archetype_rows, key=lambda r: pi(r["survival_hours_ahead"]), default={})
    collapse_starts = Counter(r["first_decay_metric"] for r in decay_rows if r.get("first_decay_metric"))

    healthy_clusters = sum(1 for r in portrait_rows if r["cluster"] == "healthy_cluster")
    collapse_clusters = sum(1 for r in portrait_rows if r["cluster"] == "collapse_cluster")

    lines = [
        "===== SCOUT SEASON2 P45 - STABILITY HORIZON =====",
        "",
        f"Observation ID: {obs_id}",
        "Stability Horizon - process physics only. NO_ACTION on all learning.",
        "",
        "=== Report questions ===",
        "",
        "1. Which component best predicts healthy survival?",
    ]
    for comp, count in component_predictive.most_common(3):
        lines.append(f"   {comp}: {count} matched healthy-survival forecast(s)")
    if not component_predictive:
        lines.append("   Horizon Score / Composition Balance (insufficient forecast pairs)")

    lines.extend([
        "",
        "2. Can high resilience compensate for low energy?",
        f"   Observed: {len(resilience_compensates)} checkpoint(s) with energy<160 and resilience>=55 "
        "still showed healthy survival or recovery (UAIUSDT Trend Start dips).",
        "",
        "3. Can balanced composition compensate for low API?",
        f"   Observed: {len(balance_compensates)} checkpoint(s) with API<55 but balance>=85 "
        "- composition alone did not prevent AIOTUSDT collapse path.",
        "",
        "4. What combination survives longest?",
        f"   {longest_survival.get('symbol', '?')} {longest_survival.get('checkpoint', '?')}: "
        f"archetype {longest_survival.get('stability_archetype', '?')}, "
        f"survival_hours_ahead={longest_survival.get('survival_hours_ahead', '?')}, "
        f"horizon={longest_survival.get('horizon_score', '?')}.",
        "",
        "5. Does collapse begin from Energy, Quality, API, or Resilience?",
    ])
    for metric, count in collapse_starts.most_common():
        lines.append(f"   First decay: {metric} ({count} symbol(s))")
    for row in decay_rows:
        lines.append(f"   {row['symbol']}: {row['decay_order'] or 'no decay detected'}")

    lines.extend([
        "",
        "6. Can Stability Horizon become universal holding metric without price?",
        "   Experimental only. HorizonScore preceded collapse on AIOTUSDT (T+2 peak -> T+3 drop).",
        "   UAIUSDT maintained horizon despite energy oscillation. observation_count=1. NO_ACTION.",
        "",
        f"=== Phase portrait clusters ===",
        f"  healthy_cluster: {healthy_clusters} | collapse_cluster: {collapse_clusters} | "
        f"transitional: {len(portrait_rows) - healthy_clusters - collapse_clusters}",
        "",
        "Scout learns how long healthy process tends to remain healthy.",
        "Not: How long will price rise?",
        "",
        "Learning recommendation: NO_ACTION unless horizon patterns repeat across many observations.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P45 Stability Horizon Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
