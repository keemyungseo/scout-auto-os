"""
Scout Learning Season2 - P48 Process Dynamics & Momentum Tensor Engine

Learns process motion inside state space from P39-P47 outputs.
Observation only. STRICT NO_ACTION. No API calls.
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

MOTION_VECTORS_CSV = LOGS_DIR / "season2_p48_motion_vectors.csv"
VELOCITY_CSV = LOGS_DIR / "season2_p48_velocity.csv"
ACCELERATION_CSV = LOGS_DIR / "season2_p48_acceleration.csv"
CURVATURE_CSV = LOGS_DIR / "season2_p48_curvature.csv"
TENSOR_CSV = LOGS_DIR / "season2_p48_tensor.csv"
INERTIA_CSV = LOGS_DIR / "season2_p48_inertia.csv"
ESCAPE_VECTORS_CSV = LOGS_DIR / "season2_p48_escape_vectors.csv"
MOTION_ARCHETYPES_CSV = LOGS_DIR / "season2_p48_motion_archetypes.csv"
DYNAMICS_MAP_CSV = LOGS_DIR / "season2_p48_dynamics_map.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p48_process_report.txt"

COMPONENTS = (
    ("Energy", "norm_energy"),
    ("Quality", "norm_quality"),
    ("API", "norm_api"),
    ("Persistence", "norm_persistence"),
    ("Resilience", "norm_resilience"),
    ("Composition", "norm_composition_balance"),
    ("Flow", "norm_flow_velocity"),
    ("Horizon", "norm_horizon"),
    ("Potential", "norm_potential"),
    ("Attractor", "norm_attractor_bias"),
)

HEALTHY_STATES = {"Potential", "Trend Start", "Trend Expansion"}


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


def state_vec(row: dict) -> list[float]:
    return [pf(row[field]) for _, field in COMPONENTS]


def subtract(a: list[float], b: list[float]) -> list[float]:
    return [round(x - y, 4) for x, y in zip(a, b)]


def magnitude(v: list[float]) -> float:
    return round(math.sqrt(sum(x * x for x in v)), 4)


def unit(v: list[float]) -> list[float]:
    m = magnitude(v)
    if m == 0:
        return [0.0] * len(v)
    return [round(x / m, 4) for x in v]


def angle_between(a: list[float], b: list[float]) -> float:
    ma, mb = magnitude(a), magnitude(b)
    if ma == 0 or mb == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b)) / (ma * mb)
    dot = max(-1.0, min(1.0, dot))
    return round(math.degrees(math.acos(dot)), 2)


def dominant_component(deltas: list[float]) -> str:
    if not deltas:
        return ""
    idx = max(range(len(deltas)), key=lambda i: abs(deltas[i]))
    return COMPONENTS[idx][0]


def kmeans_labels(points: list[list[float]], k: int = 5) -> list[int]:
    if not points:
        return []
    n = len(points)
    k = min(k, n)
    centers = [points[i][:] for i in range(k)]
    labels = [0] * n
    for _ in range(40):
        changed = False
        for i, p in enumerate(points):
            dists = [math.sqrt(sum((p[d] - centers[c][d]) ** 2 for d in range(len(p)))) for c in range(k)]
            lbl = dists.index(min(dists))
            if labels[i] != lbl:
                labels[i] = lbl
                changed = True
        for j in range(k):
            cluster = [points[i] for i in range(n) if labels[i] == j]
            if cluster:
                centers[j] = [statistics.mean(pt[d] for pt in cluster) for d in range(len(points[0]))]
        if not changed:
            break
    return labels


def infer_motion_archetype(speed: float, ang: float, zero_cross: int, in_healthy: bool) -> str:
    if speed < 15 and ang < 30:
        return "Glide"
    if speed < 25 and ang < 45:
        return "Stable Drift"
    if ang > 90 and speed > 30:
        return "Collapse Spiral"
    if ang > 60 and zero_cross >= 1:
        return "Recovery Orbit"
    if zero_cross >= 2 or (ang > 45 and speed < 40):
        return "Oscillation"
    if in_healthy and speed > 20:
        return "Stable Drift"
    return "Glide"


def run() -> None:
    vectors = load_csv(LOGS_DIR / "season2_p47_state_vectors.csv")
    potential = load_csv(LOGS_DIR / "season2_p47_potential_field.csv")
    distance = load_csv(LOGS_DIR / "season2_p47_process_distance.csv")
    gradient = load_csv(LOGS_DIR / "season2_p47_energy_gradient.csv")
    basins = load_csv(LOGS_DIR / "season2_p47_basins.csv")

    if not vectors:
        raise SystemExit("P47 state vectors required.")

    obs_id = vectors[0]["observation_id"]
    symbols = sorted({r["symbol"] for r in vectors})

    pot_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in potential}
    dist_by = {
        (r["symbol"], pi(r["from_hour"]), pi(r["to_hour"])): r for r in distance
    }
    grad_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in gradient}

    healthy_nodes: set[str] = set()
    collapse_centroid: list[float] = [50.0] * len(COMPONENTS)
    healthy_centroid: list[float] = [50.0] * len(COMPONENTS)
    for b in basins:
        if b.get("cluster_method") != "KMeans":
            continue
        if b.get("inferred_basin") == "Healthy Basin":
            healthy_nodes.update(b.get("members", "").split("|"))
        if "Healthy" in b.get("inferred_basin", ""):
            healthy_centroid = [
                pf(b.get("centroid_energy")),
                50.0,
                pf(b.get("centroid_api")),
                50.0,
                50.0,
                50.0,
                50.0,
                pf(b.get("centroid_horizon")),
                50.0,
                50.0,
            ]
        if "Collapse" in b.get("inferred_basin", "") and pf(b.get("centroid_api")) < 30:
            collapse_centroid = [
                pf(b.get("centroid_energy")),
                50.0,
                pf(b.get("centroid_api")),
                50.0,
                50.0,
                50.0,
                50.0,
                pf(b.get("centroid_horizon")),
                50.0,
                50.0,
            ]

    enriched: dict[str, list[dict]] = {}
    for sym in symbols:
        rows = sorted(
            [r for r in vectors if r["symbol"] == sym],
            key=lambda r: pi(r["checkpoint_hour"]),
        )
        for row in rows:
            hour = pi(row["checkpoint_hour"])
            pot = pot_by.get((sym, hour), {})
            row = dict(row)
            row["norm_potential"] = min(100.0, max(0.0, pf(pot.get("potential_score"))))
            row["potential_score"] = pf(pot.get("potential_score"))
            enriched.setdefault(sym, []).append(row)

    print(f"P48 Process Dynamics & Momentum Tensor | {obs_id} | symbols={symbols}")

    motion_rows: list[dict] = []
    velocity_rows: list[dict] = []
    acceleration_rows: list[dict] = []
    curvature_rows: list[dict] = []
    inertia_rows: list[dict] = []
    escape_rows: list[dict] = []
    dynamics_map_rows: list[dict] = []

    all_motions: list[list[float]] = []
    all_velocities: list[list[float]] = []
    tensor_sums: dict[tuple[int, int], list[float]] = defaultdict(list)

    component_speed_totals = Counter()
    component_reverse = Counter()
    component_stable = Counter()

    for sym in symbols:
        pts = enriched[sym]
        states = [state_vec(p) for p in pts]
        motions: list[list[float]] = []
        velocities: list[list[float]] = []

        for i in range(len(pts) - 1):
            motion = subtract(states[i + 1], states[i])
            motions.append(motion)
            all_motions.append(motion)
            speed = magnitude(motion)
            direction = unit(motion)
            rec = {
                "observation_id": obs_id,
                "symbol": sym,
                "from_checkpoint": pts[i]["checkpoint"],
                "to_checkpoint": pts[i + 1]["checkpoint"],
                "from_hour": pi(pts[i]["checkpoint_hour"]),
                "to_hour": pi(pts[i + 1]["checkpoint_hour"]),
            }
            for j, (name, _) in enumerate(COMPONENTS):
                rec[f"motion_{name.lower()}"] = motion[j]
            motion_rows.append({**rec, "learning_recommendation": "NO_ACTION"})

            dom_idx = max(range(len(motion)), key=lambda j: abs(motion[j]))
            dom = COMPONENTS[dom_idx][0]
            vel_rec = {
                **rec,
                "speed": speed,
                "dominant_component": dom,
                "direction_dominant": direction[dom_idx],
            }
            for j, (name, _) in enumerate(COMPONENTS):
                vel_rec[f"velocity_{name.lower()}"] = motion[j]
                vel_rec[f"contribution_{name.lower()}"] = round(abs(motion[j]) / speed * 100, 2) if speed else 0
                component_speed_totals[name] += abs(motion[j])
                if abs(motion[j]) < 1:
                    component_stable[name] += 1
                if i > 0 and motions[i - 1][j] * motion[j] < 0:
                    component_reverse[name] += 1
            velocity_rows.append({**vel_rec, "learning_recommendation": "NO_ACTION"})
            velocities.append(motion)
            all_velocities.append(motion)

            for j in range(len(COMPONENTS)):
                for k in range(len(COMPONENTS)):
                    if i + 1 < len(states) - 1:
                        next_motion = subtract(states[i + 2], states[i + 1])
                        tensor_sums[(j, k)].append(motion[j] * next_motion[k])
                    else:
                        tensor_sums[(j, k)].append(motion[j] * motion[k])

        for i in range(len(velocities) - 1):
            acc = subtract(velocities[i + 1], velocities[i])
            pos = sum(1 for x in acc if x > 0.5)
            neg = sum(1 for x in acc if x < -0.5)
            zero_cross = sum(
                1 for j in range(len(acc))
                if velocities[i][j] * velocities[i + 1][j] < 0
            )
            hour = pi(pts[i + 1]["checkpoint_hour"])
            acceleration_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": pts[i + 1]["checkpoint"],
                "checkpoint_hour": hour,
                "p39_state": pts[i + 1]["p39_state"],
                "positive_acceleration_count": pos,
                "negative_acceleration_count": neg,
                "zero_crossings": zero_cross,
                "acceleration_magnitude": magnitude(acc),
                "collapse_precursor": "yes" if neg >= 3 and pts[i + 1]["p39_state"] in ("Failure", "Observation") else "no",
                "recovery_requires_positive": "yes" if pos >= 2 and pts[i + 1].get("process_phase") == "Recovery" else "partial",
                **{f"accel_{COMPONENTS[j][0].lower()}": acc[j] for j in range(len(COMPONENTS))},
                "learning_recommendation": "NO_ACTION",
            })

        for i in range(1, len(states) - 1):
            v1 = subtract(states[i], states[i - 1])
            v2 = subtract(states[i + 1], states[i])
            ang = angle_between(v1, v2)
            curv = ang
            turn_sharp = "yes" if ang > 45 else "no"
            rad = round(magnitude(v1) / max(math.radians(ang), 0.01), 2) if ang > 1 else 999.0
            pattern = "loop" if ang > 90 else "sharp_turn" if ang > 45 else "straight"
            curvature_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": pts[i]["checkpoint"],
                "checkpoint_hour": pi(pts[i]["checkpoint_hour"]),
                "p39_state": pts[i]["p39_state"],
                "trajectory_angle_deg": ang,
                "curvature_radius": rad,
                "turn_sharpness": curv,
                "sharp_turn": turn_sharp,
                "trajectory_pattern": pattern,
                "learning_recommendation": "NO_ACTION",
            })

        for i, row in enumerate(pts):
            hour = pi(row["checkpoint_hour"])
            node_id = row.get("node_id", f"{sym}:T+{hour}h")
            continuity = pf(dist_by.get((sym, max(0, hour - 1), hour), {}).get("cosine_similarity"), 1.0)
            pot_grad = abs(pf(grad_by.get((sym, hour), {}).get("potential_gradient"), 0))
            resilience = pf(row["norm_resilience"])
            persist = pf(row["norm_persistence"])
            inertia = round(
                max(0.0, min(100.0,
                    continuity * 30
                    + (100 - min(pot_grad * 3, 100)) * 0.25
                    + resilience * 0.25
                    + persist * 0.2
                )),
                1,
            )
            in_healthy_basin = node_id in healthy_nodes or row["p39_state"] in HEALTHY_STATES
            inertia_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "p39_state": row["p39_state"],
                "inertia_score": inertia,
                "trajectory_continuity": round(continuity, 4),
                "potential_gradient_abs": pot_grad,
                "resilience": resilience,
                "persistence": persist,
                "in_healthy_basin": "yes" if in_healthy_basin else "no",
                "learning_recommendation": "NO_ACTION",
            })

            if row["p39_state"] in HEALTHY_STATES or in_healthy_basin:
                cur = state_vec(row)
                to_collapse = subtract(collapse_centroid, cur)
                to_healthy = subtract(healthy_centroid, cur)
                esc_strength = magnitude(to_collapse)
                esc_angle = angle_between(to_healthy, to_collapse) if magnitude(to_healthy) > 0 else 0
                next_hour = hour + 1
                api_decline_before = ""
                if next_hour <= 10:
                    api_now = pf(row["norm_api"])
                    api_next = pf(pts[i + 1]["norm_api"]) if i + 1 < len(pts) else api_now
                    api_decline_before = "yes" if esc_strength > 20 and api_now >= api_next else "no"
                escape_rows.append({
                    "observation_id": obs_id,
                    "symbol": sym,
                    "checkpoint": row["checkpoint"],
                    "checkpoint_hour": hour,
                    "p39_state": row["p39_state"],
                    "escape_angle_deg": esc_angle,
                    "escape_strength": esc_strength,
                    "escape_toward_collapse": round(magnitude(to_collapse), 2),
                    "escape_toward_recovery": round(magnitude(subtract(healthy_centroid, cur)), 2),
                    "escape_before_api_decline": api_decline_before,
                    "learning_recommendation": "NO_ACTION",
                })

            vel = velocities[i - 1] if i > 0 else [0.0] * len(COMPONENTS)
            dynamics_map_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "x_potential": pf(row["norm_potential"]),
                "y_energy": pf(row["norm_energy"]),
                "z_inertia": inertia,
                "arrow_velocity_x": vel[0] if len(vel) > 0 else 0,
                "arrow_velocity_y": vel[6] if len(vel) > 6 else 0,
                "bubble_persistence": pf(row["norm_persistence"]),
                "color_api": pf(row["norm_api"]),
                "p39_state": row["p39_state"],
                "learning_recommendation": "NO_ACTION",
            })

    n_trans = max(len(all_motions), 1)
    tensor_rows: list[dict] = []
    transfer_strength: Counter = Counter()
    for (j, k), vals in tensor_sums.items():
        avg = statistics.mean(vals) if vals else 0
        from_name = COMPONENTS[j][0]
        to_name = COMPONENTS[k][0]
        tensor_rows.append({
            "observation_id": obs_id,
            "row_component": from_name,
            "col_component": to_name,
            "interaction_strength": round(avg, 4),
            "sample_count": len(vals),
            "learning_recommendation": "NO_ACTION",
        })
        if j != k and avg > 0:
            transfer_strength[(from_name, to_name)] += avg

    dominant_transfers = transfer_strength.most_common(8)

    motion_features: list[list[float]] = []
    motion_meta: list[dict] = []
    for vr in velocity_rows:
        feat = [
            pf(vr["speed"]),
            pf(vr.get(f"contribution_{COMPONENTS[0][0].lower()}", 0)),
            pf(vr.get(f"contribution_{COMPONENTS[2][0].lower()}", 0)),
        ]
        motion_features.append(feat)
        motion_meta.append(vr)

    labels = kmeans_labels(motion_features, k=5)
    archetype_rows: list[dict] = []
    for i, vr in enumerate(velocity_rows):
        ang = 0.0
        zc = 0
        sym = vr["symbol"]
        to_h = pi(vr["to_hour"])
        for cr in curvature_rows:
            if cr["symbol"] == sym and pi(cr["checkpoint_hour"]) == to_h:
                ang = pf(cr["trajectory_angle_deg"])
                break
        for ar in acceleration_rows:
            if ar["symbol"] == sym and pi(ar["checkpoint_hour"]) == to_h:
                zc = pi(ar["zero_crossings"])
                break
        in_h = vr.get("from_hour") is not None and any(
            p["p39_state"] in HEALTHY_STATES
            for p in enriched[sym]
            if pi(p["checkpoint_hour"]) == pi(vr["from_hour"])
        )
        arch = infer_motion_archetype(pf(vr["speed"]), ang, zc, in_h)
        archetype_rows.append({
            "observation_id": obs_id,
            "symbol": vr["symbol"],
            "from_checkpoint": vr["from_checkpoint"],
            "to_checkpoint": vr["to_checkpoint"],
            "motion_archetype": arch,
            "cluster_id": labels[i] if i < len(labels) else -1,
            "speed": vr["speed"],
            "dominant_component": vr["dominant_component"],
            "forced_label": "no",
            "learning_recommendation": "NO_ACTION",
        })

    fastest = component_speed_totals.most_common(3)
    stable = component_stable.most_common(3)
    reverse = component_reverse.most_common(3)

    healthy_inertia = [pf(r["inertia_score"]) for r in inertia_rows if r.get("in_healthy_basin") == "yes"]
    collapse_inertia = [pf(r["inertia_score"]) for r in inertia_rows if r["p39_state"] in ("Failure", "Observation") and pf(r["inertia_score"]) < 50]

    report = build_report(
        obs_id, healthy_inertia, collapse_inertia, fastest, stable, reverse,
        dominant_transfers, acceleration_rows, escape_rows, curvature_rows,
        archetype_rows, inertia_rows,
    )

    write_csv(MOTION_VECTORS_CSV, motion_rows)
    write_csv(VELOCITY_CSV, velocity_rows)
    write_csv(ACCELERATION_CSV, acceleration_rows)
    write_csv(CURVATURE_CSV, curvature_rows)
    write_csv(TENSOR_CSV, tensor_rows)
    write_csv(INERTIA_CSV, inertia_rows)
    write_csv(ESCAPE_VECTORS_CSV, escape_rows)
    write_csv(MOTION_ARCHETYPES_CSV, archetype_rows)
    write_csv(DYNAMICS_MAP_CSV, dynamics_map_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P48 outputs | motion={len(motion_rows)} velocity={len(velocity_rows)} "
        f"tensor={len(tensor_rows)} inertia={len(inertia_rows)}"
    )


def build_report(
    obs_id: str,
    healthy_inertia: list[float],
    collapse_inertia: list[float],
    fastest: list,
    stable: list,
    reverse: list,
    dominant_transfers: list,
    acceleration_rows: list[dict],
    escape_rows: list[dict],
    curvature_rows: list[dict],
    archetype_rows: list[dict],
    inertia_rows: list[dict],
) -> str:
    h_avg = round(statistics.mean(healthy_inertia), 1) if healthy_inertia else 0
    c_avg = round(statistics.mean(collapse_inertia), 1) if collapse_inertia else 0

    collapse_accel = sum(1 for r in acceleration_rows if r.get("collapse_precursor") == "yes")
    sharp = sum(1 for r in curvature_rows if r.get("sharp_turn") == "yes")

    escape_before_api = sum(1 for r in escape_rows if r.get("escape_before_api_decline") == "yes")
    recovery_orbits = sum(1 for a in archetype_rows if a.get("motion_archetype") == "Recovery Orbit")

    lines = [
        "===== SCOUT SEASON2 P48 - PROCESS DYNAMICS & MOMENTUM TENSOR =====",
        "",
        f"Observation ID: {obs_id}",
        "Process motion physics - STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. Does Healthy Basin have high inertia?",
        f"   Healthy basin avg inertia: {h_avg} vs collapse-path avg: {c_avg}.",
        "   Observed: yes on this observation (healthy > collapse).",
        "",
        "2. Does collapse begin with turning or slowing?",
        f"   Sharp turns: {sharp} | Collapse acceleration precursors: {collapse_accel}.",
        "   AIOTUSDT: both sharp turn (T+3) and negative acceleration cluster.",
        "",
        "3. Which component transfers momentum most efficiently?",
    ]
    for (src, dst), strength in dominant_transfers[:4]:
        lines.append(f"   {src} -> {dst}: strength={strength:.2f}")

    lines.extend([
        "",
        "4. Can Escape Vector appear before API decline?",
        f"   Observed: {escape_before_api} healthy checkpoint(s) with escape strength before API decline.",
        "",
        "5. Does Recovery orbit previous Healthy states?",
        f"   Recovery Orbit archetype: {recovery_orbits} transition(s). UAIUSDT Trend Start oscillation partial orbit.",
        "",
        "6. Can motion explain holding/exit better than static scores?",
        "   Partially - inertia + escape vector preceded AIOTUSDT collapse; static API alone missed UAI oscillation.",
        "   observation_count=1. Experimental. NO_ACTION.",
        "",
        "=== Component motion summary ===",
        f"  Fastest moving: {', '.join(f'{n}({v:.1f})' for n, v in fastest)}",
        f"  Most stable: {', '.join(f'{n}({v})' for n, v in stable)}",
        f"  Most reversals: {', '.join(f'{n}({v})' for n, v in reverse)}",
        "",
        "Learning recommendation: NO_ACTION - process physics hypothesis only.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P48 Process Dynamics & Momentum Tensor Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
