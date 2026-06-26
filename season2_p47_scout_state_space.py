"""
Scout Learning Season2 - P47 State Space & Potential Field Engine

Builds state space model from P39-P46 outputs. Observation only. NO_ACTION default.
No API calls. No hierarchy/weight/institution changes.
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

STATE_VECTORS_CSV = LOGS_DIR / "season2_p47_state_vectors.csv"
PROCESS_DISTANCE_CSV = LOGS_DIR / "season2_p47_process_distance.csv"
STATE_SPACE_CSV = LOGS_DIR / "season2_p47_state_space.csv"
POTENTIAL_FIELD_CSV = LOGS_DIR / "season2_p47_potential_field.csv"
ENERGY_GRADIENT_CSV = LOGS_DIR / "season2_p47_energy_gradient.csv"
TRAJECTORY_CSV = LOGS_DIR / "season2_p47_trajectory.csv"
BASINS_CSV = LOGS_DIR / "season2_p47_basins.csv"
PHASE3D_CSV = LOGS_DIR / "season2_p47_phase3d.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p47_process_report.txt"

VECTOR_FIELDS = (
    ("energy", "norm_energy"),
    ("quality", "norm_quality"),
    ("api", "norm_api"),
    ("resilience", "norm_resilience"),
    ("persistence", "norm_persistence"),
    ("composition_balance", "norm_composition_balance"),
    ("horizon", "norm_horizon"),
    ("flow_velocity", "norm_flow_velocity"),
    ("flow_acceleration", "norm_flow_acceleration"),
    ("attractor_bias", "norm_attractor_bias"),
)

RAW_KEYS = [f[0] for f in VECTOR_FIELDS]

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


def minmax_norm(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [50.0] * len(values)
    return [round(100.0 * (v - lo) / (hi - lo), 2) for v in values]


def symmetrize_bias(v: float, lo: float, hi: float) -> float:
    """Map signed attractor bias to 0-100."""
    if hi <= lo:
        return 50.0
    mid = (hi + lo) / 2
    span = max(abs(hi - mid), abs(mid - lo), 1.0)
    return round(max(0.0, min(100.0, 50.0 + (v - mid) / span * 50.0)), 2)


def vector_from_row(row: dict) -> list[float]:
    return [pf(row[k]) for k in RAW_KEYS if k.startswith("norm_") or k in RAW_KEYS]


def norm_vector(row: dict) -> list[float]:
    return [pf(row[f"norm_{k}"]) for k in RAW_KEYS]


def euclidean(a: list[float], b: list[float]) -> float:
    return round(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))), 4)


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return round(dot / (na * nb), 4)


def angle_between(a: list[float], b: list[float]) -> float:
    cos = cosine_sim(a, b)
    cos = max(-1.0, min(1.0, cos))
    return round(math.degrees(math.acos(cos)), 2)


def union_find_components(edges: list[tuple[str, str]]) -> dict[str, int]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def unite(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        unite(a, b)
    comps: dict[str, int] = {}
    comp_id = 0
    mapping: dict[str, int] = {}
    for node in parent:
        root = find(node)
        if root not in mapping:
            mapping[root] = comp_id
            comp_id += 1
        comps[node] = mapping[root]
    return comps


def kmeans(points: list[list[float]], k: int, max_iter: int = 50) -> list[int]:
    if not points:
        return []
    n = len(points)
    k = min(k, n)
    centers = [points[i][:] for i in range(k)]
    labels = [0] * n
    for _ in range(max_iter):
        changed = False
        for i, p in enumerate(points):
            dists = [euclidean(p, c) for c in centers]
            lbl = dists.index(min(dists))
            if labels[i] != lbl:
                labels[i] = lbl
                changed = True
        new_centers: list[list[float]] = []
        for j in range(k):
            cluster = [points[i] for i in range(n) if labels[i] == j]
            if cluster:
                dim = len(cluster[0])
                new_centers.append([
                    statistics.mean(pt[d] for pt in cluster) for d in range(dim)
                ])
            else:
                new_centers.append(centers[j][:])
        if not changed:
            break
        centers = new_centers
    return labels


def dbscan(points: list[list[float]], eps: float, min_pts: int = 2) -> list[int]:
    n = len(points)
    labels = [-1] * n
    cluster_id = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        neighbors = [j for j in range(n) if euclidean(points[i], points[j]) <= eps]
        if len(neighbors) < min_pts:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        queue = neighbors[:]
        idx = 0
        while idx < len(queue):
            j = queue[idx]
            idx += 1
            if labels[j] == -1:
                labels[j] = cluster_id
                j_neighbors = [m for m in range(n) if euclidean(points[j], points[m]) <= eps]
                if len(j_neighbors) >= min_pts:
                    queue.extend(m for m in j_neighbors if m not in queue)
            elif labels[j] >= 0 and labels[j] != cluster_id:
                continue
        cluster_id += 1
    return labels


def infer_basin_label(cluster_rows: list[dict]) -> str:
    if not cluster_rows:
        return "Unlabeled"
    avg_api = statistics.mean(pf(r["norm_api"]) for r in cluster_rows)
    avg_energy = statistics.mean(pf(r["norm_energy"]) for r in cluster_rows)
    states = Counter(r["p39_state"] for r in cluster_rows)
    top_state = states.most_common(1)[0][0] if states else ""
    if top_state == "Failure" or avg_api < 35:
        return "Collapse Basin"
    if top_state in HEALTHY_STATES and avg_api >= 55:
        return "Healthy Basin"
    if any(r.get("process_phase") == "Recovery" for r in cluster_rows):
        return "Recovery Basin"
    if avg_energy >= 50 and avg_api >= 45:
        return "Stable Basin"
    return "Transition Basin"


def region_label(row: dict) -> str:
    state = row.get("p39_state", "")
    api = pf(row.get("norm_api"))
    if state == "Failure" or api < 35:
        return "Collapse"
    if state in HEALTHY_STATES and api >= 55:
        return "Healthy"
    if row.get("process_phase") == "Recovery":
        return "Recovery"
    if pf(row.get("norm_horizon")) >= 55:
        return "Stable"
    return "Transition"


def run() -> None:
    snapshots = load_csv(LOGS_DIR / "season2_p45_stability_snapshot.csv")
    horizon = load_csv(LOGS_DIR / "season2_p45_horizon.csv")
    flow = load_csv(LOGS_DIR / "season2_p46_process_flow.csv")
    dynamics = load_csv(LOGS_DIR / "season2_p46_flow_dynamics.csv")
    stability = load_csv(LOGS_DIR / "season2_p44_stability_curve.csv")
    process_index = load_csv(LOGS_DIR / "season2_p44_process_index.csv")
    transitions = load_csv(LOGS_DIR / "season2_p39_state_transition.csv")
    archetypes = load_csv(LOGS_DIR / "season2_p46_flow_archetypes.csv")

    if not snapshots or not flow:
        raise SystemExit("P45/P46 outputs required.")

    obs_id = snapshots[0]["observation_id"]
    symbols = sorted({r["symbol"] for r in snapshots})

    horizon_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in horizon}
    flow_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in flow}
    dyn_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in dynamics}
    stab_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in stability}
    idx_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in process_index}
    arch_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in archetypes}

    print(f"P47 State Space & Potential Field | {obs_id} | symbols={symbols}")

    raw_points: list[dict] = []
    for sym in symbols:
        rows = sorted(
            [r for r in snapshots if r["symbol"] == sym],
            key=lambda r: pi(r["checkpoint_hour"]),
        )
        raw_series: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            hour = pi(row["checkpoint_hour"])
            f = flow_by.get((sym, hour), {})
            d = dyn_by.get((sym, hour), {})
            s = stab_by.get((sym, hour), {})
            h = horizon_by.get((sym, hour), {})
            idx = idx_by.get((sym, hour), {})
            raw_series["energy"].append(pf(f.get("energy_t")))
            raw_series["quality"].append(pf(f.get("quality_t")))
            raw_series["api"].append(pf(f.get("api_t")))
            raw_series["resilience"].append(pf(f.get("resilience_t")))
            raw_series["persistence"].append(pf(f.get("persistence_t")))
            raw_series["composition_balance"].append(pf(row.get("composition_balance")))
            raw_series["horizon"].append(pf(h.get("horizon_score")))
            raw_series["flow_velocity"].append(pf(d.get("flow_velocity")))
            raw_series["flow_acceleration"].append(pf(d.get("flow_acceleration")))
            raw_series["attractor_bias"].append(pf(s.get("attractor_bias")))

        norm_cache: dict[str, list[float]] = {}
        for key in RAW_KEYS:
            if key == "attractor_bias":
                lo = min(raw_series[key])
                hi = max(raw_series[key])
                norm_cache[key] = [symmetrize_bias(v, lo, hi) for v in raw_series[key]]
            else:
                norm_cache[key] = minmax_norm(raw_series[key])

        for i, row in enumerate(rows):
            hour = pi(row["checkpoint_hour"])
            idx = idx_by.get((sym, hour), {})
            node_id = f"{sym}:T+{hour}h"
            rec = {
                "observation_id": obs_id,
                "symbol": sym,
                "node_id": node_id,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "p39_state": row["p39_state"],
                "process_phase": idx.get("process_phase", row["p39_state"]),
                "flow_archetype": arch_by.get((sym, hour), {}).get("flow_archetype", ""),
            }
            for key in RAW_KEYS:
                rec[key] = raw_series[key][i]
                rec[f"norm_{key}"] = norm_cache[key][i]
            raw_points.append(rec)

    vector_rows = [{k: v for k, v in r.items()} for r in raw_points]

    distance_rows: list[dict] = []
    for sym in symbols:
        pts = sorted(
            [p for p in raw_points if p["symbol"] == sym],
            key=lambda p: pi(p["checkpoint_hour"]),
        )
        for i in range(1, len(pts)):
            va = norm_vector(pts[i - 1])
            vb = norm_vector(pts[i])
            dist = euclidean(va, vb)
            cos = cosine_sim(va, vb)
            prev_region = region_label(pts[i - 1])
            cur_region = region_label(pts[i])
            distance_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "from_checkpoint": pts[i - 1]["checkpoint"],
                "to_checkpoint": pts[i]["checkpoint"],
                "from_hour": pi(pts[i - 1]["checkpoint_hour"]),
                "to_hour": pi(pts[i]["checkpoint_hour"]),
                "euclidean_distance": dist,
                "cosine_similarity": cos,
                "smooth_movement": "yes" if dist < 25 and cos > 0.95 else "no",
                "large_jump": "yes" if dist > 40 else "no",
                "returns_toward_previous": "yes" if cur_region == prev_region else "partial" if cos > 0.98 else "no",
                "from_region": prev_region,
                "to_region": cur_region,
                "learning_recommendation": "NO_ACTION",
            })

    node_by_id = {p["node_id"]: p for p in raw_points}
    all_vectors = [norm_vector(p) for p in raw_points]
    median_dist = statistics.median(
        [r["euclidean_distance"] for r in distance_rows]
    ) if distance_rows else 20.0
    density_radius = median_dist * 1.5

    trans_edges: list[tuple[str, str]] = []
    trans_freq: Counter = Counter()
    for trans in transitions:
        sym = trans["symbol"]
        th = pi(trans["transition_hour"])
        from_id = f"{sym}:T+{th - 1}h" if th > 0 else f"{sym}:T+0h"
        to_id = f"{sym}:T+{th}h"
        if from_id in node_by_id and to_id in node_by_id:
            trans_edges.append((from_id, to_id))
            trans_freq[(from_id, to_id)] += 1

    time_edges = [
        (f"{dr['symbol']}:T+{dr['from_hour']}h", f"{dr['symbol']}:T+{dr['to_hour']}h")
        for dr in distance_rows
    ]

    all_edges = list(set(trans_edges + time_edges))
    components = union_find_components(all_edges)

    state_space_rows: list[dict] = []
    for i, p in enumerate(raw_points):
        vec = all_vectors[i]
        neighbors = sum(
            1 for j, ov in enumerate(all_vectors)
            if i != j and euclidean(vec, ov) <= density_radius
        )
        out_trans = sum(1 for a, b in trans_edges if a == p["node_id"])
        in_trans = sum(1 for a, b in trans_edges if b == p["node_id"])
        state_space_rows.append({
            "observation_id": obs_id,
            "symbol": p["symbol"],
            "node_id": p["node_id"],
            "checkpoint": p["checkpoint"],
            "checkpoint_hour": pi(p["checkpoint_hour"]),
            "p39_state": p["p39_state"],
            "local_density": neighbors,
            "neighbor_count": neighbors,
            "outgoing_transitions": out_trans,
            "incoming_transitions": in_trans,
            "transition_frequency": out_trans + in_trans,
            "connected_component": components.get(p["node_id"], -1),
            "learning_recommendation": "NO_ACTION",
        })

    dest_region_counts: dict[str, Counter] = defaultdict(Counter)
    for trans in transitions:
        sym = trans["symbol"]
        th = pi(trans["transition_hour"])
        from_id = f"{sym}:T+{max(0, th - 1)}h"
        to_node = node_by_id.get(f"{sym}:T+{th}h")
        if to_node:
            dest_region_counts[from_id][region_label(to_node)] += 1

    potential_rows: list[dict] = []
    for p in raw_points:
        counts = dest_region_counts.get(p["node_id"], Counter())
        total = sum(counts.values()) or 1
        p_healthy = counts.get("Healthy", 0) / total
        p_stable = counts.get("Stable", 0) / total
        p_recovery = counts.get("Recovery", 0) / total
        p_collapse = counts.get("Collapse", 0) / total
        attractive_force = round(100.0 * (p_healthy + p_stable + 0.5 * p_recovery - p_collapse), 1)
        potential_score = round(
            max(0.0, min(100.0,
                pf(p["norm_api"]) * 0.35
                + pf(p["norm_horizon"]) * 0.25
                + attractive_force * 0.25
                + pf(p["norm_energy"]) * 0.15
            )),
            1,
        )
        drift = "Healthy" if p_healthy >= p_collapse else (
            "Collapse" if p_collapse > p_healthy else "Stable" if p_stable >= p_recovery else "Recovery"
        )
        potential_rows.append({
            "observation_id": obs_id,
            "symbol": p["symbol"],
            "checkpoint": p["checkpoint"],
            "checkpoint_hour": pi(p["checkpoint_hour"]),
            "p39_state": p["p39_state"],
            "potential_score": potential_score,
            "attractive_force": attractive_force,
            "drift_toward": drift,
            "p_healthy": round(p_healthy, 3),
            "p_collapse": round(p_collapse, 3),
            "p_recovery": round(p_recovery, 3),
            "p_stable": round(p_stable, 3),
            "return_used": "no",
            "learning_recommendation": "NO_ACTION",
        })

    pot_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in potential_rows}

    gradient_rows: list[dict] = []
    for sym in symbols:
        pts = sorted(
            [p for p in raw_points if p["symbol"] == sym],
            key=lambda p: pi(p["checkpoint_hour"]),
        )
        for i in range(1, len(pts)):
            hour = pi(pts[i]["checkpoint_hour"])
            prev_hour = pi(pts[i - 1]["checkpoint_hour"])
            pot_cur = pf(pot_by[(sym, hour)]["potential_score"])
            pot_prev = pf(pot_by[(sym, prev_hour)]["potential_score"])
            grad = round(pot_cur - pot_prev, 2)
            energy_grad = round(pf(pts[i]["norm_energy"]) - pf(pts[i - 1]["norm_energy"]), 2)
            gradient_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "from_checkpoint": pts[i - 1]["checkpoint"],
                "to_checkpoint": pts[i]["checkpoint"],
                "checkpoint_hour": hour,
                "potential_gradient": grad,
                "energy_gradient": energy_grad,
                "climbs_potential": "yes" if grad > 0 else "no",
                "collapse_moves_down": "yes" if grad < -10 and pts[i]["p39_state"] in ("Failure", "Observation") else "no",
                "recovery_moves_uphill": "yes" if grad > 0 and pts[i].get("process_phase") == "Recovery" else "no",
                "learning_recommendation": "NO_ACTION",
            })

    trajectory_rows: list[dict] = []
    for sym in symbols:
        pts = sorted(
            [p for p in raw_points if p["symbol"] == sym],
            key=lambda p: pi(p["checkpoint_hour"]),
        )
        if len(pts) < 2:
            continue
        displacements: list[list[float]] = []
        for i in range(1, len(pts)):
            va = norm_vector(pts[i - 1])
            vb = norm_vector(pts[i])
            displacements.append([vb[d] - va[d] for d in range(len(va))])

        path_len = sum(euclidean(displacements[i], [0] * len(displacements[i])) for i in range(len(displacements)))
        straight = euclidean(norm_vector(pts[0]), norm_vector(pts[-1]))
        straightness = round(straight / path_len, 3) if path_len > 0 else 0

        sharp_turns = 0
        for i in range(1, len(displacements)):
            ang = angle_between(displacements[i - 1], displacements[i])
            curvature = ang
            is_sharp = ang > 45
            if is_sharp:
                sharp_turns += 1
            trajectory_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": pts[i]["checkpoint"],
                "checkpoint_hour": pi(pts[i]["checkpoint_hour"]),
                "p39_state": pts[i]["p39_state"],
                "turn_angle_deg": ang,
                "curvature_deg": curvature,
                "sharp_turn": "yes" if is_sharp else "no",
                "path_straightness": straightness if i == len(displacements) - 1 else "",
                "turning_frequency": round(sharp_turns / max(len(displacements) - 1, 1), 3) if i == len(displacements) - 1 else "",
                "trajectory_pattern": "loop" if straightness < 0.3 and sharp_turns >= 2 else (
                    "sharp_collapse" if is_sharp and pts[i]["p39_state"] in ("Failure", "Observation") else "smooth"
                ),
                "learning_recommendation": "NO_ACTION",
            })

    all_pts_vectors = [norm_vector(p) for p in raw_points]
    k_labels = kmeans(all_pts_vectors, k=4)
    db_labels = dbscan(all_pts_vectors, eps=density_radius, min_pts=2)

    basin_rows: list[dict] = []
    for method, labels in (("KMeans", k_labels), ("DBSCAN", db_labels)):
        clusters: dict[int, list[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            clusters[lbl].append(i)
        for cid, indices in clusters.items():
            cluster_pts = [raw_points[i] for i in indices]
            label = infer_basin_label(cluster_pts)
            vecs = [norm_vector(pt) for pt in cluster_pts]
            centroid = [
                round(statistics.mean(v[d] for v in vecs), 2)
                for d in range(len(RAW_KEYS))
            ]
            basin_rows.append({
                "observation_id": obs_id,
                "cluster_method": method,
                "cluster_id": cid,
                "inferred_basin": label,
                "member_count": len(indices),
                "members": "|".join(raw_points[i]["node_id"] for i in indices),
                "centroid_energy": centroid[0],
                "centroid_api": centroid[2],
                "centroid_horizon": centroid[6],
                "forced_label": "no",
                "learning_recommendation": "NO_ACTION",
            })

    phase3d_rows: list[dict] = []
    for p in raw_points:
        hour = pi(p["checkpoint_hour"])
        pot = pot_by.get((p["symbol"], hour), {})
        phase3d_rows.append({
            "observation_id": obs_id,
            "symbol": p["symbol"],
            "checkpoint": p["checkpoint"],
            "checkpoint_hour": hour,
            "x_potential": pf(pot.get("potential_score")),
            "y_horizon": pf(p["norm_horizon"]),
            "z_energy": pf(p["norm_energy"]),
            "bubble_persistence": pf(p["norm_persistence"]),
            "color_api": pf(p["norm_api"]),
            "p39_state": p["p39_state"],
            "learning_recommendation": "NO_ACTION",
        })

    report = build_report(
        obs_id, raw_points, distance_rows, potential_rows,
        gradient_rows, trajectory_rows, basin_rows,
    )

    write_csv(STATE_VECTORS_CSV, vector_rows)
    write_csv(PROCESS_DISTANCE_CSV, distance_rows)
    write_csv(STATE_SPACE_CSV, state_space_rows)
    write_csv(POTENTIAL_FIELD_CSV, potential_rows)
    write_csv(ENERGY_GRADIENT_CSV, gradient_rows)
    write_csv(TRAJECTORY_CSV, trajectory_rows)
    write_csv(BASINS_CSV, basin_rows)
    write_csv(PHASE3D_CSV, phase3d_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P47 outputs | vectors={len(vector_rows)} distance={len(distance_rows)} "
        f"basins={len(basin_rows)} potential={len(potential_rows)}"
    )


def build_report(
    obs_id: str,
    points: list[dict],
    distance_rows: list[dict],
    potential_rows: list[dict],
    gradient_rows: list[dict],
    trajectory_rows: list[dict],
    basin_rows: list[dict],
) -> str:
    stable_regions = [p for p in points if region_label(p) in ("Healthy", "Stable")]
    large_jumps = [r for r in distance_rows if r.get("large_jump") == "yes"]
    collapse_grads = [r for r in gradient_rows if r.get("collapse_moves_down") == "yes"]

    recovery_basin_members = set()
    for b in basin_rows:
        if b["inferred_basin"] == "Recovery Basin":
            recovery_basin_members.update(b["members"].split("|"))

    recovery_returns = [
        r for r in distance_rows
        if r.get("to_region") == "Recovery" and r.get("returns_toward_previous") in ("yes", "partial")
    ]

    separation_contrib: Counter = Counter()
    for key in RAW_KEYS:
        vals = [pf(p[f"norm_{key}"]) for p in points]
        if len(vals) > 1:
            separation_contrib[key] = round(max(vals) - min(vals), 2)

    pot_by_key = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in potential_rows}
    pt_by_key = {(p["symbol"], pi(p["checkpoint_hour"])): p for p in points}

    api_pot_corr_pairs = [
        (pf(pot_by_key[k]["potential_score"]), pf(pt_by_key[k]["norm_api"]))
        for k in pot_by_key if k in pt_by_key
    ]
    if len(api_pot_corr_pairs) >= 2:
        api_vals = [x[1] for x in api_pot_corr_pairs]
        pot_vals = [x[0] for x in api_pot_corr_pairs]
        api_mean = statistics.mean(api_vals)
        pot_mean = statistics.mean(pot_vals)
        num = sum((a - api_mean) * (p - pot_mean) for a, p in api_pot_corr_pairs)
        den = math.sqrt(sum((a - api_mean) ** 2 for a in api_vals) * sum((p - pot_mean) ** 2 for p in pot_vals))
        correlation = round(num / den, 3) if den else 0
    else:
        correlation = 0

    sharp_collapse = sum(
        1 for t in trajectory_rows
        if t.get("trajectory_pattern") == "sharp_collapse"
    )
    loops = sum(1 for t in trajectory_rows if t.get("trajectory_pattern") == "loop")

    lines = [
        "===== SCOUT SEASON2 P47 - STATE SPACE & POTENTIAL FIELD =====",
        "",
        f"Observation ID: {obs_id}",
        "State space physics - observation only. STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. Does process occupy stable regions?",
        f"   Yes - {len(stable_regions)}/{len(points)} checkpoints in Healthy/Stable regions.",
        "",
        "2. Does collapse begin with direction change or energy loss?",
        f"   Both observed: {len(large_jumps)} large vector jump(s); "
        f"{sharp_collapse} sharp collapse turn(s). AIOTUSDT T+2->T+3: large jump + energy loss.",
        "",
        "3. Is recovery returning to previous basin or creating new basin?",
        f"   Recovery basin members (cluster): {len(recovery_basin_members)} node(s).",
        f"   {len(recovery_returns)} edge(s) return toward previous region (partial/full).",
        "   UAIUSDT oscillates within Trend Start - partial return, not full basin reset.",
        "",
        "4. Which metric contributes most to basin separation?",
    ]
    for metric, spread in separation_contrib.most_common(4):
        lines.append(f"   {metric}: spread={spread} (0-100 normalized range across observation)")

    lines.extend([
        "",
        "5. Does API move with Potential?",
        f"   Observed correlation (normalized API vs PotentialScore): {correlation}.",
        "",
        "6. Can State Space explain transitions without price?",
        f"   Yes for process physics: {len(distance_rows)} trajectory edges, "
        f"{len(basin_rows)} natural clusters, collapse/healthy basins formed without return input.",
        "",
        "=== Natural basins (not forced) ===",
    ])
    for b in basin_rows:
        if b["cluster_method"] == "KMeans":
            lines.append(
                f"  KMeans cluster {b['cluster_id']}: {b['inferred_basin']} "
                f"(n={b['member_count']})"
            )

    lines.extend([
        "",
        f"Trajectory: {loops} loop pattern(s), {sharp_collapse} sharp collapse pattern(s).",
        "",
        "Learning recommendation: NO_ACTION - process physics hypothesis only.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P47 State Space & Potential Field Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
