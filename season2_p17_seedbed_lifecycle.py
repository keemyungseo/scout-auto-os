"""
Scout Learning Season2 - P17 Seedbed Lifecycle & Temporal Transition Layer

Temporal layer on P15 + P16. Tracks how situations and seedbeds evolve — not snapshots.
Optimizes for early recognition of empirical transitions, not prediction accuracy.

Governed by Scout Research Constitution and Scout Mission.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from season2_p10_situation_evolution import build_edges
from season2_p14_regime_memory_bank import build_expanded_records
from season2_p15_situation_output import enrich_record_stack
from season2_p16_opportunity_field import attach_forward, build_evaluations
from season2_p6_market_memory import attach_forward_targets
from season2_p7_behaviour_grammar import enrich_physics
from season2_p8_participant_state import enrich_panel_fields
from season2_p9_conditional_interaction_mining import prepare_records
from season2_regime_core import assign_regimes
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LIFECYCLE_CSV = LOGS_DIR / "season2_p17_seedbed_lifecycle.csv"
TRANSITIONS_CSV = LOGS_DIR / "season2_p17_seedbed_transitions.csv"
TRAJECTORIES_CSV = LOGS_DIR / "season2_p17_symbol_trajectories.csv"
PRE_EXPANSION_CSV = LOGS_DIR / "season2_p17_pre_expansion_signatures.csv"
PRE_EXHAUSTION_CSV = LOGS_DIR / "season2_p17_pre_exhaustion_signatures.csv"
FAKE_GENUINE_CSV = LOGS_DIR / "season2_p17_fake_vs_genuine_paths.csv"
OBSERVATIONS_CSV = LOGS_DIR / "season2_p17_temporal_observations.csv"
SIMILAR_PATHS_CSV = LOGS_DIR / "season2_p17_similar_paths.csv"
REPORT_TXT = LOGS_DIR / "season2_p17_research_report.txt"

MIN_N = 4
MIN_ARC = 2
FERTILE = {"Very fertile", "Fertile"}
EXPANSION_F6 = 3.0
EXHAUSTION_SITUATIONS = {"Distribution", "Late Trend", "Exhaustion"}


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def pf(val, default=None):
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


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


def scan_gap_hours(t1: str, t2: str) -> float:
    try:
        a = datetime.strptime(t1, "%Y-%m-%d %H:%M:%S")
        b = datetime.strptime(t2, "%Y-%m-%d %H:%M:%S")
        return (b - a).total_seconds() / 3600.0
    except ValueError:
        return 0.0


def symbol_set(symbols_str: str) -> set[str]:
    if not symbols_str:
        return set()
    return {s.strip() for s in symbols_str.split("|") if s.strip()}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def load_seedbeds() -> list[dict]:
    path = LOGS_DIR / "season2_p16_seedbed_quality.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if int(r.get("sample_size") or 0) >= MIN_ARC]


def load_cluster_members() -> dict[tuple[str, str], list[dict]]:
    path = LOGS_DIR / "season2_p16_field_clusters.csv"
    if not path.exists():
        return {}
    out: dict[tuple[str, str], list] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["cluster_id"], row["scan_time"])].append(row)
    return out


def link_seedbed_arcs(seedbeds: list[dict]) -> list[dict]:
    """Link seedbed clusters across consecutive scans by symbol overlap + structure."""
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for sb in seedbeds:
        by_scan[sb["scan_time"]].append(sb)

    scan_times = sorted(by_scan.keys())
    active: dict[str, dict] = {}
    completed: list[dict] = []
    arc_counter = 0

    for i, scan_time in enumerate(scan_times):
        current = by_scan[scan_time]
        prev_scan = scan_times[i - 1] if i > 0 else None
        next_active: dict[str, dict] = {}

        for sb in current:
            syms = symbol_set(sb.get("symbols", ""))
            sig = sb.get("structure_signature", "")
            best_arc = None
            best_score = 0.0

            if prev_scan:
                for arc_id, arc in active.items():
                    if arc.get("last_scan") != prev_scan:
                        continue
                    overlap = jaccard(syms, arc.get("symbols", set()))
                    prev_sig = arc.get("structure_signature", "")
                    sig_match = 1.0 if sig == prev_sig else 0.4 if sig.split("|")[0] == prev_sig.split("|")[0] else 0.0
                    score = overlap * 0.65 + sig_match * 0.35
                    if score > best_score and score >= 0.3:
                        best_score = score
                        best_arc = arc_id

            if best_arc:
                arc = active[best_arc]
                arc["steps"].append(sb)
                arc["last_scan"] = scan_time
                arc["symbols"] = syms or arc.get("symbols", set())
                arc["structure_signature"] = sig or arc.get("structure_signature", "")
                arc["length"] = len(arc["steps"])
                next_active[best_arc] = arc
            else:
                arc_counter += 1
                arc_id = f"arc_{arc_counter}"
                next_active[arc_id] = {
                    "arc_id": arc_id,
                    "birth_scan": scan_time,
                    "last_scan": scan_time,
                    "structure_signature": sig,
                    "symbols": syms,
                    "steps": [sb],
                    "length": 1,
                }

        for arc_id, arc in active.items():
            if arc_id not in next_active:
                completed.append(arc)
        active = next_active

    completed.extend(active.values())
    return completed


def fertile_streak(steps: list[dict]) -> int:
    streak = 0
    for sb in reversed(steps):
        if sb.get("seedbed_quality") in FERTILE:
            streak += 1
        else:
            break
    return streak


def total_fertile_scans(steps: list[dict]) -> int:
    return sum(1 for sb in steps if sb.get("seedbed_quality") in FERTILE)


def arc_lifecycle_rows(arcs: list[dict], field_index: dict) -> list[dict]:
    rows = []
    for arc in arcs:
        steps = arc["steps"]
        if len(steps) < MIN_ARC:
            continue

        qualities = [s.get("seedbed_quality", "Unknown") for s in steps]
        birth_q = qualities[0]
        current_q = qualities[-1]
        fertile_n = total_fertile_scans(steps)
        streak = fertile_streak(steps)

        birth_sig = "fertile_birth" if birth_q in FERTILE else "neutral_birth" if birth_q == "Neutral" else "other_birth"
        if birth_q not in FERTILE and any(q in FERTILE for q in qualities[1:]):
            birth_sig = "fertile_emergence"

        path = " -> ".join(qualities)
        first = steps[0]
        last = steps[-1]
        field_last = field_index.get(last.get("field_id", ""), {})

        rows.append(
            {
                "arc_id": arc["arc_id"],
                "birth_scan": arc["birth_scan"],
                "last_scan": arc["last_scan"],
                "arc_length_scans": len(steps),
                "fertile_scan_count": fertile_n,
                "fertile_streak": streak,
                "birth_quality": birth_q,
                "current_quality": current_q,
                "birth_signature": birth_sig,
                "quality_path": path,
                "structure_signature": arc.get("structure_signature", ""),
                "symbols": "|".join(sorted(arc.get("symbols", set()))[:8]),
                "dominant_situation_first": first.get("dominant_situation"),
                "dominant_situation_last": last.get("dominant_situation"),
                "real_avg_first": pf(first.get("real_avg")),
                "real_avg_last": pf(last.get("real_avg")),
                "fake_avg_first": pf(first.get("fake_avg")),
                "fake_avg_last": pf(last.get("fake_avg")),
                "pressure_first": pf(first.get("field_pressure_avg")),
                "pressure_last": pf(last.get("field_pressure_avg")),
                "expansion_field_last": field_last.get("expansion_field", "Unknown"),
                "confidence": "high" if len(steps) >= 4 and fertile_n >= 2 else "medium" if len(steps) >= 3 else "hypothesis",
            }
        )
    return rows


def seedbed_transition_matrix(arcs: list[dict]) -> list[dict]:
    counts: Counter[tuple[str, str]] = Counter()
    persist: Counter[tuple[str, str]] = Counter()
    fertile_dur: dict[str, list[int]] = defaultdict(list)

    for arc in arcs:
        steps = arc["steps"]
        for i in range(len(steps) - 1):
            fr = steps[i].get("seedbed_quality", "Unknown")
            to = steps[i + 1].get("seedbed_quality", "Unknown")
            counts[(fr, to)] += 1
            if fr == to:
                persist[(fr, to)] += 1

        streak = 0
        for sb in steps:
            if sb.get("seedbed_quality") in FERTILE:
                streak += 1
            else:
                if streak:
                    fertile_dur[steps[0].get("dominant_situation", "?")].append(streak)
                streak = 0
        if streak:
            fertile_dur[steps[-1].get("dominant_situation", "?")].append(streak)

    from_totals: Counter[str] = Counter()
    for (fr, _), c in counts.items():
        from_totals[fr] += c

    rows = []
    for (fr, to), count in counts.most_common():
        prob = round(count / from_totals[fr] * 100, 1) if from_totals[fr] else 0
        rows.append(
            {
                "from_quality": fr,
                "to_quality": to,
                "transition_count": count,
                "transition_probability_pct": prob,
                "persistence": "stable" if fr == to and prob >= 40 else "evolving",
                "confidence": "high" if count >= 8 else "medium" if count >= MIN_N else "hypothesis",
            }
        )

    for sit, durations in fertile_dur.items():
        if len(durations) < 2:
            continue
        rows.append(
            {
                "from_quality": f"fertile_duration_{sit}",
                "to_quality": "median_scans",
                "transition_count": len(durations),
                "transition_probability_pct": round(statistics.median(durations), 1),
                "persistence": "fertility_duration",
                "confidence": "medium" if len(durations) >= 4 else "hypothesis",
            }
        )
    return rows


def symbol_temporal_trajectories(evaluations: list[dict], cluster_rows: list[dict]) -> list[dict]:
    sym_seedbed: dict[tuple[str, str], str] = {}
    for c in cluster_rows:
        sym_seedbed[(c["scan_time"], c["symbol"])] = c.get("seedbed_quality", "Unknown")

    edges = build_edges(evaluations)
    trajectories = []

    for e in edges:
        cur, nxt = e["from_record"], e["to_record"]
        key_fr = (cur["scan_time"], cur["symbol"])
        key_to = (nxt["scan_time"], nxt["symbol"])

        real_d = pf(nxt.get("real_trend_score"), 0) - pf(cur.get("real_trend_score"), 0)
        fake_d = pf(nxt.get("fake_trend_score"), 0) - pf(cur.get("fake_trend_score"), 0)
        press_d = pf(nxt.get("pressure_score"), 0) - pf(cur.get("pressure_score"), 0)

        sit_fr = cur.get("situation", "Unknown")
        sit_to = nxt.get("situation", "Unknown")
        change = "stable" if sit_fr == sit_to else "transitioning"

        trajectories.append(
            {
                "symbol": cur["symbol"],
                "from_scan": cur["scan_time"],
                "to_scan": nxt["scan_time"],
                "hours_gap": round(scan_gap_hours(cur["scan_time"], nxt["scan_time"]), 1),
                "from_situation": sit_fr,
                "to_situation": sit_to,
                "from_seedbed": sym_seedbed.get(key_fr, "Unknown"),
                "to_seedbed": sym_seedbed.get(key_to, "Unknown"),
                "real_delta": round(real_d, 1),
                "fake_delta": round(fake_d, 1),
                "pressure_delta": round(press_d, 1),
                "situation_change": change,
                "forward_6h_at_from": pf(cur.get("forward_6h")),
                "path_signature": f"{sit_fr}->{sit_to}",
            }
        )
    return trajectories


def pre_expansion_signatures(trajectories: list[dict]) -> list[dict]:
    """Empirical signatures before expansion — not price forecast."""
    expansion_steps = [t for t in trajectories if pf(t.get("forward_6h_at_from")) is not None and pf(t["forward_6h_at_from"]) >= EXPANSION_F6]
    non_exp = [t for t in trajectories if pf(t.get("forward_6h_at_from")) is not None and pf(t["forward_6h_at_from"]) < EXPANSION_F6]

    rows = []
    if expansion_steps:
        rows.append(
            {
                "signature_type": "pre_expansion_aggregate",
                "expansion_n": len(expansion_steps),
                "non_expansion_n": len(non_exp),
                "avg_real_delta": round(statistics.mean(t["real_delta"] for t in expansion_steps), 2),
                "avg_fake_delta": round(statistics.mean(t["fake_delta"] for t in expansion_steps), 2),
                "avg_pressure_delta": round(statistics.mean(t["pressure_delta"] for t in expansion_steps), 2),
                "from_situations": str(Counter(t["from_situation"] for t in expansion_steps).most_common(3)),
                "to_situations": str(Counter(t["to_situation"] for t in expansion_steps).most_common(3)),
                "verdict": "conditional" if len(expansion_steps) >= MIN_N else "Unknown",
            }
        )
        for sit, _ in Counter(t["from_situation"] for t in expansion_steps).most_common(5):
            subset = [t for t in expansion_steps if t["from_situation"] == sit]
            if len(subset) < MIN_N:
                continue
            rows.append(
                {
                    "signature_type": "pre_expansion_by_situation",
                    "from_situation": sit,
                    "sample_size": len(subset),
                    "avg_real_delta": round(statistics.mean(t["real_delta"] for t in subset), 2),
                    "avg_pressure_delta": round(statistics.mean(t["pressure_delta"] for t in subset), 2),
                    "dominant_to_situation": Counter(t["to_situation"] for t in subset).most_common(1)[0][0],
                    "verdict": "empirical_signature",
                }
            )
    return rows


def pre_exhaustion_signatures(trajectories: list[dict]) -> list[dict]:
    exhaustion_trans = [
        t for t in trajectories
        if t["to_situation"] in EXHAUSTION_SITUATIONS or t["to_situation"] == "Distribution"
    ]
    rows = []
    if not exhaustion_trans:
        return rows

    rows.append(
        {
            "signature_type": "pre_exhaustion_aggregate",
            "sample_size": len(exhaustion_trans),
            "avg_fake_delta": round(statistics.mean(t["fake_delta"] for t in exhaustion_trans), 2),
            "avg_pressure_delta": round(statistics.mean(t["pressure_delta"] for t in exhaustion_trans), 2),
            "from_situations": str(Counter(t["from_situation"] for t in exhaustion_trans).most_common(4)),
            "verdict": "conditional" if len(exhaustion_trans) >= MIN_N else "Unknown",
        }
    )
    for path, grp in Counter(t["path_signature"] for t in exhaustion_trans).most_common(8):
        members = [t for t in exhaustion_trans if t["path_signature"] == path]
        if len(members) < MIN_N:
            continue
        rows.append(
            {
                "signature_type": "exhaustion_path",
                "path_signature": path,
                "sample_size": len(members),
                "avg_fake_delta": round(statistics.mean(t["fake_delta"] for t in members), 2),
                "avg_pressure_delta": round(statistics.mean(t["pressure_delta"] for t in members), 2),
                "verdict": "empirical_signature",
            }
        )
    return rows


def fake_vs_genuine_paths(lifecycle_rows: list[dict]) -> list[dict]:
    rows = []
    for arc in lifecycle_rows:
        real_d = pf(arc.get("real_avg_last"), 0) - pf(arc.get("real_avg_first"), 0)
        fake_d = pf(arc.get("fake_avg_last"), 0) - pf(arc.get("fake_avg_first"), 0)
        press_d = pf(arc.get("pressure_last"), 0) - pf(arc.get("pressure_first"), 0)

        if fake_d > real_d + 15 and arc.get("current_quality") in ("Exhausted", "Dangerous"):
            path_type = "fake_seedbed_arc"
        elif real_d >= 0 and fake_d < 10 and arc.get("fertile_scan_count", 0) >= 2:
            path_type = "genuine_seedbed_arc"
        elif arc.get("fertile_scan_count", 0) >= 1 and arc.get("current_quality") in FERTILE:
            path_type = "ongoing_fertile_arc"
        else:
            path_type = "ambiguous_arc"

        rows.append(
            {
                "arc_id": arc["arc_id"],
                "path_type": path_type,
                "arc_length_scans": arc["arc_length_scans"],
                "fertile_scan_count": arc["fertile_scan_count"],
                "quality_path": arc["quality_path"],
                "real_delta": round(real_d, 1) if real_d is not None else "",
                "fake_delta": round(fake_d, 1) if fake_d is not None else "",
                "pressure_delta": round(press_d, 1) if press_d is not None else "",
                "confidence": arc.get("confidence", "hypothesis"),
            }
        )
    return rows


def temporal_observations(
    evaluations: list[dict],
    trajectories: list[dict],
    lifecycle_rows: list[dict],
) -> list[dict]:
    """Each row: where from, where now, how changing, similar paths."""
    path_counts = Counter(t["path_signature"] for t in trajectories)
    arc_by_symbol: dict[str, list] = defaultdict(list)
    for arc in lifecycle_rows:
        for sym in (arc.get("symbols") or "").split("|"):
            if sym:
                arc_by_symbol[sym].append(arc)

    obs = []
    for e in evaluations[-60:]:
        sym = e["symbol"]
        sym_traj = [t for t in trajectories if t["symbol"] == sym and t["to_scan"] == e["scan_time"]]
        prev = sym_traj[0] if sym_traj else None

        where_from = prev["from_situation"] if prev else "Unknown"
        where_now = e.get("situation", "Unknown")
        if prev:
            changing = f"situation {prev['from_situation']}->{prev['to_situation']}; real_d={prev['real_delta']}; press_d={prev['pressure_delta']}"
        else:
            changing = "Unknown"

        path_sig = prev["path_signature"] if prev else where_now
        similar_count = path_counts.get(path_sig, 0)
        confidence = "unknown"
        if similar_count >= 12:
            confidence = "high"
        elif similar_count >= 5:
            confidence = "medium"
        elif similar_count >= 2:
            confidence = "hypothesis"

        arcs = arc_by_symbol.get(sym, [])
        arc_note = arcs[0]["quality_path"] if arcs else "no_arc"

        obs.append(
            {
                "date": e["date"],
                "symbol": sym,
                "scan_time": e["scan_time"],
                "where_from": where_from,
                "where_now": where_now,
                "how_changing": changing,
                "similar_path": path_sig,
                "similar_path_count": similar_count,
                "seedbed_arc_path": arc_note,
                "trend_maturity": e.get("trend_maturity"),
                "real_trend_score": e.get("real_trend_score"),
                "fake_trend_score": e.get("fake_trend_score"),
                "confidence": confidence,
            }
        )
    return obs


def similar_path_library(trajectories: list[dict]) -> list[dict]:
    groups: dict[str, list] = defaultdict(list)
    for t in trajectories:
        groups[t["path_signature"]].append(t)

    rows = []
    for path, grp in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(grp) < MIN_N:
            continue
        f6 = [pf(t["forward_6h_at_from"]) for t in grp if pf(t.get("forward_6h_at_from")) is not None]
        rows.append(
            {
                "path_signature": path,
                "observation_count": len(grp),
                "median_forward_6h": round(statistics.median(f6), 2) if f6 else "Unknown",
                "avg_real_delta": round(statistics.mean(t["real_delta"] for t in grp), 2),
                "avg_fake_delta": round(statistics.mean(t["fake_delta"] for t in grp), 2),
                "avg_pressure_delta": round(statistics.mean(t["pressure_delta"] for t in grp), 2),
                "confidence": "high" if len(grp) >= 15 else "medium" if len(grp) >= MIN_N else "hypothesis",
                "note": "repeated empirical path — not a forecast",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Recompute P15 evaluations")
    args = parser.parse_args()

    records = build_expanded_records()
    enrich_physics(records)
    enrich_panel_fields(records)
    attach_forward_targets(records)
    prepare_records(records)
    enrich_record_stack(records)
    assign_regimes(records)

    evaluations = build_evaluations(records, use_cache=not args.refresh)
    attach_forward(records, evaluations)

    seedbeds = load_seedbeds()
    if not seedbeds:
        from season2_p16_opportunity_field import main as run_p16
        run_p16()
        seedbeds = load_seedbeds()

    cluster_path = LOGS_DIR / "season2_p16_field_clusters.csv"
    cluster_rows = list(csv.DictReader(cluster_path.open(encoding="utf-8"))) if cluster_path.exists() else []

    field_rows = []
    fields_path = LOGS_DIR / "season2_p16_opportunity_fields.csv"
    if fields_path.exists():
        field_rows = list(csv.DictReader(fields_path.open(encoding="utf-8")))
    field_index = {f["field_id"]: f for f in field_rows}

    arcs = link_seedbed_arcs(seedbeds)
    lifecycle = arc_lifecycle_rows(arcs, field_index)
    transitions = seedbed_transition_matrix(arcs)
    trajectories = symbol_temporal_trajectories(evaluations, cluster_rows)
    pre_exp = pre_expansion_signatures(trajectories)
    pre_exh = pre_exhaustion_signatures(trajectories)
    fake_genuine = fake_vs_genuine_paths(lifecycle)
    observations = temporal_observations(evaluations, trajectories, lifecycle)
    similar = similar_path_library(trajectories)

    write_csv(LIFECYCLE_CSV, lifecycle)
    write_csv(TRANSITIONS_CSV, transitions)
    write_csv(TRAJECTORIES_CSV, trajectories)
    write_csv(PRE_EXPANSION_CSV, pre_exp)
    write_csv(PRE_EXHAUSTION_CSV, pre_exh)
    write_csv(FAKE_GENUINE_CSV, fake_genuine)
    write_csv(OBSERVATIONS_CSV, observations)
    write_csv(SIMILAR_PATHS_CSV, similar)

    fertile_arcs = [a for a in lifecycle if a.get("fertile_scan_count", 0) >= 1]
    genuine = sum(1 for r in fake_genuine if r["path_type"] == "genuine_seedbed_arc")
    fake = sum(1 for r in fake_genuine if r["path_type"] == "fake_seedbed_arc")
    birth_emergence = sum(1 for a in lifecycle if a.get("birth_signature") == "fertile_emergence")

    lines = [
        "===== SCOUT SEASON2 P17 - SEEDBED LIFECYCLE & TEMPORAL LAYER =====",
        "",
        f"Symbol trajectories: {len(trajectories)} | Seedbed arcs: {len(arcs)} | Lifecycle arcs: {len(lifecycle)}",
        f"Fertile arcs: {len(fertile_arcs)} | Fertile births (emergence): {birth_emergence}",
        f"Genuine paths: {genuine} | Fake paths: {fake}",
        "",
        "Objective: early recognition of transitions — not price prediction",
        "",
        "--- How fertile seedbeds are born ---",
    ]
    births = Counter(a.get("birth_signature") for a in lifecycle)
    for sig, n in births.most_common():
        lines.append(f"  {sig}: {n}")

    dur_rows = [r for r in transitions if r.get("persistence") == "fertility_duration"]
    lines.extend(["", "--- Fertile duration (median scans) ---"])
    for r in dur_rows[:5]:
        lines.append(f"  {r['from_quality']}: {r['transition_probability_pct']} scans n={r['transition_count']}")

    lines.extend(["", "--- Pre-expansion signatures ---"])
    if pre_exp:
        agg = pre_exp[0]
        lines.append(f"  expansion_n={agg.get('expansion_n')} avg_press_d={agg.get('avg_pressure_delta')}")

    lines.extend(["", "--- Pre-exhaustion paths ---"])
    for r in pre_exh[1:4]:
        lines.append(f"  {r.get('path_signature')}: n={r.get('sample_size')} fake_d={r.get('avg_fake_delta')}")

    lines.extend(["", "--- Similar empirical paths ---"])
    for r in similar[:5]:
        lines.append(f"  {r['path_signature']}: n={r['observation_count']} conf={r['confidence']}")

    lines.extend([
        "",
        "Every observation: where_from | where_now | how_changing | similar_path_count",
        "Confidence increases only through repeated evidence across time",
    ])
    lines.extend(mission_summary_lines())
    lines.extend(["", f"Observations: {OBSERVATIONS_CSV}", "=" * 58])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P17 SEEDBED LIFECYCLE =====")
    print(f"Trajectories: {len(trajectories)} | Arcs: {len(lifecycle)} | Observations: {len(observations)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
