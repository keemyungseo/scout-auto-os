"""
Scout Learning Season2 - P10 Situation Evolution Engine & Future Context Discovery

Research only. Learn how situations evolve — not static price prediction.
"""

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p5_historical_expansion import (
    STEP1_DATES,
    build_unified_records,
    collect_missing,
    global_baseline_mae,
)
from season2_p6_market_memory import attach_forward_targets
from season2_p7_behaviour_grammar import enrich_physics
from season2_p8_participant_state import enrich_panel_fields
from season2_p9_conditional_interaction_mining import (
    load_participant_families,
    prepare_records,
)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SITUATION_TRANSITION_CSV = LOGS_DIR / "season2_p10_situation_transitions.csv"
PARTICIPANT_TRANSITION_CSV = LOGS_DIR / "season2_p10_participant_transitions.csv"
GRAMMAR_TRANSITION_CSV = LOGS_DIR / "season2_p10_grammar_transitions.csv"
EVOLUTION_FAMILY_CSV = LOGS_DIR / "season2_p10_evolution_families.csv"
CONDITIONAL_EVOLUTION_CSV = LOGS_DIR / "season2_p10_conditional_evolution.csv"
REGISTRY_CSV = LOGS_DIR / "season2_p10_evolution_registry.csv"
ENGINE_CSV = LOGS_DIR / "season2_p10_evolution_engine_output.csv"
REPORT_TXT = LOGS_DIR / "season2_p10_research_report.txt"

MIN_EDGE = 5
P9_INTERACTIONS = LOGS_DIR / "season2_p9_interactions.csv"
P9_REGISTRY = LOGS_DIR / "season2_p9_hypothesis_registry.csv"


def load_interactions() -> list[dict]:
    if not P9_INTERACTIONS.exists():
        return []
    with P9_INTERACTIONS.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fallback_situation(record: dict) -> str:
    supply = (record.get("supply_label") or "unknown").upper()
    ps = record.get("participant_state") or ""
    cluster = record.get("trend_cluster") or ""

    if supply == "MID_SUPPLY":
        return "healthy_trend"
    if ps == "late_chasing" or cluster == "late_trend":
        return "late_chasing_zone"
    if ps in ("scan_compression",) or cluster == "accumulation":
        return "early_accumulation"
    if ps == "scan_exhaustion" or cluster == "distribution":
        return "distribution"
    if record.get("collapse_label") == "YES" or supply == "COLLAPSE":
        return "forced_liquidation_zone"
    if cluster == "explosive":
        return "explosive_continuation"
    if cluster == "rotation":
        return "rotation_chop"
    return "unclassified_situation"


def match_situation(record: dict, interactions: list[dict]) -> dict | None:
    best = None
    best_score = -1.0
    for row in interactions:
        parts = []
        for segment in row.get("situation_key", "").split(" + "):
            if "=" in segment:
                d, v = segment.split("=", 1)
                parts.append((d.strip(), v.strip()))
        if not parts:
            continue
        from season2_p9_conditional_interaction_mining import filter_combo
        if len(filter_combo([record], tuple(parts))) == 1:
            score = float(row.get("information_score") or 0)
            if score > best_score:
                best_score = score
                best = row
    return best


def label_records(records: list[dict], interactions: list[dict]) -> None:
    top = interactions[:80] if interactions else []
    for record in records:
        match = match_situation(record, top) if top else None
        if match:
            record["situation_archetype"] = match["situation_description"]
            record["situation_key"] = match["situation_key"][:120]
            record["situation_confidence"] = match["confidence"]
        else:
            record["situation_archetype"] = fallback_situation(record)
            record["situation_key"] = ""
            record["situation_confidence"] = "hypothesis"


def build_edges(records: list[dict]) -> list[dict]:
    by_sym: dict[str, list] = defaultdict(list)
    for r in records:
        by_sym[r["symbol"]].append(r)
    edges: list[dict] = []
    for sym, group in by_sym.items():
        group.sort(key=lambda x: x["scan_time"])
        for i in range(len(group) - 1):
            cur, nxt = group[i], group[i + 1]
            edges.append(
                {
                    "symbol": sym,
                    "from_record": cur,
                    "to_record": nxt,
                    "same_date": cur["date"] == nxt["date"],
                    "hours_gap": _scan_gap_hours(cur["scan_time"], nxt["scan_time"]),
                }
            )
    return edges


def _scan_gap_hours(t1: str, t2: str) -> float:
    try:
        from datetime import datetime
        a = datetime.strptime(t1, "%Y-%m-%d %H:%M:%S")
        b = datetime.strptime(t2, "%Y-%m-%d %H:%M:%S")
        return (b - a).total_seconds() / 3600.0
    except ValueError:
        return 0.0


def transition_matrix(
    edges: list[dict],
    from_field: str,
    to_field: str,
    cross_day_only: bool = False,
) -> list[dict]:
    filtered = [e for e in edges if not cross_day_only or not e["same_date"]]
    if not filtered:
        filtered = edges

    counts: Counter[tuple[str, str]] = Counter()
    f6_from: dict[tuple[str, str], list] = defaultdict(list)
    f6_to: dict[tuple[str, str], list] = defaultdict(list)
    collapse_to: Counter[tuple[str, str]] = Counter()

    for e in filtered:
        fr = e["from_record"].get(from_field) or "unknown"
        to = e["to_record"].get(to_field) or "unknown"
        if fr == "unknown" or to == "unknown":
            continue
        key = (fr, to)
        counts[key] += 1
        if e["from_record"].get("target_f6") is not None:
            f6_from[key].append(e["from_record"]["target_f6"])
        if e["to_record"].get("target_f6") is not None:
            f6_to[key].append(e["to_record"]["target_f6"])
        if e["to_record"].get("collapse_label") == "YES":
            collapse_to[key] += 1

    from_totals: Counter[str] = Counter()
    for (fr, _), c in counts.items():
        from_totals[fr] += c

    rows = []
    for (fr, to), count in counts.most_common():
        if count < MIN_EDGE and from_totals[fr] < MIN_EDGE * 2:
            continue
        prob = round(count / from_totals[fr] * 100, 1) if from_totals[fr] else 0
        stable = "stable" if fr == to and prob >= 40 else "evolving" if prob >= 20 else "fragile"
        f6f = f6_from.get((fr, to), [])
        f6t = f6_to.get((fr, to), [])
        rows.append(
            {
                "from_state": fr,
                "to_state": to,
                "transition_count": count,
                "transition_probability_pct": prob,
                "from_total": from_totals[fr],
                "stability": stable,
                "median_return_at_from": round(statistics.median(f6f), 2) if f6f else "",
                "median_return_at_to": round(statistics.median(f6t), 2) if f6t else "",
                "collapse_rate_at_to_pct": round(collapse_to.get((fr, to), 0) / count * 100, 1),
                "confidence": "high" if count >= 15 else "medium" if count >= MIN_EDGE else "hypothesis",
            }
        )
    return rows


def persistence_stats(edges: list[dict], field: str) -> list[dict]:
    """State persistence: P(stay same) per state."""
    counts: Counter[str] = Counter()
    stay: Counter[str] = Counter()
    for e in edges:
        fr = e["from_record"].get(field) or "unknown"
        to = e["to_record"].get(field) or "unknown"
        counts[fr] += 1
        if fr == to:
            stay[fr] += 1
    rows = []
    for state, total in counts.most_common():
        if total < MIN_EDGE:
            continue
        rows.append(
            {
                "state": state,
                "persistence_pct": round(stay[state] / total * 100, 1),
                "sample_transitions": total,
                "confidence": "high" if total >= 20 else "medium" if total >= MIN_EDGE else "hypothesis",
            }
        )
    return rows


def next_state_lookup(matrix: list[dict], from_state: str) -> dict | None:
    candidates = [r for r in matrix if r["from_state"] == from_state]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x["transition_probability_pct"])


def evolution_families(edges: list[dict]) -> list[dict]:
    """Cluster symbols by situation transition path similarity."""
    sym_paths: dict[str, list[str]] = defaultdict(list)
    sym_meta: dict[str, list] = defaultdict(list)

    for e in edges:
        sym = e["symbol"]
        sym_paths[sym].append(f"{e['from_record'].get('situation_archetype')}->{e['to_record'].get('situation_archetype')}")
        sym_meta[sym].append(e["from_record"])

    def path_vec(sym: str) -> Counter:
        return Counter(sym_paths[sym])

    symbols = list(sym_paths.keys())
    assigned: set[str] = set()
    families: dict[str, list[str]] = {}

    for sym in sorted(symbols, key=lambda s: -len(sym_paths[s])):
        if sym in assigned:
            continue
        fid = f"evolution_family_{len(families) + 1}"
        families[fid] = [sym]
        assigned.add(sym)
        va = path_vec(sym)
        for other in symbols:
            if other in assigned:
                continue
            vb = path_vec(other)
            keys = set(va) | set(vb)
            dot = sum(va.get(k, 0) * vb.get(k, 0) for k in keys)
            na = math.sqrt(sum(v * v for v in va.values()))
            nb = math.sqrt(sum(v * v for v in vb.values()))
            sim = dot / (na * nb) if na and nb else 0
            if sim >= 0.5 and len(sym_paths[other]) >= 2:
                families[fid].append(other)
                assigned.add(other)

    rows = []
    for fid, members in sorted(families.items(), key=lambda x: -len(x[1])):
        all_paths: Counter = Counter()
        for sym in members:
            all_paths.update(sym_paths[sym])
        top_paths = all_paths.most_common(3)
        recs = [r for sym in members for r in sym_meta[sym]]
        f6 = [r.get("target_f6") for r in recs if r.get("target_f6") is not None]
        collapses = sum(1 for r in recs if r.get("collapse_label") == "YES")
        rows.append(
            {
                "family_id": fid,
                "symbols": "|".join(members[:12]),
                "symbol_count": len(members),
                "dominant_path": top_paths[0][0] if top_paths else "",
                "path_frequency": top_paths[0][1] if top_paths else 0,
                "top_paths": "|".join(f"{p}({c})" for p, c in top_paths),
                "median_forward_6h": round(statistics.median(f6), 2) if f6 else "",
                "collapse_rate_pct": round(collapses / len(recs) * 100, 1) if recs else 0,
            }
        )
    return rows


def mine_conditional_evolution(edges: list[dict]) -> list[dict]:
    """Situation + context -> next situation rules."""
    rules: Counter[tuple] = Counter()
    metrics: dict[tuple, list] = defaultdict(list)
    collapse: Counter[tuple] = Counter()

    for e in edges:
        cur = e["from_record"]
        nxt = e["to_record"]
        ctx = (
            cur.get("situation_archetype"),
            cur.get("supply_label"),
            cur.get("memory"),
            cur.get("participant_state"),
        )
        to_sit = nxt.get("situation_archetype")
        key = (*ctx, to_sit)
        rules[key] += 1
        if nxt.get("target_f6") is not None:
            metrics[key].append(nxt["target_f6"])
        if nxt.get("collapse_label") == "YES":
            collapse[key] += 1

    ctx_totals: Counter[tuple] = Counter()
    for key, c in rules.items():
        ctx_totals[key[:4]] += c

    rows = []
    for key, count in rules.most_common():
        if count < MIN_EDGE:
            continue
        ctx_key = key[:4]
        prob = round(count / ctx_totals[ctx_key] * 100, 1) if ctx_totals[ctx_key] else 0
        f6 = metrics.get(key, [])
        med = round(statistics.median(f6), 2) if f6 else ""
        coll = round(collapse[key] / count * 100, 1)
        from_sit, supply, mem, ps = ctx_key
        to_sit = key[4]
        rule_id = f"{from_sit}+{supply}+{mem}->{to_sit}"
        gain_flag = prob >= 25 and count >= MIN_EDGE
        status = "ACTIVE" if prob >= 40 and count >= 12 else "CONDITIONAL" if gain_flag else "RETIRED" if prob >= 15 else "DEAD_FOR_NOW"

        rows.append(
            {
                "evolution_rule": rule_id,
                "from_situation": from_sit,
                "context_supply": supply,
                "context_memory": mem,
                "context_participant_state": ps,
                "to_situation": to_sit,
                "transition_probability_pct": prob,
                "sample_size": count,
                "expected_return_at_next": med,
                "collapse_at_next_pct": coll,
                "confidence": "high" if count >= 20 else "medium" if count >= MIN_EDGE else "hypothesis",
                "status": status,
            }
        )
    return sorted(rows, key=lambda x: (-x["transition_probability_pct"], -x["sample_size"]))


def evolution_registry(
    sit_matrix: list[dict],
    cond_rules: list[dict],
    p9_registry: list[dict],
) -> list[dict]:
    rows = []
    for rule in cond_rules[:50]:
        rows.append(
            {
                "item_type": "evolution_rule",
                "item_name": rule["evolution_rule"],
                "prior_status": "OPEN",
                "current_status": rule["status"],
                "sample_size": rule["sample_size"],
                "transition_probability_pct": rule["transition_probability_pct"],
                "decision_reason": f"conditional evolution {rule['from_situation']} -> {rule['to_situation']}",
            }
        )
    for tr in sit_matrix:
        if tr["transition_count"] < MIN_EDGE:
            continue
        name = f"{tr['from_state']}->{tr['to_state']}"
        status = "ACTIVE" if tr["transition_probability_pct"] >= 35 and tr["transition_count"] >= 12 else "CONDITIONAL" if tr["transition_probability_pct"] >= 20 else "RETIRED" if tr["transition_count"] >= MIN_EDGE else "DEAD_FOR_NOW"
        rows.append(
            {
                "item_type": "situation_transition",
                "item_name": name,
                "prior_status": "OPEN",
                "current_status": status,
                "sample_size": tr["transition_count"],
                "transition_probability_pct": tr["transition_probability_pct"],
                "decision_reason": f"stability={tr['stability']} collapse_to={tr['collapse_rate_at_to_pct']}%",
            }
        )
    if p9_registry:
        for item in p9_registry[:30]:
            rows.append(
                {
                    "item_type": "p9_carryforward",
                    "item_name": item.get("hypothesis_name", item.get("item_name", "")),
                    "prior_status": item.get("current_status", "OPEN"),
                    "current_status": item.get("current_status", "OPEN"),
                    "sample_size": item.get("global_sample", item.get("sample_size", "")),
                    "transition_probability_pct": "",
                    "decision_reason": "carried from P9; retest on evolution batch",
                }
            )
    return rows


def evolution_action(
    current_situation: str,
    next_situation: str,
    trans_prob: float,
    collapse_next: float,
    med_next: float,
) -> tuple[str, str]:
    if next_situation in ("forced_liquidation_zone", "distribution") and trans_prob >= 25:
        return "Reduce", f"evolving toward {next_situation} ({trans_prob}% prob)"
    if collapse_next >= 30:
        return "Avoid", f"next situation collapse risk {collapse_next}%"
    if current_situation == "early_accumulation" and next_situation == "healthy_trend":
        return "Buy", "accumulation evolving to healthy trend"
    if current_situation == "healthy_trend" and next_situation == "healthy_trend" and trans_prob >= 40:
        return "Hold", "situation persisting healthily"
    if current_situation == "healthy_trend" and next_situation == "late_chasing_zone":
        return "Watch", "trend maturing into late chase phase"
    if next_situation == "late_chasing_zone" and med_next >= 4:
        return "Buy", f"late chase still productive med={med_next}%"
    if med_next >= 8 and collapse_next < 10:
        return "Strong Buy", f"next situation expectancy med={med_next}%"
    if med_next >= 3 and collapse_next < 15:
        return "Buy", f"positive evolution expectancy"
    return "Watch", f"likely next: {next_situation} ({trans_prob}%)"


def engine_output(
    records: list[dict],
    sit_matrix: list[dict],
    part_matrix: list[dict],
    gram_matrix: list[dict],
) -> list[dict]:
    rows = []
    for record in records[-40:]:
        cur_sit = record.get("situation_archetype", "unclassified_situation")
        cur_ps = record.get("participant_state", "")
        cur_gram = record.get("grammar", "")

        next_sit_row = next_state_lookup(sit_matrix, cur_sit)
        next_ps_row = next_state_lookup(part_matrix, cur_ps)
        next_gram_row = next_state_lookup(gram_matrix, cur_gram)

        next_sit = next_sit_row["to_state"] if next_sit_row else "unknown"
        trans_prob = next_sit_row["transition_probability_pct"] if next_sit_row else 0
        collapse_next = next_sit_row["collapse_rate_at_to_pct"] if next_sit_row else 0
        med_next = next_sit_row["median_return_at_to"] if next_sit_row and next_sit_row["median_return_at_to"] != "" else 0
        try:
            med_next = float(med_next)
        except (TypeError, ValueError):
            med_next = 0

        persist = next_sit_row["transition_probability_pct"] if next_sit_row and next_sit == cur_sit else 100 - trans_prob if next_sit_row else 50
        action, reason = evolution_action(cur_sit, next_sit, trans_prob, collapse_next, med_next)

        rows.append(
            {
                "date": record["date"],
                "symbol": record["symbol"],
                "scan_time": record["scan_time"],
                "current_situation": cur_sit,
                "current_participant_state": cur_ps,
                "current_grammar": cur_gram,
                "current_memory": record.get("memory"),
                "symbol_family": record.get("symbol_family"),
                "most_likely_next_situation": next_sit,
                "transition_probability_pct": trans_prob,
                "most_likely_next_participant_state": next_ps_row["to_state"] if next_ps_row else "",
                "most_likely_next_grammar": next_gram_row["to_state"] if next_gram_row else "",
                "expected_persistence_pct": persist,
                "expected_return_6h_pct": med_next,
                "collapse_probability_pct": collapse_next,
                "confidence": next_sit_row["confidence"] if next_sit_row else "hypothesis",
                "recommended_action": action,
                "reason": reason,
                "risk_factors": "|".join(
                    x for x in [
                        "late_cycle" if cur_sit == "late_chasing_zone" else "",
                        "distribution_path" if next_sit == "distribution" else "",
                        "collapse_path" if next_sit == "forced_liquidation_zone" else "",
                    ] if x
                ),
            }
        )
    return rows


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
    baseline = global_baseline_mae(records)

    edges = build_edges(records)
    sit_matrix = transition_matrix(edges, "situation_archetype", "situation_archetype")
    part_matrix = transition_matrix(edges, "participant_state", "participant_state")
    gram_matrix = transition_matrix(edges, "grammar", "grammar")
    sit_persist = persistence_stats(edges, "situation_archetype")
    part_persist = persistence_stats(edges, "participant_state")

    evo_families = evolution_families(edges)
    cond_evolution = mine_conditional_evolution(edges)

    p9_reg = []
    if P9_REGISTRY.exists():
        with P9_REGISTRY.open(encoding="utf-8") as f:
            p9_reg = list(csv.DictReader(f))
    registry = evolution_registry(sit_matrix, cond_evolution, p9_reg)
    engine = engine_output(records, sit_matrix, part_matrix, gram_matrix)

    write_csv(SITUATION_TRANSITION_CSV, sit_matrix)
    part_rows = [{**r, "row_type": "transition"} for r in part_matrix]
    part_rows += [{**r, "row_type": "persistence", "from_state": r["state"], "to_state": r["state"]} for r in part_persist]
    write_csv(PARTICIPANT_TRANSITION_CSV, part_rows)
    write_csv(GRAMMAR_TRANSITION_CSV, gram_matrix)
    write_csv(EVOLUTION_FAMILY_CSV, evo_families)
    write_csv(CONDITIONAL_EVOLUTION_CSV, cond_evolution)
    write_csv(REGISTRY_CSV, registry)
    write_csv(ENGINE_CSV, engine)

    dates = sorted({r["date"] for r in records})
    lines = [
        "===== SCOUT SEASON2 P10 - SITUATION EVOLUTION ENGINE =====",
        "",
        f"Sample: {len(records)} records | {dates[0]}..{dates[-1]} ({len(dates)} days)",
        f"Collected: {collected or '(none)'} | Transition edges: {len(edges)}",
        f"Situation transitions: {len(sit_matrix)} | Conditional rules: {len(cond_evolution)}",
        "",
        "--- Task 1: Situation transition matrix (top) ---",
    ]
    for row in sorted(sit_matrix, key=lambda x: -x["transition_count"])[:12]:
        lines.append(
            f"  {row['from_state']} -> {row['to_state']}: "
            f"p={row['transition_probability_pct']}% n={row['transition_count']} "
            f"collapse_to={row['collapse_rate_at_to_pct']}% [{row['stability']}]"
        )

    lines.extend(["", "--- Task 2: Participant state evolution (top) ---"])
    for row in sorted(part_matrix, key=lambda x: -x["transition_count"])[:8]:
        lines.append(
            f"  {row['from_state']} -> {row['to_state']}: p={row['transition_probability_pct']}% n={row['transition_count']}"
        )
    for row in part_persist[:5]:
        lines.append(f"  persistence {row['state']}: {row['persistence_pct']}% (n={row['sample_transitions']})")

    lines.extend(["", "--- Task 3: Grammar evolution (top) ---"])
    for row in sorted(gram_matrix, key=lambda x: -x["transition_count"])[:6]:
        lines.append(
            f"  {row['from_state'][:35]} -> {row['to_state'][:35]}: "
            f"p={row['transition_probability_pct']}% n={row['transition_count']}"
        )

    lines.extend(["", "--- Task 4: Evolution families ---"])
    for row in evo_families[:5]:
        lines.append(
            f"  {row['family_id']}: {row['symbol_count']} symbols path={row['dominant_path'][:55]}"
        )

    lines.extend(["", "--- Task 5: Conditional evolution rules ---"])
    for row in cond_evolution[:8]:
        lines.append(
            f"  {row['evolution_rule'][:70]}: p={row['transition_probability_pct']}% "
            f"n={row['sample_size']} [{row['status']}]"
        )

    lines.extend(["", "--- Philosophy ---",
        " Q1: What situation is this?",
        " Q2: What is this situation becoming?",
        " Scout = situation evolution engine, not price predictor",
        "",
        f"Engine: {ENGINE_CSV}",
        f"Registry: {REGISTRY_CSV}",
        "=" * 58,
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P10 SITUATION EVOLUTION =====")
    print(f"Records: {len(records)} | Edges: {len(edges)} | Rules: {len(cond_evolution)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
