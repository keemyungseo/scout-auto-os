"""
Scout Learning Season2 - P9 Conditional Interaction Mining & Situation Intelligence

Research only. Discover weak-alone / strong-together conditional contexts.
"""

import argparse
import csv
import itertools
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from season2_p5_historical_expansion import (
    SEEDS,
    STEP1_DATES,
    build_unified_records,
    collect_missing,
    compression_ctx,
    global_baseline_mae,
    loo_mae,
    vol_zone,
)
from season2_p6_market_memory import attach_forward_targets, symbol_archetype
from season2_p7_behaviour_grammar import assign_cluster, build_sentence, enrich_physics
from season2_p8_participant_state import (
    assign_participant_state,
    best_memory_per_symbol,
    enrich_panel_fields,
    load_grammar_families,
)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

INTERACTION_CSV = LOGS_DIR / "season2_p9_interactions.csv"
ARCHETYPE_CSV = LOGS_DIR / "season2_p9_situation_archetypes.csv"
REGISTRY_CSV = LOGS_DIR / "season2_p9_hypothesis_registry.csv"
REVIVAL_CSV = LOGS_DIR / "season2_p9_revival_candidates.csv"
ENGINE_CSV = LOGS_DIR / "season2_p9_situation_engine_output.csv"
REPORT_TXT = LOGS_DIR / "season2_p9_research_report.txt"

MIN_N = 8
BEAM_WIDTH = 30
MAX_DEPTH = 4

DIM_NAMES = [
    "participant_state",
    "memory",
    "grammar",
    "symbol_family",
    "trend_cluster",
    "market_physics",
    "collapse_type",
    "historical_behaviour",
]

SITUATION_NAMES = [
    "early_accumulation",
    "healthy_trend",
    "late_markup",
    "profit_taking",
    "distribution",
    "recovery",
    "false_breakout",
    "hidden_accumulation",
    "rotation_chop",
    "forced_liquidation_zone",
    "explosive_continuation",
    "unclassified_situation",
]


@dataclass
class Combo:
    parts: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    score: float = 0.0

    def key(self) -> str:
        return " + ".join(f"{d}={v}" for d, v in self.parts)

    def dims_used(self) -> set[str]:
        return {d for d, _ in self.parts}


def load_participant_families() -> dict[str, str]:
    path = LOGS_DIR / "season2_p8_participant_families.csv"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for sym in (row.get("symbols") or "").split("|"):
                if sym:
                    out[sym] = row["family_id"]
    return out


def physics_tag(record: dict) -> str:
    tags = []
    if record.get("fast_rejection"):
        tags.append("frej")
    if record.get("vol_exhaustion"):
        tags.append("vol_exh")
    if compression_ctx(record):
        tags.append("compress")
    if record.get("range_expand"):
        tags.append("range_exp")
    if record.get("vol_shock"):
        tags.append("vol_shock")
    if record.get("inside_bar"):
        tags.append("inside")
    return "|".join(tags) if tags else "neutral_physics"


def collapse_tag(record: dict) -> str:
    if record.get("collapse_label") == "YES":
        return "collapse_yes"
    label = record.get("supply_label") or "unknown"
    return f"supply_{label.lower()}"


def historical_tag(record: dict) -> str:
    arch = record.get("symbol_archetype") or "unknown"
    apps = record.get("symbol_appearances") or 1
    freq = "repeat" if apps >= 3 else "early"
    return f"{arch}_{freq}"


def prepare_records(records: list[dict]) -> None:
    grammar_fam = load_grammar_families()
    participant_fam = load_participant_families()
    sym_memory = best_memory_per_symbol(records)
    global_mem = "7d"

    sym_groups: dict[str, list] = defaultdict(list)
    for r in records:
        sym_groups[r["symbol"]].append(r)
    sym_arch = {sym: symbol_archetype(grp) for sym, grp in sym_groups.items()}
    sym_apps = {sym: len(grp) for sym, grp in sym_groups.items()}

    for record in records:
        primary, score, secondary = assign_participant_state(record)
        record["participant_state"] = primary
        record["state_score"] = score
        record["secondary_state"] = secondary
        sent = build_sentence(record)
        record.update(sent)
        record["grammar"] = (record.get("primary_trigram") or "unknown")[:45]
        gf = grammar_fam.get(record["symbol"], "unknown")
        pf = participant_fam.get(record["symbol"], "unknown")
        record["symbol_family"] = gf if gf != "unknown" else pf
        record["memory"] = sym_memory.get(record["symbol"], global_mem)
        record["trend_cluster"] = assign_cluster(record.get("tokens", []))
        record["market_physics"] = physics_tag(record)
        record["collapse_type"] = collapse_tag(record)
        record["symbol_archetype"] = sym_arch.get(record["symbol"], {}).get("archetype", "unknown")
        record["symbol_appearances"] = sym_apps.get(record["symbol"], 1)
        record["historical_behaviour"] = historical_tag(record)


def dimension_fn(name: str):
    return lambda r, n=name: str(r.get(n, "unknown"))[:50]


DIMENSIONS = {n: dimension_fn(n) for n in DIM_NAMES}


def filter_combo(records: list[dict], parts: tuple[tuple[str, str], ...]) -> list[dict]:
    subset = records
    for dim, val in parts:
        fn = DIMENSIONS[dim]
        subset = [r for r in subset if fn(r) == val]
    return subset


def evaluate_subset(subset: list[dict], baseline_mae: float) -> dict | None:
    if len(subset) < MIN_N:
        return None
    f6 = [r["target_f6"] for r in subset if r.get("target_f6") is not None]
    if len(f6) < MIN_N:
        return None

    key = lambda r: "ctx"
    ctx_mae = loo_mae(subset, key)[0]
    mae_gain = baseline_mae - ctx_mae
    spread = round(max(f6) - min(f6), 2) if len(f6) >= 2 else 0.0
    med = round(statistics.median(f6), 2)
    mean = round(statistics.mean(f6), 2)
    persist = round(sum(1 for v in f6 if v >= 0) / len(f6) * 100, 1)
    collapses = sum(1 for r in subset if r.get("collapse_label") == "YES")
    collapse_pct = round(collapses / len(subset) * 100, 1)

    # Information gain score: prioritise MAE improvement + spread + sample
    score = mae_gain * 2.0 + spread * 0.05 + min(len(subset), 50) * 0.02
    if med >= 3 and collapse_pct < 20:
        score += 1.0
    if collapse_pct >= 35:
        score -= 1.5

    conf = "high" if len(subset) >= 25 and mae_gain >= 0.5 else "medium" if len(subset) >= MIN_N and mae_gain >= 0.25 else "hypothesis"

    return {
        "sample_size": len(subset),
        "expected_return_6h": med,
        "mean_return_6h": mean,
        "collapse_probability_pct": collapse_pct,
        "persistence_pct": persist,
        "context_mae": round(ctx_mae, 2),
        "mae_improvement": round(mae_gain, 2),
        "spread_f6": spread,
        "confidence": conf,
        "score": round(score, 3),
    }


def single_value_candidates(records: list[dict], baseline: float) -> list[tuple[tuple[tuple[str, str], ...], dict]]:
    found: list[tuple[tuple[tuple[str, str], ...], dict]] = []
    for dim in DIM_NAMES:
        groups: dict[str, list] = defaultdict(list)
        fn = DIMENSIONS[dim]
        for r in records:
            groups[fn(r)].append(r)
        for val, grp in groups.items():
            if val in ("unknown", "neutral_physics", ""):
                continue
            metrics = evaluate_subset(grp, baseline)
            if metrics and (metrics["mae_improvement"] >= 0.15 or metrics["expected_return_6h"] >= 4):
                found.append((((dim, val),), metrics))
    found.sort(key=lambda x: -x[1]["score"])
    return found[:BEAM_WIDTH * 2]


def beam_mine(records: list[dict], baseline: float) -> list[dict]:
    """Greedy beam: singles -> pairs -> triples -> quads by information gain."""
    seeds = single_value_candidates(records, baseline)
    beam: list[Combo] = []
    seen: set[str] = set()
    all_rows: list[dict] = []

    for parts, metrics in seeds:
        c = Combo(parts=parts, score=metrics["score"])
        if c.key() not in seen:
            beam.append(c)
            seen.add(c.key())

    beam.sort(key=lambda x: -x.score)
    beam = beam[:BEAM_WIDTH]

    for depth in range(1, MAX_DEPTH):
        next_beam: list[Combo] = []
        for combo in beam:
            used = combo.dims_used()
            remaining = [d for d in DIM_NAMES if d not in used]
            for dim in remaining:
                vals = Counter(DIMENSIONS[dim](r) for r in filter_combo(records, combo.parts))
                for val, _ in vals.most_common(12):
                    if val in ("unknown", "neutral_physics", ""):
                        continue
                    new_parts = combo.parts + ((dim, val),)
                    key = " + ".join(f"{d}={v}" for d, v in new_parts)
                    if key in seen:
                        continue
                    subset = filter_combo(records, new_parts)
                    metrics = evaluate_subset(subset, baseline)
                    if not metrics:
                        continue
                    if metrics["mae_improvement"] < 0.1 and depth >= 2:
                        continue
                    seen.add(key)
                    nc = Combo(parts=new_parts, score=metrics["score"])
                    next_beam.append(nc)
                    all_rows.append(build_interaction_row(new_parts, metrics, baseline, depth + 1))

        next_beam.sort(key=lambda x: -x.score)
        beam = next_beam[:BEAM_WIDTH]
        if not beam:
            break

    for parts, metrics in seeds:
        all_rows.append(build_interaction_row(parts, metrics, baseline, len(parts)))

    # dedupe by situation key, keep best score
    best: dict[str, dict] = {}
    for row in all_rows:
        k = row["situation_key"]
        if k not in best or row["information_score"] > best[k]["information_score"]:
            best[k] = row
    return sorted(best.values(), key=lambda x: -x["information_score"])


def build_interaction_row(parts: tuple, metrics: dict, baseline: float, depth: int) -> dict:
    situation = infer_situation_name(parts, metrics)
    risks = []
    if metrics["collapse_probability_pct"] >= 25:
        risks.append("high_collapse")
    if metrics["sample_size"] < 12:
        risks.append("small_sample")
    if any(d == "late_chasing" or v == "late_chasing" for d, v in parts):
        risks.append("late_cycle")
    action = infer_action(metrics, situation)

    return {
        "situation_key": " + ".join(f"{d}={v}" for d, v in parts),
        "situation_description": situation,
        "interaction_depth": depth,
        "dimensions": "|".join(d for d, _ in parts),
        "sample_size": metrics["sample_size"],
        "expected_return_6h": metrics["expected_return_6h"],
        "mean_return_6h": metrics["mean_return_6h"],
        "collapse_probability_pct": metrics["collapse_probability_pct"],
        "persistence_pct": metrics["persistence_pct"],
        "baseline_mae": round(baseline, 2),
        "context_mae": metrics["context_mae"],
        "mae_improvement": metrics["mae_improvement"],
        "spread_f6": metrics["spread_f6"],
        "confidence": metrics["confidence"],
        "information_score": metrics["score"],
        "risk_factors": "|".join(risks) if risks else "",
        "action_candidate": action,
        "status": classify_interaction_status(metrics),
    }


def infer_situation_name(parts: tuple[tuple[str, str], ...], metrics: dict) -> str:
    text = " ".join(f"{d} {v}" for d, v in parts).lower()
    med = metrics["expected_return_6h"]
    collapse = metrics["collapse_probability_pct"]

    if "late_chasing" in text or "late_markup" in text:
        if "explosive" in text:
            return "late_markup"
        return "late_chasing_zone"
    if "compress" in text or "accumulation" in text or "hidden" in text:
        return "hidden_accumulation" if "hidden" in text else "early_accumulation"
    if "distribution" in text or ("vol_exh" in text and collapse >= 15):
        return "distribution"
    if "profit" in text or ("frej" in text and collapse < 15):
        return "profit_taking"
    if "collapse" in text and collapse >= 30:
        return "forced_liquidation_zone"
    if "recovery" in text or "inventory" in text:
        return "recovery"
    if "false" in text or "frej" in text:
        return "false_breakout"
    if med >= 4 and collapse < 15 and "explosive" in text:
        return "explosive_continuation"
    if med >= 2 and collapse < 12:
        return "healthy_trend"
    if "rotation" in text or "choppy" in text:
        return "rotation_chop"
    return "unclassified_situation"


def infer_action(metrics: dict, situation: str) -> str:
    med = metrics["expected_return_6h"]
    collapse = metrics["collapse_probability_pct"]
    if collapse >= 35 or situation in ("forced_liquidation_zone", "distribution"):
        return "Avoid"
    if med >= 8 and collapse < 10:
        return "Strong Buy"
    if med >= 4 and collapse < 15:
        return "Buy"
    if med >= 0 and collapse < 20:
        return "Hold"
    if situation in ("late_markup", "late_chasing_zone", "false_breakout"):
        return "Watch"
    return "Watch"


def classify_interaction_status(metrics: dict) -> str:
    if metrics["mae_improvement"] >= 0.5 and metrics["sample_size"] >= 20:
        return "ACTIVE"
    if metrics["mae_improvement"] >= 0.25 or (metrics["expected_return_6h"] >= 5 and metrics["sample_size"] >= MIN_N):
        return "CONDITIONAL"
    if metrics["mae_improvement"] >= 0.1:
        return "RETIRED"
    return "DEAD_FOR_NOW"


def load_all_registries() -> list[dict]:
    rows = []
    paths = [
        (LOGS_DIR / "season2_p6_hypothesis_registry.csv", "feature", "hypothesis_name"),
        (LOGS_DIR / "season2_p7_grammar_registry.csv", "grammar", "grammar_sequence"),
        (LOGS_DIR / "season2_p8_hypothesis_registry.csv", "participant_state", "hypothesis_name"),
    ]
    for path, kind, name_col in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "kind": kind,
                        "name": row.get(name_col, ""),
                        "prior_status": row.get("current_status", "OPEN"),
                        "source": path.name,
                    }
                )
    for seed in SEEDS:
        rows.append(
            {
                "kind": "feature",
                "name": seed.feature,
                "prior_status": seed.prior_status,
                "source": "p5_seeds",
            }
        )
    return rows


def retest_registry(records: list[dict], interactions: list[dict], baseline: float) -> tuple[list[dict], list[dict]]:
    registry_items = load_all_registries()
    revival_rows = []
    registry_rows = []

    interaction_by_key = {r["situation_key"]: r for r in interactions}

    for item in registry_items:
        name = item["name"]
        kind = item["kind"]
        best_gain = 0.0
        best_ctx = ""
        best_status = item["prior_status"]
        best_med = None

        if kind == "grammar":
            matching = [r for r in interactions if f"grammar={name}" in r["situation_key"] or name in r["situation_key"]]
        elif kind == "participant_state":
            matching = [r for r in interactions if f"participant_state={name}" in r["situation_key"]]
        else:
            matching = [r for r in interactions if name in r["situation_key"]]

        if matching:
            best = max(matching, key=lambda x: x["mae_improvement"])
            best_gain = best["mae_improvement"]
            best_ctx = best["situation_key"][:80]
            best_med = best["expected_return_6h"]
            if best_gain >= 0.35:
                best_status = "REVIVED" if item["prior_status"] in ("RETIRED", "DEAD_FOR_NOW") else "CONDITIONAL"
            elif best_gain >= 0.25:
                best_status = "CONDITIONAL"
            elif best_gain >= 0.1:
                best_status = "RETIRED" if item["prior_status"] != "ACTIVE" else "ACTIVE"

        subset = [r for r in records if name in (r.get("grammar", ""), r.get("participant_state", ""), r.get("dominant_trigger", ""))]
        global_mae = loo_mae(subset, lambda r: "x")[0] if len(subset) >= MIN_N else baseline
        global_med = round(statistics.median([r["target_f6"] for r in subset if r.get("target_f6") is not None]), 2) if subset else None

        registry_rows.append(
            {
                "hypothesis_kind": kind,
                "hypothesis_name": name,
                "prior_status": item["prior_status"],
                "current_status": best_status,
                "global_sample": len(subset),
                "global_mae": round(global_mae, 2) if subset else "",
                "best_context": best_ctx,
                "best_mae_improvement": best_gain,
                "best_context_return": best_med,
                "decision_reason": f"retest on {len(records)} records; best gain {best_gain:.2f}",
            }
        )
        if item["prior_status"] in ("RETIRED", "DEAD_FOR_NOW") and best_status in ("REVIVED", "CONDITIONAL"):
            revival_rows.append({**registry_rows[-1], "revival": "YES"})

    return registry_rows, revival_rows


def cluster_archetypes(interactions: list[dict]) -> list[dict]:
    """Task 7: group top interactions into situation archetypes."""
    top = [r for r in interactions if r["sample_size"] >= MIN_N][:40]
    archetype_groups: dict[str, list] = defaultdict(list)
    for row in top:
        archetype_groups[row["situation_description"]].append(row)

    rows = []
    for arch, group in sorted(archetype_groups.items(), key=lambda x: -len(x[1])):
        n = sum(r["sample_size"] for r in group)
        meds = [r["expected_return_6h"] for r in group]
        collapses = [r["collapse_probability_pct"] for r in group]
        best = max(group, key=lambda x: x["information_score"])
        rows.append(
            {
                "situation_archetype": arch,
                "interaction_variants": len(group),
                "total_sample": n,
                "median_expected_return": round(statistics.median(meds), 2),
                "median_collapse_pct": round(statistics.median(collapses), 1),
                "best_situation_key": best["situation_key"][:100],
                "best_mae_improvement": best["mae_improvement"],
                "best_action": best["action_candidate"],
                "confidence": best["confidence"],
            }
        )
    return rows


def match_record_to_situation(record: dict, interactions: list[dict]) -> dict | None:
    best = None
    best_score = -1.0
    for row in interactions:
        parts = []
        for segment in row["situation_key"].split(" + "):
            if "=" in segment:
                d, v = segment.split("=", 1)
                parts.append((d.strip(), v.strip()))
        subset = filter_combo([record], tuple(parts))
        if len(subset) == 1:
            if row["information_score"] > best_score:
                best_score = row["information_score"]
                best = row
    return best


def engine_rows(records: list[dict], interactions: list[dict]) -> list[dict]:
    top_interactions = interactions[:60]
    rows = []
    for record in records[-40:]:
        match = match_record_to_situation(record, top_interactions)
        if match:
            action = match["action_candidate"]
            reason = f"situation={match['situation_description']}; mae_gain={match['mae_improvement']}"
            conf = match["confidence"]
            med = match["expected_return_6h"]
            collapse = match["collapse_probability_pct"]
        else:
            action = "Watch"
            reason = "no strong conditional match"
            conf = "hypothesis"
            med = 0
            collapse = 0

        rows.append(
            {
                "date": record["date"],
                "symbol": record["symbol"],
                "scan_time": record["scan_time"],
                "participant_state": record.get("participant_state"),
                "memory": record.get("memory"),
                "grammar": record.get("grammar"),
                "symbol_family": record.get("symbol_family"),
                "trend_cluster": record.get("trend_cluster"),
                "market_physics": record.get("market_physics"),
                "matched_situation": match["situation_description"] if match else "unmatched",
                "situation_key": match["situation_key"][:120] if match else "",
                "expected_return_6h": med,
                "collapse_probability_pct": collapse,
                "persistence_pct": match["persistence_pct"] if match else "",
                "confidence": conf,
                "recommended_action": action,
                "reason": reason,
                "risk_factors": match["risk_factors"] if match else "unclassified",
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
    baseline = global_baseline_mae(records)

    interactions = beam_mine(records, baseline)
    archetypes = cluster_archetypes(interactions)
    registry_rows, revival_rows = retest_registry(records, interactions, baseline)
    engine = engine_rows(records, interactions)

    write_csv(INTERACTION_CSV, interactions)
    write_csv(ARCHETYPE_CSV, archetypes)
    write_csv(REGISTRY_CSV, registry_rows)
    write_csv(REVIVAL_CSV, revival_rows)
    write_csv(ENGINE_CSV, engine)

    dates = sorted({r["date"] for r in records})
    top = interactions[:10]
    lines = [
        "===== SCOUT SEASON2 P9 - CONDITIONAL INTERACTION MINING =====",
        "",
        f"Sample: {len(records)} records | {dates[0]}..{dates[-1]} ({len(dates)} days)",
        f"Collected: {collected or '(none)'} | Baseline MAE: {baseline:.2f}%",
        f"Interactions discovered: {len(interactions)} | Revivals: {len(revival_rows)}",
        "",
        "--- Top conditional interactions (weak alone, strong together) ---",
    ]
    for row in top:
        lines.append(
            f"  [{row['situation_description']}] n={row['sample_size']} "
            f"ret={row['expected_return_6h']}% collapse={row['collapse_probability_pct']}% "
            f"mae_gain={row['mae_improvement']} [{row['status']}]"
        )
        lines.append(f"    {row['situation_key'][:90]}")

    lines.extend(["", "--- Situation archetypes ---"])
    for row in archetypes[:8]:
        lines.append(
            f"  {row['situation_archetype']}: variants={row['interaction_variants']} "
            f"n={row['total_sample']} med_ret={row['median_expected_return']}% "
            f"action={row['best_action']}"
        )

    lines.extend(["", "--- Registry revivals ---"])
    for row in revival_rows[:8]:
        lines.append(f"  {row['hypothesis_name']}: {row['prior_status']} -> {row['current_status']} | {row['best_context'][:60]}")

    if not revival_rows:
        lines.append("  (no RETIRED/DEAD_FOR_NOW items revived above threshold)")

    lines.extend([
        "",
        "--- Research notes ---",
        " Objective: situation understanding, not universal prediction",
        " Beam search over 8 dimensions; depth up to 4-way interactions",
        " Never permanently discard — registry retest each batch",
        "",
        f"Interactions: {INTERACTION_CSV}",
        f"Engine: {ENGINE_CSV}",
        "=" * 58,
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P9 CONDITIONAL INTERACTION MINING =====")
    print(f"Records: {len(records)} | Interactions: {len(interactions)} | Revivals: {len(revival_rows)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
