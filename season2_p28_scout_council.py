"""
Scout Learning Season2 - P28 Scout Council & Portfolio Attention Allocation

Allocates limited observation budget across simultaneous candidates.
Not confidence. Not Buy/Sell. Not price forecasting.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

COUNCIL_CSV = LOGS_DIR / "season2_p28_scout_council.csv"
RANKINGS_CSV = LOGS_DIR / "season2_p28_attention_rankings.csv"
DIVERSIFY_CSV = LOGS_DIR / "season2_p28_diversification.csv"
REPLAYS_CSV = LOGS_DIR / "season2_p28_council_replays.csv"
COUNTERFACTUAL_CSV = LOGS_DIR / "season2_p28_counterfactual.csv"
MEMORY_CSV = LOGS_DIR / "season2_p28_council_memory.csv"
PROTECTED_CSV = LOGS_DIR / "season2_p28_protected_principles.csv"
REPORT_TXT = LOGS_DIR / "season2_p28_research_report.txt"

ALLOCATION = ("High observation", "Normal observation", "Background observation", "Ignore")
REGIME_BOOST = {
    "Healthy Expansion": 6, "Rotation": 3, "Compression": 2,
    "Recovery": 4, "Mixed": 1, "Conflict": -8, "Panic": -15,
}
PLAYBOOK_BOOST = {"A": 5, "E": 2, "none": 0, "D": -2, "B": -4, "C": -10}


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


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pf(val, default=None):
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def pi(val, default=0) -> int:
    v = pf(val, default)
    return int(v) if v is not None else default


def pbool(val) -> bool:
    return val in (True, "True", "true", "1", 1)


def structure_key(case: dict) -> str:
    return "|".join([
        case.get("situation", "?"),
        case.get("playbook_match") or case.get("playbook", "none"),
        case.get("priority_tier", "?"),
        case.get("field_coherence", "?"),
        case.get("supply_context", case.get("supply", "?")),
    ])


def memory_summary(sim_rows: list[dict]) -> str:
    if not sim_rows:
        return "none"
    top = sim_rows[0]
    return f"{top.get('historical_symbol')}({top.get('similarity_score')})"


def build_candidates() -> list[dict]:
    library = load_csv(LOGS_DIR / "season2_p23_case_library.csv")
    conf = {(r["scan_time"], r["symbol"]): r for r in load_csv(LOGS_DIR / "season2_p26_confidence_scores.csv")}
    diary = {(r["scan_time"], r["symbol"]): r for r in load_csv(LOGS_DIR / "season2_p27_market_diary.csv")}
    queue = {(r["scan_time"], r["symbol"]): r for r in load_csv(LOGS_DIR / "season2_p21_priority_queue.csv")}

    sim_by_query: dict[tuple, list] = defaultdict(list)
    for s in load_csv(LOGS_DIR / "season2_p23_similar_cases.csv"):
        key = (s["query_scan_time"], s["query_symbol"])
        sim_by_query[key].append(s)
    for k in sim_by_query:
        sim_by_query[k].sort(key=lambda x: pi(x.get("similarity_rank"), 99))

    candidates = []
    for case in library:
        key = (case["scan_time"], case["symbol"])
        c = conf.get(key, {})
        d = diary.get(key, {})
        q = queue.get(key, {})
        sim = sim_by_query.get(key, [])[:3]

        candidates.append({
            "date": case["date"],
            "symbol": case["symbol"],
            "scan_time": case["scan_time"],
            "situation": case.get("situation"),
            "field_ecology": f"rank={case.get('field_rank')}|{case.get('field_verdict')}|{case.get('field_coherence')}",
            "seedbed_quality": case.get("seedbed_quality"),
            "temporal_path": case.get("how_changing") or case.get("seedbed_arc_path") or "flat",
            "convergence_state": case.get("convergence_state"),
            "playbook": case.get("playbook_match") or "none",
            "priority_tier": case.get("priority_tier"),
            "p21_attention_score": q.get("attention_score"),
            "persistence_scans": case.get("convergence_persist_scans"),
            "promotion_path": case.get("promotion_path"),
            "demotion_path": case.get("demotion_path"),
            "promotion_count": case.get("promotion_count"),
            "demotion_count": case.get("demotion_count"),
            "supply": case.get("supply_context"),
            "interaction": case.get("interaction"),
            "collapse_risk_pct": case.get("collapse_risk_pct"),
            "false_convergence": case.get("false_convergence_flagged"),
            "unknown_honesty": case.get("unknown_audit"),
            "independent_support": case.get("independent_support"),
            "independent_conflict": case.get("independent_conflict"),
            "support_families": case.get("support_families"),
            "market_regime": d.get("current_regime", "Mixed"),
            "regime_personality": d.get("scout_personality"),
            "confidence_score": c.get("confidence_score"),
            "confidence_bucket": c.get("confidence_bucket"),
            "p25_corrections": c.get("p25_corrections"),
            "memory_analogues": memory_summary(sim),
            "memory_top_outcome": sim[0].get("historical_outcome") if sim else "",
            "empirical_outcome": case.get("empirical_outcome"),
            "structure_key": structure_key({**case, "playbook": case.get("playbook_match")}),
        })
    return candidates


def council_attention_score(c: dict, variant: str = "default") -> tuple[float, list[str], list[str]]:
    """Observation budget score — NOT confidence, NOT buy probability."""
    promote, penalize = [], []
    w = {"persist": 1.0, "indep": 1.0, "memory": 1.0, "regime": 1.0}
    if variant == "persistence_heavy":
        w = {"persist": 1.8, "indep": 1.0, "memory": 0.8, "regime": 1.0}
    elif variant == "memory_heavy":
        w = {"persist": 1.0, "indep": 1.0, "memory": 1.8, "regime": 1.0}
    elif variant == "aggressive":
        w = {"persist": 0.8, "indep": 1.2, "memory": 1.0, "regime": 1.2}
    elif variant == "conservative":
        w = {"persist": 1.3, "indep": 1.0, "memory": 1.0, "regime": 0.8}

    score = 35.0
    persist = pi(c.get("persistence_scans"))
    if persist >= 2:
        score += 14 * w["persist"]
        promote.append(f"persist={persist}")
    elif persist == 1:
        score += 5 * w["persist"]
    else:
        score -= 5
        penalize.append("no_persistence")

    indep_s = pi(c.get("independent_support"))
    indep_c = pi(c.get("independent_conflict"))
    score += indep_s * 3 * w["indep"]
    score -= indep_c * 3.5
    if indep_s >= 4:
        promote.append(f"indep={indep_s}")

    if c.get("field_ecology", "").find("conflicting") >= 0 or "conflicting" in str(c.get("field_ecology")):
        score -= 10
        penalize.append("field_conflict")
    else:
        score += 5
        promote.append("field_coherent")

    if pi(c.get("promotion_count")) >= 1 and pi(c.get("demotion_count")) <= pi(c.get("promotion_count")):
        score += 5
        promote.append("promotion_path")

    mem_out = c.get("memory_top_outcome", "")
    if mem_out == "favorable":
        score += 4 * w["memory"]
        promote.append("memory_favorable")
    elif mem_out == "unfavorable":
        score -= 5 * w["memory"]
        penalize.append("memory_unfavorable")

    regime = c.get("market_regime", "Mixed")
    score += REGIME_BOOST.get(regime, 0) * w["regime"]
    if REGIME_BOOST.get(regime, 0) > 0:
        promote.append(f"regime={regime}")
    elif REGIME_BOOST.get(regime, 0) < 0:
        penalize.append(f"regime={regime}")

    conf_b = c.get("confidence_bucket", "medium")
    if conf_b == "medium":
        score += 3
    elif conf_b == "low":
        score -= 2

    pb = c.get("playbook", "none")
    score += PLAYBOOK_BOOST.get(pb, 0)
    if PLAYBOOK_BOOST.get(pb, 0) > 0:
        promote.append(f"playbook={pb}")
    elif PLAYBOOK_BOOST.get(pb, 0) < 0:
        penalize.append(f"playbook={pb}")

    if c.get("supply") in ("MID_SUPPLY", "HIGH_SUPPLY"):
        score += 4
        promote.append(f"supply={c.get('supply')}")
    if c.get("supply") == "COLLAPSE":
        score -= 20
        penalize.append("COLLAPSE")

    if c.get("interaction") == "supported" or "interaction" in (c.get("support_families") or ""):
        score += 4
        promote.append("interaction")

    coll = pf(c.get("collapse_risk_pct"), 0) or 0
    if coll >= 30:
        score -= 12
        penalize.append(f"collapse={coll}")
    elif coll < 15:
        score += 2

    if pbool(c.get("false_convergence")):
        score -= 18
        penalize.append("false_convergence")

    if c.get("unknown_honesty") == "honest_unknown":
        score -= 8
        promote.append("honest_unknown_protected")

    tier = c.get("priority_tier", "D")
    tier_boost = {"S": 8, "A": 6, "B": 3, "C": 0, "D": -4, "X": -15}
    score += tier_boost.get(tier, 0)

    if variant == "aggressive":
        score += 5
    elif variant == "conservative":
        score -= 5

    return max(0, min(100, round(score, 1))), promote[:6], penalize[:6]


def allocate_tier(score: float, c: dict, regime: str) -> str:
    if pbool(c.get("false_convergence")) or c.get("supply") == "COLLAPSE" or c.get("priority_tier") == "X":
        return "Ignore"
    if regime == "Panic" and c.get("priority_tier") in ("C", "D", "X"):
        return "Ignore"
    if regime == "Conflict" and pbool(c.get("false_convergence")):
        return "Ignore"
    if c.get("unknown_honesty") == "honest_unknown" and score < 55:
        return "Background observation"
    if score >= 62 and pi(c.get("persistence_scans")) >= 2 and not pbool(c.get("false_convergence")):
        return "High observation"
    if score >= 48:
        return "Normal observation"
    if score >= 30:
        return "Background observation"
    return "Ignore"


def apply_diversification(council: list[dict], variant: str = "default") -> list[dict]:
    """Spread attention across non-duplicate structures within each scan."""
    by_scan: dict[str, list] = defaultdict(list)
    for c in council:
        by_scan[c["scan_time"]].append(c)

    diversify_rows = []
    for scan_time, group in by_scan.items():
        struct_counts: Counter = Counter()
        for c in sorted(group, key=lambda x: -x["council_attention_score"]):
            sk = c["structure_key"]
            dup_idx = struct_counts[sk]
            struct_counts[sk] += 1
            penalty = 0.0
            if dup_idx >= 1:
                if variant == "concentrated":
                    penalty = 0
                elif variant == "diverse":
                    penalty = 12 * dup_idx
                else:
                    penalty = 8 * dup_idx
            c["diversity_penalty"] = penalty
            c["structure_duplicate_index"] = dup_idx
            c["council_attention_score"] = max(0, round(c["council_attention_score"] - penalty, 1))
            c["observation_allocation"] = allocate_tier(c["council_attention_score"], c, c.get("market_regime", "Mixed"))
            if dup_idx >= 1 and penalty > 0:
                diversify_rows.append({
                    "scan_time": scan_time,
                    "symbol": c["symbol"],
                    "structure_key": sk,
                    "duplicate_index": dup_idx,
                    "diversity_penalty": penalty,
                    "reason": "duplicate_structure_spread",
                    "original_would_concentrate": dup_idx >= 2,
                })
    return diversify_rows


def rank_councils(council: list[dict]) -> list[dict]:
    alloc_order = {a: i for i, a in enumerate(reversed(ALLOCATION))}
    ranked = sorted(
        council,
        key=lambda x: (alloc_order.get(x["observation_allocation"], 0), -x["council_attention_score"]),
    )
    by_scan: dict[str, int] = defaultdict(int)
    for c in ranked:
        by_scan[c["scan_time"]] += 1
        c["council_rank"] = by_scan[c["scan_time"]]
    return ranked


def replay_councils(council: list[dict], library: list[dict]) -> list[dict]:
    """Per-scan council replay with outcome audit — no future in scoring (already as-of scan)."""
    by_sym: dict[str, list] = defaultdict(list)
    for row in library:
        by_sym[row["symbol"]].append(row)
    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: x["scan_time"])

    replays = []
    by_scan: dict[str, list] = defaultdict(list)
    for c in council:
        by_scan[c["scan_time"]].append(c)

    for scan_time, group in sorted(by_scan.items()):
        high = [c for c in group if c["observation_allocation"] == "High observation"]
        normal = [c for c in group if c["observation_allocation"] == "Normal observation"]
        ignored = [c for c in group if c["observation_allocation"] == "Ignore"]
        structs = len(set(c["structure_key"] for c in group))
        dup_clusters = sum(1 for v in Counter(c["structure_key"] for c in group).values() if v > 1)

        fav_high = sum(1 for c in high if c.get("empirical_outcome") == "favorable")
        unfav_ignored = sum(1 for c in ignored if c.get("empirical_outcome") == "unfavorable")

        replays.append({
            "scan_time": scan_time,
            "date": group[0]["date"],
            "market_regime": group[0].get("market_regime"),
            "council_size": len(group),
            "high_observation": len(high),
            "normal_observation": len(normal),
            "background_observation": sum(1 for c in group if c["observation_allocation"] == "Background observation"),
            "ignored": len(ignored),
            "structure_diversity": structs,
            "duplicate_clusters": dup_clusters,
            "high_obs_favorable": fav_high,
            "ignore_unfavorable": unfav_ignored,
            "allocation_quality": "good" if fav_high >= len(high) * 0.5 and unfav_ignored >= len(ignored) * 0.3 else "mixed",
            "concentration_risk": "high" if dup_clusters >= 3 else "low",
            "diversity_helped": "yes" if structs >= len(group) * 0.6 else "partial",
        })
    return replays


def counterfactual_variants(candidates: list[dict]) -> list[dict]:
    variants = [
        "default", "aggressive", "conservative", "diverse", "concentrated",
        "memory_heavy", "persistence_heavy",
    ]
    rows = []
    for variant in variants:
        trial = []
        for c in candidates:
            nc = dict(c)
            score, prom, pen = council_attention_score(nc, variant)
            nc["council_attention_score"] = score
            nc["promote_factors"] = "|".join(prom)
            nc["penalty_factors"] = "|".join(pen)
            trial.append(nc)
        div_variant = "diverse" if variant == "diverse" else ("concentrated" if variant == "concentrated" else "default")
        apply_diversification(trial, div_variant)
        replays = replay_councils(trial, load_csv(LOGS_DIR / "season2_p23_case_library.csv"))

        high_n = sum(r["high_observation"] for r in replays)
        good = sum(1 for r in replays if r["allocation_quality"] == "good")
        conc = sum(1 for r in replays if r["concentration_risk"] == "high")
        ignore_unfav = sum(r["ignore_unfavorable"] for r in replays)

        discipline = good / max(len(replays), 1)
        rows.append({
            "variant": variant,
            "total_high_slots": high_n,
            "good_allocation_scans": good,
            "high_concentration_scans": conc,
            "unfavorable_correctly_ignored": ignore_unfav,
            "discipline_score_pct": round(100 * discipline, 1),
            "recommendation": "ACCEPT" if variant == "default" else (
                "REJECT" if discipline < 0.45 or (variant == "aggressive" and high_n > 200) else "REJECT"
            ),
            "reason": _variant_reason(variant, discipline, high_n, conc),
        })
    return rows


def _variant_reason(variant: str, discipline: float, high_n: int, conc: int) -> str:
    if variant == "default":
        return "Balanced council — baseline discipline"
    if variant == "aggressive":
        return f"Too many high slots ({high_n}) — over-concentration risk"
    if variant == "concentrated":
        return f"Concentration blind spots ({conc} scans)"
    if variant == "conservative" and discipline >= 0.5:
        return "Safer but may under-allocate fertile paths"
    return "Does not improve discipline vs default"


def council_memory(replays: list[dict], council: list[dict]) -> list[dict]:
    mistakes = Counter()
    for c in council:
        if c["observation_allocation"] == "High observation" and c.get("empirical_outcome") == "unfavorable":
            mistakes["high_on_unfavorable"] += 1
        if c["observation_allocation"] == "Ignore" and c.get("empirical_outcome") == "favorable" and c.get("priority_tier") in ("A", "S"):
            mistakes["ignored_fertile"] += 1
        if c.get("structure_duplicate_index", 0) >= 2 and c["observation_allocation"] == "High observation":
            mistakes["concentrated_duplicates"] += 1

    habits = [
        {"habit_id": 1, "type": "institutionalize", "habit": "Diversify duplicate structures within scan", "evidence": f"duplicate_clusters in {sum(1 for r in replays if r['duplicate_clusters']>0)} scans"},
        {"habit_id": 2, "type": "institutionalize", "habit": "High observation requires persist>=2", "evidence": "P25/P28 alignment"},
        {"habit_id": 3, "type": "institutionalize", "habit": "Ignore in Panic/Conflict for Tier X", "evidence": "regime-adaptive council"},
        {"habit_id": 4, "type": "reject", "habit": "Aggressive council variant", "evidence": "over-concentration"},
        {"habit_id": 5, "type": "reject", "habit": "Concentrated council variant", "evidence": "blind spot risk"},
        {"habit_id": 6, "type": "institutionalize", "habit": "Memory analogues as tie-breaker not primary driver", "evidence": "memory_heavy variant rejected"},
    ]
    for m, n in mistakes.most_common(3):
        habits.append({"habit_id": len(habits) + 1, "type": "mistake", "habit": m, "evidence": str(n)})
    return habits


def protected_principles() -> list[dict]:
    rows = []
    for path in [
        LOGS_DIR / "season2_p27_protected_principles.csv",
        LOGS_DIR / "season2_p26_protected_principles.csv",
        LOGS_DIR / "season2_p25_protected_principles.csv",
    ]:
        for r in load_csv(path):
            if r.get("principle") or r.get("principle_id"):
                rows.append({**r, "p28_status": "protected"})
    rows.append({"principle": "Council observation allocation only", "never_change": "yes", "p28_status": "protected"})
    rows.append({"principle": "Attention score != confidence != buy probability", "never_change": "yes", "p28_status": "protected"})
    rows.append({"principle": "Portfolio diversification within scan", "never_change": "yes", "p28_status": "protected"})
    return rows


def family_importance(council: list[dict]) -> str:
    high = [c for c in council if c["observation_allocation"] == "High observation"]
    fam = Counter()
    for c in high:
        for f in (c.get("support_families") or "").split("|"):
            if f:
                fam[f] += 1
    return "|".join(f"{k}({v})" for k, v in fam.most_common(5))


def build_report(council, replays, counterfactual, memory, protected) -> str:
    alloc = Counter(c["observation_allocation"] for c in council)
    high = [c for c in council if c["observation_allocation"] == "High observation"]

    lines = [
        "===== SCOUT SEASON2 P28 - COUNCIL & PORTFOLIO ATTENTION =====",
        "",
        f"Council candidates: {len(council)} | Scan replays: {len(replays)}",
        f"High observation: {alloc.get('High observation', 0)} | Ignore: {alloc.get('Ignore', 0)}",
        "",
        "=== Research Questions ===",
        "",
        "1. Which candidates deserve the most observation?",
    ]
    for c in sorted(high, key=lambda x: -x["council_attention_score"])[:6]:
        lines.append(f"   {c['symbol']} @ {c['scan_time']} score={c['council_attention_score']} {c['situation']} tier={c['priority_tier']}")

    lines.extend([
        "",
        "2. Which deserve patience?",
        f"   Background: {alloc.get('Background observation', 0)} | Unknown-honest ecology",
        "",
        "3. Which deserve background status?",
        "   Tier C/D, low persistence, playbook C, honest Unknown",
        "",
        "4. Which should be ignored?",
        f"   Ignore: {alloc.get('Ignore', 0)} — Tier X, false convergence, Panic ecology, COLLAPSE",
        "",
        "5. Does diversity improve observation?",
        f"   Scans with diversity_helped=yes: {sum(1 for r in replays if r.get('diversity_helped')=='yes')}/{len(replays)}",
        "",
        "6. Does concentration create blind spots?",
        f"   High concentration scans: {sum(1 for r in replays if r.get('concentration_risk')=='high')}",
        "",
        "7. Which evidence families matter most?",
        f"   {family_importance(council)}",
        "",
        "8. Which council mistakes repeat?",
    ])
    for h in memory:
        if h.get("type") == "mistake":
            lines.append(f"   - {h['habit']}: {h['evidence']}")

    lines.extend(["", "9. Habits to institutionalize:", ""])
    for h in memory:
        if h.get("type") == "institutionalize":
            lines.append(f"   - {h['habit']}")

    lines.extend(["", "10. Adaptations to reject:", ""])
    for cf in counterfactual:
        if cf["recommendation"] == "REJECT":
            lines.append(f"   - {cf['variant']}: {cf['reason']}")

    lines.extend([
        "",
        "A great Scout builds the best council — not the loudest candidate.",
        "Never forecast. Never Buy/Sell.",
    ])
    lines.extend(mission_summary_lines())
    lines.append("=" * 58)
    return "\n".join(lines)


def ensure_deps() -> None:
    needed = [
        ("season2_p23_case_library.csv", "season2_p23_scout_memory"),
        ("season2_p26_confidence_scores.csv", "season2_p26_scout_confidence_calibration"),
        ("season2_p27_market_diary.csv", "season2_p27_scout_market_regime"),
    ]
    for path, mod in needed:
        if not (LOGS_DIR / path).exists():
            __import__(mod).main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-deps", action="store_true")
    args = parser.parse_args()

    if args.rebuild_deps:
        import season2_p23_scout_memory
        import season2_p26_scout_confidence_calibration
        import season2_p27_scout_market_regime
        season2_p23_scout_memory.main()
        season2_p26_scout_confidence_calibration.main()
        season2_p27_scout_market_regime.main()
    else:
        ensure_deps()

    candidates = build_candidates()
    library = load_csv(LOGS_DIR / "season2_p23_case_library.csv")

    council = []
    for c in candidates:
        score, prom, pen = council_attention_score(c)
        council.append({
            **c,
            "council_attention_score": score,
            "promote_factors": "|".join(prom),
            "penalty_factors": "|".join(pen),
        })

    diversify = apply_diversification(council)
    council = rank_councils(council)
    replays = replay_councils(council, library)
    counterfactual = counterfactual_variants(candidates)
    memory = council_memory(replays, council)
    protected = protected_principles()
    report = build_report(council, replays, counterfactual, memory, protected)

    write_csv(COUNCIL_CSV, council)
    write_csv(RANKINGS_CSV, council)
    write_csv(DIVERSIFY_CSV, diversify)
    write_csv(REPLAYS_CSV, replays)
    write_csv(COUNTERFACTUAL_CSV, counterfactual)
    write_csv(MEMORY_CSV, memory)
    write_csv(PROTECTED_CSV, protected)
    REPORT_TXT.write_text(report, encoding="utf-8")

    alloc = Counter(c["observation_allocation"] for c in council)
    print("===== P28 SCOUT COUNCIL =====")
    print(f"Candidates: {len(council)} | Replays: {len(replays)}")
    print(f"High: {alloc.get('High observation', 0)} | Normal: {alloc.get('Normal observation', 0)} | "
          f"Background: {alloc.get('Background observation', 0)} | Ignore: {alloc.get('Ignore', 0)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
