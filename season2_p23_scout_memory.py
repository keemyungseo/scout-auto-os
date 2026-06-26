"""
Scout Learning Season2 - P23 Scout Memory & Historical Case Library

"Have we seen something like this before?" — not "what will happen."
Builds empirical memory and retrieves structural analogues from P15-P22.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CASE_LIBRARY_CSV = LOGS_DIR / "season2_p23_case_library.csv"
SIMILAR_CASES_CSV = LOGS_DIR / "season2_p23_similar_cases.csv"
GOOD_BAD_CSV = LOGS_DIR / "season2_p23_good_vs_bad_comparison.csv"
CASE_MEMORY_CSV = LOGS_DIR / "season2_p23_case_memory.csv"
CURRENT_HISTORY_CSV = LOGS_DIR / "season2_p23_current_vs_history.csv"
REPORT_TXT = LOGS_DIR / "season2_p23_research_report.txt"

TOP_K = 5
EVIDENCE_DOMAINS = (
    "situation", "health", "pressure", "participant", "supply", "grammar",
    "interaction", "field_environment", "temporal_lifecycle", "collapse_risk",
    "false_convergence", "unknown_audit", "persistence", "priority_tier",
    "promotion_history", "demotion_history", "playbook",
)
CAT_WEIGHTS = {
    "situation": 12, "supply_context": 8, "priority_tier": 10, "playbook_match": 8,
    "convergence_state": 8, "field_verdict": 6, "field_coherence": 5,
    "seedbed_quality": 6, "health_class": 5, "pressure_band": 4,
    "scout_stance": 5, "unknown_audit": 6,
}
FAMILY_WEIGHT = 30


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


def bucket_collapse(v) -> str:
    x = pf(v, 0) or 0
    if x < 15:
        return "low"
    if x < 30:
        return "mid"
    return "high"


def bucket_persist(v) -> str:
    x = pi(v)
    if x == 0:
        return "none"
    if x == 1:
        return "single"
    return "multi"


def family_set(case: dict) -> set[str]:
    s = set()
    for part in (case.get("support_families") or "").split("|"):
        if part:
            s.add(part)
    if case.get("health_class"):
        s.add("health")
    if case.get("pressure_band"):
        s.add("pressure")
    if case.get("participant_state"):
        s.add("participant")
    if case.get("supply_context"):
        s.add("supply")
    if case.get("grammar"):
        for g in case["grammar"].split("|"):
            if g:
                s.add("grammar")
    if case.get("interaction") == "supported":
        s.add("interaction")
    if case.get("field_verdict"):
        s.add("field_environment")
    if case.get("how_changing") or case.get("seedbed_arc_path"):
        s.add("temporal_lifecycle")
    if bucket_collapse(case.get("collapse_risk_pct")) != "low":
        s.add("collapse_risk")
    if pbool(case.get("false_convergence_flagged")):
        s.add("false_convergence")
    if case.get("unknown_audit") not in ("", "neutral"):
        s.add("unknown_audit")
    if bucket_persist(case.get("convergence_persist_scans")) != "none":
        s.add("persistence")
    return s


def structure_signature(case: dict) -> str:
    return "|".join([
        f"sit={case.get('situation', '?')}",
        f"supply={case.get('supply_context', '?')}",
        f"tier={case.get('priority_tier', '?')}",
        f"pb={case.get('playbook_match') or 'none'}",
        f"conv={case.get('convergence_state', '?')}",
        f"field={case.get('field_verdict', '?')}",
        f"seed={case.get('seedbed_quality') or 'Unknown'}",
        f"fc={case.get('false_convergence_flagged', False)}",
        f"coll={bucket_collapse(case.get('collapse_risk_pct'))}",
        f"persist={bucket_persist(case.get('convergence_persist_scans'))}",
    ])


def evidence_composition(case: dict) -> str:
    fam = sorted(family_set(case))
    return "|".join(fam) if fam else "sparse"


def unknown_audit_label(row: dict) -> str:
    v = row.get("audit_verdict", "")
    if v == "correct_unknown":
        return "honest_unknown"
    if v in ("missed_fertile", "late_recognition", "premature_confidence", "false_convergence_penalized"):
        return v
    if row.get("scout_confidence") == "Unknown":
        return "unknown_active"
    return "neutral"


def build_history_maps() -> tuple[dict, dict]:
    changes = load_csv(LOGS_DIR / "season2_p22_attention_changes.csv")
    promo_hist: dict[tuple, dict] = defaultdict(lambda: {"count": 0, "path": []})
    demo_hist: dict[tuple, dict] = defaultdict(lambda: {"count": 0, "path": []})
    by_sym: dict[str, list] = defaultdict(list)
    for c in changes:
        by_sym[c["symbol"]].append(c)

    scan_times: dict[str, list[str]] = defaultdict(list)
    queue = load_csv(LOGS_DIR / "season2_p21_priority_queue.csv")
    for r in queue:
        scan_times[r["symbol"]].append(r["scan_time"])
    for sym in scan_times:
        scan_times[sym] = sorted(set(scan_times[sym]))

    for sym, scans in scan_times.items():
        series = sorted(by_sym.get(sym, []), key=lambda x: x["to_scan"])
        idx = 0
        pn, dn = 0, 0
        pp, dp = [], []
        for scan in scans:
            while idx < len(series) and series[idx]["to_scan"] <= scan:
                c = series[idx]
                if c["direction"] == "promoted":
                    pn += 1
                    pp.append(c["transition"])
                elif c["direction"] == "demoted":
                    dn += 1
                    dp.append(c["transition"])
                idx += 1
            promo_hist[(sym, scan)] = {"count": pn, "path": ">".join(pp[-5:])}
            demo_hist[(sym, scan)] = {"count": dn, "path": ">".join(dp[-5:])}
    return promo_hist, demo_hist


def build_cases() -> list[dict]:
    p15_idx = {(r["scan_time"], r["symbol"]): r for r in load_csv(LOGS_DIR / "season2_p15_operational_scores.csv")}
    queue = load_csv(LOGS_DIR / "season2_p21_priority_queue.csv")
    if not queue:
        return []

    promo_hist, demo_hist = build_history_maps()
    cases = []
    for q in queue:
        key = (q["scan_time"], q["symbol"])
        p15 = p15_idx.get(key, {})
        ph = promo_hist.get((q["symbol"], q["scan_time"]), {"count": 0, "path": ""})
        dh = demo_hist.get((q["symbol"], q["scan_time"]), {"count": 0, "path": ""})

        case = {
            "case_id": f"{q['symbol']}_{q['scan_time'].replace(' ', '_').replace(':', '')}",
            "date": q["date"],
            "symbol": q["symbol"],
            "scan_time": q["scan_time"],
            "situation": q.get("situation"),
            "supply_context": q.get("supply_context"),
            "health_class": p15.get("health_class", ""),
            "pressure_band": p15.get("pressure_band", ""),
            "participant_state": p15.get("participant_state", ""),
            "grammar": p15.get("grammar", ""),
            "field_verdict": q.get("field_verdict"),
            "field_coherence": q.get("field_coherence"),
            "field_rank": q.get("field_rank"),
            "seedbed_quality": q.get("seedbed_quality") or ("Fertile" if pbool(q.get("in_fertile_seedbed")) else "Unknown"),
            "how_changing": q.get("how_changing", ""),
            "seedbed_arc_path": q.get("seedbed_arc_path", ""),
            "convergence_state": q.get("convergence_state"),
            "convergence_score": q.get("convergence_score"),
            "independent_support": q.get("independent_support"),
            "independent_conflict": q.get("independent_conflict"),
            "support_families": q.get("support_families"),
            "false_convergence_flagged": q.get("false_convergence_flagged"),
            "collapse_risk_pct": q.get("collapse_risk_pct"),
            "convergence_persist_scans": q.get("convergence_persist_scans"),
            "persist_6h_pct": p15.get("persist_6h_pct", ""),
            "priority_tier": q.get("priority_tier"),
            "attention_score": q.get("attention_score"),
            "scout_stance": q.get("scout_stance"),
            "playbook_match": q.get("playbook_match") or "none",
            "promotion_count": ph["count"],
            "demotion_count": dh["count"],
            "promotion_path": ph["path"],
            "demotion_path": dh["path"],
            "unknown_audit": unknown_audit_label(q),
            "audit_verdict": q.get("audit_verdict"),
            "empirical_outcome": q.get("empirical_outcome") or "unknown",
            "recommended_action": q.get("recommended_action"),
            "interaction": "supported" if "interaction" in (q.get("support_families") or "") else "absent",
        }
        case["structure_signature"] = structure_signature(case)
        case["evidence_composition"] = evidence_composition(case)
        cases.append(case)
    return cases


def evolution_summary(case: dict, by_sym: dict[str, list[dict]], horizon: int = 3) -> dict:
    series = by_sym.get(case["symbol"], [])
    idx = next((i for i, c in enumerate(series) if c["scan_time"] == case["scan_time"]), None)
    if idx is None:
        return {"evolution": "unknown", "final_tier": case["priority_tier"], "tier_path": case["priority_tier"]}
    future = series[idx + 1: idx + 1 + horizon]
    if not future:
        return {"evolution": "no_followup", "final_tier": case["priority_tier"], "tier_path": case["priority_tier"]}
    tiers = [case["priority_tier"]] + [f["priority_tier"] for f in future]
    outcomes = [f.get("empirical_outcome") for f in future]
    promoted = any(pi(future[i]["promotion_count"]) > pi(case["promotion_count"]) for i in range(len(future)))
    demoted = any(pi(future[i]["demotion_count"]) > pi(case["demotion_count"]) for i in range(len(future)))
    ev = "stable"
    if promoted and not demoted:
        ev = "promoted_path"
    elif demoted and not promoted:
        ev = "demoted_path"
    elif promoted and demoted:
        ev = "volatile_path"
    return {
        "evolution": ev,
        "final_tier": future[-1]["priority_tier"],
        "tier_path": ">".join(tiers),
        "followup_outcomes": "|".join(outcomes),
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(query: dict, candidate: dict) -> tuple[float, list[str], list[str]]:
    """Structural similarity 0-100. No price features."""
    score = 0.0
    matches: list[str] = []
    diffs: list[str] = []

    fam_q, fam_c = family_set(query), family_set(candidate)
    fam_sim = jaccard(fam_q, fam_c)
    score += fam_sim * FAMILY_WEIGHT
    shared = fam_q & fam_c
    if shared:
        matches.append(f"families={','.join(sorted(shared)[:5])}")

    for field, weight in CAT_WEIGHTS.items():
        qv = str(query.get(field if field != "unknown_audit" else "unknown_audit", query.get(field, "")))
        cv = str(candidate.get(field if field != "playbook_match" else "playbook_match", candidate.get(
            "playbook_match" if field == "playbook_match" else field, ""
        )))
        if field == "playbook_match":
            qv, cv = str(query.get("playbook_match", "none")), str(candidate.get("playbook_match", "none"))
        if qv == cv and qv not in ("", "?"):
            score += weight
            matches.append(f"{field}={qv}")
        elif qv != cv:
            diffs.append(f"{field}:{qv}!={cv}")

    if pbool(query.get("false_convergence_flagged")) == pbool(candidate.get("false_convergence_flagged")):
        score += 4
    else:
        diffs.append("false_convergence_mismatch")

    if bucket_collapse(query.get("collapse_risk_pct")) == bucket_collapse(candidate.get("collapse_risk_pct")):
        score += 3
    if bucket_persist(query.get("convergence_persist_scans")) == bucket_persist(candidate.get("convergence_persist_scans")):
        score += 4
        matches.append(f"persist={bucket_persist(query.get('convergence_persist_scans'))}")
    else:
        diffs.append("persistence_mismatch")

    indep_q, indep_c = pi(query.get("independent_support")), pi(candidate.get("independent_support"))
    if abs(indep_q - indep_c) <= 1:
        score += 4
    if query.get("seedbed_quality") == candidate.get("seedbed_quality"):
        score += 3

    return min(round(score, 1), 100.0), matches[:6], diffs[:6]


def archetype_label(case: dict) -> str:
    pb = case.get("playbook_match", "none")
    if pb != "none":
        return f"playbook_{pb}"
    sit = case.get("situation", "")
    if pbool(case.get("false_convergence_flagged")):
        return "false_convergence_archetype"
    if case.get("unknown_audit") == "honest_unknown":
        return "honest_unknown_archetype"
    if case.get("seedbed_quality") in ("Fertile", "Very fertile") and sit == "Accumulation":
        return "early_fertile_archetype"
    if sit in ("Early Trend", "Healthy Trend") and case.get("priority_tier") in ("A", "S"):
        return "attention_candidate_archetype"
    if case.get("priority_tier") == "X":
        return "avoid_attention_archetype"
    return "background_archetype"


def scout_memory_recommendation(query: dict, similar: list[dict]) -> str:
    if not similar:
        return "Insufficient history — remain Unknown"
    outcomes = Counter(s["historical_outcome"] for s in similar)
    fav = outcomes.get("favorable", 0)
    unfav = outcomes.get("unfavorable", 0)
    false_conv = sum(1 for s in similar if pbool(s.get("historical_false_convergence")))
    promoted = sum(1 for s in similar if "promoted" in (s.get("historical_evolution") or ""))

    if false_conv >= 3:
        return "Resembles false convergence history — maintain Unknown or Watch, no prediction"
    if fav >= 3 and unfav <= 1 and not pbool(query.get("false_convergence_flagged")):
        return "Historical analogues mixed favorable — maintain Watch, monitor persistence"
    if unfav >= 3:
        return "Historical analogues often unfavorable — low attention warranted"
    if promoted >= 2:
        return "Similar cases promoted with persistence — Watch closely, not confident"
    if query.get("unknown_audit") == "honest_unknown":
        return "Unknown remains valid — historical noise without forced clarity"
    return "Maintain Watch — empirical analogues inconclusive"


def enrich_library(cases: list[dict]) -> list[dict]:
    by_sym: dict[str, list] = defaultdict(list)
    for c in cases:
        by_sym[c["symbol"]].append(c)
    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: x["scan_time"])

    library = []
    for c in cases:
        evo = evolution_summary(c, by_sym)
        library.append({
            **c,
            "current_structure": c["structure_signature"],
            "temporal_state": c.get("how_changing") or c.get("seedbed_arc_path") or "flat",
            "promotion_status": f"count={c['promotion_count']}|path={c['promotion_path']}",
            "demotion_status": f"count={c['demotion_count']}|path={c['demotion_path']}",
            "outcome": c["empirical_outcome"],
            "persistence": f"scans={c.get('convergence_persist_scans')}|6h={c.get('persist_6h_pct')}%",
            "false_convergence": c.get("false_convergence_flagged"),
            "unknown_audit": c.get("unknown_audit"),
            "collapse_risk": c.get("collapse_risk_pct"),
            "archetype": archetype_label(c),
            "evolution": evo["evolution"],
            "final_tier_after": evo["final_tier"],
            "tier_path": evo["tier_path"],
            "followup_outcomes": evo.get("followup_outcomes", ""),
        })
    return library


def find_similar(query: dict, library: list[dict], k: int = TOP_K) -> list[dict]:
    candidates = [
        c for c in library
        if c["scan_time"] < query["scan_time"] and not (c["symbol"] == query["symbol"] and c["scan_time"] == query["scan_time"])
    ]
    scored = []
    for c in candidates:
        sim, matches, diffs = similarity(query, c)
        scored.append((sim, c, matches, diffs))
    scored.sort(key=lambda x: -x[0])
    results = []
    for rank, (sim, c, matches, diffs) in enumerate(scored[:k], 1):
        results.append({
            "query_symbol": query["symbol"],
            "query_scan_time": query["scan_time"],
            "query_date": query["date"],
            "query_tier": query["priority_tier"],
            "query_archetype": archetype_label(query),
            "similarity_rank": rank,
            "similarity_score": sim,
            "historical_symbol": c["symbol"],
            "historical_date": c["date"],
            "historical_scan_time": c["scan_time"],
            "structural_match": "|".join(matches),
            "historical_outcome": c["empirical_outcome"],
            "historical_final_tier": c.get("final_tier_after", c["priority_tier"]),
            "historical_evolution": c.get("evolution", ""),
            "promotion_path": c.get("promotion_path", ""),
            "demotion_path": c.get("demotion_path", ""),
            "key_evidence_families": c.get("evidence_composition", ""),
            "important_differences": "|".join(diffs),
            "historical_tier_at_match": c["priority_tier"],
            "historical_playbook": c.get("playbook_match"),
            "historical_false_convergence": c.get("false_convergence_flagged"),
            "historical_scout_stance": c.get("scout_stance"),
        })
    return results


def good_vs_bad_comparison(query: dict, similar: list[dict]) -> dict:
    fav = [s for s in similar if s["historical_outcome"] == "favorable"]
    unfav = [s for s in similar if s["historical_outcome"] == "unfavorable"]
    mixed = [s for s in similar if s["historical_outcome"] == "mixed"]

    fav_tiers = Counter(s["historical_tier_at_match"] for s in fav)
    unfav_tiers = Counter(s["historical_tier_at_match"] for s in unfav)
    fav_fc = sum(1 for s in fav if pbool(s.get("historical_false_convergence")))
    unfav_fc = sum(1 for s in unfav if pbool(s.get("historical_false_convergence")))

    return {
        "query_symbol": query["symbol"],
        "query_scan_time": query["scan_time"],
        "favorable_count": len(fav),
        "unfavorable_count": len(unfav),
        "mixed_count": len(mixed),
        "favorable_common_tier": fav_tiers.most_common(1)[0][0] if fav_tiers else "",
        "unfavorable_common_tier": unfav_tiers.most_common(1)[0][0] if unfav_tiers else "",
        "favorable_false_convergence_pct": round(100 * fav_fc / max(len(fav), 1), 1),
        "unfavorable_false_convergence_pct": round(100 * unfav_fc / max(len(unfav), 1), 1),
        "structural_separation": _structural_separation(query, fav, unfav),
        "missing_evidence_in_unfav": _missing_evidence(query, unfav),
        "empirical_explanation": _empirical_explanation(query, fav, unfav, mixed),
    }


_library_lookup: list[dict] = []


def _structural_separation(query: dict, fav: list, unfav: list) -> str:
    parts = []
    if fav and not unfav:
        parts.append("only_favorable_analogues")
    elif unfav and not fav:
        parts.append("only_unfavorable_analogues")
    else:
        fav_promo = sum(1 for s in fav if s.get("promotion_path"))
        unfav_promo = sum(1 for s in unfav if s.get("promotion_path"))
        if fav_promo > unfav_promo:
            parts.append("favorable_had_more_promotion_paths")
        fav_fc = sum(1 for s in fav if pbool(s.get("historical_false_convergence")))
        unfav_fc = sum(1 for s in unfav if pbool(s.get("historical_false_convergence")))
        if unfav_fc > fav_fc:
            parts.append("unfavorable_had_more_false_convergence")
        if pi(query.get("convergence_persist_scans")) == 0:
            parts.append("current_lacks_persistence")
    return "|".join(parts) if parts else "inconclusive"


def _missing_evidence(query: dict, unfav: list) -> str:
    if not unfav:
        return ""
    qf = family_set(query)
    unfav_fams = set()
    for s in unfav:
        for f in (s.get("key_evidence_families") or "").split("|"):
            if f:
                unfav_fams.add(f)
    missing = unfav_fams - qf
    return "|".join(sorted(missing)[:5])


def _empirical_explanation(query: dict, fav: list, unfav: list, mixed: list) -> str:
    if len(fav) >= 2 and len(unfav) >= 2:
        return "Current resembles both favorable and false structures — main uncertainty: field persistence"
    if len(fav) >= 3:
        return "Historical analogues skew favorable — maintain Watch without prediction"
    if len(unfav) >= 3:
        return "Historical analogues skew unfavorable — patience or low attention warranted"
    if pbool(query.get("false_convergence_flagged")):
        return "False convergence pattern in history — Unknown was often correct"
    if mixed:
        return "Mixed empirical outcomes — no forced certainty"
    return "Sparse analogue history — remain Unknown"


def build_case_memory(library: list[dict]) -> list[dict]:
    archetypes: dict[str, list] = defaultdict(list)
    for c in library:
        archetypes[c["archetype"]].append(c)

    rows = []
    for arch, items in sorted(archetypes.items(), key=lambda x: -len(x[1])):
        outcomes = Counter(i["empirical_outcome"] for i in items)
        tiers = Counter(i["priority_tier"] for i in items)
        evo = Counter(i.get("evolution", "") for i in items)
        fam_counter = Counter()
        for i in items:
            for f in (i.get("evidence_composition") or "").split("|"):
                if f:
                    fam_counter[f] += 1
        rows.append({
            "archetype": arch,
            "case_count": len(items),
            "common_situation": Counter(i["situation"] for i in items).most_common(1)[0][0],
            "common_tier": tiers.most_common(1)[0][0],
            "favorable_pct": round(100 * outcomes.get("favorable", 0) / len(items), 1),
            "unfavorable_pct": round(100 * outcomes.get("unfavorable", 0) / len(items), 1),
            "mixed_pct": round(100 * outcomes.get("mixed", 0) / len(items), 1),
            "promoted_path_pct": round(100 * evo.get("promoted_path", 0) / len(items), 1),
            "demoted_path_pct": round(100 * evo.get("demoted_path", 0) / len(items), 1),
            "top_evidence_families": "|".join(f"{k}({v})" for k, v in fam_counter.most_common(5)),
            "false_convergence_pct": round(100 * sum(1 for i in items if pbool(i.get("false_convergence_flagged"))) / len(items), 1),
            "scout_lesson": _archetype_lesson(arch, outcomes, items),
            "recommended_stance": _archetype_stance(arch),
        })
    return rows


def _archetype_lesson(arch: str, outcomes: Counter, items: list) -> str:
    if "false" in arch:
        return "Cosmetic agreement — families align but outcomes mixed; Unknown often correct"
    if "honest_unknown" in arch:
        return "Forcing clarity wasted attention — patience validated empirically"
    if "early_fertile" in arch:
        return "Fertile seedbed attracts Watch — promotion needs 2+ scan persistence"
    if "avoid" in arch:
        return "High conflict or collapse — attention poorly spent here"
    fav_rate = outcomes.get("favorable", 0) / max(len(items), 1)
    if fav_rate > 0.5:
        return "Structure often followed favorable path — still Watch not Buy"
    return "Background ecology — check only if time permits"


def _archetype_stance(arch: str) -> str:
    if arch in ("playbook_C", "false_convergence_archetype", "honest_unknown_archetype"):
        return "Unknown"
    if arch in ("playbook_A", "playbook_B", "playbook_E", "early_fertile_archetype", "attention_candidate_archetype"):
        return "Watch"
    if arch == "avoid_attention_archetype":
        return "Avoid attention"
    return "Background"


def example_narrative(symbol: str, library: list[dict]) -> str | None:
    """Formatted analogue narrative for report (e.g. AIOTUSDT-style example)."""
    sym_cases = [c for c in library if c["symbol"] == symbol and c["date"] >= "2026-06-08"]
    if not sym_cases:
        return None
    query = sym_cases[-1]
    sim = find_similar(query, library, 3)
    if not sim:
        return None
    lines = [f"{symbol} @ {query['scan_time']}", "Most similar historical cases:"]
    for s in sim:
        lines.append(
            f"  {s['similarity_rank']}. {s['historical_symbol']} sim={s['similarity_score']} "
            f"outcome={s['historical_outcome']} evolution={s['historical_evolution']} "
            f"tier={s['historical_tier_at_match']}"
        )
    comp = good_vs_bad_comparison(query, sim)
    rec = scout_memory_recommendation(query, sim)
    lines.append(f"Scout Summary: {comp['empirical_explanation']}. {rec}. No prediction.")
    return "\n".join(lines)


def build_report(
    library: list[dict],
    similar_all: list[dict],
    comparisons: list[dict],
    current_rows: list[dict],
    memory: list[dict],
) -> str:
    latest = max(c["date"] for c in library)
    lines = [
        "===== SCOUT SEASON2 P23 - MEMORY & CASE LIBRARY =====",
        "",
        f"Case library: {len(library)} observations | Similar retrievals: {len(similar_all)}",
        f"Latest date queries: {sum(1 for c in current_rows if c.get('query_date') == latest)}",
        "",
        "--- Task 1: Similar historical structures ---",
        "Top analogue retrievals computed per latest observation (structural, not price).",
        "",
        "--- Task 2: How similar cases evolved ---",
    ]
    evo_dist = Counter(c.get("evolution") for c in library)
    for k, v in evo_dist.most_common():
        lines.append(f"  {k}: {v}")

    lines.extend(["", "--- Task 3: Favorable vs unfavorable analogues ---"])
    fav = sum(1 for c in library if c["empirical_outcome"] == "favorable")
    unfav = sum(1 for c in library if c["empirical_outcome"] == "unfavorable")
    mixed = sum(1 for c in library if c["empirical_outcome"] == "mixed")
    lines.append(f"  Library outcomes: favorable={fav} unfavorable={unfav} mixed={mixed}")

    lines.extend(["", "--- Task 4: Good vs bad structural separation ---"])
    for comp in comparisons[:5]:
        lines.append(f"  {comp['query_symbol']}: {comp['empirical_explanation']}")

    lines.extend(["", "--- Task 5: Historical archetypes ---"])
    for m in memory[:6]:
        lines.append(f"  {m['archetype']}: n={m['case_count']} stance={m['recommended_stance']}")

    lines.extend(["", "--- Task 6: Example Scout memory (latest) ---"])
    for cr in current_rows[:4]:
        lines.append(f"  {cr['query_symbol']}: {cr['scout_summary'][:130]}")

    lines.extend(["", "--- Worked example (structural retrieval) ---"])
    for sym in ("AIOTUSDT", "STGUSDT", "ALLOUSDT"):
        ex = example_narrative(sym, library)
        if ex:
            lines.append(ex)
            lines.append("")
            break

    lines.extend([
        "",
        "--- Final principles ---",
        "Never forecast prices. Retrieve empirical analogues only.",
        "Unknown remains valid. A great Scout remembers.",
    ])
    lines.extend(mission_summary_lines())
    lines.append("=" * 58)
    return "\n".join(lines)


def ensure_deps() -> None:
    if not (LOGS_DIR / "season2_p21_priority_queue.csv").exists():
        import season2_p21_priority_queue
        season2_p21_priority_queue.main()
    if not (LOGS_DIR / "season2_p22_attention_changes.csv").exists():
        import season2_p22_scout_promotion_demotion
        season2_p22_scout_promotion_demotion.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-deps", action="store_true")
    args = parser.parse_args()

    if args.rebuild_deps:
        ensure_deps()
    else:
        ensure_deps()

    global _library_lookup
    cases = build_cases()
    if not cases:
        print("Run P15-P22 first")
        return

    library = enrich_library(cases)
    _library_lookup = library

    latest_date = max(c["date"] for c in library)
    current_cases = [c for c in library if c["date"] == latest_date]

    similar_all: list[dict] = []
    comparisons: list[dict] = []
    current_rows: list[dict] = []

    for query in current_cases:
        sim = find_similar(query, library, TOP_K)
        similar_all.extend(sim)
        comp = good_vs_bad_comparison(query, sim)
        comparisons.append(comp)

        top_lines = []
        for s in sim[:3]:
            top_lines.append(
                f"{s['historical_symbol']} sim={s['similarity_score']} outcome={s['historical_outcome']} "
                f"evolution={s['historical_evolution']} tier={s['historical_tier_at_match']}"
            )
        rec = scout_memory_recommendation(query, sim)
        current_rows.append({
            "query_date": query["date"],
            "query_symbol": query["symbol"],
            "query_scan_time": query["scan_time"],
            "query_structure": query["structure_signature"],
            "query_tier": query["priority_tier"],
            "query_playbook": query["playbook_match"],
            "query_archetype": archetype_label(query),
            "query_outcome_unknown": "yes — retrieval only",
            "top_similar_cases": " || ".join(top_lines),
            "favorable_analogues": comp["favorable_count"],
            "unfavorable_analogues": comp["unfavorable_count"],
            "structural_separation": comp["structural_separation"],
            "empirical_explanation": comp["empirical_explanation"],
            "scout_recommendation": rec,
            "scout_summary": f"{comp['empirical_explanation']}. {rec}. No prediction.",
            "should_remain_unknown": "yes" if "Unknown" in rec else "no",
            "should_remain_watch": "yes" if "Watch" in rec else "optional",
            "more_attention": "yes" if "Watch closely" in rec else "no",
        })

    memory = build_case_memory(library)
    report = build_report(library, similar_all, comparisons, current_rows, memory)

    write_csv(CASE_LIBRARY_CSV, library)
    write_csv(SIMILAR_CASES_CSV, similar_all)
    write_csv(GOOD_BAD_CSV, comparisons)
    write_csv(CASE_MEMORY_CSV, memory)
    write_csv(CURRENT_HISTORY_CSV, current_rows)
    REPORT_TXT.write_text(report, encoding="utf-8")

    print("===== P23 SCOUT MEMORY & CASE LIBRARY =====")
    print(f"Cases: {len(library)} | Similar pairs: {len(similar_all)} | Archetypes: {len(memory)}")
    print(f"Latest queries: {len(current_cases)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
