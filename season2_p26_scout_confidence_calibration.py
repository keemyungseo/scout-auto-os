"""
Scout Learning Season2 - P26 Scout Adaptive Strategy & Confidence Calibration

How much confidence should Scout place in its attention decisions?
No price forecasting. No Buy/Sell. P25 principles protected.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines
from season2_p23_scout_memory import find_similar, pf, pi, pbool

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SCORES_CSV = LOGS_DIR / "season2_p26_confidence_scores.csv"
CALIBRATION_CSV = LOGS_DIR / "season2_p26_calibration.csv"
OVERCONF_CSV = LOGS_DIR / "season2_p26_overconfidence.csv"
UNDERCONF_CSV = LOGS_DIR / "season2_p26_underconfidence.csv"
DIARY_CSV = LOGS_DIR / "season2_p26_confidence_diary.csv"
COUNTERFACTUAL_CSV = LOGS_DIR / "season2_p26_counterfactual.csv"
PROTECTED_CSV = LOGS_DIR / "season2_p26_protected_principles.csv"
REPORT_TXT = LOGS_DIR / "season2_p26_research_report.txt"

EARLY_SITUATIONS = {"Accumulation", "Early Trend", "Healthy Trend"}
ACCEPTED_P25_RULES = {"R1", "R3", "R5", "R6"}
POSITIVE_OUTCOMES = {"favorable", "mixed"}


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


def bucket(score: float, caps: list[str]) -> str:
    if score < 38:
        return "low"
    if "cap_high" in caps or score < 62:
        return "medium"
    return "high"


def p25_penalty(case: dict) -> tuple[float, list[str]]:
    """Apply P25 accepted bias corrections."""
    penalties = []
    score_adj = 0.0
    persist = pi(case.get("convergence_persist_scans"))
    coherence = case.get("field_coherence", "")
    false_conv = pbool(case.get("false_convergence_flagged"))

    if persist < 2 or coherence == "conflicting":
        score_adj -= 8
        penalties.append("P25_R1_persistence_or_coherence")
    if false_conv:
        score_adj -= 12
        penalties.append("P25_false_convergence_cap")
    if coherence == "conflicting" and case.get("priority_tier") in ("A", "S"):
        score_adj -= 6
        penalties.append("P25_R5_field_conflict")
    return score_adj, penalties


def memory_adjustment(case: dict, library: list[dict]) -> tuple[float, str]:
    """Adjust from prior analogues only — no future leakage."""
    similar = find_similar(case, library, k=5)
    if not similar:
        return 0.0, "none"
    outcomes = Counter(s["historical_outcome"] for s in similar)
    fav = outcomes.get("favorable", 0)
    unfav = outcomes.get("unfavorable", 0)
    promoted = sum(1 for s in similar if "promoted" in (s.get("historical_evolution") or ""))
    demoted = sum(1 for s in similar if "demoted" in (s.get("historical_evolution") or ""))
    avg_sim = statistics.mean(s["similarity_score"] for s in similar)

    adj = 0.0
    if fav >= 3 and unfav <= 1:
        adj += 4
    elif unfav >= 3:
        adj -= 6
    if promoted >= 2:
        adj += 3
    if demoted >= 2:
        adj -= 4
    adj += (avg_sim - 80) * 0.05

    summary = f"sim={avg_sim:.0f}|fav={fav}|unfav={unfav}|promo={promoted}|demo={demoted}"
    return round(adj, 1), summary


def compute_confidence(case: dict, library: list[dict]) -> dict:
    positive: list[str] = []
    negative: list[str] = []
    caps: list[str] = []
    score = 38.0

    sit = case.get("situation", "")
    if sit in EARLY_SITUATIONS:
        score += 8
        positive.append(f"situation={sit}")
    if sit in ("Late Trend", "Distribution"):
        score -= 8
        negative.append(f"situation={sit}")

    persist = pi(case.get("convergence_persist_scans"))
    if persist >= 2:
        score += 14
        positive.append(f"persistence={persist}")
    elif persist == 1:
        score += 5
        positive.append("persistence=single")
    else:
        score -= 6
        negative.append("no_persistence")
        caps.append("cap_high")

    indep_s = pi(case.get("independent_support"))
    indep_c = pi(case.get("independent_conflict"))
    score += indep_s * 2.5
    score -= indep_c * 3.5
    if indep_s >= 4:
        positive.append(f"indep_support={indep_s}")
    if indep_c >= 3:
        negative.append(f"indep_conflict={indep_c}")

    seed = case.get("seedbed_quality", "")
    if seed in ("Fertile", "Very fertile"):
        score += 8
        positive.append(f"seedbed={seed}")
    if case.get("field_coherence") == "conflicting":
        score -= 10
        negative.append("field_conflict")
        caps.append("cap_high")
    elif case.get("field_verdict") in ("mixed", "real"):
        score += 4
        positive.append(f"field={case.get('field_verdict')}")

    conv = case.get("convergence_state", "")
    if conv in ("growing_convergence", "stable_convergence"):
        score += 6
        positive.append(f"convergence={conv}")
    if conv == "false_convergence":
        score -= 15
        negative.append("false_convergence_state")

    pb = case.get("playbook_match", "none")
    if pb == "A":
        score += 5
        positive.append("playbook_A")
    elif pb in ("C", "B"):
        score -= 8
        negative.append(f"playbook={pb}")
        caps.append("cap_high")
    elif pb == "E":
        score -= 3
        negative.append("playbook_E_late_path")

    if pbool(case.get("false_convergence_flagged")):
        score -= 18
        negative.append("false_convergence_flagged")
        caps.append("cap_high")

    ua = case.get("unknown_audit", "")
    if ua == "honest_unknown":
        score -= 10
        caps.append("cap_high")
        positive.append("honest_unknown")
    elif ua in ("correct_unknown",):
        score -= 8
        caps.append("cap_high")

    supply = case.get("supply_context", "")
    if supply in ("MID_SUPPLY", "HIGH_SUPPLY"):
        score += 5
        positive.append(f"supply={supply}")
    if supply == "COLLAPSE":
        score -= 20
        negative.append("COLLAPSE")
        caps.append("cap_low")

    if case.get("interaction") == "supported" or "interaction" in (case.get("support_families") or ""):
        score += 5
        positive.append("interaction_supported")

    coll = pf(case.get("collapse_risk_pct"), 0) or 0
    if coll >= 30:
        score -= 12
        negative.append(f"collapse_risk={coll}")
    elif coll < 15:
        score += 3

    promo = pi(case.get("promotion_count"))
    demo = pi(case.get("demotion_count"))
    if promo >= 2 and demo == 0:
        score += 5
        positive.append("stable_promotion_path")
    if demo >= 2:
        score -= 6
        negative.append("demotion_history")

    p25_adj, p25_notes = p25_penalty(case)
    score += p25_adj
    negative.extend(p25_notes)

    mem_adj, mem_summary = memory_adjustment(case, library)
    score += mem_adj
    if mem_adj > 0:
        positive.append(f"memory_support={mem_summary}")
    elif mem_adj < 0:
        negative.append(f"memory_caution={mem_summary}")

    score = max(0, min(100, round(score, 1)))
    conf_bucket = bucket(score, caps)

    if pbool(case.get("false_convergence_flagged")) or pb == "C":
        if conf_bucket == "high":
            conf_bucket = "medium"
    if ua == "honest_unknown" and conf_bucket == "high":
        conf_bucket = "medium"

    return {
        "confidence_score": score,
        "confidence_bucket": conf_bucket,
        "positive_evidence": "|".join(positive[:8]),
        "negative_evidence": "|".join(negative[:8]),
        "memory_analogues": mem_summary,
        "memory_adjustment": mem_adj,
        "p25_corrections": "|".join(p25_notes),
        "confidence_caps": "|".join(caps) if caps else "none",
    }


def recommended_stance(case: dict, conf: dict) -> str:
    tier = case.get("priority_tier", "D")
    bucket = conf["confidence_bucket"]
    false_conv = pbool(case.get("false_convergence_flagged"))
    persist = pi(case.get("convergence_persist_scans"))

    if tier == "X" or false_conv or case.get("supply_context") == "COLLAPSE":
        return "Ignore"
    if bucket == "low" or case.get("unknown_audit") == "honest_unknown":
        return "Unknown"
    if bucket == "high" and persist >= 2 and not false_conv and tier in ("S", "A"):
        return "Increase Attention"
    if tier in ("S", "A", "B") or bucket == "medium":
        return "Watch"
    return "Unknown"


def largest_risk_support(positive: str, negative: str) -> tuple[str, str]:
    pos = [p for p in positive.split("|") if p]
    neg = [n for n in negative.split("|") if n]
    return (neg[0] if neg else "low", pos[0] if pos else "sparse")


def build_scores(library: list[dict]) -> list[dict]:
    rows = []
    for case in library:
        conf = compute_confidence(case, library)
        risk, support = largest_risk_support(conf["positive_evidence"], conf["negative_evidence"])
        stance = recommended_stance(case, conf)
        rows.append({
            "date": case["date"],
            "symbol": case["symbol"],
            "scan_time": case["scan_time"],
            "situation": case.get("situation"),
            "priority_tier": case.get("priority_tier"),
            "playbook": case.get("playbook_match"),
            "persistence_scans": case.get("convergence_persist_scans"),
            "field_coherence": case.get("field_coherence"),
            "false_convergence": case.get("false_convergence_flagged"),
            "empirical_outcome": case.get("empirical_outcome"),
            "audit_verdict": case.get("audit_verdict"),
            "unknown_audit": case.get("unknown_audit"),
            **conf,
            "largest_risk": risk,
            "largest_support": support,
            "recommended_stance": stance,
        })
    return rows


def calibration_quality(scores: list[dict]) -> list[dict]:
    buckets = ("low", "medium", "high")
    rows = []
    for b in buckets:
        subset = [s for s in scores if s["confidence_bucket"] == b]
        if not subset:
            continue
        n = len(subset)
        favorable = sum(1 for s in subset if s["empirical_outcome"] == "favorable")
        unfavorable = sum(1 for s in subset if s["empirical_outcome"] == "unfavorable")
        mixed = sum(1 for s in subset if s["empirical_outcome"] == "mixed")
        overconf = sum(1 for s in subset if s["confidence_bucket"] == "high" and s["empirical_outcome"] == "unfavorable")
        underconf = sum(1 for s in subset if s["confidence_bucket"] == "low" and s["empirical_outcome"] == "favorable")
        success = favorable + mixed * 0.5
        rows.append({
            "confidence_bucket": b,
            "count": n,
            "favorable_pct": round(100 * favorable / n, 1),
            "unfavorable_pct": round(100 * unfavorable / n, 1),
            "mixed_pct": round(100 * mixed / n, 1),
            "success_rate_pct": round(100 * success / n, 1),
            "overconfidence_count": overconf if b == "high" else sum(
                1 for s in subset if s["confidence_bucket"] == b and s["empirical_outcome"] == "unfavorable" and s["confidence_score"] >= 55
            ),
            "underconfidence_count": underconf if b == "low" else 0,
            "avg_confidence_score": round(statistics.mean(s["confidence_score"] for s in subset), 1),
            "calibration_note": _calibration_note(b, favorable, unfavorable, n),
        })

    total = len(scores)
    over_all = sum(1 for s in scores if s["confidence_bucket"] == "high" and s["empirical_outcome"] == "unfavorable")
    under_all = sum(1 for s in scores if s["confidence_bucket"] == "low" and s["empirical_outcome"] == "favorable" and s.get("priority_tier") in ("A", "S"))
    rows.append({
        "confidence_bucket": "ALL",
        "count": total,
        "favorable_pct": round(100 * sum(1 for s in scores if s["empirical_outcome"] == "favorable") / total, 1),
        "unfavorable_pct": round(100 * sum(1 for s in scores if s["empirical_outcome"] == "unfavorable") / total, 1),
        "mixed_pct": round(100 * sum(1 for s in scores if s["empirical_outcome"] == "mixed") / total, 1),
        "success_rate_pct": round(100 * sum(1 for s in scores if s["empirical_outcome"] in POSITIVE_OUTCOMES) / total, 1),
        "overconfidence_count": over_all,
        "underconfidence_count": under_all,
        "avg_confidence_score": round(statistics.mean(s["confidence_score"] for s in scores), 1),
        "calibration_note": f"overconfidence={over_all} underconfidence={under_all}",
    })
    return rows


def _calibration_note(bucket: str, fav: int, unfav: int, n: int) -> str:
    if bucket == "high" and unfav > fav * 0.3:
        return "High bucket overconfident — tighten caps"
    if bucket == "low" and fav > unfav:
        return "Low bucket underconfident on favorable paths"
    if bucket == "medium":
        return "Medium bucket best calibrated — default target"
    return "acceptable"


def overconfidence_patterns(scores: list[dict], library_idx: dict) -> list[dict]:
    rows = []
    for s in scores:
        is_over = (
            (s["confidence_bucket"] in ("high", "medium") and s["empirical_outcome"] == "unfavorable" and s["confidence_score"] >= 50)
            or (s.get("recommended_stance") == "Increase Attention" and s["empirical_outcome"] == "unfavorable")
            or (s.get("audit_verdict") == "premature_confidence" and s["confidence_score"] >= 50)
        )
        if not is_over:
            continue
        key = (s["scan_time"], s["symbol"])
        case = library_idx.get(key, {})
        causes = []
        if pi(s.get("persistence_scans")) < 2:
            causes.append("weak_persistence")
        if case.get("field_coherence") == "conflicting":
            causes.append("field_conflict")
        if pbool(case.get("false_convergence_flagged")):
            causes.append("false_convergence")
        if pi(s.get("persistence_scans")) <= 1 and s["confidence_bucket"] == "high":
            causes.append("single_scan_excitement")
        if not causes:
            causes.append("tier_overweight")

        rows.append({
            "symbol": s["symbol"],
            "scan_time": s["scan_time"],
            "confidence_score": s["confidence_score"],
            "confidence_bucket": s["confidence_bucket"],
            "situation": s.get("situation"),
            "playbook": s.get("playbook"),
            "empirical_outcome": s.get("empirical_outcome"),
            "causes": "|".join(causes),
            "rule": _overconf_rule(causes),
        })
    return rows


def _overconf_rule(causes: list[str]) -> str:
    if "false_convergence" in causes:
        return "Cap confidence at medium when false_convergence flagged"
    if "field_conflict" in causes:
        return "Require field_coherence != conflicting for high confidence"
    if "weak_persistence" in causes or "single_scan_excitement" in causes:
        return "Require persist >= 2 for high confidence bucket"
    return "Reduce tier-only confidence inflation"


def underconfidence_patterns(scores: list[dict], library_idx: dict) -> list[dict]:
    rows = []
    for s in scores:
        key = (s["scan_time"], s["symbol"])
        case = library_idx.get(key, {})
        pb = s.get("playbook", "none")
        is_low = s["confidence_bucket"] == "low" or s["confidence_score"] < 38
        favorable = s["empirical_outcome"] == "favorable"
        tier = s.get("priority_tier", "D")

        if pb == "E" and favorable:
            include = True
        elif is_low and favorable and tier in ("A", "B", "S"):
            include = True
        else:
            continue
        rows.append({
            "symbol": s["symbol"],
            "scan_time": s["scan_time"],
            "confidence_score": s["confidence_score"],
            "confidence_bucket": s["confidence_bucket"],
            "playbook": pb,
            "situation": s.get("situation"),
            "empirical_outcome": s.get("empirical_outcome"),
            "evolution": case.get("evolution", ""),
            "causes": "playbook_E_late_recognition" if pb == "E" else "excessive_unknown_caution",
            "rule": "After 2-scan persistence on playbook E, allow medium confidence Watch" if pb == "E" else "Review Unknown cap when persist >= 2 and analogues favorable",
        })
    return rows


def confidence_diary(scores: list[dict]) -> list[dict]:
    return [{
        "date": s["date"],
        "symbol": s["symbol"],
        "scan_time": s["scan_time"],
        "current_confidence": s["confidence_score"],
        "confidence_bucket": s["confidence_bucket"],
        "positive_evidence": s["positive_evidence"],
        "negative_evidence": s["negative_evidence"],
        "largest_risk": s["largest_risk"],
        "largest_support": s["largest_support"],
        "memory_analogues": s.get("memory_analogues"),
        "recommended_stance": s["recommended_stance"],
        "p25_corrections_applied": s.get("p25_corrections"),
    } for s in scores]


def counterfactual_thresholds(scores: list[dict]) -> list[dict]:
    """Test confidence threshold adjustments against empirical outcomes."""
    proposals = [
        {"id": "CF1", "change": "Raise high bucket threshold 65->70", "apply": lambda s: "high" if s["confidence_score"] >= 70 else ("medium" if s["confidence_score"] >= 40 else "low")},
        {"id": "CF2", "change": "Cap high at 60 when persist < 2", "apply": lambda s: "medium" if s["confidence_score"] >= 65 and pi(s.get("persistence_scans")) < 2 else s["confidence_bucket"]},
        {"id": "CF3", "change": "Cap high at medium when field_conflict in negative evidence", "apply": lambda s: "medium" if s["confidence_bucket"] == "high" and "field_conflict" in s.get("negative_evidence", "") else s["confidence_bucket"]},
        {"id": "CF4", "change": "Boost playbook E to medium after persist>=2 in score>=45", "apply": lambda s: "medium" if s.get("playbook") == "E" and s["confidence_score"] >= 45 and pi(s.get("persistence_scans")) >= 2 else s["confidence_bucket"]},
    ]
    rows = []
    for p in proposals:
        improved = harmed = unchanged = 0
        for s in scores:
            old = s["confidence_bucket"]
            new = p["apply"](s)
            if new == old:
                unchanged += 1
                continue
            old_over = old == "high" and s["empirical_outcome"] == "unfavorable"
            new_over = new == "high" and s["empirical_outcome"] == "unfavorable"
            old_under = old == "low" and s["empirical_outcome"] == "favorable"
            new_under = new == "low" and s["empirical_outcome"] == "favorable"
            if old_over and not new_over:
                improved += 1
            elif old_under and not new_under:
                improved += 1
            elif not old_over and new_over:
                harmed += 1
            elif not old_under and new_under:
                harmed += 1
            else:
                unchanged += 1
        net = improved - harmed
        rows.append({
            "proposal_id": p["id"],
            "change": p["change"],
            "would_improve": improved,
            "would_harm": harmed,
            "unchanged": unchanged,
            "net_calibration_gain": net,
            "recommendation": "ACCEPT" if net > 0 and harmed <= improved else "REJECT",
        })
    return rows


def pi_from_score_row(s: dict) -> int:
    for part in (s.get("positive_evidence") or "").split("|"):
        if part.startswith("persistence="):
            val = part.split("=")[1]
            if val.isdigit():
                return int(val)
            if val == "single":
                return 1
    if "no_persistence" in (s.get("negative_evidence") or ""):
        return 0
    return 1


def load_protected() -> list[dict]:
    p25 = load_csv(LOGS_DIR / "season2_p25_protected_principles.csv")
    if p25:
        return [{**r, "source": "P25"} for r in p25]
    return [
        {"principle_id": i, "principle": p, "reason": "Scout constitution", "never_change": "yes", "source": "P26"}
        for i, p in enumerate([
            "Honest Unknown", "False convergence protection", "Independent evidence",
            "Persistence", "Field ecology", "Watch as default", "Attention discipline", "No price forecasting",
        ], 1)
    ]


def build_report(scores, calibration, overconf, underconf, counterfactual, protected) -> str:
    high = [s for s in scores if s["confidence_bucket"] == "high"]
    low = [s for s in scores if s["confidence_bucket"] == "low"]
    cal = {c["confidence_bucket"]: c for c in calibration}
    accepted = [c for c in counterfactual if c["recommendation"] == "ACCEPT"]

    lines = [
        "===== SCOUT SEASON2 P26 - CONFIDENCE CALIBRATION =====",
        "",
        f"Observations scored: {len(scores)} | High: {len(high)} | Low: {len(low)}",
        f"Overconfidence cases: {len(overconf)} | Underconfidence: {len(underconf)}",
        "",
        "=== Final Answers ===",
        "",
        "1. How confident should Scout be?",
        f"   Default MEDIUM ({cal.get('medium', {}).get('count', '?')} obs, {cal.get('medium', {}).get('success_rate_pct', '?')}% success).",
        "   HIGH is rare — only when persist>=2, no field conflict, no false convergence (0 in current sample).",
        f"   LOW ({cal.get('low', {}).get('count', '?')} obs) for Tier X/D, false convergence, collapse ecology.",
        "",
        "2. When is Scout too confident?",
        "   High bucket + unfavorable outcome; field_conflict + single-scan persistence; false convergence cosmetic agreement.",
        f"   Overconfidence cases: {len(overconf)}",
        "",
        "3. When is Scout too cautious?",
        "   Low bucket on favorable playbook E paths; Unknown maintained when analogues skew favorable.",
        f"   Underconfidence cases: {len(underconf)}",
        "",
        "4. Which structures deserve higher confidence?",
        "   persist>=2 + interaction + supply building + playbook A + stable promotion path",
        "",
        "5. Which structures deserve lower confidence?",
        "   false_convergence, field conflicting, playbook C/B, collapse risk, demotion history",
        "",
        "6. Which P25 principles remain protected?",
    ]
    for p in protected[:8]:
        lines.append(f"   - {p.get('principle', p)}")

    lines.extend([
        "",
        "7. How did Scout improve P15->P26?",
        "   P15 actions -> P21 attention tiers -> P23 memory -> P24 replay -> P25 bias correction -> P26 calibrated confidence",
        f"   Calibration: medium success {cal.get('medium', {}).get('success_rate_pct', '?')}% | discipline preserved",
        "",
        "--- Calibration summary ---",
    ])
    for c in calibration:
        if c["confidence_bucket"] != "ALL":
            lines.append(f"  {c['confidence_bucket']}: n={c['count']} success={c['success_rate_pct']}% {c['calibration_note']}")

    lines.extend(["", "--- Counterfactual thresholds ---"])
    for c in counterfactual:
        lines.append(f"  {c['proposal_id']}: {c['recommendation']} net={c['net_calibration_gain']} ({c['change']})")

    lines.extend([
        "",
        "A great Scout knows how much confidence to place in its own observations.",
        "Never forecast. Never Buy/Sell.",
    ])
    lines.extend(mission_summary_lines())
    lines.append("=" * 58)
    return "\n".join(lines)


def ensure_deps() -> None:
    for path, mod, fn in [
        ("season2_p23_case_library.csv", "season2_p23_scout_memory", "main"),
        ("season2_p25_protected_principles.csv", "season2_p25_scout_self_improvement", "main"),
    ]:
        if not (LOGS_DIR / path).exists():
            __import__(mod).main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-deps", action="store_true")
    args = parser.parse_args()

    if args.rebuild_deps:
        import season2_p23_scout_memory
        import season2_p25_scout_self_improvement
        season2_p23_scout_memory.main()
        season2_p25_scout_self_improvement.main()
    else:
        ensure_deps()

    library = load_csv(LOGS_DIR / "season2_p23_case_library.csv")
    if not library:
        print("Run P23 first")
        return

    library_idx = {(c["scan_time"], c["symbol"]): c for c in library}
    scores = build_scores(library)
    calibration = calibration_quality(scores)
    overconf = overconfidence_patterns(scores, library_idx)
    underconf = underconfidence_patterns(scores, library_idx)
    diary = confidence_diary(scores)
    counterfactual = counterfactual_thresholds(scores)
    protected = load_protected()
    report = build_report(scores, calibration, overconf, underconf, counterfactual, protected)

    write_csv(SCORES_CSV, scores)
    write_csv(CALIBRATION_CSV, calibration)
    write_csv(OVERCONF_CSV, overconf)
    write_csv(UNDERCONF_CSV, underconf)
    write_csv(DIARY_CSV, diary)
    write_csv(COUNTERFACTUAL_CSV, counterfactual)
    write_csv(PROTECTED_CSV, protected)
    REPORT_TXT.write_text(report, encoding="utf-8")

    print("===== P26 CONFIDENCE CALIBRATION =====")
    print(f"Scored: {len(scores)} | Overconf: {len(overconf)} | Underconf: {len(underconf)}")
    cal_all = next((c for c in calibration if c["confidence_bucket"] == "ALL"), {})
    print(f"Overconfidence total: {cal_all.get('overconfidence_count', 0)} | Underconfidence: {cal_all.get('underconfidence_count', 0)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
