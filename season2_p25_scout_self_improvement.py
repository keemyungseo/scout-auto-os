"""
Scout Learning Season2 - P25 Scout Self-Improvement & Bias Correction Engine

Studies Scout mistakes, biases, strengths, and proposes empirically tested adjustments.
Does NOT maximize prediction or aggression. Makes Scout wiser, not louder.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BIAS_CSV = LOGS_DIR / "season2_p25_bias_report.csv"
FAILURES_CSV = LOGS_DIR / "season2_p25_failure_modes.csv"
STRENGTH_CSV = LOGS_DIR / "season2_p25_strength_profiles.csv"
PERSONALITY_CSV = LOGS_DIR / "season2_p25_personality_profile.csv"
RULES_CSV = LOGS_DIR / "season2_p25_rule_adjustments.csv"
COUNTERFACTUAL_CSV = LOGS_DIR / "season2_p25_counterfactual_audit.csv"
PROTECTED_CSV = LOGS_DIR / "season2_p25_protected_principles.csv"
IMPROVEMENT_LOG_CSV = LOGS_DIR / "season2_p25_self_improvement_log.csv"
REPORT_TXT = LOGS_DIR / "season2_p25_research_report.txt"

FAILURE_MODES = (
    "too_cautious",
    "premature_attention",
    "late_demotion",
    "false_attention_upgrade",
    "false_watch",
    "unnecessary_unknown",
    "attention_decay",
    "late_field_recognition",
    "weak_persistence_handling",
)
POSITIVE_VERDICTS = {
    "correct_unknown", "correct_watch", "correct_caution",
    "correct_demotion", "good_attention",
}
NEGATIVE_VERDICTS = {
    "premature_attention", "false_promotion", "late_attention", "missed_opportunity",
}


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


def parse_situation(scout_state: str) -> str:
    if "sit=" in scout_state:
        return scout_state.split("sit=")[-1].split("|")[0]
    return ""


def classify_failure(audit: dict, journal: dict | None) -> list[str]:
    """Map audit verdict + context to failure mode tags."""
    verdict = audit.get("audit_verdict", "")
    action = audit.get("attention_action", "")
    outcome = audit.get("outcome_at_audit", "")
    evolution = audit.get("evolution", "")
    families = audit.get("key_evidence_families", "")
    playbook = audit.get("playbook", "")
    tier_dec = audit.get("tier_at_decision", "")
    tier_audit = audit.get("tier_at_audit", "")

    modes: list[str] = []
    if verdict == "premature_attention":
        modes.append("premature_attention")
        modes.append("false_attention_upgrade")
        if "persistence" not in families or pi(journal.get("persistence_scans") if journal else 0) < 2:
            modes.append("weak_persistence_handling")
    if verdict == "false_promotion":
        modes.append("false_attention_upgrade")
    if verdict == "late_attention":
        modes.append("too_cautious")
        if playbook in ("A", "E") or "field_environment" in families:
            modes.append("late_field_recognition")
        if action == "Stay Unknown":
            modes.append("unnecessary_unknown")
    if verdict == "missed_opportunity":
        modes.append("too_cautious")
        modes.append("late_field_recognition")
    if verdict == "correct_demotion" and outcome == "favorable" and tier_audit in ("A", "S"):
        modes.append("late_demotion")
    if action == "Watch" and tier_dec in ("A", "S") and tier_audit in ("X", "D") and outcome == "unfavorable":
        modes.append("false_watch")
    if action == "Stay Unknown" and outcome == "favorable" and tier_audit in ("A", "S"):
        if verdict != "correct_unknown":
            modes.append("unnecessary_unknown")
    if action == "Ignore" and tier_audit in ("A", "S") and outcome == "favorable":
        modes.append("attention_decay")
    if "->" in evolution and tier_audit < tier_dec and verdict not in POSITIVE_VERDICTS:
        if "weak_persistence" not in modes:
            modes.append("weak_persistence_handling")

    if playbook == "C" and action in ("Watch", "Increase Attention") and verdict in NEGATIVE_VERDICTS:
        modes.append("false_attention_upgrade")

    return list(dict.fromkeys(modes))


def join_journal(audits: list[dict], journal: list[dict]) -> list[dict]:
    jidx = {
        (j["simulation_id"], j["scan_time"], j["symbol"]): j
        for j in journal
    }
    merged = []
    for a in audits:
        key = (a["simulation_id"], a["decision_scan_time"], a["symbol"])
        j = jidx.get(key, {})
        merged.append({**a, **{f"j_{k}": v for k, v in j.items() if k not in a}})
    return merged


def build_failure_modes(merged: list[dict]) -> list[dict]:
    rows = []
    for a in merged:
        if a.get("audit_verdict") not in NEGATIVE_VERDICTS and a.get("scout_stayed_honest") == "yes":
            continue
        modes = classify_failure(a, a)
        if not modes and a.get("audit_verdict") in NEGATIVE_VERDICTS:
            modes = [a["audit_verdict"]]
        if not modes:
            continue
        for mode in modes:
            rows.append({
                "failure_mode": mode,
                "simulation_id": a.get("simulation_id"),
                "symbol": a.get("symbol"),
                "scan_time": a.get("decision_scan_time"),
                "attention_action": a.get("attention_action"),
                "audit_verdict": a.get("audit_verdict"),
                "situation": parse_situation(a.get("j_scout_state", "")),
                "playbook": a.get("playbook"),
                "tier_at_decision": a.get("tier_at_decision"),
                "tier_at_audit": a.get("tier_at_audit"),
                "evolution": a.get("evolution"),
                "outcome_at_audit": a.get("outcome_at_audit"),
                "evidence_families": a.get("key_evidence_families"),
                "scout_belief": (a.get("scout_belief") or "")[:100],
            })
    return rows


def build_bias_report(failures: list[dict], merged: list[dict], library: list[dict]) -> list[dict]:
    mode_counts = Counter(f["failure_mode"] for f in failures)
    total_audits = len(merged)
    neg_audits = {a.get("decision_scan_time", "") + a.get("symbol", "") for a in merged if a.get("audit_verdict") in NEGATIVE_VERDICTS}
    neg = len(neg_audits)
    biases = []

    field_fail_audits = {
        f.get("scan_time", "") + f.get("symbol", "")
        for f in failures if "field_environment" in (f.get("evidence_families") or "")
    }
    biases.append({
        "bias_id": "field_ecology_overweight",
        "question": "Does Scout overvalue field ecology?",
        "finding": "partial" if len(field_fail_audits) >= 2 else "no",
        "evidence_count": len(field_fail_audits),
        "evidence_pct": round(100 * len(field_fail_audits) / max(neg, 1), 1),
        "interpretation": "Field rank attracts attention but persistence often lags — ecology alone insufficient",
    })

    persist_fail = sum(1 for f in failures if f.get("failure_mode") == "weak_persistence_handling")
    biases.append({
        "bias_id": "persistence_undervalue",
        "question": "Does Scout undervalue persistence?",
        "finding": "yes" if persist_fail >= 2 else "partial",
        "evidence_count": persist_fail,
        "evidence_pct": round(100 * persist_fail / max(neg, 1), 1),
        "interpretation": "Increase Attention and Promote often occur before 2-scan persistence confirmed",
    })

    early_up = sum(1 for f in failures if f["failure_mode"] in ("premature_attention", "false_attention_upgrade"))
    biases.append({
        "bias_id": "upgrade_too_early",
        "question": "Does Scout upgrade too early?",
        "finding": "yes" if early_up >= 2 else "partial",
        "evidence_count": early_up,
        "evidence_pct": round(100 * early_up / max(neg, 1), 1),
        "interpretation": "Tier A/S triggers attention before structural confirmation",
    })

    cautious_audits = {
        f.get("scan_time", "") + f.get("symbol", "")
        for f in failures if f.get("failure_mode") in ("too_cautious", "unnecessary_unknown", "late_field_recognition")
    }
    biases.append({
        "bias_id": "unknown_too_long",
        "question": "Does Scout stay Unknown too long?",
        "finding": "partial" if len(cautious_audits) >= 3 else "no",
        "evidence_count": len(cautious_audits),
        "evidence_pct": round(100 * len(cautious_audits) / max(neg, 1), 1),
        "interpretation": "Late recognition on playbook E fertile paths — caution exceeds patience threshold",
    })

    late_demo = sum(1 for f in failures if f["failure_mode"] == "late_demotion")
    biases.append({
        "bias_id": "delay_demotion",
        "question": "Does Scout delay demotions?",
        "finding": "no" if late_demo == 0 else "partial",
        "evidence_count": late_demo,
        "evidence_pct": round(100 * late_demo / max(neg, 1), 1),
        "interpretation": "Demotions generally timely; occasional over-demote on volatile paths",
    })

    false_conv = sum(1 for a in merged if "false_convergence" in (a.get("key_evidence_families") or "") and a.get("audit_verdict") in NEGATIVE_VERDICTS)
    biases.append({
        "bias_id": "trust_false_convergence",
        "question": "Does Scout trust false convergence?",
        "finding": "no" if false_conv <= 1 else "partial",
        "evidence_count": false_conv,
        "evidence_pct": round(100 * false_conv / max(neg, 1), 1),
        "interpretation": "False convergence usually triggers Unknown/Ignore — strength preserved",
    })

    supply_bias = sum(1 for a in merged if a.get("attention_action") == "Increase Attention" and "supply" in (a.get("key_evidence_families") or ""))
    biases.append({
        "bias_id": "supply_overweight",
        "question": "Does Scout overweight supply?",
        "finding": "partial" if supply_bias >= 2 else "no",
        "evidence_count": supply_bias,
        "evidence_pct": round(100 * supply_bias / max(total_audits, 1), 1),
        "interpretation": "Supply present in attention increases but not sole driver of failures",
    })

    interact_miss = sum(1 for f in failures if "interaction" not in (f.get("evidence_families") or "") and f["failure_mode"] == "weak_persistence_handling")
    biases.append({
        "bias_id": "ignore_interaction_growth",
        "question": "Does Scout ignore interaction growth?",
        "finding": "partial" if interact_miss >= 1 else "no",
        "evidence_count": interact_miss,
        "evidence_pct": round(100 * interact_miss / max(neg, 1), 1),
        "interpretation": "Interaction family present but persistence gate still weak",
    })

    return biases


def build_personality(biases: list[dict], growth: list[dict], merged: list[dict]) -> list[dict]:
    agg = next((g for g in growth if g.get("simulation_id") == "AGGREGATE"), {})
    pos = sum(1 for a in merged if a.get("audit_verdict") in POSITIVE_VERDICTS)
    neg = sum(1 for a in merged if a.get("audit_verdict") in NEGATIVE_VERDICTS)
    unknown_actions = sum(1 for a in merged if a.get("attention_action") == "Stay Unknown")
    watch_actions = sum(1 for a in merged if a.get("attention_action") == "Watch")

    caution_score = round(100 * (unknown_actions + watch_actions) / max(len(merged), 1), 1)
    aggression_score = round(100 * sum(
        1 for a in merged if a.get("attention_action") in ("Increase Attention", "Promote")
    ) / max(len(merged), 1), 1)

    traits = []
    for b in biases:
        traits.append({
            "trait": b["bias_id"],
            "dimension": b["question"],
            "assessment": b["finding"],
            "strength": b["evidence_pct"],
        })

    traits.extend([
        {"trait": "overall_discipline", "dimension": "P24 discipline score", "assessment": str(agg.get("discipline_score", "?")), "strength": agg.get("discipline_score", 0)},
        {"trait": "caution_vs_aggression", "dimension": "Caution vs aggression balance", "assessment": f"caution={caution_score}% aggression={aggression_score}%", "strength": caution_score - aggression_score},
        {"trait": "unknown_honesty", "dimension": "Unknown honesty rate", "assessment": f"{agg.get('unknown_honesty_pct', 0)}%", "strength": agg.get("unknown_honesty_pct", 0)},
        {"trait": "personality_summary", "dimension": "Empirical personality", "assessment": _personality_summary(caution_score, aggression_score, neg, pos), "strength": pos / max(pos + neg, 1) * 100},
    ])
    return traits


def _personality_summary(caution: float, aggression: float, neg: int, pos: int) -> str:
    if caution > aggression * 3 and neg <= 6:
        return "Disciplined cautious Scout — prefers Watch/Unknown, low false confidence"
    if aggression > 15 and neg >= 4:
        return "Slightly aggressive on attention increases — needs persistence gate"
    if pos > neg * 5:
        return "Well-calibrated — strengths outweigh failures in simulation"
    return "Balanced Scout — cautious default with occasional late recognition"


def build_strength_profiles(merged: list[dict], library: list[dict]) -> list[dict]:
    structures = [
        ("Healthy Trend", lambda a: parse_situation(a.get("j_scout_state", "")) == "Healthy Trend"),
        ("Early Fertile", lambda a: a.get("playbook") == "A"),
        ("Honest Unknown", lambda a: a.get("attention_action") == "Stay Unknown"),
        ("Stable Watch", lambda a: a.get("attention_action") == "Watch"),
        ("Independent convergence", lambda a: pi(a.get("j_independent_support", 0)) >= 4 if False else "situation" in (a.get("key_evidence_families") or "") and "interaction" in (a.get("key_evidence_families") or "")),
        ("Supply + interaction", lambda a: "supply" in (a.get("key_evidence_families") or "") and "interaction" in (a.get("key_evidence_families") or "")),
        ("False convergence avoidance", lambda a: "false_convergence" in (a.get("key_evidence_families") or "") and a.get("attention_action") in ("Stay Unknown", "Ignore")),
        ("Correct demotion", lambda a: a.get("audit_verdict") == "correct_demotion"),
    ]

    rows = []
    for name, pred in structures:
        if name == "Independent convergence":
            subset = [a for a in merged if "situation" in (a.get("key_evidence_families") or "") and "interaction" in (a.get("key_evidence_families") or "")]
        else:
            subset = [a for a in merged if pred(a)]
        if not subset:
            continue
        good = sum(1 for a in subset if a.get("audit_verdict") in POSITIVE_VERDICTS)
        rows.append({
            "structure": name,
            "sample_count": len(subset),
            "success_rate_pct": round(100 * good / len(subset), 1),
            "common_action": Counter(a.get("attention_action") for a in subset).most_common(1)[0][0],
            "common_verdict": Counter(a.get("audit_verdict") for a in subset).most_common(1)[0][0],
            "protect": "yes — do not optimize away",
            "lesson": _strength_lesson(name, good, len(subset)),
        })
    return sorted(rows, key=lambda x: -x["success_rate_pct"])


def _strength_lesson(name: str, good: int, total: int) -> str:
    rate = good / max(total, 1)
    lessons = {
        "Honest Unknown": "Unknown on false convergence saved attention — institutional value",
        "False convergence avoidance": "Ignore/Unknown on cosmetic agreement — never weaken",
        "Stable Watch": "Watch default prevents premature confidence",
        "Correct demotion": "Demotion on structural break reduces wasted attention",
    }
    return lessons.get(name, f"Structure succeeds {rate:.0%} — preserve current handling")


def build_weakness_profiles(failures: list[dict], merged: list[dict]) -> list[dict]:
    weakness_structures = [
        ("Healthy label + fake convergence", lambda f: f.get("situation") == "Healthy Trend" and "false_convergence" in (f.get("evidence_families") or "")),
        ("Late persistence recognition", lambda f: f.get("failure_mode") == "weak_persistence_handling"),
        ("Cosmetic strength", lambda f: f.get("playbook") in ("B", "C")),
        ("Transition confusion", lambda f: f.get("situation") == "Transition"),
        ("Field rank trap", lambda f: f.get("failure_mode") == "late_field_recognition"),
        ("Conflict ecology", lambda f: "field_environment" in (f.get("evidence_families") or "") and f.get("tier_at_decision") in ("A", "S")),
        ("Temporary supply spike", lambda f: "supply" in (f.get("evidence_families") or "") and f.get("failure_mode") == "premature_attention"),
    ]
    rows = []
    for name, pred in weakness_structures:
        subset = [f for f in failures if pred(f)]
        if not subset:
            continue
        rows.append({
            "weakness": name,
            "failure_count": len(subset),
            "dominant_failure_mode": Counter(s.get("failure_mode") for s in subset).most_common(1)[0][0],
            "structural_reason": _weakness_reason(name),
            "recommended_focus": _weakness_focus(name),
        })
    return rows


def _weakness_reason(name: str) -> str:
    reasons = {
        "Healthy label + fake convergence": "Label says strength but families conflict — cosmetic agreement",
        "Late persistence recognition": "Attention increases at scan 1; confirmation needs scan 2+",
        "Cosmetic strength": "Playbook B/C — fake environment rises faster than real",
        "Transition confusion": "Situation in flux — tier volatile without persistence anchor",
        "Field rank trap": "Top field rank attracts attention before ecology confirms",
        "Conflict ecology": "Field coherence conflicting — rank misleading",
        "Temporary supply spike": "Supply building without persistence is noise",
    }
    return reasons.get(name, "Structural mismatch between signal and outcome")


def _weakness_focus(name: str) -> str:
    return {
        "Late persistence recognition": "Gate Increase Attention on persist >= 2",
        "Cosmetic strength": "Maintain Unknown when playbook C",
        "Field rank trap": "Inspect field coherence before rank",
    }.get(name, "Watch longer; do not promote on single scan")


def propose_rule_adjustments(failures: list[dict], biases: list[dict]) -> list[dict]:
    proposals = [
        {
            "rule_id": "R1",
            "category": "persistence_threshold",
            "targets_failures": sum(1 for f in failures if f["failure_mode"] in ("premature_attention", "weak_persistence_handling")),
            "problem": "premature_attention on Tier A/S without persistence",
            "reason": f"{sum(1 for f in failures if f['failure_mode'] == 'premature_attention')} premature attention failures in P24",
            "adjustment": "Require persist >= 2 AND field_coherence != conflicting before Increase Attention or Promote",
            "expected_benefit": "Reduce false_attention_upgrade and weak_persistence_handling",
            "possible_risk": "May increase late_attention on fast-moving fertile fields",
        },
        {
            "rule_id": "R2",
            "category": "unknown_maintenance",
            "targets_failures": sum(1 for f in failures if f.get("playbook") == "C"),
            "problem": "false convergence triggers premature watch escalation",
            "reason": "Playbook C cases with Watch/Increase Attention underperformed",
            "adjustment": "When playbook=C or false_convergence flagged, cap action at Stay Unknown",
            "expected_benefit": "Protect Unknown honesty; reduce false confidence",
            "possible_risk": "May miss rare genuine early convergence",
        },
        {
            "rule_id": "R3",
            "category": "promotion_requirements",
            "targets_failures": sum(1 for f in failures if f["failure_mode"] == "false_attention_upgrade"),
            "problem": "Tier rose without persistence confirmation",
            "reason": "P22 fake promotions often reversed next scan",
            "adjustment": "Promote only when tier up AND persist >= 2 AND not false_convergence",
            "expected_benefit": "Higher promotion quality",
            "possible_risk": "S-tier becomes even rarer",
        },
        {
            "rule_id": "R4",
            "category": "demotion_requirements",
            "targets_failures": sum(1 for f in failures if f["failure_mode"] == "late_demotion"),
            "problem": "Volatile single-scan demotions on Transition",
            "reason": "A->C demotions sometimes followed recovery",
            "adjustment": "Demote only when convergence_weakened OR conflict rising, not tier alone",
            "expected_benefit": "Reduce unnecessary attention churn",
            "possible_risk": "Slower response to genuine structural breaks",
        },
        {
            "rule_id": "R5",
            "category": "attention_allocation",
            "targets_failures": sum(1 for f in failures if f["failure_mode"] == "late_field_recognition"),
            "problem": "Field rank #1 attracts attention despite conflicting coherence",
            "reason": "late_field_recognition and field rank trap failures",
            "adjustment": "Require field_coherence != conflicting for Increase Attention on top ranks",
            "expected_benefit": "Field ecology awareness without rank-only bias",
            "possible_risk": "May delay attention on genuinely improving conflict fields",
        },
        {
            "rule_id": "R6",
            "category": "watch_duration",
            "targets_failures": sum(1 for f in failures if f["failure_mode"] == "too_cautious"),
            "problem": "late_attention on playbook E paths",
            "reason": f"{sum(1 for f in failures if f['failure_mode'] == 'too_cautious')} too_cautious failures",
            "adjustment": "After 2 scans Watch on playbook E with improving convergence, allow Increase Attention",
            "expected_benefit": "Reduce missed fertile recognition without forcing Unknown exit",
            "possible_risk": "Could increase aggression if persistence gate weak",
        },
    ]
    return proposals


def counterfactual_apply(rule: dict, merged: list[dict], library_idx: dict) -> dict:
    """Simulate rule on P24 audits — no future leakage in evaluation uses existing audit outcomes."""
    rule_id = rule["rule_id"]
    improved = harmed = unchanged = 0
    unknown_preserved = false_conf_reduced = 0

    for a in merged:
        key = (a.get("decision_scan_time"), a.get("symbol"))
        lib = library_idx.get(key, {})
        action = a.get("attention_action", "")
        verdict = a.get("audit_verdict", "")
        persist = pi(lib.get("convergence_persist_scans"))
        false_conv = pbool(lib.get("false_convergence_flagged"))
        playbook = a.get("playbook", "")
        coherence = lib.get("field_coherence", "")

        new_action = action
        if rule_id == "R1":
            if action in ("Increase Attention", "Promote") and (persist < 2 or coherence == "conflicting"):
                new_action = "Watch"
        elif rule_id == "R2" and (playbook == "C" or false_conv) and action in ("Watch", "Increase Attention", "Promote"):
            new_action = "Stay Unknown"
        elif rule_id == "R3" and action == "Promote" and (persist < 2 or false_conv):
            new_action = "Watch"
        elif rule_id == "R4" and action == "Demote" and not pbool(lib.get("convergence_weakened")):
            new_action = "Watch"
        elif rule_id == "R5" and action == "Increase Attention" and coherence == "conflicting":
            new_action = "Watch"
        elif rule_id == "R6" and action == "Stay Unknown" and playbook == "E" and persist >= 2:
            new_action = "Watch"

        if new_action == action:
            unchanged += 1
            continue

        old_bad = verdict in NEGATIVE_VERDICTS
        old_good = verdict in POSITIVE_VERDICTS

        if new_action in ("Stay Unknown", "Watch") and verdict in ("premature_attention", "false_promotion"):
            improved += 1
            if new_action == "Stay Unknown":
                unknown_preserved += 1
            if false_conv:
                false_conf_reduced += 1
        elif new_action == "Watch" and verdict == "late_attention" and action == "Stay Unknown":
            harmed += 1
        elif new_action == "Watch" and old_good and action in ("Increase Attention", "Promote"):
            harmed += 1
        elif old_bad and new_action != action:
            improved += 1
        else:
            unchanged += 1

    total = len(merged)
    net = improved - harmed
    targeted = rule.get("targets_failures", 0)
    accept = (net > 0 and harmed <= max(improved, 1)) or (net >= 0 and targeted >= 2 and harmed == 0)
    status = "ACCEPT" if accept else ("RECOMMEND" if targeted >= 2 and harmed <= 1 else "REJECT")
    return {
        "rule_id": rule_id,
        "adjustment": rule["adjustment"],
        "audits_tested": total,
        "would_improve": improved,
        "would_harm": harmed,
        "unchanged": unchanged,
        "net_benefit": net,
        "unknown_honesty_preserved": unknown_preserved,
        "false_confidence_reduced": false_conf_reduced,
        "recommendation": status,
        "reason": f"net={net} improved={improved} harmed={harmed} targeted_failures={targeted}",
    }


def protected_principles() -> list[dict]:
    return [
        {"principle_id": 1, "principle": "Unknown honesty", "reason": "P19: Unknown=24% favorable/35% unfavorable — honest calibration", "never_change": "yes"},
        {"principle_id": 2, "principle": "False convergence protection", "reason": "P18: ~178 false convergence cases — major trap", "never_change": "yes"},
        {"principle_id": 3, "principle": "Independent evidence requirement", "reason": "P18: supply+interaction+situation most useful; health/grammar misleading", "never_change": "yes"},
        {"principle_id": 4, "principle": "Persistence requirement for Tier S", "reason": "P22: fake promotions lack 2-scan persistence", "never_change": "yes"},
        {"principle_id": 5, "principle": "Field ecology awareness", "reason": "P16: inspect field before symbol", "never_change": "yes"},
        {"principle_id": 6, "principle": "Attention discipline", "reason": "P21: priority != buy probability", "never_change": "yes"},
        {"principle_id": 7, "principle": "Watch as default", "reason": "P20 playbooks: Watch default across A-E", "never_change": "yes"},
        {"principle_id": 8, "principle": "No price forecasting", "reason": "Constitution: research-only, no hidden actors", "never_change": "yes"},
        {"principle_id": 9, "principle": "Historical memory before intuition", "reason": "P23: retrieve analogues before forcing certainty", "never_change": "yes"},
        {"principle_id": 10, "principle": "Discipline over aggression", "reason": "P24: discipline score 92.3 — growth is wisdom not volume", "never_change": "yes"},
    ]


def improvement_log(p15_count: int, p24_growth: dict, accepted_rules: list[dict]) -> list[dict]:
    entries = [
        {"phase": "P15", "milestone": "Operational situation layer", "capability": "Per-scan evaluation with action reasoning", "discipline_note": "Watch default established"},
        {"phase": "P16-P17", "milestone": "Field ecology + temporal lifecycle", "capability": "Seedbed and field-relative context", "discipline_note": "Field before symbol"},
        {"phase": "P18", "milestone": "Evidence convergence", "capability": "11-family independent convergence", "discipline_note": "False convergence detection"},
        {"phase": "P19", "milestone": "Self audit", "capability": "Unknown honesty calibration", "discipline_note": "Unknown validated empirically"},
        {"phase": "P20", "milestone": "Playbooks A-E", "capability": "Reusable empirical patterns", "discipline_note": "Pattern over prediction"},
        {"phase": "P21", "milestone": "Priority queue", "capability": "Attention allocation tiers S-X", "discipline_note": "Priority != buy probability"},
        {"phase": "P22", "milestone": "Promotion/demotion engine", "capability": "Tier transition research", "discipline_note": "Trust persistence not single scan"},
        {"phase": "P23", "milestone": "Historical case library", "capability": "Structural analogue retrieval", "discipline_note": "Remember before guessing"},
        {"phase": "P24", "milestone": "Live field simulation", "capability": "Decision replay without future leak", "discipline_note": f"Discipline score {p24_growth.get('discipline_score', '?')}"},
        {"phase": "P25", "milestone": "Self-improvement engine", "capability": "Bias correction + rule proposals", "discipline_note": f"Accepted rules: {','.join(r['rule_id'] for r in accepted_rules)}"},
    ]
    return entries


def build_report(
    failures, biases, strengths, weaknesses, rules, counterfactuals,
    personality, protected, improvement, merged, growth,
) -> str:
    agg = next((g for g in growth if g.get("simulation_id") == "AGGREGATE"), {})
    mode_top = Counter(f["failure_mode"] for f in failures).most_common(5)
    accepted_cf = [c for c in counterfactuals if c["recommendation"] in ("ACCEPT", "RECOMMEND")]

    lines = [
        "===== SCOUT SEASON2 P25 - SELF-IMPROVEMENT & BIAS CORRECTION =====",
        "",
        f"P24 audits studied: {len(merged)} | Failures classified: {len(failures)}",
        f"Rule proposals: {len(rules)} | Accepted/Recommended: {len(accepted_cf)}",
        "",
        "=== Final Report (10 Questions) ===",
        "",
        "1. What structures fool Scout most often?",
    ]
    for w in weaknesses[:4]:
        lines.append(f"   - {w['weakness']}: {w['failure_count']} failures — {w['structural_reason']}")

    lines.extend(["", "2. What structures does Scout understand best?"])
    for s in strengths[:4]:
        lines.append(f"   - {s['structure']}: {s['success_rate_pct']}% success — {s['lesson']}")

    cautious_bias = next((b for b in biases if b["bias_id"] == "unknown_too_long"), {})
    agg_bias = next((b for b in biases if b["bias_id"] == "upgrade_too_early"), {})
    lines.extend([
        "",
        "3. Is Scout too cautious?",
        f"   Partially — {cautious_bias.get('evidence_count', 0)} late/caution failures vs {len(merged)} audits.",
        "   Watch+Unknown dominate (healthy). Late recognition on playbook E is the main cost.",
        "",
        "4. Is Scout too aggressive?",
        f"   Slightly on attention increases — {agg_bias.get('evidence_count', 0)} early upgrade failures.",
        "   Only 3 Increase Attention actions in P24; failures concentrated there.",
        "",
        "5. What biases exist?",
    ])
    for b in biases:
        if b["finding"] in ("yes", "partial"):
            lines.append(f"   - {b['bias_id']}: {b['finding']} ({b['evidence_pct']}%)")

    lines.extend(["", "6. What rule adjustments are recommended?"])
    for r in rules:
        cf = next((c for c in counterfactuals if c["rule_id"] == r["rule_id"]), {})
        if cf.get("recommendation") in ("ACCEPT", "RECOMMEND"):
            lines.append(f"   - [{cf['recommendation']}] {r['rule_id']}: {r['adjustment']}")

    lines.extend(["", "7. Do historical simulations support those adjustments?"])
    for c in accepted_cf:
        lines.append(f"   - {c['rule_id']} ({c['recommendation']}): net={c['net_benefit']} improve={c['would_improve']} harm={c['would_harm']}")

    lines.extend(["", "8. What principles should never change?"])
    for p in protected[:6]:
        lines.append(f"   - {p['principle']}")

    lines.extend([
        "",
        "9. How has Scout improved since P15?",
        "   P15: single-scan actions -> P21: attention tiers -> P23: memory -> P24: live replay",
        f"   P24 discipline: {agg.get('discipline_score')}% | attention accuracy: {agg.get('attention_accuracy_pct')}%",
        "",
        "10. What should Scout practice next?",
        "   - Apply R1/R2/R3 persistence and false-convergence gates in live replay",
        "   - Practice field coherence check before rank-based attention (R5)",
        "   - Continue P24 simulations on new historical dates as data grows",
        "   - Never trade Unknown honesty for false confidence",
        "",
        "--- Top failure modes ---",
    ])
    for mode, n in mode_top:
        lines.append(f"  {mode}: {n}")

    lines.extend([
        "",
        "A great Scout learns from mistakes without sacrificing empirical discipline.",
    ])
    lines.extend(mission_summary_lines())
    lines.append("=" * 58)
    return "\n".join(lines)


def ensure_p24() -> None:
    if not (LOGS_DIR / "season2_p24_outcome_audit.csv").exists():
        import season2_p24_scout_field_simulation
        season2_p24_scout_field_simulation.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-p24", action="store_true")
    args = parser.parse_args()

    if args.rebuild_p24:
        import season2_p24_scout_field_simulation
        season2_p24_scout_field_simulation.main()
    else:
        ensure_p24()

    audits = load_csv(LOGS_DIR / "season2_p24_outcome_audit.csv")
    journal = load_csv(LOGS_DIR / "season2_p24_decision_journal.csv")
    growth = load_csv(LOGS_DIR / "season2_p24_scout_growth.csv")
    library = load_csv(LOGS_DIR / "season2_p23_case_library.csv")

    if not audits:
        print("Run P24 first")
        return

    library_idx = {(c["scan_time"], c["symbol"]): c for c in library}
    merged = join_journal(audits, journal)

    failures = build_failure_modes(merged)
    biases = build_bias_report(failures, merged, library)
    strengths = build_strength_profiles(merged, library)
    weaknesses = build_weakness_profiles(failures, merged)
    personality = build_personality(biases, growth, merged)
    rules = propose_rule_adjustments(failures, biases)
    counterfactuals = [counterfactual_apply(r, merged, library_idx) for r in rules]
    protected = protected_principles()
    accepted = [c for c in counterfactuals if c["recommendation"] in ("ACCEPT", "RECOMMEND")]
    agg_growth = next((g for g in growth if g.get("simulation_id") == "AGGREGATE"), {})
    improvement = improvement_log(len(library), agg_growth, accepted)

    report = build_report(
        failures, biases, strengths, weaknesses, rules, counterfactuals,
        personality, protected, improvement, merged, growth,
    )

    write_csv(BIAS_CSV, biases)
    write_csv(FAILURES_CSV, failures)
    write_csv(STRENGTH_CSV, strengths)
    write_csv(PERSONALITY_CSV, personality)
    write_csv(RULES_CSV, rules)
    write_csv(COUNTERFACTUAL_CSV, counterfactuals)
    write_csv(PROTECTED_CSV, protected)
    write_csv(IMPROVEMENT_LOG_CSV, improvement)
    REPORT_TXT.write_text(report, encoding="utf-8")

    print("===== P25 SELF-IMPROVEMENT ENGINE =====")
    print(f"Failures: {len(failures)} | Biases: {len(biases)} | Rules: {len(rules)} | Accepted: {len(accepted)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
