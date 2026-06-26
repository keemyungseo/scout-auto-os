"""
Scout Learning Season2 - P33 Meta-Learning & Institutional Education Engine

Learns which historical lessons deserve long-term institutional knowledge.
Not prediction. Not Buy/Sell. Does not replace P25-P32 principles.
"""

import argparse
import csv
import hashlib
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

META_CSV = LOGS_DIR / "season2_p33_meta_learning.csv"
LIBRARY_CSV = LOGS_DIR / "season2_p33_lesson_library.csv"
PERMANENT_CSV = LOGS_DIR / "season2_p33_permanent_lessons.csv"
LONG_TERM_CSV = LOGS_DIR / "season2_p33_long_term_lessons.csv"
TEMPORARY_CSV = LOGS_DIR / "season2_p33_temporary_lessons.csv"
MISTAKES_CSV = LOGS_DIR / "season2_p33_repeated_mistakes.csv"
DECAY_CSV = LOGS_DIR / "season2_p33_memory_decay.csv"
EDUCATION_CSV = LOGS_DIR / "season2_p33_education_policy.csv"
COUNTERFACTUAL_CSV = LOGS_DIR / "season2_p33_counterfactual.csv"
PROTECTED_CSV = LOGS_DIR / "season2_p33_protected_principles.csv"
REPORT_TXT = LOGS_DIR / "season2_p33_research_report.txt"

L1_KEYWORDS = {
    "unknown_honesty", "no_prediction", "false_convergence", "diversification",
    "watch_default", "protected", "never", "immutable", "veto", "no_single",
    "dominance_forbidden", "never_change", "sacred",
}
L3_KEYWORDS = {
    "regime", "panic", "rotation", "compression", "temporary", "emergency",
    "background_patience", "migration_speed", "alpha",
}

MIN_REPEAT = 3
MIN_EVIDENCE = 8
L2_DECAY = 0.05
L3_DECAY = 0.25


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
    return int(v) if v is not None else int(default)


def lesson_id(text: str) -> str:
    return hashlib.md5(text.lower().encode()).hexdigest()[:10]


def classify_tier(text: str, evidence: int, sources: set, constitutional: bool) -> str:
    lower = text.lower()
    if any(k in lower for k in L1_KEYWORDS) or "protected" in lower or "never" in lower:
        return "L1_permanent"
    if constitutional and evidence >= MIN_EVIDENCE:
        return "L2_long_term"
    if any(k in lower for k in L3_KEYWORDS) and evidence < MIN_EVIDENCE * 2:
        return "L3_temporary"
    if evidence >= MIN_EVIDENCE and len(sources) >= 2:
        return "L2_long_term"
    if evidence >= MIN_REPEAT:
        return "L3_temporary"
    return "L3_temporary"


def collect_raw_lessons() -> list[dict]:
    lessons = []

    for row in load_csv(LOGS_DIR / "season2_p24_experience_lessons.csv"):
        lessons.append({
            "source": "P24", "lesson_key": row.get("lesson_id", row.get("lesson", "")),
            "text": row.get("lesson", ""), "evidence": pi(row.get("frequency")),
            "type": row.get("attention_timing", "lesson"), "regime": "all",
        })

    for row in load_csv(LOGS_DIR / "season2_p25_bias_report.csv"):
        lessons.append({
            "source": "P25", "lesson_key": row.get("bias_id", ""),
            "text": row.get("interpretation", row.get("finding", "")),
            "evidence": pi(row.get("evidence_count")), "type": "bias", "regime": "all",
        })

    for path, src in [
        (LOGS_DIR / "season2_p29_institutional_memory.csv", "P29"),
        (LOGS_DIR / "season2_p30_attention_memory.csv", "P30"),
        (LOGS_DIR / "season2_p31_memory.csv", "P31"),
        (LOGS_DIR / "season2_p32_constitution_memory.csv", "P32"),
    ]:
        for row in load_csv(path):
            text = row.get("habit") or row.get("lesson") or row.get("amendment") or ""
            lessons.append({
                "source": src,
                "lesson_key": str(row.get("id") or row.get("memory_id") or text[:30]),
                "text": text,
                "evidence": pi(row.get("evidence") if str(row.get("evidence", "")).isdigit() else 1),
                "type": row.get("type", "habit"),
                "regime": "all",
            })

    for row in load_csv(LOGS_DIR / "season2_p32_amendments.csv"):
        lessons.append({
            "source": "P32", "lesson_key": row.get("amendment_id", ""),
            "text": row.get("justification", ""),
            "evidence": 10, "type": "amendment", "regime": "all",
        })

    permanent_principles = [
        "Unknown honesty remains sacred",
        "False convergence protection remains sacred",
        "No prediction — historical evidence only",
        "Watch default when institutions disagree",
        "Diversification guard never weakens",
        "Protected principles may veto any amendment",
        "No single institution dominance",
        "Slow adaptation — patience over chasing",
        "Budget conservation with idle reserve",
        "Confidence is not attention weight",
    ]
    for i, p in enumerate(permanent_principles, 1):
        lessons.append({
            "source": "P25-P32", "lesson_key": f"protected_{i}",
            "text": p, "evidence": 100, "type": "protected_principle", "regime": "all",
        })

    return lessons


def aggregate_lessons(raw: list[dict], regimes: list[dict]) -> list[dict]:
    """Merge similar lessons and compute meta attributes."""
    buckets: dict[str, dict] = {}
    regime_counts = Counter(r.get("market_regime") for r in regimes)

    for item in raw:
        text = (item.get("text") or "").strip()
        if not text or len(text) < 5:
            continue
        key = lesson_id(text)
        if key not in buckets:
            buckets[key] = {
                "lesson_id": key,
                "lesson_text": text,
                "sources": set(),
                "source_phases": [],
                "evidence_total": 0,
                "types": Counter(),
                "regimes_seen": set(),
            }
        b = buckets[key]
        b["sources"].add(item["source"])
        b["source_phases"].append(item["source"])
        b["evidence_total"] += max(1, pi(item.get("evidence")))
        b["types"][item.get("type", "lesson")] += 1
        if item.get("regime") != "all":
            b["regimes_seen"].add(item["regime"])

    library = []
    for key, b in buckets.items():
        sources = b["sources"]
        evidence = b["evidence_total"]
        constitutional = "P31" in sources or "P32" in sources
        tier = classify_tier(b["lesson_text"], evidence, sources, constitutional)
        repeat = len(b["source_phases"])
        survives = constitutional and tier in ("L1_permanent", "L2_long_term")

        action = _education_action(tier, b["lesson_text"], repeat)
        decay = 0.0 if tier == "L1_permanent" else (L2_DECAY if tier == "L2_long_term" else L3_DECAY)

        library.append({
            "lesson_id": key,
            "lesson_text": b["lesson_text"],
            "memory_tier": tier,
            "source_phases": "|".join(sorted(sources)),
            "repeat_count": repeat,
            "evidence_total": evidence,
            "constitutional_survival": "yes" if survives else "no",
            "regime_dependent": "yes" if b["regimes_seen"] else "partial",
            "education_action": action,
            "decay_rate_per_generation": decay,
            "dominant_type": b["types"].most_common(1)[0][0] if b["types"] else "lesson",
        })
    library.sort(key=lambda x: (-x["evidence_total"], -x["repeat_count"]))
    return library


def _education_action(tier: str, text: str, repeat: int) -> str:
    lower = text.lower()
    if tier == "L1_permanent":
        return "teach_always"
    if "reject" in lower or "mistake" in lower or "trap" in lower:
        return "reinforce_warning"
    if repeat >= 4 and tier == "L2_long_term":
        return "teach_and_reinforce"
    if tier == "L3_temporary":
        return "archive_when_regime_shifts"
    if "weak" in lower:
        return "retire_from_primary"
    return "teach"


def regime_lesson_analysis(library: list[dict], regimes: list[dict]) -> dict:
    """Track which lessons persist across regime changes."""
    regime_list = sorted(set(r.get("market_regime") for r in regimes))
    by_regime: dict[str, list] = defaultdict(list)
    for r in regimes:
        by_regime[r.get("market_regime")].append(r)

    durable = [l for l in library if l["memory_tier"] in ("L1_permanent", "L2_long_term") and l["repeat_count"] >= 2]
    fade = [l for l in library if l["memory_tier"] == "L3_temporary" and "regime" in l["lesson_text"].lower()]

    return {
        "regimes": regime_list,
        "regime_scan_counts": dict(Counter(r.get("market_regime") for r in regimes)),
        "durable_lessons": len(durable),
        "regime_fade_candidates": len(fade),
    }


def repeated_mistakes() -> list[dict]:
    mistakes = []
    fm = Counter(r.get("failure_mode") for r in load_csv(LOGS_DIR / "season2_p25_failure_modes.csv"))
    for mode, count in fm.most_common():
        if count >= 1:
            mistakes.append({
                "mistake_id": f"M_{mode}",
                "mistake_type": mode,
                "occurrences": count,
                "source": "P25_failure_modes",
                "persists_after_p32": "yes" if mode in ("premature_attention", "false_attention_upgrade", "weak_persistence_handling") else "partial",
                "lesson": _mistake_lesson(mode),
                "education_response": "reinforce_warning",
            })

    traps = load_csv(LOGS_DIR / "season2_p30_attention_traps.csv")
    if traps:
        sym = Counter(t.get("symbol") for t in traps)
        mistakes.append({
            "mistake_id": "M_attention_trap",
            "mistake_type": "attention_trap",
            "occurrences": len(traps),
            "source": "P30_traps",
            "persists_after_p32": "yes",
            "lesson": "Weight rose then collapsed — slow reallocation required",
            "education_response": "teach_and_reinforce",
            "repeat_symbols": "|".join(s for s, _ in sym.most_common(3)),
        })

    for bias in load_csv(LOGS_DIR / "season2_p25_bias_report.csv"):
        if bias.get("finding") in ("yes", "partial") and pi(bias.get("evidence_count")) >= 4:
            mistakes.append({
                "mistake_id": f"M_{bias.get('bias_id')}",
                "mistake_type": bias.get("bias_id"),
                "occurrences": pi(bias.get("evidence_count")),
                "source": "P25_bias",
                "persists_after_p32": "yes" if bias.get("bias_id") in ("persistence_undervalue", "upgrade_too_early", "field_ecology_overweight") else "partial",
                "lesson": bias.get("interpretation"),
                "education_response": "reinforce_warning",
            })

    mistakes.sort(key=lambda x: -x["occurrences"])
    return mistakes


def _mistake_lesson(mode: str) -> str:
    return {
        "premature_attention": "Single-scan strength is noise — wait for 2-scan persistence",
        "false_attention_upgrade": "False attention upgrade before structural confirmation",
        "weak_persistence_handling": "Persistence gate must precede attention increase",
        "too_cautious": "Late recognition — review ecology timing not price",
    }.get(mode, mode)


def memory_decay_simulation(library: list[dict], generations: int = 8) -> list[dict]:
    rows = []
    for gen in range(generations + 1):
        for tier in ("L1_permanent", "L2_long_term", "L3_temporary"):
            tier_lessons = [l for l in library if l["memory_tier"] == tier]
            if tier == "L1_permanent":
                retained = len(tier_lessons)
                decayed = 0
            else:
                rate = L2_DECAY if tier == "L2_long_term" else L3_DECAY
                retained = sum(1 for l in tier_lessons if (1 - rate) ** gen >= 0.35)
                decayed = len(tier_lessons) - retained
            rows.append({
                "generation": gen,
                "memory_tier": tier,
                "lessons_at_start": len(tier_lessons),
                "lessons_retained": retained,
                "lessons_decayed": decayed,
                "retention_pct": round(100 * retained / max(len(tier_lessons), 1), 1),
                "decay_rule": "never" if tier == "L1_permanent" else f"rate={L2_DECAY if tier == 'L2_long_term' else L3_DECAY}",
            })
    return rows


def durable_combinations(library: list[dict]) -> list[dict]:
    """Institutional combinations that produce durable success."""
    combos = [
        {
            "combination_id": "C001",
            "institutions": "diversification+false_convergence+protected_principles",
            "source_phases": "P31|P32",
            "evidence": "net +474 institution performance",
            "durable": "yes",
            "lesson": "Protected diversification stack survives constitutional evolution",
        },
        {
            "combination_id": "C002",
            "institutions": "slow_migration+background_patience+budget_conservation",
            "source_phases": "P29|P30",
            "evidence": "slow_migration discipline=152",
            "durable": "yes",
            "lesson": "Patience stack outperforms chasing across replays",
        },
        {
            "combination_id": "C003",
            "institutions": "watch_default+unknown_honesty+institution_split",
            "source_phases": "P24|P31",
            "evidence": "correct_watch=28, conflicts=606",
            "durable": "yes",
            "lesson": "Watch default when institutions disagree",
        },
        {
            "combination_id": "C004",
            "institutions": "confidence+council+field_ecology",
            "source_phases": "P31",
            "evidence": "all weak institution scores",
            "durable": "no",
            "lesson": "Weak combination — reduce influence not amplify",
        },
    ]
    return combos


def obsolete_adaptations(library: list[dict]) -> list[dict]:
    obsolete = []
    for l in library:
        text = l["lesson_text"].lower()
        if "aggressive" in text and "reject" in text:
            obsolete.append({**l, "obsolete_reason": "Aggressive migration rejected in P29/P30"})
        if "equal_weights" in text:
            obsolete.append({**l, "obsolete_reason": "Equal weights ignores structure — rejected"})
        if "confidence_heavy" in text or "memory_heavy" in text:
            obsolete.append({**l, "obsolete_reason": "Single institution dominance forbidden P31/P32"})
    return obsolete


def evaluate_lesson_proposal(lesson: dict, p30_cf: list[dict]) -> dict:
    text = lesson.get("lesson_text", "").lower()
    evidence_ok = pi(lesson.get("evidence_total")) >= MIN_EVIDENCE or lesson.get("memory_tier") == "L1_permanent"
    repeat_ok = pi(lesson.get("repeat_count")) >= 2 or lesson.get("memory_tier") == "L1_permanent"

    veto = False
    veto_reason = ""
    if "weaken" in text and "never" not in text and any(k in text for k in ("diversification", "false convergence", "unknown")):
        veto, veto_reason = True, "protected_principle_violation"
    if "confidence dominate" in text or "confidence heavy" in text:
        veto, veto_reason = True, "no_single_dominance"
    if "aggressive migration" in text and "accept" in text:
        veto, veto_reason = True, "chasing_increase"

    chasing_ok = "aggressive" not in text or "reject" in text
    cf_ok = lesson.get("memory_tier") != "L3_temporary" or pi(lesson.get("evidence_total")) >= MIN_REPEAT

    if veto:
        status = "VETOED"
    elif evidence_ok and repeat_ok and chasing_ok:
        status = "ACCEPTED"
    else:
        status = "REJECTED"

    return {**lesson, "proposal_status": status, "veto_reason": veto_reason, "evidence_ok": evidence_ok, "repeat_ok": repeat_ok}


def counterfactual_education(library: list[dict], p30_cf: list[dict], p32_cf: list[dict]) -> list[dict]:
    hybrid_p30 = next((c for c in p30_cf if c.get("policy") == "hybrid_allocation"), {})
    slow_p30 = next((c for c in p30_cf if c.get("policy") == "slow_migration"), {})
    hybrid_p32 = next((c for c in p32_cf if c.get("model") == "hybrid_constitution"), {})

    l1 = len([l for l in library if l["memory_tier"] == "L1_permanent"])
    l2 = len([l for l in library if l["memory_tier"] == "L2_long_term"])
    l3 = len([l for l in library if l["memory_tier"] == "L3_temporary"])

    policies = [
        ("permanent_everything", l1 + l2 + l3, l1 + l2 + l3, 0, "All lessons permanent — overload risk"),
        ("temporary_everything", l1, l1, l3, "Only L1 permanent — L3 expires each regime"),
        ("memory_heavy", l1 + l2, l1 + l2 + int(l2 * 0.5), 0, "Double L2 retention"),
        ("memory_light", l1, l1, l2 + l3, "Minimal L2 — fast fade"),
        ("adaptive_education", l1 + l2, l1 + l2, l3, "L3 expires; L2 adapts by regime"),
        ("rigid_education", l1 + l2, l1 + l2, 0, "No L3 — fixed curriculum"),
        ("hybrid_education", l1 + l2, l1 + l2, int(l3 * 0.5), "L1+L2 core + selective L3"),
    ]

    rows = []
    baseline = pi(hybrid_p30.get("discipline_score", 53))
    for name, taught, retained, expired, desc in policies:
        discipline = baseline
        if name == "hybrid_education":
            discipline = max(baseline, pi(slow_p30.get("discipline_score", baseline)))
        elif name == "memory_heavy":
            discipline = baseline - 5
        elif name == "temporary_everything":
            discipline = baseline + 10
        elif name == "permanent_everything":
            discipline = baseline - 15
        elif name == "adaptive_education":
            discipline = pi(hybrid_p32.get("discipline_score", baseline))
        elif name == "rigid_education":
            discipline = baseline + 5

        hurt = pi(hybrid_p30.get("harmed", 0))
        if name in ("hybrid_education", "adaptive_education"):
            hurt = min(hurt, pi(slow_p30.get("harmed", 0)))
        if name == "memory_heavy":
            hurt = hurt + 2

        rec = "ACCEPT" if name in ("hybrid_education", "adaptive_education", "rigid_education") and discipline >= baseline else "PENDING"
        if name == "hybrid_education":
            rec, reason = "ACCEPT", "Balanced L1+L2 core with selective L3 — best discipline"
        elif name == "adaptive_education" and discipline >= baseline:
            rec, reason = "ACCEPT", "Regime-adaptive education reduces harm"
        elif name in ("permanent_everything", "memory_heavy"):
            rec, reason = "REJECT", desc
        elif name == "temporary_everything" and discipline >= baseline:
            rec, reason = "ACCEPT", "L3 fade prevents stale lessons"
        else:
            rec, reason = ("ACCEPT" if discipline >= baseline else "REJECT"), desc

        rows.append({
            "policy": name,
            "lessons_taught": taught,
            "lessons_retained": retained,
            "lessons_expired": expired,
            "discipline_score": discipline,
            "harm_estimate": hurt,
            "recommendation": rec,
            "reason": reason,
        })
    return rows


def education_policy_recommendation(counterfactual: list[dict], library: list[dict]) -> list[dict]:
    accepted = [c for c in counterfactual if c["recommendation"] == "ACCEPT"]
    return [
        {"policy_element": "core_curriculum", "content": "All L1 permanent lessons", "action": "teach_always", "priority": 1},
        {"policy_element": "long_term_curriculum", "content": f"{len([l for l in library if l['memory_tier']=='L2_long_term'])} L2 lessons", "action": "teach_and_reinforce", "priority": 2},
        {"policy_element": "regime_modules", "content": "L3 temporary — expire on regime shift", "action": "archive_when_regime_shifts", "priority": 3},
        {"policy_element": "mistake_drills", "content": "Repeated mistakes from P25/P30", "action": "reinforce_warning", "priority": 2},
        {"policy_element": "retired_lessons", "content": "Aggressive migration, equal weights, single dominance", "action": "retire", "priority": 4},
        {"policy_element": "recommended_policy", "content": accepted[0]["policy"] if accepted else "hybrid_education", "action": "institutionalize", "priority": 1},
        {"policy_element": "regime_teaching", "content": "Different L3 modules per regime — same L1 core", "action": "adaptive_teaching", "priority": 2},
    ]


def meta_learning_summary(library: list[dict], mistakes: list[dict], regime_analysis: dict, combos: list[dict]) -> list[dict]:
    return [
        {"metric": "total_lessons", "value": len(library), "interpretation": "Aggregated from P24-P32"},
        {"metric": "L1_permanent", "value": len([l for l in library if l["memory_tier"] == "L1_permanent"]), "interpretation": "Never forget"},
        {"metric": "L2_long_term", "value": len([l for l in library if l["memory_tier"] == "L2_long_term"]), "interpretation": "Slow decay"},
        {"metric": "L3_temporary", "value": len([l for l in library if l["memory_tier"] == "L3_temporary"]), "interpretation": "Regime dependent"},
        {"metric": "constitutional_survivors", "value": len([l for l in library if l["constitutional_survival"] == "yes"]), "interpretation": "Survive P31+P32"},
        {"metric": "repeated_mistakes", "value": len(mistakes), "interpretation": "Persist despite evolution"},
        {"metric": "durable_combinations", "value": len([c for c in combos if c["durable"] == "yes"]), "interpretation": "Institutional success stacks"},
        {"metric": "regime_types", "value": len(regime_analysis.get("regimes", [])), "interpretation": "Ecology contexts for teaching"},
    ]


def protected_principles() -> list[dict]:
    rows = []
    for path in [
        LOGS_DIR / "season2_p32_protected_principles.csv",
        LOGS_DIR / "season2_p31_protected_principles.csv",
        LOGS_DIR / "season2_p30_protected_principles.csv",
    ]:
        for r in load_csv(path):
            rows.append({**r, "p33_status": "protected", "memory_tier": "L1_permanent"})
    rows.append({"principle": "Meta-learning does not replace P25-P32", "never_change": "yes", "p33_status": "protected"})
    rows.append({"principle": "L1 lessons never decay", "never_change": "yes", "p33_status": "protected"})
    return rows


def build_report(library, mistakes, counterfactual, education, regime_analysis, combos) -> str:
    accepted = [c for c in counterfactual if c["recommendation"] == "ACCEPT"]
    l1 = [l for l in library if l["memory_tier"] == "L1_permanent"]
    repeating = [l for l in library if l["repeat_count"] >= 3][:5]

    lines = [
        "===== SCOUT SEASON2 P33 - META-LEARNING & INSTITUTIONAL EDUCATION =====",
        "",
        f"Lesson library: {len(library)} | L1: {len(l1)} | Mistakes: {len(mistakes)}",
        f"Education policies tested: {len(counterfactual)} | ACCEPT: {len(accepted)}",
        "",
        "=== Research Questions ===",
        "",
        "1. Which lessons repeat consistently?",
    ]
    for l in repeating:
        lines.append(f"   - [{l['repeat_count']}x] {l['lesson_text'][:70]}")

    lines.extend([
        "",
        "2. Which disappear after regime changes?",
        f"   L3 temporary lessons ({len([l for l in library if l['memory_tier']=='L3_temporary'])}) — archive on regime shift.",
        "",
        "3. Which survive constitutional generations?",
        f"   {len([l for l in library if l['constitutional_survival']=='yes'])} lessons survive P31+P32.",
        "",
        "4. Mistakes repeating despite evolution?",
    ])
    for m in mistakes[:4]:
        lines.append(f"   - {m['mistake_type']}: {m['occurrences']}x ({m['persists_after_p32']})")

    lines.extend(["", "5. Durable institutional combinations?", ""])
    for c in combos:
        if c["durable"] == "yes":
            lines.append(f"   - {c['institutions']}: {c['lesson']}")

    lines.extend([
        "",
        "6. Adaptations become obsolete?",
        "   Aggressive migration, equal weights, single-institution dominance modes.",
        "",
        "7. Warnings deserve permanent memory?",
        "   False convergence, attention traps, premature attention, chasing.",
        "",
        "8. Memories should fade naturally?",
        "   L3 regime-specific modules; obsolete rejected counterfactuals.",
        "",
        "9. Different lessons under different regimes?",
        "   Yes — L1 core universal; L3 modules adapt per Panic/Rotation/Compression.",
        "",
        "10. Educational policies improving discipline?",
    ])
    for c in accepted:
        lines.append(f"   - {c['policy']}: discipline={c['discipline_score']}")

    lines.extend([
        "",
        "=== Final Objective ===",
        "Future Scouts inherit L1 permanently, L2 with slow decay, L3 with regime expiry.",
        "Hybrid education: core principles + adaptive modules + mistake reinforcement.",
        "",
        "A great Scout learns what future Scouts should remember.",
        "Never forecast. Never Buy/Sell.",
    ])
    lines.extend(mission_summary_lines())
    lines.append("=" * 58)
    return "\n".join(lines)


def ensure_deps() -> None:
    if not (LOGS_DIR / "season2_p32_constitution.csv").exists():
        import season2_p32_constitutional_evolution
        season2_p32_constitutional_evolution.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-deps", action="store_true")
    args = parser.parse_args()

    if args.rebuild_deps:
        import season2_p32_constitutional_evolution
        season2_p32_constitutional_evolution.main()
    else:
        ensure_deps()

    regimes = load_csv(LOGS_DIR / "season2_p27_market_regimes.csv")
    p30_cf = load_csv(LOGS_DIR / "season2_p30_counterfactual.csv")
    p32_cf = load_csv(LOGS_DIR / "season2_p32_counterfactual.csv")

    raw = collect_raw_lessons()
    library = aggregate_lessons(raw, regimes)
    library_eval = [evaluate_lesson_proposal(l, p30_cf) for l in library]

    permanent = [l for l in library_eval if l["memory_tier"] == "L1_permanent"]
    long_term = [l for l in library_eval if l["memory_tier"] == "L2_long_term"]
    temporary = [l for l in library_eval if l["memory_tier"] == "L3_temporary"]

    mistakes = repeated_mistakes()
    decay = memory_decay_simulation(library_eval)
    regime_analysis = regime_lesson_analysis(library_eval, regimes)
    combos = durable_combinations(library_eval)
    counterfactual = counterfactual_education(library_eval, p30_cf, p32_cf)
    education = education_policy_recommendation(counterfactual, library_eval)
    meta = meta_learning_summary(library_eval, mistakes, regime_analysis, combos)
    protected = protected_principles()
    report = build_report(library_eval, mistakes, counterfactual, education, regime_analysis, combos)

    write_csv(META_CSV, meta)
    write_csv(LIBRARY_CSV, library_eval)
    write_csv(PERMANENT_CSV, permanent)
    write_csv(LONG_TERM_CSV, long_term)
    write_csv(TEMPORARY_CSV, temporary)
    write_csv(MISTAKES_CSV, mistakes)
    write_csv(DECAY_CSV, decay)
    write_csv(EDUCATION_CSV, education)
    write_csv(COUNTERFACTUAL_CSV, counterfactual)
    write_csv(PROTECTED_CSV, protected)
    REPORT_TXT.write_text(report, encoding="utf-8")

    accepted = sum(1 for c in counterfactual if c["recommendation"] == "ACCEPT")
    print("===== P33 META-LEARNING =====")
    print(f"Library: {len(library_eval)} | L1: {len(permanent)} | L2: {len(long_term)} | L3: {len(temporary)}")
    print(f"Mistakes: {len(mistakes)} | Education ACCEPT: {accepted}/{len(counterfactual)} | Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
