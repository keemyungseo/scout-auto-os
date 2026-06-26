"""
Scout Learning Season2 - P20 Scout Playbook & Practical Pattern Mining

Extracts reusable empirical playbooks from P15-P19 experience.
Field handbook > prediction model. Unknown > false certainty.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PLAYBOOKS_CSV = LOGS_DIR / "season2_p20_scout_playbooks.csv"
GOOD_CSV = LOGS_DIR / "season2_p20_good_patterns.csv"
BAD_CSV = LOGS_DIR / "season2_p20_bad_patterns.csv"
UNKNOWN_CSV = LOGS_DIR / "season2_p20_unknown_patterns.csv"
LATE_CSV = LOGS_DIR / "season2_p20_late_recognition_patterns.csv"
FALSE_CONF_CSV = LOGS_DIR / "season2_p20_false_confidence_patterns.csv"
HANDBOOK_CSV = LOGS_DIR / "season2_p20_field_handbook.csv"
REPORT_TXT = LOGS_DIR / "season2_p20_research_report.txt"

MIN_PATTERN_N = 5
MIN_PLAYBOOK_N = 8


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


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def structure_key(audit: dict, p15: dict, field: dict | None) -> str:
    parts = [
        f"sit={audit.get('scout_thought_situation', '?')}",
        f"supply={p15.get('supply_context', '?')}",
        f"press={p15.get('pressure_band', '?')}",
        f"conv={audit.get('convergence_state', '?')}",
        f"fake={audit.get('false_convergence_flagged', 'False')}",
        f"fertile={audit.get('in_fertile_seedbed', 'False')}",
    ]
    if field:
        parts.append(f"field={field.get('environment_verdict', '?')}")
    return "|".join(parts)


def enrich_rows() -> list[dict]:
    audit = load_csv(LOGS_DIR / "scout_self_audit.csv")
    p15_idx = {(r["scan_time"], r["symbol"]): r for r in load_csv(LOGS_DIR / "season2_p15_operational_scores.csv")}
    field_idx = {r["scan_time"]: r for r in load_csv(LOGS_DIR / "season2_p16_opportunity_fields.csv")}
    conv_idx = {(r["scan_time"], r["symbol"]): r for r in load_csv(LOGS_DIR / "season2_p18_convergence_scores.csv")}

    rows = []
    for a in audit:
        key = (a["scan_time"], a["symbol"])
        p15 = p15_idx.get(key, {})
        field = field_idx.get(a["scan_time"])
        conv = conv_idx.get(key, {})
        rows.append(
            {
                **a,
                "supply_context": p15.get("supply_context", ""),
                "pressure_band": p15.get("pressure_band", ""),
                "health_class": p15.get("health_class", ""),
                "fake_trend_score": pf(p15.get("fake_trend_score")),
                "real_trend_score": pf(p15.get("real_trend_score")),
                "participant_state": p15.get("participant_state", ""),
                "grammar": (p15.get("grammar") or "")[:30],
                "field_verdict": field.get("environment_verdict", "") if field else "",
                "field_coherence": field.get("coherence", "") if field else "",
                "independent_support": conv.get("independent_support_count", ""),
                "support_families": conv.get("support_families", ""),
                "structure_key": structure_key(a, p15, field),
            }
        )
    return rows


def pattern_stats(group: list[dict]) -> dict:
    n = len(group)
    favorable = sum(1 for r in group if r.get("empirical_outcome") == "favorable")
    unfavorable = sum(1 for r in group if r.get("empirical_outcome") in ("unfavorable", "collapse"))
    mixed = n - favorable - unfavorable
    f6 = [pf(r.get("forward_6h")) for r in group if pf(r.get("forward_6h")) is not None]
    audits = Counter(r.get("audit_verdict") for r in group)
    transitions = Counter(r.get("convergence_state") for r in group)
    return {
        "frequency": n,
        "favorable_rate_pct": round(favorable / n * 100, 1) if n else 0,
        "unfavorable_rate_pct": round(unfavorable / n * 100, 1) if n else 0,
        "mixed_rate_pct": round(mixed / n * 100, 1) if n else 0,
        "median_forward_6h": round(statistics.median(f6), 2) if f6 else "",
        "top_audit_verdict": audits.most_common(1)[0][0] if audits else "",
        "common_transition": transitions.most_common(1)[0][0] if transitions else "",
        "confidence": "high" if n >= 20 else "medium" if n >= MIN_PATTERN_N else "hypothesis",
    }


def mine_patterns(rows: list[dict], filter_fn, pattern_type: str) -> list[dict]:
    subset = [r for r in rows if filter_fn(r)]
    groups: dict[str, list] = defaultdict(list)
    for r in subset:
        groups[r["structure_key"]].append(r)

    patterns = []
    for key, group in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(group) < MIN_PATTERN_N:
            continue
        stats = pattern_stats(group)
        patterns.append(
            {
                "pattern_type": pattern_type,
                "structure_key": key,
                "observed_structure": key.replace("|", " + "),
                **stats,
                "failure_mode": _failure_mode(pattern_type, stats),
                "recommended_behavior": _recommend_behavior(pattern_type, stats, key),
                "sample_symbols": "|".join(sorted({r["symbol"] for r in group})[:5]),
            }
        )
    return patterns


def _failure_mode(ptype: str, stats: dict) -> str:
    if ptype == "good":
        return "missed_if_too_cautious" if stats["favorable_rate_pct"] >= 50 else "mixed_outcomes"
    if ptype == "bad":
        return "premature_engagement" if stats["unfavorable_rate_pct"] >= 40 else "false_signal"
    if ptype == "unknown":
        return "would_have_lost_if_engaged" if stats["unfavorable_rate_pct"] >= 40 else "genuine_ambiguity"
    if ptype == "late":
        return "recognition_delay_cost"
    if ptype == "false_confidence":
        return "confidence_exceeded_evidence"
    return "unknown"


def _recommend_behavior(ptype: str, stats: dict, key: str) -> str:
    if ptype == "good":
        if stats["favorable_rate_pct"] >= 45 and "fertile=True" in key:
            return "Watch closely — early fertile candidate; confidence medium if independent support >= 3"
        return "Watch — conditional fertile signal"
    if ptype == "bad":
        if "fake=True" in key:
            return "Watch — false convergence risk; do not promote to Buy"
        return "Avoid premature confidence — cosmetic strength"
    if ptype == "unknown":
        return "Unknown — correct to stay disengaged"
    if ptype == "late":
        return "Watch earlier scans in same symbol arc — fertility may precede recognition"
    if ptype == "false_confidence":
        return "Reduce confidence tier — wait for persistent independent convergence"
    return "Watch"


def build_playbooks(
    good: list[dict],
    bad: list[dict],
    unknown: list[dict],
    late: list[dict],
    false_conf: list[dict],
) -> list[dict]:
    playbooks = []

    # Playbook A — early fertile candidate
    fertile_good = [p for p in good if "fertile=True" in p["structure_key"] and "Accumulation" in p["structure_key"]]
    if fertile_good:
        p = fertile_good[0]
        playbooks.append(
            {
                "playbook_id": "A",
                "name": "Early Fertile Candidate",
                "observed_structure": "Accumulation + fertile seedbed + low collapse + supply building",
                "supporting_evidence": "P16 fertile cluster + P15 Accumulation + P18 independent support",
                "frequency": p["frequency"],
                "success_rate_pct": p["favorable_rate_pct"],
                "failure_mode": "Scout too cautious (Unknown) when outcome favorable",
                "common_transition": p.get("common_transition", "growing_convergence"),
                "recommended_scout_behavior": "Watch closely. Early fertile candidate. Confidence medium only after 2+ scans persistence.",
                "confidence": p["confidence"],
            }
        )

    # Playbook B — cosmetic strength
    cosmetic = [p for p in good if "Healthy Trend" in p["structure_key"] and "fake=True" in p["structure_key"]]
    if not cosmetic:
        cosmetic = [p for p in bad if "Healthy Trend" in p["structure_key"]]
    if cosmetic:
        p = cosmetic[0]
        playbooks.append(
            {
                "playbook_id": "B",
                "name": "Cosmetic Strength",
                "observed_structure": "Healthy trend label + FRAGILE health + high fake score + false convergence",
                "supporting_evidence": "P15 label/health mismatch + P18 false convergence + P16 fake_bias field",
                "frequency": p["frequency"],
                "success_rate_pct": p["favorable_rate_pct"],
                "failure_mode": "Scout Watch was correct — apparent strength may be cosmetic (P19 penalized false conv)",
                "common_transition": p.get("common_transition", "false_convergence"),
                "recommended_scout_behavior": "Watch. Possible cosmetic strength. Avoid premature confidence.",
                "confidence": p["confidence"],
            }
        )

    # Playbook C — false convergence
    fc = [p for p in bad if "fake=True" in p["structure_key"]] or false_conf
    if fc:
        p = fc[0]
        playbooks.append(
            {
                "playbook_id": "C",
                "name": "False Convergence Trap",
                "observed_structure": "Multiple families agree BUT fake environment dominates",
                "supporting_evidence": "P18 false_convergence + P16 fake_bias field + fake_trend > real_trend",
                "frequency": p["frequency"],
                "success_rate_pct": p.get("favorable_rate_pct", 100 - p.get("unfavorable_rate_pct", 0)),
                "failure_mode": "Apparent agreement from correlated families",
                "common_transition": "false_convergence",
                "recommended_scout_behavior": "Watch. False convergence risk. Do not upgrade confidence.",
                "confidence": p["confidence"],
            }
        )

    # Playbook D — correct unknown
    if unknown:
        p = max(unknown, key=lambda x: x["frequency"])
        playbooks.append(
            {
                "playbook_id": "D",
                "name": "Honest Unknown",
                "observed_structure": "Transition/Late + conflicting convergence + mixed field",
                "supporting_evidence": "P19 correct_unknown audit + unfavorable > favorable when engaged",
                "frequency": p["frequency"],
                "success_rate_pct": 100 - p.get("unfavorable_rate_pct", 0),
                "failure_mode": "N/A — staying unknown was correct",
                "common_transition": p.get("common_transition", "weak_convergence"),
                "recommended_scout_behavior": "Unknown. Correct uncertainty. Do not force classification.",
                "confidence": p["confidence"],
            }
        )

    # Playbook E — late recognition
    if late:
        p = late[0]
        playbooks.append(
            {
                "playbook_id": "E",
                "name": "Late Recognition Pattern",
                "observed_structure": "Fertile seedbed appeared 2+ scans before Scout recognition",
                "supporting_evidence": "P17 lifecycle arc + P19 late_recognition",
                "frequency": p["frequency"],
                "success_rate_pct": p.get("favorable_rate_pct", ""),
                "failure_mode": "Evidence growth slower than opportunity field",
                "common_transition": "breaking_convergence after peak",
                "recommended_scout_behavior": "Watch field ecology earlier — symbol-level evaluation lags field.",
                "confidence": p["confidence"],
            }
        )

    return playbooks


def build_handbook(rows: list[dict], playbooks: list[dict]) -> list[dict]:
    lessons = [
        {
            "lesson_id": 1,
            "topic": "Independent evidence",
            "practical_wisdom": "Supply + interaction families are most useful; field_environment + health mislead most often (P19).",
            "source": "P19 self audit",
            "confidence": "high",
        },
        {
            "lesson_id": 2,
            "topic": "False convergence",
            "practical_wisdom": "When families agree but fake_trend > real_trend, downgrade — do not treat as Buy signal.",
            "source": "P18 + P19",
            "confidence": "high",
        },
        {
            "lesson_id": 3,
            "topic": "Correct Unknown",
            "practical_wisdom": "Unknown confidence with 35% unfavorable rate is honest calibration — not a failure to fix.",
            "source": "P19 calibration",
            "confidence": "high",
        },
        {
            "lesson_id": 4,
            "topic": "Field before symbol",
            "practical_wisdom": "Evaluate opportunity field ecology before ranking individual symbols within scan.",
            "source": "P16 + P17",
            "confidence": "medium",
        },
        {
            "lesson_id": 5,
            "topic": "Persistence over strength",
            "practical_wisdom": "Fertile seedbed lasting 2+ scans outweighs single-scan high convergence score.",
            "source": "P17 lifecycle",
            "confidence": "medium",
        },
        {
            "lesson_id": 6,
            "topic": "Watch not Buy",
            "practical_wisdom": "Healthy Trend + FRAGILE health → Watch only. Label is not health.",
            "source": "P11 + P20 patterns",
            "confidence": "high",
        },
    ]

    for pb in playbooks:
        lessons.append(
            {
                "lesson_id": f"playbook_{pb['playbook_id']}",
                "topic": pb["name"],
                "practical_wisdom": pb["recommended_scout_behavior"],
                "source": f"Playbook {pb['playbook_id']}",
                "confidence": pb["confidence"],
            }
        )

    # Situation-level wisdom from aggregated rows
    by_sit = defaultdict(list)
    for r in rows:
        by_sit[r.get("scout_thought_situation", "?")].append(r)
    for sit, grp in sorted(by_sit.items(), key=lambda x: -len(x[1])):
        if len(grp) < MIN_PATTERN_N:
            continue
        stats = pattern_stats(grp)
        lessons.append(
            {
                "lesson_id": f"situation_{sit.replace(' ', '_')}",
                "topic": f"Situation: {sit}",
                "practical_wisdom": (
                    f"Observed {stats['frequency']}x: favorable={stats['favorable_rate_pct']}% "
                    f"unfavorable={stats['unfavorable_rate_pct']}%. "
                    f"Default: {'Watch' if stats['favorable_rate_pct'] < 50 else 'Watch closely'}."
                ),
                "source": "P20 pattern mining",
                "confidence": stats["confidence"],
            }
        )
    return lessons


def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    rows = enrich_rows()
    if not rows:
        print("Run P15-P19 first")
        return

    good = mine_patterns(
        rows,
        lambda r: r.get("empirical_outcome") == "favorable"
        or r.get("audit_verdict") in ("missed_fertile", "correct_caution")
        or (r.get("in_fertile_seedbed") in (True, "True") and r.get("empirical_outcome") != "collapse"),
        "good",
    )

    bad = mine_patterns(
        rows,
        lambda r: r.get("audit_verdict") in (
            "premature_confidence",
            "false_convergence_validated",
            "early_false_confidence",
        )
        or r.get("false_convergence_flagged") in (True, "True")
        and r.get("empirical_outcome") in ("unfavorable", "collapse"),
        "bad",
    )

    unknown = mine_patterns(
        rows,
        lambda r: r.get("audit_verdict") == "correct_unknown",
        "unknown",
    )

    late = mine_patterns(
        rows,
        lambda r: r.get("audit_verdict") in ("late_recognition",)
        or r.get("timing_verdict") in ("too_late_or_missed", "too_cautious_missed"),
        "late",
    )

    false_conf = mine_patterns(
        rows,
        lambda r: r.get("audit_verdict") in ("premature_confidence", "early_false_confidence", "false_convergence_validated"),
        "false_confidence",
    )

    playbooks = build_playbooks(good, bad, unknown, late, false_conf)
    handbook = build_handbook(rows, playbooks)

    write_csv(PLAYBOOKS_CSV, playbooks)
    write_csv(GOOD_CSV, good)
    write_csv(BAD_CSV, bad)
    write_csv(UNKNOWN_CSV, unknown)
    write_csv(LATE_CSV, late)
    write_csv(FALSE_CONF_CSV, false_conf)
    write_csv(HANDBOOK_CSV, handbook)

    lines = [
        "===== SCOUT SEASON2 P20 - SCOUT PLAYBOOK =====",
        "",
        f"Enriched observations: {len(rows)}",
        f"Playbooks: {len(playbooks)} | Good patterns: {len(good)} | Bad: {len(bad)}",
        f"Unknown patterns: {len(unknown)} | Late: {len(late)} | False confidence: {len(false_conf)}",
        "",
        "--- Empirical playbooks ---",
    ]
    for pb in playbooks:
        lines.append(f"  Playbook {pb['playbook_id']}: {pb['name']}")
        lines.append(f"    Behavior: {pb['recommended_scout_behavior']}")
        lines.append(f"    Success rate: {pb.get('success_rate_pct')}% | n={pb.get('frequency')} | conf={pb['confidence']}")

    lines.extend(["", "--- Top good patterns ---"])
    for p in good[:4]:
        lines.append(f"  {p['structure_key'][:80]}: favorable={p['favorable_rate_pct']}% n={p['frequency']}")

    lines.extend(["", "--- Top bad patterns ---"])
    for p in bad[:4]:
        lines.append(f"  {p['structure_key'][:80]}: unfavorable={p['unfavorable_rate_pct']}% n={p['frequency']}")

    lines.extend([
        "",
        "Final question: What would an experienced Scout teach a new Scout?",
        "  1. Unknown is often correct — do not fix it by being more aggressive",
        "  2. False convergence is the main trap — families agree but fake environment dominates",
        "  3. Supply + interaction are the most honest families",
        "  4. Field ecology precedes symbol ranking",
        "  5. Persistence across scans beats single-scan convergence score",
        "  6. Watch is the default — Buy requires persistent independent convergence",
    ])
    lines.extend(mission_summary_lines())
    lines.extend(["", f"Handbook: {HANDBOOK_CSV}", "=" * 58])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P20 SCOUT PLAYBOOK =====")
    print(f"Playbooks: {len(playbooks)} | Patterns: good={len(good)} bad={len(bad)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
