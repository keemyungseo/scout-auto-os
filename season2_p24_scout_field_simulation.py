"""
Scout Learning Season2 - P24 Scout Field Simulation & Live Decision Replay

Train Scout attention discipline through historical replay.
No future leakage. No Buy/Sell. No price forecasting.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from season2_scout_mission import mission_summary_lines
from season2_p23_scout_memory import (
    archetype_label,
    build_cases,
    enrich_library,
    find_similar,
    pf,
    pi,
    pbool,
    scout_memory_recommendation,
    similarity,
)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LIVE_CSV = LOGS_DIR / "season2_p24_live_simulations.csv"
JOURNAL_CSV = LOGS_DIR / "season2_p24_decision_journal.csv"
ATTENTION_CSV = LOGS_DIR / "season2_p24_attention_changes.csv"
AUDIT_CSV = LOGS_DIR / "season2_p24_outcome_audit.csv"
LESSONS_CSV = LOGS_DIR / "season2_p24_experience_lessons.csv"
GROWTH_CSV = LOGS_DIR / "season2_p24_scout_growth.csv"
REPORT_TXT = LOGS_DIR / "season2_p24_research_report.txt"

DEFAULT_ANCHORS = (
    "2026-06-13 11:00:00",
    "2026-06-08 11:00:00",
    "2026-06-14 15:00:00",
)
TIME_OFFSETS_H = (0, 2, 4, 12, 24)
AUDIT_HORIZON_H = 24
TIER_ORDER = {"S": 6, "A": 5, "B": 4, "C": 3, "D": 2, "X": 1}
ACTIONS = (
    "Stay Unknown", "Watch", "Increase Attention",
    "Promote", "Demote", "Ignore",
)


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


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def all_scan_times(library: list[dict]) -> list[str]:
    return sorted(set(c["scan_time"] for c in library))


def scan_at_offset(anchor: str, offset_h: int, scans: list[str]) -> str | None:
    target = parse_ts(anchor) + timedelta(hours=offset_h)
    candidates = [s for s in scans if parse_ts(s) >= target]
    if not candidates:
        return scans[-1] if scans and parse_ts(scans[-1]) >= parse_ts(anchor) else None
    return candidates[0]


def scan_after(anchor: str, horizon_h: int, scans: list[str]) -> str | None:
    return scan_at_offset(anchor, horizon_h, scans)


def library_as_of(library: list[dict], as_of: str) -> list[dict]:
    """Strict no-leak filter: only data visible at or before as_of."""
    return [c for c in library if c["scan_time"] <= as_of]


def obs_at(library: list[dict], symbol: str, scan_time: str) -> dict | None:
    for c in library:
        if c["symbol"] == symbol and c["scan_time"] == scan_time:
            return c
    return None


def similar_as_of(query: dict, library: list[dict], k: int = 3) -> list[dict]:
    past = [c for c in library if c["scan_time"] < query["scan_time"]]
    scored = []
    for c in past:
        if c["symbol"] == query["symbol"] and c["scan_time"] == query["scan_time"]:
            continue
        sim, matches, diffs = similarity(query, c)
        scored.append({
            "similarity_score": sim,
            "historical_symbol": c["symbol"],
            "historical_scan_time": c["scan_time"],
            "historical_outcome": c.get("empirical_outcome"),
            "historical_tier": c.get("priority_tier"),
            "structural_match": "|".join(matches[:3]),
            "important_differences": "|".join(diffs[:2]),
        })
    scored.sort(key=lambda x: -x["similarity_score"])
    return scored[:k]


def live_panel(obs: dict, similar: list[dict]) -> dict:
    risks = []
    if pbool(obs.get("false_convergence_flagged")):
        risks.append("false_convergence")
    if pi(obs.get("independent_conflict")) >= 4:
        risks.append("high_conflict")
    if pf(obs.get("collapse_risk_pct"), 0) and pf(obs.get("collapse_risk_pct"), 0) >= 30:
        risks.append("collapse_risk")
    if pi(obs.get("convergence_persist_scans")) == 0:
        risks.append("no_persistence")

    sim_summary = " || ".join(
        f"{s['historical_symbol']}({s['similarity_score']})" for s in similar[:3]
    ) or "none"

    return {
        "where_are_we": f"{obs['situation']}|tier={obs['priority_tier']}|field={obs.get('field_verdict')}",
        "tier": obs["priority_tier"],
        "playbook": obs.get("playbook_match", "none"),
        "field_ecology": f"rank={obs.get('field_rank')}|{obs.get('field_verdict')}|{obs.get('field_coherence')}",
        "convergence_state": obs.get("convergence_state"),
        "persistence": f"scans={obs.get('convergence_persist_scans')}|6h={obs.get('persist_6h_pct')}%",
        "risks": "|".join(risks) if risks else "low",
        "similar_cases": sim_summary,
        "attention_score": obs.get("attention_score"),
        "unknown_honesty": obs.get("unknown_audit"),
    }


def scout_decision(
    obs: dict,
    prev_tier: str | None,
    prev_action: str | None,
    similar: list[dict],
) -> tuple[str, str, str]:
    """Attention action + reason + confidence. Never Buy/Sell."""
    tier = obs["priority_tier"]
    false_conv = pbool(obs.get("false_convergence_flagged"))
    persist = pi(obs.get("convergence_persist_scans"))
    unknown = obs.get("unknown_audit") in ("honest_unknown", "unknown_active")
    playbook = obs.get("playbook_match", "none")
    collapse = pf(obs.get("collapse_risk_pct"), 0) or 0

    if tier == "X" or (false_conv and persist == 0 and collapse >= 25):
        return "Ignore", "Tier X or false convergence with collapse — avoid unnecessary attention", "low"

    if prev_tier and TIER_ORDER.get(tier, 0) > TIER_ORDER.get(prev_tier, 0):
        if persist >= 2 and not false_conv:
            return "Promote", f"Tier {prev_tier}->{tier} with persistence={persist} and clean convergence", "hypothesis"
        if false_conv or persist < 2:
            return "Watch", f"Tier rose {prev_tier}->{tier} but persistence={persist} — wait for confirmation", "low"

    if prev_tier and TIER_ORDER.get(tier, 0) < TIER_ORDER.get(prev_tier, 0):
        return "Demote", f"Tier fell {prev_tier}->{tier} — reduce attention per structural break", "medium"

    if unknown or tier == "D" or playbook == "C":
        return "Stay Unknown", "Honest Unknown or false-convergence playbook — no forced clarity", "honest_unknown"

    if tier == "S":
        return "Increase Attention", "Tier S with persistent independent improvement — limited attention here first", "medium"

    if tier == "A":
        if persist >= 1 and playbook in ("A", "E"):
            return "Increase Attention", f"Tier A fertile/watch pattern playbook={playbook}", "hypothesis"
        return "Watch", "Tier A high interest — monitor field and persistence", "hypothesis"

    if similar and sum(1 for s in similar if s["similarity_score"] >= 85) >= 2:
        mem = scout_memory_recommendation(obs, similar)
        if "Unknown" in mem:
            return "Stay Unknown", mem, "honest_unknown"
        if "low attention" in mem:
            return "Ignore", mem, "low"

    if tier == "B":
        return "Watch", "Default watchlist tier — attention earned not assumed", "hypothesis"

    return "Watch", "Background ecology — light watch unless field improves", "low"


def next_audit_scan(
    symbol: str,
    step_scan: str,
    scans: list[str],
    sym_series: list[dict],
    prefer_h: int = 24,
    min_h: int = 2,
) -> str | None:
    """Prefer +24h audit window; fall back to next symbol scan without leaking decision-time data."""
    preferred = scan_after(step_scan, prefer_h, scans)
    if preferred and obs_at_series(sym_series, preferred):
        return preferred
    step_dt = parse_ts(step_scan)
    for c in sym_series:
        if c["scan_time"] <= step_scan:
            continue
        gap_h = (parse_ts(c["scan_time"]) - step_dt).total_seconds() / 3600
        if gap_h >= min_h:
            return c["scan_time"]
    return None


def obs_at_series(sym_series: list[dict], scan_time: str) -> dict | None:
    for c in sym_series:
        if c["scan_time"] == scan_time:
            return c
    return None


def tier_change_label(prev: str | None, curr: str) -> str:
    if not prev:
        return "initial"
    if TIER_ORDER.get(curr, 0) > TIER_ORDER.get(prev, 0):
        return "promoted"
    if TIER_ORDER.get(curr, 0) < TIER_ORDER.get(prev, 0):
        return "demoted"
    return "stable"


def audit_decision(
    action: str,
    obs_decision: dict,
    obs_future: dict | None,
    tier_at_decision: str,
    tier_future: str | None,
) -> tuple[str, str]:
    """Evaluate Scout quality — not profits."""
    if obs_future is None:
        return "insufficient_followup", "No future scan available for audit"

    outcome = obs_future.get("empirical_outcome", "unknown")
    persist_f = pi(obs_future.get("convergence_persist_scans"))
    false_f = pbool(obs_future.get("false_convergence_flagged"))
    tier_up = TIER_ORDER.get(tier_future or "", 0) > TIER_ORDER.get(tier_at_decision, 0)
    tier_down = TIER_ORDER.get(tier_future or "", 0) < TIER_ORDER.get(tier_at_decision, 0)

    if action == "Stay Unknown":
        if false_f or tier_future in ("D", "X") or outcome == "unfavorable":
            return "correct_unknown", "Unknown stance matched noisy or unfavorable evolution"
        if tier_future in ("A", "S") and persist_f >= 2 and outcome == "favorable":
            return "missed_opportunity", "Structure improved — attention could have increased earlier with persistence"
        if tier_up and outcome == "favorable":
            return "late_attention", "Favorable path emerged — patience was cautious but late"
        return "correct_caution", "Unknown remained appropriate given mixed evolution"

    if action == "Watch":
        if outcome == "favorable" and tier_future in ("A", "S"):
            return "correct_watch", "Watch matched improving structure without overconfidence"
        if false_f and outcome == "unfavorable":
            return "correct_caution", "Watch avoided forced promotion into false structure"
        if tier_up and persist_f >= 2:
            return "late_attention", "Watch was safe but promotion came late"
        return "correct_watch", "Watch default appropriate"

    if action == "Increase Attention":
        if false_f or tier_down:
            return "premature_attention", "Attention increased into weakening or false structure"
        if tier_future in ("A", "S") or outcome == "favorable":
            return "good_attention", "Attention aligned with favorable evolution"
        return "correct_watch", "Attention elevated but evolution mixed"

    if action == "Promote":
        if false_f or (tier_down and outcome == "unfavorable"):
            return "false_promotion", "Promotion not sustained — cosmetic or temporary upgrade"
        if tier_future in ("A", "S") or persist_f >= 2:
            return "good_attention", "Promotion justified by sustained structure"
        return "premature_attention", "Promotion ahead of persistence confirmation"

    if action == "Demote":
        if tier_down or false_f or outcome == "unfavorable":
            return "correct_demotion", "Demotion matched structural deterioration"
        if tier_up and outcome == "favorable":
            return "late_attention", "Demotion was cautious — structure recovered"
        return "correct_demotion", "Demotion reduced wasted attention"

    if action == "Ignore":
        if tier_future in ("X", "D") or false_f:
            return "correct_caution", "Ignore saved attention on low-value ecology"
        if tier_future in ("A", "S") and outcome == "favorable":
            return "missed_opportunity", "Ignored symbol later became attention-worthy"
        return "correct_caution", "Ignore appropriate for background noise"

    return "neutral", "No strong audit signal"


def extract_lessons(audits: list[dict]) -> list[dict]:
    by_verdict: dict[str, list] = defaultdict(list)
    for a in audits:
        by_verdict[a["audit_verdict"]].append(a)

    lessons = []
    templates = [
        ("correct_unknown", "Unknown honesty validated — false convergence and conflict mean patience wins"),
        ("correct_watch", "Watch default correct — persistence confirms before promotion"),
        ("good_attention", "Attention increase justified when persistence >= 2 and families align"),
        ("late_attention", "Scout missed early persistence — review field ecology timing not price"),
        ("premature_attention", "Single-scan strength is noise — wait for 2-scan persistence"),
        ("false_promotion", "Cosmetic convergence — families agree but fake environment rises"),
        ("correct_demotion", "Demotion prevented wasted attention on breaking convergence"),
        ("missed_opportunity", "Fertile field present but Unknown too long — check playbook A with persistence gate"),
        ("correct_caution", "Caution appropriate — mixed outcomes in similar historical cases"),
    ]
    for verdict, lesson_text in templates:
        items = by_verdict.get(verdict, [])
        if not items:
            continue
        playbooks = Counter(a.get("playbook") for a in items)
        families = Counter()
        for a in items:
            for f in (a.get("key_evidence_families") or "").split("|"):
                if f:
                    families[f] += 1
        lessons.append({
            "lesson_id": verdict,
            "frequency": len(items),
            "lesson": lesson_text,
            "common_playbook": playbooks.most_common(1)[0][0] if playbooks else "",
            "evidence_that_mattered": "|".join(f"{k}({v})" for k, v in families.most_common(4)),
            "evidence_that_failed": "false_convergence" if verdict in ("false_promotion", "premature_attention") else "",
            "should_unknown_remain": "yes" if verdict in ("correct_unknown", "correct_caution") else "conditional",
            "attention_timing": verdict,
        })
    return sorted(lessons, key=lambda x: -x["frequency"])


def scout_growth_metrics(journal: list[dict], audits: list[dict]) -> list[dict]:
    sim_ids = sorted(set(j["simulation_id"] for j in journal))
    rows = []
    for sim_id in sim_ids:
        j_items = [j for j in journal if j["simulation_id"] == sim_id]
        a_items = [a for a in audits if a["simulation_id"] == sim_id]
        if not a_items:
            continue
        verdicts = Counter(a["audit_verdict"] for a in a_items)
        total = len(a_items)
        rows.append({
            "simulation_id": sim_id,
            "anchor_scan": j_items[0]["anchor_scan"],
            "decision_steps": len(j_items),
            "audited_steps": total,
            "attention_accuracy_pct": round(100 * (
                verdicts.get("good_attention", 0) + verdicts.get("correct_watch", 0) +
                verdicts.get("correct_demotion", 0) + verdicts.get("correct_caution", 0)
            ) / total, 1),
            "unknown_honesty_pct": round(100 * verdicts.get("correct_unknown", 0) / max(
                sum(1 for j in j_items if j["attention_action"] == "Stay Unknown"), 1
            ), 1),
            "promotion_quality_pct": round(100 * verdicts.get("good_attention", 0) / max(
                sum(1 for j in j_items if j["attention_action"] == "Promote"), 1
            ), 1),
            "demotion_quality_pct": round(100 * verdicts.get("correct_demotion", 0) / max(
                sum(1 for j in j_items if j["attention_action"] == "Demote"), 1
            ), 1),
            "late_recognition_count": verdicts.get("late_attention", 0),
            "premature_confidence_count": verdicts.get("premature_attention", 0) + verdicts.get("false_promotion", 0),
            "false_convergence_resistance": verdicts.get("correct_unknown", 0) + verdicts.get("correct_caution", 0),
            "persistence_awareness": sum(1 for j in j_items if "persistence" in j.get("decision_reason", "")),
            "playbook_alignment": sum(1 for j in j_items if j.get("playbook") not in ("none", "")),
            "case_memory_usage": sum(1 for j in j_items if j.get("similar_cases_used") not in ("", "none")),
            "missed_opportunity_count": verdicts.get("missed_opportunity", 0),
            "discipline_score": round(100 * (
                verdicts.get("correct_unknown", 0) + verdicts.get("correct_watch", 0) +
                verdicts.get("correct_demotion", 0) + verdicts.get("correct_caution", 0) +
                verdicts.get("good_attention", 0)
            ) / total - verdicts.get("false_promotion", 0) * 5, 1),
        })

    if len(rows) > 1:
        rows.append({
            "simulation_id": "AGGREGATE",
            "anchor_scan": "all",
            "decision_steps": sum(r["decision_steps"] for r in rows),
            "audited_steps": sum(r["audited_steps"] for r in rows),
            "attention_accuracy_pct": round(statistics.mean(r["attention_accuracy_pct"] for r in rows), 1),
            "unknown_honesty_pct": round(statistics.mean(r["unknown_honesty_pct"] for r in rows), 1),
            "promotion_quality_pct": round(statistics.mean(r["promotion_quality_pct"] for r in rows), 1),
            "demotion_quality_pct": round(statistics.mean(r["demotion_quality_pct"] for r in rows), 1),
            "late_recognition_count": sum(r["late_recognition_count"] for r in rows),
            "premature_confidence_count": sum(r["premature_confidence_count"] for r in rows),
            "false_convergence_resistance": sum(r["false_convergence_resistance"] for r in rows),
            "persistence_awareness": sum(r["persistence_awareness"] for r in rows),
            "playbook_alignment": sum(r["playbook_alignment"] for r in rows),
            "case_memory_usage": sum(r["case_memory_usage"] for r in rows),
            "missed_opportunity_count": sum(r["missed_opportunity_count"] for r in rows),
            "discipline_score": round(statistics.mean(r["discipline_score"] for r in rows), 1),
        })
    return rows


def run_simulations(
    library: list[dict],
    anchors: tuple[str, ...],
    offsets_h: tuple[int, ...],
) -> tuple[list, list, list, list, list]:
    scans = all_scan_times(library)
    live_rows: list[dict] = []
    journal: list[dict] = []
    attention_changes: list[dict] = []
    audits: list[dict] = []

    sym_series: dict[str, list] = defaultdict(list)
    for c in library:
        sym_series[c["symbol"]].append(c)
    for sym in sym_series:
        sym_series[sym].sort(key=lambda x: x["scan_time"])

    for anchor in anchors:
        if anchor not in scans:
            closest = min(scans, key=lambda s: abs((parse_ts(s) - parse_ts(anchor)).total_seconds()))
            anchor = closest
        sim_id = f"sim_{anchor.replace(' ', '_').replace(':', '')}"
        symbols_at_anchor = [c["symbol"] for c in library if c["scan_time"] == anchor]
        if not symbols_at_anchor:
            continue

        prev_state: dict[str, dict] = {}

        for offset in offsets_h:
            step_scan = scan_at_offset(anchor, offset, scans)
            if not step_scan or parse_ts(step_scan) < parse_ts(anchor):
                continue
            as_of_lib = library_as_of(library, step_scan)

            for symbol in symbols_at_anchor:
                obs = obs_at(library, symbol, step_scan)
                if not obs:
                    continue

                similar = similar_as_of(obs, as_of_lib, 3)
                panel = live_panel(obs, similar)
                prev = prev_state.get(symbol, {})
                prev_tier = prev.get("tier")
                prev_action = prev.get("action")

                action, reason, confidence = scout_decision(obs, prev_tier, prev_action, similar)
                change = tier_change_label(prev_tier, obs["priority_tier"])

                live_rows.append({
                    "simulation_id": sim_id,
                    "anchor_scan": anchor,
                    "step_offset_h": offset,
                    "scan_time": step_scan,
                    "symbol": symbol,
                    "future_hidden": "yes",
                    "where_are_we": panel["where_are_we"],
                    "tier": panel["tier"],
                    "playbook": panel["playbook"],
                    "field_ecology": panel["field_ecology"],
                    "convergence_state": panel["convergence_state"],
                    "persistence": panel["persistence"],
                    "risks": panel["risks"],
                    "similar_cases": panel["similar_cases"],
                    "attention_score": panel["attention_score"],
                    "unknown_honesty": panel["unknown_honesty"],
                    "deserves_attention": "yes" if action in ("Increase Attention", "Promote", "Watch") and obs["priority_tier"] in ("S", "A") else "optional",
                    "deserves_patience": "yes" if action == "Stay Unknown" else "no",
                    "deserves_caution": "yes" if action in ("Ignore", "Demote") or panel["risks"] != "low" else "no",
                })

                journal.append({
                    "simulation_id": sim_id,
                    "anchor_scan": anchor,
                    "step_offset_h": offset,
                    "scan_time": step_scan,
                    "symbol": symbol,
                    "scout_state": f"tier={obs['priority_tier']}|sit={obs['situation']}",
                    "attention_action": action,
                    "decision_reason": reason,
                    "key_evidence_families": obs.get("evidence_composition") or obs.get("support_families", ""),
                    "similar_cases_used": panel["similar_cases"],
                    "playbook": obs.get("playbook_match", "none"),
                    "tier": obs["priority_tier"],
                    "confidence": confidence,
                    "unknown_honesty": obs.get("unknown_audit"),
                    "tier_change": change,
                })

                if change in ("promoted", "demoted") and prev_tier:
                    attention_changes.append({
                        "simulation_id": sim_id,
                        "scan_time": step_scan,
                        "symbol": symbol,
                        "from_tier": prev_tier,
                        "to_tier": obs["priority_tier"],
                        "direction": change,
                        "attention_action": action,
                        "reason": reason,
                    })

                prev_state[symbol] = {"tier": obs["priority_tier"], "action": action}

                audit_scan = next_audit_scan(symbol, step_scan, scans, sym_series[symbol])
                if audit_scan:
                    obs_future = obs_at(library, symbol, audit_scan)
                    verdict, audit_reason = audit_decision(
                        action, obs, obs_future, obs["priority_tier"],
                        obs_future["priority_tier"] if obs_future else None,
                    )
                    audits.append({
                        "simulation_id": sim_id,
                        "decision_scan_time": step_scan,
                        "audit_scan_time": audit_scan,
                        "symbol": symbol,
                        "attention_action": action,
                        "scout_belief": reason[:120],
                        "tier_at_decision": obs["priority_tier"],
                        "tier_at_audit": obs_future["priority_tier"] if obs_future else "",
                        "evolution": f"{obs['priority_tier']}->{obs_future['priority_tier']}" if obs_future else "",
                        "outcome_at_audit": obs_future.get("empirical_outcome") if obs_future else "",
                        "audit_verdict": verdict,
                        "audit_reason": audit_reason,
                        "playbook": obs.get("playbook_match"),
                        "key_evidence_families": obs.get("evidence_composition", ""),
                        "scout_stayed_honest": "yes" if verdict in (
                            "correct_unknown", "correct_watch", "correct_caution", "correct_demotion", "good_attention"
                        ) else "review",
                        "what_scout_learned": audit_reason,
                    })

    return live_rows, journal, attention_changes, audits


def build_report(live, journal, audits, lessons, growth) -> str:
    verdicts = Counter(a["audit_verdict"] for a in audits)
    actions = Counter(j["attention_action"] for j in journal)
    lines = [
        "===== SCOUT SEASON2 P24 - FIELD SIMULATION & LIVE REPLAY =====",
        "",
        f"Live steps: {len(live)} | Journal entries: {len(journal)} | Audits: {len(audits)}",
        f"Simulations: {len([g for g in growth if g['simulation_id'] != 'AGGREGATE'])}",
        "",
        "--- Task 1: Historical live simulation ---",
        "Future scans hidden at each step. Panel built from as-of data only.",
        "",
        "--- Task 2: Scout decisions (no Buy/Sell) ---",
    ]
    for act, n in actions.most_common():
        lines.append(f"  {act}: {n}")

    lines.extend(["", "--- Task 5: Outcome audit ---"])
    for v, n in verdicts.most_common():
        lines.append(f"  {v}: {n}")

    lines.extend(["", "--- Task 6: Experience lessons ---"])
    for les in lessons[:6]:
        lines.append(f"  {les['lesson_id']}: {les['lesson']}")

    lines.extend(["", "--- Task 7: Scout growth ---"])
    agg = next((g for g in growth if g["simulation_id"] == "AGGREGATE"), None)
    if agg:
        lines.append(f"  Discipline score: {agg['discipline_score']}")
        lines.append(f"  Attention accuracy: {agg['attention_accuracy_pct']}%")
        lines.append(f"  Unknown honesty: {agg['unknown_honesty_pct']}%")
        lines.append(f"  Premature confidence: {agg['premature_confidence_count']}")

    lines.extend(["", "--- Final questions (per simulation) ---"])
    if live:
        sample = live[0]
        lines.append(f"  Where are we? {sample['where_are_we']}")
        lines.append(f"  Seen before? {sample['similar_cases']}")
        lines.append(f"  Attention? {sample['deserves_attention']} | Patience? {sample['deserves_patience']}")
        lines.append(f"  Caution? {sample['deserves_caution']}")

    lines.extend([
        "",
        "Never forecast. Never leak future. Unknown remains valid.",
        "Attention is earned. Persistence beats excitement.",
    ])
    lines.extend(mission_summary_lines())
    lines.append("=" * 58)
    return "\n".join(lines)


def ensure_deps() -> None:
    if not (LOGS_DIR / "season2_p23_case_library.csv").exists():
        import season2_p23_scout_memory
        season2_p23_scout_memory.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", nargs="*", default=list(DEFAULT_ANCHORS))
    parser.add_argument("--rebuild-deps", action="store_true")
    args = parser.parse_args()

    if args.rebuild_deps:
        ensure_deps()
    else:
        ensure_deps()

    library = load_csv(LOGS_DIR / "season2_p23_case_library.csv")
    if not library:
        cases = build_cases()
        library = enrich_library(cases)
    if not library:
        print("Run P15-P23 first")
        return

    live, journal, attention_changes, audits = run_simulations(
        library, tuple(args.anchors), TIME_OFFSETS_H,
    )
    lessons = extract_lessons(audits)
    growth = scout_growth_metrics(journal, audits)
    report = build_report(live, journal, audits, lessons, growth)

    write_csv(LIVE_CSV, live)
    write_csv(JOURNAL_CSV, journal)
    write_csv(ATTENTION_CSV, attention_changes)
    write_csv(AUDIT_CSV, audits)
    write_csv(LESSONS_CSV, lessons)
    write_csv(GROWTH_CSV, growth)
    REPORT_TXT.write_text(report, encoding="utf-8")

    print("===== P24 FIELD SIMULATION =====")
    print(f"Live steps: {len(live)} | Journal: {len(journal)} | Audits: {len(audits)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
