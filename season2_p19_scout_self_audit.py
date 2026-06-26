"""
Scout Learning Season2 - P19 Scout Self Audit & Missed Opportunity Research

The Scout studies its own past decisions vs empirical outcomes.
NOT prediction optimization — honest self-improvement.

Governed by Scout Research Constitution and Scout Mission.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p14_regime_memory_bank import build_expanded_records
from season2_p15_situation_output import enrich_record_stack
from season2_p6_market_memory import attach_forward_targets
from season2_p7_behaviour_grammar import enrich_physics
from season2_p8_participant_state import enrich_panel_fields
from season2_p9_conditional_interaction_mining import prepare_records
from season2_regime_core import assign_regimes
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SELF_AUDIT_CSV = LOGS_DIR / "scout_self_audit.csv"
MISSED_CSV = LOGS_DIR / "missed_opportunities.csv"
CORRECT_UNKNOWN_CSV = LOGS_DIR / "correct_unknowns.csv"
LATE_CSV = LOGS_DIR / "late_recognition.csv"
EARLY_FALSE_CSV = LOGS_DIR / "early_false_confidence.csv"
CALIBRATION_CSV = LOGS_DIR / "confidence_calibration.csv"
REPORT_TXT = LOGS_DIR / "self_learning_report.txt"

FERTILE = {"Very fertile", "Fertile"}
EARLY_SITUATIONS = {"Accumulation", "Early Trend", "Healthy Trend"}
POSITIVE_F6 = 3.0
NEGATIVE_F6 = -5.0
COLLAPSE_RISK_HIGH = 50.0


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


def empirical_outcome(forward_6h: float | None, collapse_risk: float, collapse_label: str) -> str:
    if collapse_label == "YES" or (collapse_risk is not None and collapse_risk >= COLLAPSE_RISK_HIGH):
        return "collapse"
    if forward_6h is None:
        return "unknown"
    if forward_6h >= POSITIVE_F6:
        return "favorable"
    if forward_6h <= NEGATIVE_F6:
        return "unfavorable"
    return "mixed"


def scout_stance_bucket(action: str, confidence: str) -> str:
    if action in ("Strong Buy", "Buy"):
        return "engaged"
    if confidence in ("medium", "hypothesis") and action not in ("Avoid", "Reduce"):
        return "cautious_positive"
    if confidence == "Unknown" or action in ("Unknown", "Hold", "Wait"):
        return "uncertain"
    if action in ("Reduce", "Avoid"):
        return "defensive"
    return "watch"


def ideal_stance(outcome: str, situation: str) -> str:
    if outcome == "favorable" and situation in EARLY_SITUATIONS:
        return "cautious_positive"
    if outcome == "collapse" or outcome == "unfavorable":
        return "defensive"
    if outcome == "mixed":
        return "uncertain"
    return "uncertain"


def timing_verdict(scout_bucket: str, ideal: str, outcome: str) -> str:
    if outcome == "unknown":
        return "unknown"
    if scout_bucket == "engaged" and outcome in ("unfavorable", "collapse"):
        return "too_early_or_wrong"
    if scout_bucket == "uncertain" and outcome == "favorable":
        return "too_late_or_missed"
    if scout_bucket == "defensive" and outcome == "favorable":
        return "too_cautious_missed"
    if scout_bucket == "uncertain" and outcome in ("unfavorable", "collapse"):
        return "correctly_uncertain"
    if scout_bucket == "defensive" and outcome in ("unfavorable", "collapse"):
        return "correctly_defensive"
    if scout_bucket == "watch" and outcome == "mixed":
        return "correctly_ambiguous"
    return "aligned"


def audit_verdict(
    timing: str,
    false_conv: bool,
    confidence: str,
    outcome: str,
    in_fertile: bool,
    scout_bucket: str,
) -> str:
    if timing == "correctly_uncertain":
        return "correct_unknown"
    if timing == "correctly_defensive" or timing == "correctly_ambiguous":
        return "correct_caution"
    if false_conv and outcome in ("unfavorable", "collapse"):
        return "false_convergence_validated"
    if false_conv and outcome == "favorable":
        return "false_convergence_penalized"  # we were right to flag it
    if confidence in ("medium", "hypothesis") and outcome in ("unfavorable", "collapse"):
        return "premature_confidence"
    if in_fertile and outcome == "favorable" and scout_bucket in ("uncertain", "watch", "defensive"):
        return "missed_fertile"
    if timing == "too_early_or_wrong":
        return "early_false_confidence"
    if timing == "too_late_or_missed" or timing == "too_cautious_missed":
        return "late_recognition"
    return "neutral"


def build_outcome_index(records: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (r["scan_time"], r["symbol"]): {
            "forward_6h": r.get("target_f6") or r.get("forward_6h"),
            "forward_12h": r.get("forward_12h"),
            "collapse_label": r.get("collapse_label", ""),
        }
        for r in records
    }


def fertile_symbol_index(seedbeds: list[dict], watchlist: list[dict]) -> dict[tuple[str, str], dict]:
    idx: dict[tuple, dict] = {}
    for sb in seedbeds:
        if sb.get("seedbed_quality") not in FERTILE:
            continue
        for sym in (sb.get("symbols") or "").split("|"):
            sym = sym.strip()
            if sym:
                idx[(sb["scan_time"], sym)] = sb
    for w in watchlist:
        for sym in (w.get("symbols") or "").split("|"):
            sym = sym.strip()
            if sym:
                key = (w["scan_time"], sym)
                if key not in idx:
                    idx[key] = w
    return idx


def first_recognition_scan(symbol: str, confidence_rows: list[dict]) -> str | None:
    sym_rows = sorted([r for r in confidence_rows if r["symbol"] == symbol], key=lambda x: x["scan_time"])
    for r in sym_rows:
        if r.get("scout_confidence") in ("medium", "hypothesis") or r.get("answer") == "partial_yes":
            return r["scan_time"]
    return None


def first_fertile_scan(symbol: str, fertile_idx: dict) -> str | None:
    scans = sorted(st for (st, sym) in fertile_idx if sym == symbol)
    return scans[0] if scans else None


def evidence_family_audit(independent: list[dict], outcomes: dict) -> tuple[list[dict], list[dict]]:
    """Which families were misleading vs useful."""
    misleading = []
    useful = []
    by_obs: dict[tuple, list] = defaultdict(list)
    for row in independent:
        by_obs[(row["scan_time"], row["symbol"])].append(row)

    for (scan_time, symbol), votes in by_obs.items():
        outcome = outcomes.get((scan_time, symbol), {}).get("empirical_outcome", "unknown")
        for v in votes:
            if v["vote"] == "1" and outcome in ("unfavorable", "collapse"):
                misleading.append({**v, "empirical_outcome": outcome, "verdict": "misleading_support"})
            elif v["vote"] == "-1" and outcome == "favorable":
                misleading.append({**v, "empirical_outcome": outcome, "verdict": "misleading_oppose"})
            elif v["vote"] == "1" and outcome == "favorable":
                useful.append({**v, "empirical_outcome": outcome, "verdict": "useful_support"})
            elif v["vote"] == "-1" and outcome in ("unfavorable", "collapse"):
                useful.append({**v, "empirical_outcome": outcome, "verdict": "useful_oppose"})
    return misleading, useful


def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    p15 = load_csv(LOGS_DIR / "season2_p15_operational_scores.csv")
    p18 = load_csv(LOGS_DIR / "season2_p18_confidence_engine.csv")
    p18_false = {(r["scan_time"], r["symbol"]) for r in load_csv(LOGS_DIR / "season2_p18_false_convergence.csv")}
    independent = load_csv(LOGS_DIR / "season2_p18_independent_evidence.csv")
    seedbeds = load_csv(LOGS_DIR / "season2_p16_seedbed_quality.csv")
    watchlist = load_csv(LOGS_DIR / "season2_p16_watchlist.csv")
    temporal = load_csv(LOGS_DIR / "season2_p18_temporal_convergence.csv")

    if not p15 or not p18:
        print("Run P15 and P18 first")
        return

    records = build_expanded_records()
    enrich_physics(records)
    enrich_panel_fields(records)
    attach_forward_targets(records)
    prepare_records(records)
    enrich_record_stack(records)
    assign_regimes(records)
    outcome_raw = build_outcome_index(records)

    p18_idx = {(r["scan_time"], r["symbol"]): r for r in p18}
    temporal_idx = {(r["scan_time"], r["symbol"]): r for r in temporal}
    fertile_idx = fertile_symbol_index(seedbeds, watchlist)

    audit_rows = []
    missed_rows = []
    correct_unknown_rows = []
    late_rows = []
    early_false_rows = []

    outcomes_enriched: dict[tuple, dict] = {}

    for row in p15:
        key = (row["scan_time"], row["symbol"])
        raw = outcome_raw.get(key, {})
        f6 = pf(raw.get("forward_6h"))
        coll_risk = pf(row.get("collapse_risk_pct"), 0)
        coll_label = raw.get("collapse_label", "")
        outcome = empirical_outcome(f6, coll_risk, coll_label)
        outcomes_enriched[key] = {"empirical_outcome": outcome, "forward_6h": f6}

        conf_row = p18_idx.get(key, {})
        confidence = conf_row.get("scout_confidence", "Unknown")
        conv_state = conf_row.get("convergence_state", "")
        action = row.get("recommended_action", "Watch")
        scout_bucket = scout_stance_bucket(action, confidence)
        ideal = ideal_stance(outcome, row.get("situation", ""))
        timing = timing_verdict(scout_bucket, ideal, outcome)
        false_conv = key in p18_false
        in_fertile = key in fertile_idx

        verdict = audit_verdict(timing, false_conv, confidence, outcome, in_fertile, scout_bucket)

        temp = temporal_idx.get(key, {})
        conv_improved = temp.get("convergence_improved", "")
        conv_weakened = temp.get("convergence_weakened", "")

        audit_rows.append(
            {
                "date": row["date"],
                "symbol": row["symbol"],
                "scan_time": row["scan_time"],
                "scout_thought_situation": row.get("situation"),
                "scout_action": action,
                "scout_confidence": confidence,
                "convergence_state": conv_state,
                "convergence_score": conf_row.get("convergence_score", ""),
                "false_convergence_flagged": false_conv,
                "empirical_outcome": outcome,
                "forward_6h": f6 if f6 is not None else "",
                "collapse_risk_pct": coll_risk,
                "in_fertile_seedbed": in_fertile,
                "timing_verdict": timing,
                "audit_verdict": verdict,
                "where_was_i": scout_bucket,
                "where_should_have_been": ideal,
                "convergence_improved": conv_improved,
                "convergence_weakened": conv_weakened,
                "evidence_growth": "yes" if conv_improved in (True, "True") else "no",
                "evidence_decay": "yes" if conv_weakened in (True, "True") else "no",
            }
        )

        if verdict == "missed_fertile":
            missed_rows.append(
                {
                    **audit_rows[-1],
                    "seedbed_quality": fertile_idx[key].get("seedbed_quality", ""),
                    "reason": "fertile_seedbed + favorable_outcome + scout_too_cautious",
                }
            )
        if verdict == "correct_unknown":
            correct_unknown_rows.append({**audit_rows[-1], "reason": "unknown_stance + bad_or_mixed_outcome"})
        if verdict in ("late_recognition", "too_late_or_missed", "too_cautious_missed"):
            late_rows.append({**audit_rows[-1], "reason": verdict})
        if verdict in ("premature_confidence", "early_false_confidence", "false_convergence_validated"):
            early_false_rows.append({**audit_rows[-1], "reason": verdict})

    # Late recognition: fertile appeared N scans before scout recognized
    sym_scans = defaultdict(list)
    for r in p15:
        sym_scans[r["symbol"]].append(r["scan_time"])
    for sym in sym_scans:
        sym_scans[sym] = sorted(set(sym_scans[sym]))

    for sym in sym_scans:
        fert_scan = first_fertile_scan(sym, fertile_idx)
        rec_scan = first_recognition_scan(sym, p18)
        if not fert_scan:
            continue
        scans = sym_scans[sym]
        if fert_scan not in scans:
            continue
        fi = scans.index(fert_scan)
        if rec_scan and rec_scan in scans:
            ri = scans.index(rec_scan)
            delta = ri - fi
            if delta > 0:
                late_rows.append(
                    {
                        "symbol": sym,
                        "first_fertile_scan": fert_scan,
                        "first_recognition_scan": rec_scan,
                        "scans_late": delta,
                        "verdict": "delayed_recognition",
                    }
                )
        elif not rec_scan:
            late_rows.append(
                {
                    "symbol": sym,
                    "first_fertile_scan": fert_scan,
                    "first_recognition_scan": "never",
                    "scans_late": "unknown",
                    "verdict": "never_recognized",
                }
            )

    misleading, useful = evidence_family_audit(independent, outcomes_enriched)

    # Confidence calibration
    cal_groups: dict[str, list] = defaultdict(list)
    for a in audit_rows:
        conf = a["scout_confidence"]
        outcome = a["empirical_outcome"]
        cal_groups[conf].append(outcome)

    calibration_rows = []
    for conf, outcomes in sorted(cal_groups.items()):
        n = len(outcomes)
        favorable = sum(1 for o in outcomes if o == "favorable")
        unfavorable = sum(1 for o in outcomes if o in ("unfavorable", "collapse"))
        calibration_rows.append(
            {
                "scout_confidence": conf,
                "sample_size": n,
                "favorable_rate_pct": round(favorable / n * 100, 1) if n else 0,
                "unfavorable_rate_pct": round(unfavorable / n * 100, 1) if n else 0,
                "calibration": (
                    "honest" if conf == "Unknown" and unfavorable >= favorable
                    else "overconfident" if conf in ("medium", "hypothesis") and unfavorable > favorable
                    else "aligned" if favorable > unfavorable
                    else "mixed"
                ),
            }
        )

    family_mislead = Counter(r["evidence_family"] for r in misleading)
    family_useful = Counter(r["evidence_family"] for r in useful)

    write_csv(SELF_AUDIT_CSV, audit_rows)
    write_csv(MISSED_CSV, missed_rows)
    write_csv(CORRECT_UNKNOWN_CSV, correct_unknown_rows)
    write_csv(LATE_CSV, late_rows)
    write_csv(EARLY_FALSE_CSV, early_false_rows)
    write_csv(CALIBRATION_CSV, calibration_rows)

    verdict_dist = Counter(a["audit_verdict"] for a in audit_rows)
    timing_dist = Counter(a["timing_verdict"] for a in audit_rows)

    lines = [
        "===== SCOUT SEASON2 P19 - SELF AUDIT =====",
        "",
        f"Evaluations audited: {len(audit_rows)}",
        f"Missed fertile opportunities: {len(missed_rows)}",
        f"Correct Unknown decisions: {len(correct_unknown_rows)}",
        f"Late recognition events: {len(late_rows)}",
        f"Early false confidence: {len(early_false_rows)}",
        f"False convergence validated (outcome bad): {sum(1 for a in audit_rows if a['audit_verdict'] == 'false_convergence_validated')}",
        "",
        "--- Audit verdicts ---",
    ]
    for v, n in verdict_dist.most_common():
        lines.append(f"  {v}: {n}")

    lines.extend(["", "--- Timing ---"])
    for t, n in timing_dist.most_common():
        lines.append(f"  {t}: {n}")

    lines.extend(["", "--- Confidence calibration ---"])
    for r in calibration_rows:
        lines.append(
            f"  {r['scout_confidence']}: favorable={r['favorable_rate_pct']}% "
            f"unfavorable={r['unfavorable_rate_pct']}% [{r['calibration']}]"
        )

    lines.extend(["", "--- Misleading evidence families (top) ---"])
    for fam, n in family_mislead.most_common(5):
        lines.append(f"  {fam}: {n} misleading votes")

    lines.extend(["", "--- Useful evidence families (top) ---"])
    for fam, n in family_useful.most_common(5):
        lines.append(f"  {fam}: {n} useful votes")

    lines.extend([
        "",
        "Objective: fewer serious mistakes — not more Buy signals",
        "Correct uncertainty is success | False certainty is failure",
        "The Scout should become more honest, not more aggressive",
    ])
    lines.extend(mission_summary_lines())
    lines.extend(["", f"Self audit: {SELF_AUDIT_CSV}", "=" * 58])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P19 SCOUT SELF AUDIT =====")
    print(f"Audited: {len(audit_rows)} | Missed: {len(missed_rows)} | Correct Unknown: {len(correct_unknown_rows)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
