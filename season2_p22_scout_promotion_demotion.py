"""
Scout Learning Season2 - P22 Scout Promotion & Demotion Engine

Studies how attention priority changes over time — not prices.
Builds on P15-P21.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TRANSITIONS_CSV = LOGS_DIR / "season2_p22_tier_transitions.csv"
PROMOTIONS_CSV = LOGS_DIR / "season2_p22_promotions.csv"
DEMOTIONS_CSV = LOGS_DIR / "season2_p22_demotions.csv"
STABLE_CSV = LOGS_DIR / "season2_p22_stable_tiers.csv"
FAKE_PROMO_CSV = LOGS_DIR / "season2_p22_fake_promotions.csv"
CHANGES_CSV = LOGS_DIR / "season2_p22_attention_changes.csv"
RULES_CSV = LOGS_DIR / "season2_p22_scout_rules.csv"
REPORT_TXT = LOGS_DIR / "season2_p22_research_report.txt"

QUEUE_CSV = LOGS_DIR / "season2_p21_priority_queue.csv"
TIER_ORDER = {"S": 6, "A": 5, "B": 4, "C": 3, "D": 2, "X": 1}
SUPPLY_RANK = {"COLLAPSE": 0, "LOW_SUPPLY": 1, "MID_SUPPLY": 2, "HIGH_SUPPLY": 3}
KEY_PROMOTIONS = ("A->S", "B->A", "D->B", "C->B", "D->A", "X->A", "X->B")
KEY_DEMOTIONS = ("S->A", "A->C", "A->X", "B->X", "A->D", "S->X", "B->C")
STABLE_FOCUS = ("S", "A", "D")


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


def tier_direction(prev: str, curr: str) -> str:
    if TIER_ORDER.get(curr, 0) > TIER_ORDER.get(prev, 0):
        return "promoted"
    if TIER_ORDER.get(curr, 0) < TIER_ORDER.get(prev, 0):
        return "demoted"
    return "stable"


def supply_delta(prev: str, curr: str) -> str:
    p, c = SUPPLY_RANK.get(prev, 1), SUPPLY_RANK.get(curr, 1)
    if c > p:
        return "supply_building"
    if c < p:
        return "supply_weakening"
    return "unchanged"


def interaction_changed(prev: dict, curr: dict) -> str:
    prev_f = set((prev.get("support_families") or "").split("|"))
    curr_f = set((curr.get("support_families") or "").split("|"))
    if "interaction" in curr_f and "interaction" not in prev_f:
        return "interaction_added"
    if "interaction" in prev_f and "interaction" not in curr_f:
        return "interaction_lost"
    return "unchanged"


def field_improved(prev: dict, curr: dict) -> str:
    pr, cr = pi(prev.get("field_rank"), 99), pi(curr.get("field_rank"), 99)
    if cr < pr:
        return "field_improved"
    if cr > pr:
        return "field_deteriorated"
    return "unchanged"


def fake_gap(row: dict) -> float:
    fake = pf(row.get("fake_trend_score"), 50) or 50
    real = pf(row.get("real_trend_score"), 50) or 50
    return fake - real


def build_transitions(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (all_changes, promotions, demotions) from per-symbol time series."""
    by_sym: dict[str, list] = defaultdict(list)
    for r in rows:
        by_sym[r["symbol"]].append(r)

    changes: list[dict] = []
    promotions: list[dict] = []
    demotions: list[dict] = []

    for sym, series in by_sym.items():
        series.sort(key=lambda x: x["scan_time"])
        for i in range(1, len(series)):
            prev, curr = series[i - 1], series[i]
            ft, tt = prev["priority_tier"], curr["priority_tier"]
            transition = f"{ft}->{tt}"
            direction = tier_direction(ft, tt)

            pre_persist = pi(prev.get("persistence_scans"))
            post_persist = pi(curr.get("persistence_scans"))
            pre_indep = pi(prev.get("independent_support"))
            post_indep = pi(curr.get("independent_support"))
            pre_conflict = pi(prev.get("independent_conflict"))
            post_conflict = pi(curr.get("independent_conflict"))

            entry = {
                "symbol": sym,
                "from_scan": prev["scan_time"],
                "to_scan": curr["scan_time"],
                "from_tier": ft,
                "to_tier": tt,
                "transition": transition,
                "direction": direction,
                "from_score": prev.get("attention_score"),
                "to_score": curr.get("attention_score"),
                "score_delta": round(pf(curr.get("attention_score"), 0) - pf(prev.get("attention_score"), 0), 1),
                "pre_persistence": pre_persist,
                "post_persistence": post_persist,
                "persistence_delta": post_persist - pre_persist,
                "pre_independent_support": pre_indep,
                "post_independent_support": post_indep,
                "independent_support_delta": post_indep - pre_indep,
                "pre_independent_conflict": pre_conflict,
                "post_independent_conflict": post_conflict,
                "conflict_delta": post_conflict - pre_conflict,
                "pre_supply": prev.get("supply_context"),
                "post_supply": curr.get("supply_context"),
                "supply_change": supply_delta(prev.get("supply_context", ""), curr.get("supply_context", "")),
                "field_change": field_improved(prev, curr),
                "interaction_change": interaction_changed(prev, curr),
                "pre_field_rank": prev.get("field_rank"),
                "post_field_rank": curr.get("field_rank"),
                "pre_situation": prev.get("situation"),
                "post_situation": curr.get("situation"),
                "pre_collapse_risk": prev.get("collapse_risk_pct"),
                "post_collapse_risk": curr.get("collapse_risk_pct"),
                "pre_false_convergence": prev.get("false_convergence_flagged"),
                "post_false_convergence": curr.get("false_convergence_flagged"),
                "convergence_improved": curr.get("convergence_improved"),
                "convergence_weakened": curr.get("convergence_weakened"),
                "playbook_match": curr.get("playbook_match"),
                "pre_playbook": prev.get("playbook_match"),
                "empirical_outcome": curr.get("empirical_outcome"),
                "audit_verdict": curr.get("audit_verdict"),
                "promote_reasons": curr.get("promote_reasons"),
                "risk_factors": curr.get("risk_factors"),
                "fake_gap_post": round(fake_gap(curr), 1),
                "in_fertile_post": curr.get("in_fertile_seedbed"),
            }
            changes.append(entry)

            if direction == "promoted":
                entry["why_promoted"] = curr.get("promote_reasons") or curr.get("why_promoted")
                entry["trust_signals"] = _trust_signals(prev, curr)
                promotions.append(entry)
            elif direction == "demoted":
                entry["why_demoted"] = curr.get("risk_factors") or curr.get("why_demoted")
                entry["demotion_causes"] = _demotion_causes(prev, curr)
                demotions.append(entry)

    return changes, promotions, demotions


def _trust_signals(prev: dict, curr: dict) -> str:
    signals = []
    if pi(curr.get("persistence_scans")) >= 2:
        signals.append("persistence_2plus")
    if pi(curr.get("independent_support")) > pi(prev.get("independent_support")):
        signals.append("indep_growth")
    if field_improved(prev, curr) == "field_improved":
        signals.append("field_improved")
    if pbool(curr.get("convergence_improved")):
        signals.append("convergence_improving")
    if curr.get("playbook_match") in ("A", "E"):
        signals.append(f"playbook_{curr.get('playbook_match')}")
    if supply_delta(prev.get("supply_context", ""), curr.get("supply_context", "")) == "supply_building":
        signals.append("supply_building")
    return "|".join(signals) if signals else "weak_promotion"


def _demotion_causes(prev: dict, curr: dict) -> str:
    causes = []
    if pf(curr.get("collapse_risk_pct"), 0) and pf(curr.get("collapse_risk_pct"), 0) >= 30:
        causes.append("collapse_risk_high")
    if pbool(curr.get("false_convergence_flagged")):
        causes.append("false_convergence")
    if field_improved(prev, curr) == "field_deteriorated":
        causes.append("field_deteriorated")
    if pbool(curr.get("convergence_weakened")):
        causes.append("breaking_convergence")
    if pi(curr.get("persistence_scans")) < pi(prev.get("persistence_scans")):
        causes.append("persistence_lost")
    if pi(curr.get("independent_conflict")) > pi(prev.get("independent_conflict")):
        causes.append("conflict_rising")
    if fake_gap(curr) > 15:
        causes.append("fake_environment")
    return "|".join(causes) if causes else "tier_rebalance"


def aggregate_transitions(changes: list[dict]) -> list[dict]:
    groups: dict[str, list] = defaultdict(list)
    for c in changes:
        groups[c["transition"]].append(c)

    rows = []
    for transition, items in sorted(groups.items(), key=lambda x: -len(x[1])):
        pre_p = [pi(i.get("pre_persistence")) for i in items]
        post_p = [pi(i.get("post_persistence")) for i in items]
        causes = Counter()
        for i in items:
            if i["direction"] == "promoted":
                for part in (i.get("promote_reasons") or "").split("|"):
                    if part:
                        causes[part.split("=")[0]] += 1
            elif i["direction"] == "demoted":
                for part in (i.get("demotion_causes") or i.get("risk_factors") or "").split("|"):
                    if part:
                        causes[part.split("=")[0].split("_")[0]] += 1
            else:
                causes["stable"] += 1

        rows.append({
            "transition": transition,
            "direction": items[0]["direction"],
            "frequency": len(items),
            "avg_pre_persistence": round(statistics.mean(pre_p), 2) if pre_p else 0,
            "avg_post_persistence": round(statistics.mean(post_p), 2) if post_p else 0,
            "avg_indep_delta": round(statistics.mean([i["independent_support_delta"] for i in items]), 2),
            "avg_score_delta": round(statistics.mean([pf(i.get("score_delta"), 0) for i in items]), 2),
            "common_causes": "|".join(f"{k}({v})" for k, v in causes.most_common(5)),
            "supply_building_pct": round(100 * sum(1 for i in items if i.get("supply_change") == "supply_building") / len(items), 1),
            "field_improved_pct": round(100 * sum(1 for i in items if i.get("field_change") == "field_improved") / len(items), 1),
            "convergence_improved_pct": round(100 * sum(1 for i in items if pbool(i.get("convergence_improved"))) / len(items), 1),
        })
    return rows


def build_stable_runs(rows: list[dict]) -> list[dict]:
    by_sym: dict[str, list] = defaultdict(list)
    for r in rows:
        by_sym[r["symbol"]].append(r)

    stable_runs: list[dict] = []
    for sym, series in by_sym.items():
        series.sort(key=lambda x: x["scan_time"])
        if not series:
            continue
        run_start = 0
        for i in range(1, len(series) + 1):
            if i < len(series) and series[i]["priority_tier"] == series[run_start]["priority_tier"]:
                continue
            tier = series[run_start]["priority_tier"]
            run = series[run_start:i]
            duration = len(run)
            if duration < 1:
                run_start = i
                continue
            families = Counter()
            for r in run:
                for f in (r.get("support_families") or "").split("|"):
                    if f:
                        families[f] += 1
            fertile_pct = round(100 * sum(1 for r in run if pbool(r.get("in_fertile_seedbed"))) / duration, 1)
            unknown_pct = round(100 * sum(1 for r in run if r.get("scout_confidence") == "Unknown") / duration, 1)

            stable_runs.append({
                "symbol": sym,
                "tier": tier,
                "start_scan": run[0]["scan_time"],
                "end_scan": run[-1]["scan_time"],
                "duration_scans": duration,
                "avg_attention_score": round(statistics.mean([pf(r.get("attention_score"), 0) for r in run]), 1),
                "avg_persistence": round(statistics.mean([pi(r.get("persistence_scans")) for r in run]), 2),
                "avg_independent_support": round(statistics.mean([pi(r.get("independent_support")) for r in run]), 2),
                "dominant_situation": Counter(r.get("situation") for r in run).most_common(1)[0][0],
                "common_evidence_families": "|".join(f"{k}({v})" for k, v in families.most_common(4)),
                "fertile_seedbed_pct": fertile_pct,
                "unknown_confidence_pct": unknown_pct,
                "field_verdict_mode": Counter(r.get("field_verdict") for r in run).most_common(1)[0][0],
                "stable_category": f"Stable_{tier}" if tier in STABLE_FOCUS else f"Stable_{tier}",
            })
            run_start = i

    return sorted(stable_runs, key=lambda x: (-x["duration_scans"], -TIER_ORDER.get(x["tier"], 0)))


def detect_fake_promotions(
    promotions: list[dict], changes: list[dict], by_sym: dict[str, list]
) -> list[dict]:
    """Flag promotions that look impressive but lack structural trust."""
    next_transition: dict[tuple, dict] = {}
    for c in changes:
        next_transition[(c["symbol"], c["from_scan"])] = c

    fake_rows: list[dict] = []
    for p in promotions:
        flags: list[str] = []
        transition = p["transition"]

        if pf(p.get("fake_gap_post"), 0) > 10:
            flags.append("fake_environment")
        if pbool(p.get("post_false_convergence")):
            flags.append("fake_convergence")
        if transition in ("A->S", "B->A") and pi(p.get("post_persistence")) < 2:
            flags.append("premature_attention")
        if p.get("empirical_outcome") == "unfavorable":
            flags.append("unfavorable_outcome")
        if p.get("playbook_match") in ("B", "C"):
            flags.append("cosmetic_playbook")

        nxt = next_transition.get((p["symbol"], p["to_scan"]))
        if nxt and nxt["direction"] == "demoted":
            flags.append("temporary_upgrade")

        if not flags:
            continue

        fake_rows.append({
            **p,
            "fake_promotion_flags": "|".join(flags),
            "primary_flag": flags[0],
            "trust_assessment": "distrust" if len(flags) >= 2 else "caution",
            "lesson": _fake_lesson(flags, transition),
        })
    return fake_rows


def _fake_lesson(flags: list[str], transition: str) -> str:
    if "temporary_upgrade" in flags:
        return "Promotion reversed next scan — wait for 2-scan persistence before increasing attention"
    if "fake_environment" in flags and transition in ("A->S", "B->A"):
        return "Fake score rising faster than real — cosmetic promotion, keep Watch not confidence"
    if "fake_convergence" in flags:
        return "Families agree cosmetically — demotion likely; Unknown was safer"
    if "premature_attention" in flags:
        return "Single-scan promotion without persistence — patience required"
    return "Promotion lacks independent structural support"


def analyze_key_patterns(promotions: list[dict], demotions: list[dict]) -> dict:
    def summarize(items: list[dict], key_transitions: tuple) -> dict:
        subset = [i for i in items if i["transition"] in key_transitions]
        if not subset:
            return {"count": 0}
        return {
            "count": len(subset),
            "avg_indep_delta": round(statistics.mean([i["independent_support_delta"] for i in subset]), 2),
            "avg_persist_delta": round(statistics.mean([i["persistence_delta"] for i in subset]), 2),
            "convergence_improved_pct": round(100 * sum(1 for i in subset if pbool(i.get("convergence_improved"))) / len(subset), 1),
            "field_improved_pct": round(100 * sum(1 for i in subset if i.get("field_change") == "field_improved") / len(subset), 1),
            "supply_building_pct": round(100 * sum(1 for i in subset if i.get("supply_change") == "supply_building") / len(subset), 1),
            "interaction_added_pct": round(100 * sum(1 for i in subset if i.get("interaction_change") == "interaction_added") / len(subset), 1),
            "playbook_a_pct": round(100 * sum(1 for i in subset if i.get("playbook_match") == "A") / len(subset), 1),
        }

    return {
        "promo_A_S": summarize(promotions, ("A->S",)),
        "promo_B_A": summarize(promotions, ("B->A",)),
        "promo_D_B": summarize(promotions, ("D->B", "D->A")),
        "demo_S_A": summarize(demotions, ("S->A",)),
        "demo_A_C": summarize(demotions, ("A->C",)),
        "demo_A_X": summarize(demotions, ("A->X",)),
        "demo_B_X": summarize(demotions, ("B->X",)),
    }


def build_scout_rules(patterns: dict, transition_agg: list[dict], fake_promos: list[dict], stable: list[dict]) -> list[dict]:
    rules = [
        {
            "rule_id": 1,
            "question": "When should attention increase?",
            "answer": "When independent_support grows + convergence_improved + persistence >= 2 across scans",
            "evidence": f"B->A: conv_improved {patterns['promo_B_A'].get('convergence_improved_pct', 0)}% | indep_delta {patterns['promo_B_A'].get('avg_indep_delta', 0)}",
            "source": "P22 Task2",
        },
        {
            "rule_id": 2,
            "question": "When should attention stay unchanged?",
            "answer": "When tier stable 2+ scans with honest Unknown or background ecology — do not force promotion",
            "evidence": f"Stable_D runs: {sum(1 for s in stable if s['tier'] == 'D' and s['duration_scans'] >= 2)}",
            "source": "P22 Task4",
        },
        {
            "rule_id": 3,
            "question": "When should attention decrease?",
            "answer": "When false_convergence, conflict rising, persistence lost, or collapse_risk >= 30",
            "evidence": f"A->X count={patterns['demo_A_X'].get('count', 0)} | B->X count={patterns['demo_B_X'].get('count', 0)}",
            "source": "P22 Task3",
        },
        {
            "rule_id": 4,
            "question": "When should Unknown remain Unknown?",
            "answer": "When fake_environment + false_convergence coexist — P19 correct_unknown pattern",
            "evidence": f"Fake promotions flagged: {len(fake_promos)}",
            "source": "P19+P22 Task5",
        },
        {
            "rule_id": 5,
            "question": "Which promotions deserve trust?",
            "answer": "D->B/A with supply_building + field_improved + playbook A alignment",
            "evidence": f"D->B/A: field_improved {patterns['promo_D_B'].get('field_improved_pct', 0)}%",
            "source": "P22 Task2",
        },
        {
            "rule_id": 6,
            "question": "Which demotions are temporary?",
            "answer": "Single-scan demotion from A->C with flat convergence — often tier noise, not structural break",
            "evidence": "Check next-scan reversal in attention_changes",
            "source": "P22 Task3",
        },
        {
            "rule_id": 7,
            "question": "Which changes are fake?",
            "answer": "A->S or B->A with fake_gap>10, false_convergence, or reversed next scan",
            "evidence": f"fake_promotion count={len(fake_promos)}",
            "source": "P22 Task5",
        },
        {
            "rule_id": 8,
            "question": "Stable S structures",
            "answer": "Rare — require sustained indep_support >= 4 and interaction family present",
            "evidence": f"Stable_S runs: {sum(1 for s in stable if s['tier'] == 'S')}",
            "source": "P22 Task4",
        },
    ]

    top_trans = sorted(transition_agg, key=lambda x: -x["frequency"])[:3]
    if top_trans:
        rules.append({
            "rule_id": 9,
            "question": "Most common transitions",
            "answer": " | ".join(f"{t['transition']}({t['frequency']})" for t in top_trans),
            "evidence": top_trans[0].get("common_causes", ""),
            "source": "P22 Task1",
        })
    return rules


def build_report(
    changes: list[dict],
    transition_agg: list[dict],
    promotions: list[dict],
    demotions: list[dict],
    stable: list[dict],
    fake_promos: list[dict],
    patterns: dict,
    rules: list[dict],
) -> str:
    promo_n = sum(1 for c in changes if c["direction"] == "promoted")
    demo_n = sum(1 for c in changes if c["direction"] == "demoted")
    stable_n = sum(1 for c in changes if c["direction"] == "stable")

    lines = [
        "===== SCOUT SEASON2 P22 - PROMOTION & DEMOTION ENGINE =====",
        "",
        f"Tier transitions tracked: {len(changes)} | Promotions: {promo_n} | Demotions: {demo_n} | Stable steps: {stable_n}",
        f"Fake promotions flagged: {len(fake_promos)} | Stable tier runs: {len(stable)}",
        "",
        "--- Task 1: Tier transition frequency ---",
    ]
    for t in transition_agg[:12]:
        lines.append(
            f"  {t['transition']}: n={t['frequency']} "
            f"pre_persist={t['avg_pre_persistence']} post_persist={t['avg_post_persistence']} "
            f"| {t['common_causes']}"
        )

    lines.extend(["", "--- Task 2: Promotion patterns ---"])
    for key, label in [
        ("promo_A_S", "A->S"),
        ("promo_B_A", "B->A"),
        ("promo_D_B", "D->B/D->A"),
    ]:
        p = patterns[key]
        if p.get("count"):
            lines.append(
                f"  {label}: n={p['count']} indep_delta={p['avg_indep_delta']} "
                f"conv_improved={p['convergence_improved_pct']}% field_improved={p['field_improved_pct']}% "
                f"supply_building={p['supply_building_pct']}% playbook_A={p['playbook_a_pct']}%"
            )

    lines.extend(["", "--- Task 3: Demotion patterns ---"])
    for key, label in [
        ("demo_S_A", "S->A"),
        ("demo_A_C", "A->C"),
        ("demo_A_X", "A->X"),
        ("demo_B_X", "B->X"),
    ]:
        p = patterns[key]
        if p.get("count"):
            lines.append(
                f"  {label}: n={p['count']} indep_delta={p['avg_indep_delta']} "
                f"persist_delta={p.get('avg_persist_delta', 0)}"
            )

    lines.extend(["", "--- Task 4: Stable tiers (longest runs) ---"])
    for s in stable[:8]:
        lines.append(
            f"  {s['symbol']} Stable_{s['tier']} duration={s['duration_scans']} "
            f"families={s['common_evidence_families']}"
        )

    lines.extend(["", "--- Task 5: Fake promotions (sample) ---"])
    for f in fake_promos[:6]:
        lines.append(
            f"  {f['symbol']} {f['transition']}: {f['fake_promotion_flags']} | {f['lesson']}"
        )

    lines.extend(["", "--- Task 6: Scout practical lessons ---"])
    for r in rules[:6]:
        lines.append(f"  {r['question']}")
        lines.append(f"    -> {r['answer']}")

    lines.extend([
        "",
        "--- Final answer ---",
        "Empirical structures governing attention changes:",
        "  1. TRUST promotions: independent_support growth + persistence >= 2 + field_improved",
        "  2. PATIENCE: Stable D/Unknown with correct_unknown audit — do not promote on single-scan convergence",
        "  3. DEMOTION: false_convergence + conflict rise + collapse_risk — structural break not noise",
        "  4. FAKE: cosmetic A->S when fake_trend > real_trend — impressive but reversed quickly",
        "",
        "A good Scout understands promotion requires repeated evidence,",
        "patience when Unknown is honest, and demotion when convergence breaks.",
    ])
    lines.extend(mission_summary_lines())
    lines.append("=" * 58)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-p21", action="store_true", help="Re-run P21 queue before P22")
    args = parser.parse_args()

    if args.rebuild_p21 or not QUEUE_CSV.exists():
        from season2_p21_priority_queue import enrich_observations, write_csv as w21, QUEUE_CSV as Q21

        rows21 = enrich_observations()
        if not rows21:
            print("Run P15-P20 first")
            return
        w21(Q21, rows21)

    rows = load_csv(QUEUE_CSV)
    if not rows:
        print(f"Missing {QUEUE_CSV} — run P21 first")
        return

    changes, promotions, demotions = build_transitions(rows)
    transition_agg = aggregate_transitions(changes)
    stable = build_stable_runs(rows)

    by_sym: dict[str, list] = defaultdict(list)
    for r in rows:
        by_sym[r["symbol"]].append(r)
    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: x["scan_time"])

    fake_promos = detect_fake_promotions(promotions, changes, by_sym)
    patterns = analyze_key_patterns(promotions, demotions)
    rules = build_scout_rules(patterns, transition_agg, fake_promos, stable)
    report = build_report(changes, transition_agg, promotions, demotions, stable, fake_promos, patterns, rules)

    write_csv(TRANSITIONS_CSV, transition_agg)
    write_csv(PROMOTIONS_CSV, promotions)
    write_csv(DEMOTIONS_CSV, demotions)
    write_csv(STABLE_CSV, stable)
    write_csv(FAKE_PROMO_CSV, fake_promos)
    write_csv(CHANGES_CSV, changes)
    write_csv(RULES_CSV, rules)
    REPORT_TXT.write_text(report, encoding="utf-8")

    print("===== P22 PROMOTION & DEMOTION ENGINE =====")
    print(f"Transitions: {len(changes)} | Promotions: {len(promotions)} | Demotions: {len(demotions)}")
    print(f"Stable runs: {len(stable)} | Fake promotions: {len(fake_promos)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
