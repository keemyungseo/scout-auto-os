"""
Scout Learning Season2 - P18 Evidence Convergence & Scout Confidence

Convergence layer on P15 + P16 + P17.
Independent empirical evidence families — not more signals, not prediction.

Governed by Scout Research Constitution and Scout Mission.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CONVERGENCE_CSV = LOGS_DIR / "season2_p18_convergence_scores.csv"
INDEPENDENT_CSV = LOGS_DIR / "season2_p18_independent_evidence.csv"
CONFLICT_CSV = LOGS_DIR / "season2_p18_conflict_map.csv"
LIFECYCLE_CSV = LOGS_DIR / "season2_p18_convergence_lifecycle.csv"
FALSE_CONV_CSV = LOGS_DIR / "season2_p18_false_convergence.csv"
TEMPORAL_CSV = LOGS_DIR / "season2_p18_temporal_convergence.csv"
CONFIDENCE_CSV = LOGS_DIR / "season2_p18_confidence_engine.csv"
REPORT_TXT = LOGS_DIR / "season2_p18_research_report.txt"

EARLY_SITUATIONS = {"Accumulation", "Early Trend", "Healthy Trend"}
LATE_SITUATIONS = {"Late Trend", "Distribution", "Recovery"}
FERTILE = {"Very fertile", "Fertile"}

# Independent evidence families — deliberately non-overlapping domains
EVIDENCE_FAMILIES = (
    "situation",
    "health",
    "pressure",
    "participant",
    "supply",
    "grammar",
    "interaction",
    "field_environment",
    "temporal_lifecycle",
    "collapse_risk",
    "regime_context",
)

# Redundant pairs — agreement between these counts as redundant, not independent
REDUNDANT_PAIRS = {
    frozenset({"situation", "health"}),  # both often label-derived
    frozenset({"pressure", "health"}),   # co-move in fragile states
    frozenset({"participant", "situation"}),
}


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


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


def vote_situation(row: dict) -> tuple[int, str]:
    sit = row.get("situation", "Unknown")
    if sit in EARLY_SITUATIONS:
        return 1, f"situation={sit}"
    if sit in LATE_SITUATIONS:
        return -1, f"situation={sit}"
    if sit == "Transition":
        return 0, "situation=Transition"
    return 0, "situation=Unknown"


def vote_health(row: dict) -> tuple[int, str]:
    h = row.get("health_class", "")
    if h in ("IMPROVING", "STABLE", "RECOVERING"):
        return 1, f"health={h}"
    if h == "FRAGILE":
        return -1, "health=FRAGILE"
    if h == "WEAKENING":
        return -1, "health=WEAKENING"
    return 0, f"health={h or 'unknown'}"


def vote_pressure(row: dict) -> tuple[int, str]:
    band = row.get("pressure_band", "")
    score = pf(row.get("pressure_score"), 50)
    if band in ("low",) or (score is not None and score < 32):
        return 1, f"pressure={band or score}"
    if band in ("high", "explosive") or (score is not None and score >= 50):
        return -1, f"pressure={band or score}"
    return 0, f"pressure={band or 'medium'}"


def vote_participant(row: dict) -> tuple[int, str]:
    ps = row.get("participant_state") or ""
    if any(x in ps for x in ("warm", "compression", "accumulation", "markup", "expansion")):
        return 1, f"participant={ps}"
    if any(x in ps for x in ("exhaustion", "late_chasing", "late_markup", "trapped", "liquidation", "panic")):
        return -1, f"participant={ps}"
    return 0, f"participant={ps or 'unknown'}"


def vote_supply(row: dict) -> tuple[int, str]:
    s = row.get("supply_context") or ""
    if s == "MID_SUPPLY":
        return 1, "supply=MID_SUPPLY"
    if s == "HIGH_SUPPLY":
        return 1, "supply=HIGH_SUPPLY"
    if s == "COLLAPSE":
        return -1, "supply=COLLAPSE"
    if s == "LOW_SUPPLY":
        return 0, "supply=LOW_SUPPLY"
    return 0, "supply=unknown"


def vote_grammar(row: dict) -> tuple[int, str]:
    g = row.get("grammar") or ""
    if "vol_increase" in g and "exhaustion" not in g:
        return 1, "grammar=expansion"
    if "vol_exhaustion" in g or "upper_wick" in g:
        return -1, "grammar=exhaustion"
    if "compression" in g or "inside_bar" in g:
        return 1, "grammar=compression"
    return 0, "grammar=neutral"


def vote_interaction(row: dict) -> tuple[int, str]:
    ev = row.get("empirical_evidence") or ""
    if "interaction_persist" in ev and "100" in ev:
        return 1, "interaction=high_persist"
    if "evolution_conflict" in ev:
        return -1, "interaction=evolution_conflict"
    if "label_health_mismatch" in ev:
        return -1, "interaction=label_mismatch"
    if ev:
        return 0, "interaction=mixed"
    return 0, "interaction=unknown"


def vote_field(field: dict | None, seedbed: dict | None) -> tuple[int, str]:
    if not field:
        return 0, "field=unknown"
    coh = field.get("coherence", "")
    env = field.get("environment_verdict", "")
    if coh == "supportive" and field.get("fertile_seedbed_count", 0) and int(field.get("fertile_seedbed_count", 0)) >= 1:
        return 1, f"field=supportive_fertile"
    if env == "fake_bias" or coh == "conflicting":
        return -1, f"field={env or coh}"
    if env == "genuine_bias":
        return 1, "field=genuine_bias"
    return 0, f"field={coh or 'mixed'}"


def vote_temporal(temp: dict | None) -> tuple[int, str]:
    if not temp:
        return 0, "temporal=unknown"
    where_now = temp.get("where_now", "")
    path = temp.get("similar_path", "")
    count = int(temp.get("similar_path_count") or 0)
    if where_now in EARLY_SITUATIONS and "Accumulation" in path or "Early" in path:
        return 1, f"temporal={path}"
    if where_now in LATE_SITUATIONS or "Distribution" in path or "Late" in path:
        return -1, f"temporal={path}"
    if count >= 8:
        return 0, f"temporal=repeated_path_n={count}"
    return 0, f"temporal={path or 'unknown'}"


def vote_collapse(row: dict) -> tuple[int, str]:
    risk = pf(row.get("collapse_risk_pct"), 0)
    if risk >= 50:
        return -1, f"collapse_risk={risk}"
    if risk < 15:
        return 1, f"collapse_risk={risk}"
    return 0, f"collapse_risk={risk}"


def vote_regime(row: dict) -> tuple[int, str]:
    r = row.get("regime_context", "Unknown")
    if r in ("Crash", "Bear"):
        return -1, f"regime={r}"
    if r in ("Recovery", "Bull", "Alt_Season"):
        return 1, f"regime={r}"
    return 0, f"regime={r}"


def vote_real_fake(row: dict) -> tuple[int, str]:
    real = pf(row.get("real_trend_score"), 50)
    fake = pf(row.get("fake_trend_score"), 50)
    if real > fake + 8:
        return 1, f"real={real}>fake={fake}"
    if fake > real + 10:
        return -1, f"fake={fake}>real={real}"
    return 0, f"real={real} fake={fake}"


FAMILY_VOTERS = {
    "situation": vote_situation,
    "health": vote_health,
    "pressure": vote_pressure,
    "participant": vote_participant,
    "supply": vote_supply,
    "grammar": vote_grammar,
    "interaction": vote_interaction,
    "field_environment": lambda row, ctx: vote_field(ctx.get("field"), ctx.get("seedbed")),
    "temporal_lifecycle": lambda row, ctx: vote_temporal(ctx.get("temporal")),
    "collapse_risk": lambda row, ctx: vote_collapse(row),
    "regime_context": lambda row, ctx: vote_regime(row),
}


def collect_votes(row: dict, ctx: dict) -> dict[str, dict]:
    votes = {}
    for family in EVIDENCE_FAMILIES:
        voter = FAMILY_VOTERS[family]
        if family in ("field_environment", "temporal_lifecycle", "collapse_risk", "regime_context"):
            v, note = voter(row, ctx)
        else:
            v, note = voter(row)
        votes[family] = {"vote": v, "note": note}
    # real_fake folded into interaction redundancy check — separate internal
    rf_v, rf_note = vote_real_fake(row)
    votes["_real_fake"] = {"vote": rf_v, "note": rf_note}
    return votes


def count_convergence(votes: dict[str, dict]) -> dict:
    support_families = [f for f, v in votes.items() if not f.startswith("_") and v["vote"] == 1]
    oppose_families = [f for f, v in votes.items() if not f.startswith("_") and v["vote"] == -1]
    neutral_families = [f for f, v in votes.items() if not f.startswith("_") and v["vote"] == 0]

    redundant = 0
    for i, f1 in enumerate(support_families):
        for f2 in support_families[i + 1 :]:
            if frozenset({f1, f2}) in REDUNDANT_PAIRS:
                redundant += 1
    for i, f1 in enumerate(oppose_families):
        for f2 in oppose_families[i + 1 :]:
            if frozenset({f1, f2}) in REDUNDANT_PAIRS:
                redundant += 1

    indep_support = len(support_families) - redundant
    indep_conflict = len(oppose_families)

    # real_fake cross-check
    rf = votes.get("_real_fake", {}).get("vote", 0)
    if rf == -1 and indep_support >= 2:
        false_convergence_risk = True
    else:
        false_convergence_risk = rf == -1 and len(support_families) >= 3

    raw_agreement = len(support_families) + len(oppose_families)
    redundancy_pct = round(redundant / max(raw_agreement, 1) * 100, 1)

    net = indep_support - indep_conflict
    score = clamp(50 + net * 12 - redundancy_pct * 0.3)

    return {
        "independent_support_count": max(0, indep_support),
        "independent_conflict_count": indep_conflict,
        "redundant_agreement_pairs": redundant,
        "redundancy_pct": redundancy_pct,
        "neutral_family_count": len(neutral_families),
        "support_families": "|".join(support_families),
        "conflict_families": "|".join(oppose_families),
        "convergence_score": round(score, 1),
        "false_convergence_risk": false_convergence_risk,
        "votes": votes,
    }


def convergence_state(current: float, prev: float | None, false_risk: bool) -> str:
    if false_risk:
        return "false_convergence"
    if prev is None:
        return "weak_convergence" if current >= 55 else "unknown"
    delta = current - prev
    if current >= 60 and delta >= 5:
        return "growing_convergence"
    if current >= 58 and abs(delta) < 5:
        return "stable_convergence"
    if delta <= -8:
        return "breaking_convergence"
    if current < 45:
        return "weak_convergence"
    return "weak_convergence"


def scout_confidence(conv: dict, state: str, persist_scans: int) -> tuple[str, str]:
    indep = conv["independent_support_count"]
    conflict = conv["independent_conflict_count"]
    false_risk = conv["false_convergence_risk"]

    if false_risk or state == "false_convergence":
        return "Watch", "false_convergence — families agree but fake trend dominates"
    if conflict >= 3 or (conflict >= 2 and indep < 2):
        return "Unknown", f"conflicting_convergence support={indep} conflict={conflict}"
    if indep >= 4 and persist_scans >= 2 and state == "stable_convergence":
        return "medium", f"persistent_independent_convergence={indep} scans={persist_scans}"
    if indep >= 3 and state == "growing_convergence":
        return "hypothesis", f"growing_convergence indep={indep}"
    if indep >= 2 and conflict <= 1:
        return "hypothesis", f"weak_convergence indep={indep}"
    return "Unknown", "insufficient_independent_convergence"


def build_indexes(p15, p16, p17_obs, p16_seedbed, p16_clusters):
    field_by_scan = {r["scan_time"]: r for r in p16}
    temp_by_key = {(r["scan_time"], r["symbol"]): r for r in p17_obs}
    seedbed_by_key: dict[tuple, dict] = {}
    for sb in p16_seedbed:
        if sb.get("seedbed_quality") not in FERTILE:
            continue
        for sym in (sb.get("symbols") or "").split("|"):
            if sym:
                seedbed_by_key[(sb["scan_time"], sym.strip())] = sb
    cluster_q = {(r["scan_time"], r["symbol"]): r.get("seedbed_quality") for r in p16_clusters}
    return field_by_scan, temp_by_key, seedbed_by_key, cluster_q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest-only", action="store_true")
    args = parser.parse_args()

    p15 = load_csv(LOGS_DIR / "season2_p15_operational_scores.csv")
    p16 = load_csv(LOGS_DIR / "season2_p16_opportunity_fields.csv")
    p17_obs = load_csv(LOGS_DIR / "season2_p17_temporal_observations.csv")
    p16_seedbed = load_csv(LOGS_DIR / "season2_p16_seedbed_quality.csv")
    p16_clusters = load_csv(LOGS_DIR / "season2_p16_field_clusters.csv")

    if not p15:
        print("P15 scores missing — run season2_p15_situation_output.py first")
        return

    field_by_scan, temp_by_key, seedbed_by_key, cluster_q = build_indexes(
        p15, p16, p17_obs, p16_seedbed, p16_clusters
    )

    rows = p15
    if args.latest_only:
        latest = max(r["scan_time"] for r in p15)
        rows = [r for r in p15 if r["scan_time"] == latest]

    convergence_rows = []
    independent_rows = []
    conflict_rows = []
    false_rows = []

    for row in rows:
        key = (row["scan_time"], row["symbol"])
        ctx = {
            "field": field_by_scan.get(row["scan_time"]),
            "seedbed": seedbed_by_key.get(key),
            "temporal": temp_by_key.get(key),
        }
        conv = count_convergence(collect_votes(row, ctx))

        convergence_rows.append(
            {
                "date": row["date"],
                "symbol": row["symbol"],
                "scan_time": row["scan_time"],
                "situation": row.get("situation"),
                "convergence_score": conv["convergence_score"],
                "independent_support_count": conv["independent_support_count"],
                "independent_conflict_count": conv["independent_conflict_count"],
                "redundant_agreement_pairs": conv["redundant_agreement_pairs"],
                "redundancy_pct": conv["redundancy_pct"],
                "support_families": conv["support_families"],
                "conflict_families": conv["conflict_families"],
                "false_convergence_risk": conv["false_convergence_risk"],
            }
        )

        for family, v in conv["votes"].items():
            if family.startswith("_"):
                continue
            independent_rows.append(
                {
                    "symbol": row["symbol"],
                    "scan_time": row["scan_time"],
                    "evidence_family": family,
                    "vote": v["vote"],
                    "vote_label": "support" if v["vote"] == 1 else "oppose" if v["vote"] == -1 else "neutral",
                    "note": v["note"],
                    "is_independent_domain": True,
                }
            )

        if conv["conflict_families"]:
            conflict_rows.append(
                {
                    "symbol": row["symbol"],
                    "scan_time": row["scan_time"],
                    "support_families": conv["support_families"],
                    "conflict_families": conv["conflict_families"],
                    "conflict_intensity": conv["independent_conflict_count"],
                    "resolution": "Unknown" if conv["independent_conflict_count"] >= 2 else "Watch",
                }
            )

        if conv["false_convergence_risk"]:
            false_rows.append(
                {
                    "symbol": row["symbol"],
                    "scan_time": row["scan_time"],
                    "convergence_score": conv["convergence_score"],
                    "support_families": conv["support_families"],
                    "fake_trend_score": row.get("fake_trend_score"),
                    "real_trend_score": row.get("real_trend_score"),
                    "field_verdict": (ctx.get("field") or {}).get("environment_verdict", ""),
                    "verdict": "false_convergence",
                    "note": "families_agree_but_fake_environment",
                }
            )

    # Temporal convergence per symbol
    conv_by_sym: dict[str, list] = defaultdict(list)
    for r in convergence_rows:
        conv_by_sym[r["symbol"]].append(r)
    for sym in conv_by_sym:
        conv_by_sym[sym].sort(key=lambda x: x["scan_time"])

    temporal_rows = []
    lifecycle_rows = []
    confidence_rows = []

    for sym, series in conv_by_sym.items():
        persist = 0
        prev_score = None
        prev_state = "start"

        for r in series:
            false_risk = r["false_convergence_risk"]
            state = convergence_state(r["convergence_score"], prev_score, false_risk)

            if r["convergence_score"] >= 55 and not false_risk and state in ("stable_convergence", "growing_convergence"):
                persist += 1
            else:
                persist = 0

            improved = prev_score is not None and r["convergence_score"] - prev_score >= 5
            weakened = prev_score is not None and prev_score - r["convergence_score"] >= 5

            temporal_rows.append(
                {
                    "symbol": sym,
                    "scan_time": r["scan_time"],
                    "convergence_score": r["convergence_score"],
                    "prev_convergence_score": prev_score if prev_score is not None else "",
                    "convergence_state": state,
                    "convergence_improved": improved,
                    "convergence_weakened": weakened,
                    "convergence_persist_scans": persist,
                    "independent_support": r["independent_support_count"],
                    "independent_conflict": r["independent_conflict_count"],
                }
            )

            conf, reason = scout_confidence(
                {
                    "independent_support_count": r["independent_support_count"],
                    "independent_conflict_count": r["independent_conflict_count"],
                    "false_convergence_risk": false_risk,
                },
                state,
                persist,
            )

            p15_row = next((x for x in p15 if x["symbol"] == sym and x["scan_time"] == r["scan_time"]), {})
            confidence_rows.append(
                {
                    "date": p15_row.get("date", ""),
                    "symbol": sym,
                    "scan_time": r["scan_time"],
                    "situation": r.get("situation"),
                    "convergence_score": r["convergence_score"],
                    "convergence_state": state,
                    "independent_support_count": r["independent_support_count"],
                    "independent_conflict_count": r["independent_conflict_count"],
                    "redundancy_pct": r["redundancy_pct"],
                    "convergence_persist_scans": persist,
                    "scout_confidence": conf,
                    "recommended_stance": "Watch" if conf in ("Unknown", "hypothesis") and false_risk else conf,
                    "reason": reason,
                    "empirical_question": "Do independent structures agree something meaningful is happening?",
                    "answer": (
                        "partial_yes" if r["independent_support_count"] >= 3 and not false_risk
                        else "no" if r["independent_conflict_count"] >= 2
                        else "unknown"
                    ),
                }
            )

            if state != prev_state:
                lifecycle_rows.append(
                    {
                        "symbol": sym,
                        "scan_time": r["scan_time"],
                        "convergence_state": state,
                        "convergence_score": r["convergence_score"],
                        "transition": f"{prev_state}->{state}",
                    }
                )
            prev_score = r["convergence_score"]
            prev_state = state

    write_csv(CONVERGENCE_CSV, convergence_rows)
    write_csv(INDEPENDENT_CSV, independent_rows)
    write_csv(CONFLICT_CSV, conflict_rows)
    write_csv(LIFECYCLE_CSV, lifecycle_rows)
    write_csv(FALSE_CONV_CSV, false_rows)
    write_csv(TEMPORAL_CSV, temporal_rows)
    write_csv(CONFIDENCE_CSV, confidence_rows)

    states = Counter(r["convergence_state"] for r in temporal_rows)
    conf_dist = Counter(r["scout_confidence"] for r in confidence_rows)
    avg_indep = statistics.mean(r["independent_support_count"] for r in convergence_rows)

    lines = [
        "===== SCOUT SEASON2 P18 - EVIDENCE CONVERGENCE =====",
        "",
        f"Observations: {len(convergence_rows)} | Evidence family votes: {len(independent_rows)}",
        f"Conflicts: {len(conflict_rows)} | False convergence: {len(false_rows)}",
        f"Avg independent support: {avg_indep:.1f}",
        "",
        "--- Convergence states ---",
    ]
    for st, n in states.most_common():
        lines.append(f"  {st}: {n}")

    lines.extend(["", "--- Scout confidence ---"])
    for c, n in conf_dist.most_common():
        lines.append(f"  {c}: {n}")

    lines.extend(["", "--- Independent evidence families ---"])
    for fam in EVIDENCE_FAMILIES:
        lines.append(f"  {fam}")

    lines.extend([
        "",
        "Final question: Do independent empirical structures gradually agree?",
        "Confidence increases only through independent + persistent convergence",
        "Unknown preferred over false certainty",
    ])
    lines.extend(mission_summary_lines())
    lines.extend(["", f"Confidence engine: {CONFIDENCE_CSV}", "=" * 58])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P18 EVIDENCE CONVERGENCE =====")
    print(f"Observations: {len(convergence_rows)} | False convergence: {len(false_rows)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
