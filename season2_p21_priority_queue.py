"""
Scout Learning Season2 - P21 Scout Priority Queue & Attention Allocation

Decides where experienced attention should go first — not who wins.
Watch remains default. Unknown remains honest.

Builds on P15-P20.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

QUEUE_CSV = LOGS_DIR / "season2_p21_priority_queue.csv"
RANKING_CSV = LOGS_DIR / "season2_p21_attention_ranking.csv"
PROMOTIONS_CSV = LOGS_DIR / "season2_p21_promotions.csv"
DEMOTIONS_CSV = LOGS_DIR / "season2_p21_demotions.csv"
BACKGROUND_CSV = LOGS_DIR / "season2_p21_background_candidates.csv"
UNKNOWN_SAFE_CSV = LOGS_DIR / "season2_p21_unknown_safe.csv"
HANDBOOK_CSV = LOGS_DIR / "season2_p21_attention_handbook.csv"
REPORT_TXT = LOGS_DIR / "season2_p21_research_report.txt"

TIERS = ("S", "A", "B", "C", "D", "X")
EARLY_SITUATIONS = {"Accumulation", "Early Trend", "Healthy Trend"}
TIER_ORDER = {"S": 6, "A": 5, "B": 4, "C": 3, "D": 2, "X": 1}


def pf(val, default=None):
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


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


def fertile_symbols(watchlist: list[dict], seedbeds: list[dict]) -> dict[tuple[str, str], dict]:
    idx: dict[tuple, dict] = {}
    for sb in seedbeds:
        if sb.get("seedbed_quality") not in ("Fertile", "Very fertile"):
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


def attention_score(row: dict) -> tuple[float, list[str], list[str]]:
    """Attention score — NOT buy probability."""
    promote: list[str] = []
    risk: list[str] = []
    score = 40.0

    sit = row.get("situation", "")
    if sit in EARLY_SITUATIONS:
        score += 8
        promote.append(f"situation={sit}")
    if sit in ("Late Trend", "Distribution", "Recovery"):
        score -= 6
        risk.append(f"situation={sit}")

    persist = int(row.get("convergence_persist_scans") or 0)
    if persist >= 2:
        score += persist * 4
        promote.append(f"persistence={persist}_scans")
    elif persist == 0:
        risk.append("no_convergence_persistence")

    indep_s = int(row.get("independent_support") or 0)
    indep_c = int(row.get("independent_conflict") or 0)
    score += indep_s * 5
    score -= indep_c * 4
    if indep_s >= 3:
        promote.append(f"independent_support={indep_s}")
    if indep_c >= 3:
        risk.append(f"independent_conflict={indep_c}")

    if row.get("in_fertile_seedbed") in (True, "True"):
        score += 12
        promote.append("fertile_seedbed")
    if row.get("field_rank"):
        score += max(0, 10 - int(row.get("field_rank") or 10))
        promote.append(f"field_rank=#{row.get('field_rank')}")

    supply = row.get("supply_context", "")
    if supply in ("MID_SUPPLY", "HIGH_SUPPLY"):
        score += 6
        promote.append(f"supply={supply}")
    if supply == "COLLAPSE":
        score -= 20
        risk.append("COLLAPSE_supply")

    if "interaction" in (row.get("support_families") or ""):
        score += 4
        promote.append("interaction_support")

    if pbool(row.get("convergence_improved")):
        score += 10
        promote.append("convergence_improving")
    if pbool(row.get("convergence_weakened")):
        score -= 12
        risk.append("convergence_weakening")

    coll = pf(row.get("collapse_risk_pct"), 0)
    if coll is not None:
        if coll < 15:
            score += 4
        elif coll >= 40:
            score -= 15
            risk.append(f"collapse_risk={coll}")

    if pbool(row.get("false_convergence_flagged")):
        score -= 18
        risk.append("false_convergence")

    fake = pf(row.get("fake_trend_score"), 50)
    real = pf(row.get("real_trend_score"), 50)
    if fake is not None and real is not None and fake > real + 10:
        score -= 10
        risk.append("fake_environment")

    if row.get("audit_verdict") == "correct_unknown":
        score -= 8
        promote.append("honest_unknown_valid")

    if row.get("audit_verdict") == "missed_fertile":
        score += 6
        promote.append("late_recognition_risk_field_fertile")

    if row.get("playbook_match") in ("A", "E"):
        score += 5
        promote.append(f"playbook={row['playbook_match']}")
    if row.get("playbook_match") in ("C", "B"):
        score -= 4
        risk.append(f"playbook={row['playbook_match']}")

    how = (row.get("how_changing") or "").lower()
    if "improv" in how or "growing" in how or "real_d=" in how:
        score += 4
        promote.append("temporal_improvement")
    if "weaken" in how or "decay" in how or "exhaust" in how:
        score -= 6
        risk.append("temporal_decay")

    if row.get("seedbed_arc_path") and "Fertile" in (row.get("seedbed_arc_path") or ""):
        score += 3
        promote.append("seedbed_arc_fertile")

    return clamp(score), promote[:6], risk[:6]


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def assign_tier(score: float, row: dict, promote: list[str], risk: list[str]) -> str:
    false_conv = pbool(row.get("false_convergence_flagged"))
    coll = pf(row.get("collapse_risk_pct"), 0) or 0
    persist = int(row.get("convergence_persist_scans") or 0)
    indep_s = int(row.get("independent_support") or 0)
    indep_c = int(row.get("independent_conflict") or 0)
    sit = row.get("situation", "")

    if coll >= 50 or (sit == "Recovery" and row.get("supply_context") == "COLLAPSE"):
        return "X"
    if false_conv and indep_c >= 4 and coll >= 25:
        return "X"
    if row.get("audit_verdict") == "correct_unknown":
        return "D"
    if (
        score >= 78
        and persist >= 2
        and indep_s >= 4
        and indep_c <= 1
        and not false_conv
        and pbool(row.get("convergence_improved"))
    ):
        return "S"
    if score >= 62 and (row.get("in_fertile_seedbed") in (True, "True") or indep_s >= 3) and not false_conv:
        return "A"
    if score >= 52 and not false_conv and indep_s >= 2:
        return "B"
    if score < 38 or (indep_c >= 5 and not false_conv):
        return "X" if coll >= 30 or false_conv else "D"
    if score < 48 and row.get("scout_confidence") == "Unknown" and persist == 0:
        return "D"
    return "C"


def tier_confidence(tier: str, persist: int, indep_s: int) -> str:
    if tier == "S" and persist >= 2 and indep_s >= 3:
        return "medium"
    if tier in ("A", "B") and (persist >= 1 or indep_s >= 2):
        return "hypothesis"
    if tier == "D":
        return "honest_unknown"
    return "low"


def match_playbook(row: dict) -> str:
    sit = row.get("situation", "")
    if row.get("in_fertile_seedbed") in (True, "True") and sit == "Accumulation":
        return "A"
    if sit == "Healthy Trend" and pbool(row.get("false_convergence_flagged")):
        return "B"
    if pbool(row.get("false_convergence_flagged")):
        return "C"
    if row.get("audit_verdict") == "correct_unknown":
        return "D"
    if row.get("audit_verdict") in ("late_recognition", "missed_fertile"):
        return "E"
    return ""


def supporting_evidence(row: dict) -> str:
    parts = []
    if row.get("support_families"):
        parts.append(f"families={row['support_families']}")
    if row.get("convergence_state"):
        parts.append(f"conv={row['convergence_state']}")
    if row.get("field_verdict"):
        parts.append(f"field={row['field_verdict']}")
    if row.get("how_changing"):
        parts.append(f"temporal={row['how_changing'][:80]}")
    if row.get("audit_verdict"):
        parts.append(f"audit={row['audit_verdict']}")
    return "|".join(parts[:5])


def enrich_observations() -> list[dict]:
    p15 = load_csv(LOGS_DIR / "season2_p15_operational_scores.csv")
    audit_idx = {(r["scan_time"], r["symbol"]): r for r in load_csv(LOGS_DIR / "scout_self_audit.csv")}
    conv_idx = {(r["scan_time"], r["symbol"]): r for r in load_csv(LOGS_DIR / "season2_p18_convergence_scores.csv")}
    temp_idx = {(r["scan_time"], r["symbol"]): r for r in load_csv(LOGS_DIR / "season2_p18_temporal_convergence.csv")}
    conf_idx = {(r["scan_time"], r["symbol"]): r for r in load_csv(LOGS_DIR / "season2_p18_confidence_engine.csv")}
    p17_idx = {(r["scan_time"], r["symbol"]): r for r in load_csv(LOGS_DIR / "season2_p17_temporal_observations.csv")}
    field_by_scan = {r["scan_time"]: r for r in load_csv(LOGS_DIR / "season2_p16_opportunity_fields.csv")}
    false_conv_keys = {(r["scan_time"], r["symbol"]) for r in load_csv(LOGS_DIR / "season2_p18_false_convergence.csv")}
    fertile_idx = fertile_symbols(
        load_csv(LOGS_DIR / "season2_p16_watchlist.csv"),
        load_csv(LOGS_DIR / "season2_p16_seedbed_quality.csv"),
    )

    rows = []
    for p in p15:
        key = (p["scan_time"], p["symbol"])
        audit = audit_idx.get(key, {})
        conv = conv_idx.get(key, {})
        temp = temp_idx.get(key, {})
        conf = conf_idx.get(key, {})
        p17 = p17_idx.get(key, {})
        field = field_by_scan.get(p["scan_time"], {})
        fert = fertile_idx.get(key)

        row = {
            "date": p["date"],
            "symbol": p["symbol"],
            "scan_time": p["scan_time"],
            "situation": p.get("situation"),
            "supply_context": p.get("supply_context"),
            "collapse_risk_pct": p.get("collapse_risk_pct"),
            "real_trend_score": p.get("real_trend_score"),
            "fake_trend_score": p.get("fake_trend_score"),
            "recommended_action": p.get("recommended_action"),
            "in_fertile_seedbed": key in fertile_idx,
            "seedbed_quality": fert.get("seedbed_quality", "") if fert else "",
            "field_rank": field.get("field_relative_rank", ""),
            "field_score": field.get("field_relative_score", ""),
            "field_verdict": field.get("environment_verdict", ""),
            "field_coherence": field.get("coherence", ""),
            "convergence_score": conv.get("convergence_score", ""),
            "independent_support": conv.get("independent_support_count", temp.get("independent_support", "")),
            "independent_conflict": conv.get("independent_conflict_count", temp.get("independent_conflict", "")),
            "support_families": conv.get("support_families", ""),
            "false_convergence_flagged": key in false_conv_keys,
            "how_changing": p17.get("how_changing", ""),
            "seedbed_arc_path": p17.get("seedbed_arc_path", ""),
            "temporal_confidence": p17.get("confidence", ""),
            "convergence_state": conf.get("convergence_state", ""),
            "convergence_persist_scans": temp.get("convergence_persist_scans", conf.get("convergence_persist_scans", 0)),
            "convergence_improved": temp.get("convergence_improved", ""),
            "convergence_weakened": temp.get("convergence_weakened", ""),
            "scout_confidence": conf.get("scout_confidence", ""),
            "audit_verdict": audit.get("audit_verdict", ""),
            "empirical_outcome": audit.get("empirical_outcome", ""),
        }
        row["playbook_match"] = match_playbook(row)
        score, promote, risk = attention_score(row)
        tier = assign_tier(score, row, promote, risk)
        recent = (
            "improving" if pbool(row.get("convergence_improved"))
            else "weakening" if pbool(row.get("convergence_weakened"))
            else "flat"
        )
        row.update(
            {
                "attention_score": round(score, 1),
                "priority_tier": tier,
                "why_promoted": "|".join(promote) if tier in ("S", "A") else "",
                "why_demoted": "|".join(risk) if tier in ("D", "X") else "",
                "promote_reasons": "|".join(promote),
                "risk_factors": "|".join(risk),
                "supporting_evidence": supporting_evidence(row),
                "persistence_scans": row.get("convergence_persist_scans"),
                "recent_change": recent,
                "attention_confidence": tier_confidence(
                    tier, int(row.get("convergence_persist_scans") or 0), int(row.get("independent_support") or 0)
                ),
                "scout_stance": (
                    "Watch" if tier in ("S", "A", "B")
                    else "Unknown" if tier == "D"
                    else "Background" if tier == "C"
                    else "Avoid attention"
                ),
            }
        )
        rows.append(row)
    return rows


def tier_change(prev: str, curr: str) -> str:
    if TIER_ORDER.get(curr, 0) > TIER_ORDER.get(prev, 0):
        return "promoted"
    if TIER_ORDER.get(curr, 0) < TIER_ORDER.get(prev, 0):
        return "demoted"
    return "unchanged"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest-date-only", action="store_true")
    args = parser.parse_args()

    rows = enrich_observations()
    if not rows:
        print("Run P15-P20 first")
        return

    if args.latest_date_only:
        latest_date = max(r["date"] for r in rows)
        rows_focus = [r for r in rows if r["date"] == latest_date]
    else:
        rows_focus = rows

    ranked = sorted(rows_focus, key=lambda x: (-TIER_ORDER.get(x["priority_tier"], 0), -x["attention_score"]))
    for i, r in enumerate(ranked, 1):
        r["attention_rank"] = i
        r["attention_percentile"] = round((len(ranked) - i) / max(len(ranked) - 1, 1) * 100, 1)

    promotions = []
    demotions = []
    by_sym: dict[str, list] = defaultdict(list)
    for r in rows:
        by_sym[r["symbol"]].append(r)
    for sym, series in by_sym.items():
        series.sort(key=lambda x: x["scan_time"])
        for i in range(1, len(series)):
            prev, curr = series[i - 1], series[i]
            change = tier_change(prev["priority_tier"], curr["priority_tier"])
            entry = {
                "symbol": sym,
                "from_scan": prev["scan_time"],
                "to_scan": curr["scan_time"],
                "from_tier": prev["priority_tier"],
                "to_tier": curr["priority_tier"],
                "from_score": prev["attention_score"],
                "to_score": curr["attention_score"],
                "recent_change": curr["recent_change"],
                "reason": curr.get("promote_reasons") if change == "promoted" else curr.get("risk_factors"),
            }
            if change == "promoted":
                promotions.append({**entry, "why_promoted": entry["reason"]})
            elif change == "demoted":
                demotions.append({**entry, "why_demoted": entry["reason"]})

    background = [r for r in ranked if r["priority_tier"] == "C"]
    unknown_safe = [r for r in ranked if r["priority_tier"] == "D"]

    handbook = [
        {"rule": 1, "guidance": "Tier S is rare — only persistent independent improvement across 2+ scans", "source": "P18+P17"},
        {"rule": 2, "guidance": "Tier A = fertile field + early situation — inspect field before symbol", "source": "P16+P20 Playbook A"},
        {"rule": 3, "guidance": "Tier B = default active watchlist — most attention goes here", "source": "P21"},
        {"rule": 4, "guidance": "Tier C = background — check only if time permits", "source": "P21"},
        {"rule": 5, "guidance": "Tier D = honest Unknown — do not waste attention forcing clarity", "source": "P19+P20 Playbook D"},
        {"rule": 6, "guidance": "Tier X = COLLAPSE/false convergence/high conflict — avoid unnecessary attention", "source": "P18+P20 Playbook C"},
        {"rule": 7, "guidance": "Promotion requires tier jump + improving convergence — not single-scan strength", "source": "P17"},
        {"rule": 8, "guidance": "One hour rule: Tier S/A fields first, then B symbols in top field ranks", "source": "P21"},
    ]

    write_csv(QUEUE_CSV, ranked)
    write_csv(RANKING_CSV, ranked)
    write_csv(PROMOTIONS_CSV, promotions)
    write_csv(DEMOTIONS_CSV, demotions)
    write_csv(BACKGROUND_CSV, background)
    write_csv(UNKNOWN_SAFE_CSV, unknown_safe)
    write_csv(HANDBOOK_CSV, handbook)

    tier_dist = Counter(r["priority_tier"] for r in ranked)
    latest_date = max(r["date"] for r in rows)
    latest = sorted([r for r in rows if r["date"] == latest_date], key=lambda x: (-TIER_ORDER.get(x["priority_tier"], 0), -x["attention_score"]))
    improving = [r for r in ranked if r["recent_change"] == "improving"][:5]
    weakening = [r for r in ranked if r["recent_change"] == "weakening"][:5]
    noisy = [r for r in ranked if r["priority_tier"] == "D" and pbool(r.get("false_convergence_flagged"))][:5]
    persist_promo = [r for r in ranked if int(r.get("persistence_scans") or 0) >= 2 and r["priority_tier"] in ("S", "A")][:5]

    lines = [
        "===== SCOUT SEASON2 P21 - PRIORITY QUEUE =====",
        "",
        f"Observations ranked: {len(ranked)} | Promotions: {len(promotions)} | Demotions: {len(demotions)}",
        f"Latest date focus: {latest_date} ({len(latest)} scans)",
        "",
        "--- Priority tiers ---",
    ]
    for t in TIERS:
        lines.append(f"  Tier {t}: {tier_dist.get(t, 0)}")

    lines.extend(["", "--- Research Q1: Immediate attention ---"])
    for r in ranked[:6]:
        lines.append(
            f"  {r['symbol']} Tier-{r['priority_tier']} score={r['attention_score']} "
            f"| {r['why_promoted'] or r['promote_reasons']}"
        )

    lines.extend(["", "--- Research Q2: Can safely wait (Tier C/D) ---"])
    wait = [r for r in ranked if r["priority_tier"] in ("C", "D")][:5]
    for r in wait:
        lines.append(f"  {r['symbol']} Tier-{r['priority_tier']}: {r['scout_stance']}")

    lines.extend(["", "--- Research Q3: Improving rapidly ---"])
    for r in improving:
        lines.append(f"  {r['symbol']} {r['situation']} | {r['recent_change']} | {r['promote_reasons']}")

    lines.extend(["", "--- Research Q4: Becoming less attractive ---"])
    for r in weakening:
        lines.append(f"  {r['symbol']} {r['situation']} | {r['risk_factors']}")

    lines.extend(["", "--- Research Q5: Noisy — remain Unknown ---"])
    for r in noisy:
        lines.append(f"  {r['symbol']}: {r['risk_factors']} | audit={r.get('audit_verdict')}")

    lines.extend(["", "--- Research Q6: Promotion after persistence ---"])
    for r in persist_promo:
        lines.append(
            f"  {r['symbol']} persist={r['persistence_scans']} Tier-{r['priority_tier']} | {r['promote_reasons']}"
        )

    lines.extend(["", "--- One hour rule: where attention goes first ---"])
    hour_first = [r for r in latest if r["priority_tier"] in ("S", "A")][:5]
    if not hour_first:
        hour_first = [r for r in latest if r["priority_tier"] == "B"][:5]
    for r in hour_first:
        lines.append(
            f"  {r['symbol']} Tier-{r['priority_tier']} field=#{r.get('field_rank')} "
            f"{r['situation']} | {r['supporting_evidence'][:100]}"
        )

    lines.extend(["", "--- One hour rule: do not waste attention ---"])
    waste = [r for r in latest if r["priority_tier"] in ("X", "D")][:6]
    for r in waste:
        lines.append(f"  {r['symbol']} Tier-{r['priority_tier']}: {r['why_demoted'] or r['risk_factors'] or 'honest_unknown'}")

    lines.extend([
        "",
        "Priority != Buy probability. Priority = where attention goes first.",
        "Watch remains default. Unknown remains honest.",
        "Repeated evidence beats isolated evidence.",
    ])
    lines.extend(mission_summary_lines())
    lines.extend(["", f"Queue: {QUEUE_CSV}", "=" * 58])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P21 PRIORITY QUEUE =====")
    print(f"Ranked: {len(ranked)} | S={tier_dist.get('S',0)} A={tier_dist.get('A',0)} B={tier_dist.get('B',0)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
