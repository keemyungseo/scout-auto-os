"""Daily research Telegram report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_research_report_text(report: dict) -> str:
    lines = [
        "SCOUT Research Report",
        f"Date: {report.get('report_date', 'n/a')}",
        f"Samples yesterday: {report.get('samples_yesterday', 0)}",
        f"TOP20 candidates: {report.get('top20_count', 0)}",
        "",
        f"A6_CURRENT win2h: {report.get('a6_win_rate_2h', 0)}% avg2h={report.get('a6_avg_return_2h', 0)}%",
        f"Random baseline: win2h={report.get('random_win_rate_2h', 0)}% avg2h={report.get('random_avg_return_2h', 0)}%",
        "",
        "Formula League (top 5):",
    ]
    for row in (report.get("formula_league_top") or [])[:5]:
        lines.append(
            f"  {row.get('formula_name')} score={row.get('score')} win2h={row.get('win_rate_2h')}%"
        )
    lines.append("")
    lines.append("Feature League (top 5):")
    for row in (report.get("feature_league_top") or [])[:5]:
        lines.append(
            f"  {row.get('feature_name')} {row.get('condition')} win2h={row.get('win_rate_2h')}%"
        )
    lines.append("")
    lines.append("Missed Big Winners:")
    for m in (report.get("missed_big_winners") or [])[:5]:
        lines.append(f"  {m.get('scan_time_kst')} {m.get('symbol')} ret2h={m.get('return_2h')}%")
    lines.append("")
    lines.append("Trap patterns:")
    for t in (report.get("trap_patterns") or [])[:3]:
        lines.append(f"  {t.get('scan_time_kst')} {t.get('symbol')} ret2h={t.get('return_2h')}%")
    lines.append("")
    lines.append("A6 improvement candidates:")
    for c in (report.get("improvement_candidates") or [])[:5]:
        lines.append(f"  [{c.get('tier')}] {c.get('label')}: {c.get('detail')}")
    return "\n".join(lines)


def classify_improvements(formula_league: list[dict], feature_league: list[dict]) -> list[dict]:
    out: list[dict] = []
    if not formula_league:
        return out
    best = formula_league[0]
    a6 = next((x for x in formula_league if x.get("formula_name") == "A6_CURRENT"), None)
    if best and a6 and best.get("formula_name") != "A6_CURRENT":
        delta = float(best.get("win_rate_2h", 0)) - float(a6.get("win_rate_2h", 0))
        if delta >= 5:
            out.append({
                "tier": "verification_needed",
                "label": best["formula_name"],
                "detail": f"win2h +{delta:.1f}pp vs A6_CURRENT (research only)",
            })
        elif delta >= 2:
            out.append({
                "tier": "hypothesis",
                "label": best["formula_name"],
                "detail": f"win2h +{delta:.1f}pp vs A6_CURRENT",
            })
    for feat in feature_league[:3]:
        if feat.get("comment") == "apply_forbidden_trap_risk":
            out.append({
                "tier": "apply_forbidden",
                "label": f"{feat['feature_name']} {feat['condition']}",
                "detail": f"trap_rate={feat.get('trap_rate')}%",
            })
        elif feat.get("comment") == "verification_candidate":
            out.append({
                "tier": "verification_needed",
                "label": f"{feat['feature_name']} {feat['condition']}",
                "detail": f"win2h={feat.get('win_rate_2h')}% n={feat.get('sample_count')}",
            })
    return out


def build_daily_report_payload(
    store,
    formula_league: list[dict],
    feature_league: list[dict],
    forward_rows: list[dict],
    candidate_count_yesterday: int,
) -> dict:
    today = datetime.now(KST).strftime("%Y-%m-%d")
    a6 = next((x for x in formula_league if x.get("formula_name") == "A6_CURRENT"), {})
    from scout_auto_os.engine.research.formula_league import random_baseline_stats

    random_stats = random_baseline_stats(forward_rows)
    missed = [
        r for r in forward_rows
        if str(r.get("label_big_winner", "")).lower() in ("true", "1")
        and int(r.get("rank", 99)) > 5
    ]
    missed.sort(key=lambda x: float(x.get("return_2h") or 0), reverse=True)
    traps = [r for r in forward_rows if str(r.get("label_trap", "")).lower() in ("true", "1")]
    traps.sort(key=lambda x: float(x.get("return_2h") or 0))

    payload = {
        "report_date": today,
        "generated_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
        "samples_yesterday": candidate_count_yesterday,
        "top20_count": candidate_count_yesterday,
        "a6_win_rate_2h": a6.get("win_rate_2h", 0),
        "a6_avg_return_2h": a6.get("avg_return_2h", 0),
        "random_win_rate_2h": random_stats.get("win_rate_2h", 0),
        "random_avg_return_2h": random_stats.get("avg_return_2h", 0),
        "formula_league_top": formula_league[:8],
        "feature_league_top": feature_league[:10],
        "missed_big_winners": missed[:10],
        "trap_patterns": traps[:10],
        "improvement_candidates": classify_improvements(formula_league, feature_league),
    }
    return payload
