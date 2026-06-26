"""Cadence optimizer report and recommendations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def _noise_notes(interval_min: int, row: dict) -> str:
    turnover = float(row.get("turnover_rate", 0))
    reselect = int(row.get("reselect_count", 0))
    ret = float(row.get("combined_avg_return", 0))
    if interval_min <= 5:
        return f"High churn (turnover/scan={turnover:.2f}), reselect={reselect}; avg return {ret}% — likely noise-dominated"
    if interval_min <= 15:
        return f"Moderate turnover={turnover:.2f}; balance of refresh vs stability"
    if interval_min >= 60:
        return f"Low turnover={turnover:.2f}; fewer opportunities, longer holds"
    return f"turnover/scan={turnover:.2f}, avg return={ret}%"


def recommend_cadence(summaries: list[dict]) -> dict:
    if not summaries:
        return {}

    by_ret_day = max(summaries, key=lambda x: float(x.get("return_per_day", 0)))
    by_stability = max(
        summaries,
        key=lambda x: (
            float(x.get("return_per_turnover", 0)),
            -float(x.get("turnover_rate", 999)),
        ),
    )
    by_efficiency = max(summaries, key=lambda x: float(x.get("return_per_turnover", 0)))

    # LIVE heuristic: primary = best efficiency with min trades; refresh faster if <=15m stable
    candidates = [s for s in summaries if int(s["interval_min"]) >= 5 and s.get("trade_count", 0) >= 3]
    primary = by_efficiency if by_efficiency in candidates else by_stability
    refresh = min(summaries, key=lambda x: int(x["interval_min"]))
    rebalance = primary

    discard = sorted(summaries, key=lambda x: (float(x.get("return_per_turnover", 0)), float(x.get("combined_avg_return", 0))))[:2]

    return {
        "best_return_interval_min": by_ret_day["interval_min"],
        "most_stable_interval_min": by_stability["interval_min"],
        "best_efficiency_interval_min": by_efficiency["interval_min"],
        "primary_scan_interval_min": int(primary["interval_min"]),
        "candidate_refresh_interval_min": int(refresh["interval_min"]),
        "portfolio_rebalance_interval_min": int(rebalance["interval_min"]),
        "discard_intervals": [int(d["interval_min"]) for d in discard],
        "primary_reason": (
            f"Interval {primary['interval_min']}m: return/day={primary.get('return_per_day')} "
            f"return/turnover={primary.get('return_per_turnover')} turnover/scan={primary.get('turnover_rate')}"
        ),
    }


def build_cadence_report(
    meta: dict,
    summaries: list[dict],
    rec: dict,
) -> str:
    lines = [
        "# Scan Cadence Optimizer V1",
        "",
        "LIVE scan interval comparison — **no new features, rules, or prediction models**.",
        "",
        "## Data constraint",
        "",
        f"- Base universe snapshots: **every {meta.get('base_snapshot_minutes')}m** ({meta.get('base_scan_count')} scans)",
        "- Sub-120m cadences replay portfolio on synthetic ticks using **latest available snapshot** (causal, no lookahead)",
        "- Forward returns: **evaluation only**",
        "",
        "## Interval performance",
        "",
        "| Interval | Scans | Trades | Pass/day | Long avg | Short avg | Combined | Win% | MDD | Replace | Hold(min) | Occupancy | Turnover | Ret/trade | Ret/day | Ret/turnover |",
        "|----------|-------|--------|----------|----------|-----------|----------|------|-----|---------|-----------|-----------|----------|-----------|---------|--------------|",
    ]

    for r in sorted(summaries, key=lambda x: int(x["interval_min"])):
        lines.append(
            f"| {r['interval_min']}m | {r['scan_count']} | {r['trade_count']} | {r['pass_per_day']} | "
            f"{r['long_avg_return']} | {r['short_avg_return']} | {r['combined_avg_return']} | "
            f"{r['win_rate_pct']} | {r['mdd_pct']} | {r['replacement_count']} | {r['avg_hold_minutes']} | "
            f"{r['avg_slot_occupancy']} | {r['turnover_rate']} | {r['return_per_trade']} | "
            f"{r['return_per_day']} | {r['return_per_turnover']} |",
        )

    lines.extend(
        [
            "",
            "## Rankings",
            "",
            f"1. **Highest return/day:** {rec.get('best_return_interval_min')}m",
            f"2. **Most stable (return/turnover, low churn):** {rec.get('most_stable_interval_min')}m",
            f"3. **Best turnover efficiency:** {rec.get('best_efficiency_interval_min')}m",
            "",
            "## Noise analysis",
            "",
        ],
    )
    for r in sorted(summaries, key=lambda x: int(x["interval_min"])):
        lines.append(f"- **{r['interval_min']}m:** {_noise_notes(int(r['interval_min']), r)}")

    lines.extend(
        [
            "",
            "## LIVE recommendation",
            "",
            "```",
            f"primary_scan_interval = {rec.get('primary_scan_interval_min')} minutes",
            f"candidate_refresh_interval = {rec.get('candidate_refresh_interval_min')} minutes",
            f"portfolio_rebalance_interval = {rec.get('portfolio_rebalance_interval_min')} minutes",
            "```",
            "",
            f"**Reason:** {rec.get('primary_reason', '')}",
            "",
            "## Intervals to deprioritize",
            "",
        ],
    )
    for im in rec.get("discard_intervals", []):
        row = next((s for s in summaries if int(s["interval_min"]) == im), {})
        lines.append(f"- **{im}m:** return/turnover={row.get('return_per_turnover')} — { _noise_notes(im, row)}")

    lines.extend(
        [
            "",
            "Probabilistic — validate on new regimes before LIVE cadence change.",
            "",
            f"_Generated {datetime.now(KST).isoformat()}_",
        ],
    )

    try:
        from season2_scout_mission import evaluate_convergence, mission_summary_lines

        conv = evaluate_convergence(
            "supply_label",
            improves=["trend_persistence_estimation", "relative_ranking_between_candidates"],
            sample_size=int(meta.get("base_scan_count", 0)),
            confidence="medium" if len(summaries) >= 5 else "hypothesis",
        )
        lines.append(
            f"**Convergence tier:** {conv['tier']} | {', '.join(conv['convergence_criteria_met']) or 'background'}",
        )
        lines.extend(mission_summary_lines())
    except ImportError:
        pass

    return "\n".join(lines)
