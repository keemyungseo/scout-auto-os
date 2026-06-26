"""Rule Discovery Engine V1 report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_rule_discovery_report(
    meta: dict,
    v2_long: dict,
    v2_short: dict,
    top_long: list[dict],
    top_short: list[dict],
    recommendation: dict,
) -> str:
    lines = [
        "# Rule Discovery Engine V1",
        "",
        "Automatic operational entry rule search — **scan-time features only**.",
        "No lifecycle, clustering, pattern discovery, or price prediction.",
        "",
        "## Bottleneck context",
        "",
        f"- Direction Champion candidates: **{meta.get('total_champion', '?')}**",
        f"- Entry Rule V2 pass (full sample): **{meta.get('v2_pass_total', '?')}** ({meta.get('v2_coverage_pct', '?')}%)",
        "",
        "## Method",
        "",
        "- Candidate generation: threshold, ratio, diff, window-increase, scan rank, AND/OR/NOT",
        "- Thresholds mined on **train scans only**; metrics on **blind scans**",
        f"- Train/blind split: {meta.get('train_scan_count')} / {meta.get('blind_scan_count')} scans (temporal)",
        "- Primary objective: maximize **pass_per_day** subject to precision >= V2 - 2%",
        "- Secondary: maximize avg 2h return",
        "",
        "## Current Entry Rule V2 — blind validation",
        "",
        f"| Direction | Pass | Precision | Recall | Pass/day | Avg 2h | Avg 4h | Coverage |",
        f"|-----------|------|-----------|--------|----------|--------|--------|----------|",
        f"| Long | {v2_long.get('pass_count')} | {v2_long.get('precision')} | {v2_long.get('recall')} | "
        f"{v2_long.get('pass_per_day')} | {v2_long.get('avg_return_2h')} | {v2_long.get('avg_return_4h')} | "
        f"{v2_long.get('coverage_pct')}% |",
        f"| Short | {v2_short.get('pass_count')} | {v2_short.get('precision')} | {v2_short.get('recall')} | "
        f"{v2_short.get('pass_per_day')} | {v2_short.get('avg_return_2h')} | {v2_short.get('avg_return_4h')} | "
        f"{v2_short.get('coverage_pct')}% |",
        "",
    ]

    for direction, top in (("LONG", top_long), ("SHORT", top_short)):
        if not top:
            continue
        lines.extend(
            [
                f"## Top discovered rules — {direction} (blind)",
                "",
                "| Rank | Rule | Pass | Prec | Recall | Lift | Pass/day | Avg2h | Avg4h | Cov% | Floor |",
                "|------|------|------|------|--------|------|----------|-------|-------|------|-------|",
            ],
        )
        for r in top[:10]:
            lines.append(
                f"| {r.get('rank')} | `{r.get('rule_expr', '')[:55]}` | {r.get('pass_count')} | "
                f"{r.get('precision')} | {r.get('recall')} | {r.get('lift')} | {r.get('pass_per_day')} | "
                f"{r.get('avg_return_2h')} | {r.get('avg_return_4h')} | {r.get('coverage_pct')} | "
                f"{r.get('meets_precision_floor')} |",
            )
        lines.append("")

    lines.extend(
        [
            "## Recommendation",
            "",
            f"**Decision:** {recommendation.get('decision')}",
            "",
            f"- Reason: {recommendation.get('reason')}",
            f"- Mode: {recommendation.get('deployment_mode')}",
            f"- Rule: `{recommendation.get('recommended_rule_expr', 'n/a')}`",
            "",
            "## Interpretation",
            "",
            "- Precision floor is **V2 blind precision - 2%** — not a guarantee on future regimes.",
            "- Hybrid `(V2 OR candidate)` trades coverage for precision — validate before LIVE.",
            "- Reject/Needs-validation outcomes are valid — do not force rule promotion.",
            "",
            f"_Generated {datetime.now(KST).isoformat()}_",
        ],
    )

    try:
        from season2_scout_mission import evaluate_convergence, mission_summary_lines

        conv = evaluate_convergence(
            "interaction_mining",
            improves=["relative_ranking_between_candidates", "early_trend_detection"],
            sample_size=int(meta.get("total_champion", 0)),
            confidence="medium" if meta.get("blind_scan_count", 0) >= 20 else "hypothesis",
        )
        lines.append(
            f"**Convergence tier:** {conv['tier']} | {', '.join(conv['convergence_criteria_met']) or 'background'}",
        )
        lines.extend(mission_summary_lines())
    except ImportError:
        pass

    return "\n".join(lines)
