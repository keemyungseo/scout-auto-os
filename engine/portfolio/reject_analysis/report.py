"""Coverage report for reject analysis."""

from __future__ import annotations

from collections import Counter

from season2_scout_mission import mission_summary_lines


def build_coverage_report(
    meta: dict,
    feature_fail_pct: dict[str, float],
    portfolio_fail: Counter,
    near_pass_rows: list[dict],
) -> str:
    funnel = meta.get("funnel", {})
    bottleneck = meta.get("bottleneck", {})
    almost = sum(1 for r in near_pass_rows if r.get("almost_pass"))

    lines = [
        "# Coverage Report — Reject Analysis V1",
        "",
        "## Funnel (Direction Champion → Rule → Portfolio → Fill)",
        "",
        "| Stage | Count | Survival % |",
        "|-------|-------|------------|",
        f"| Direction Champion candidates | {funnel.get('direction_champion')} | 100% |",
        f"| Entry Rule V2 PASS | {funnel.get('rule_pass')} | **{funnel.get('rule_pass_rate_pct')}%** |",
        f"| Portfolio PASS / Replacement | {funnel.get('portfolio_pass')} | **{funnel.get('portfolio_pass_rate_pct')}%** |",
        "",
        f"- Scans analyzed: {meta.get('scans')} (2h interval)",
        f"- Near-pass (Almost Pass) candidates: **{almost}**",
        "",
        "## Rule Reject — condition failure share",
        "",
        "| Feature group | Fail % of rule failures |",
        "|---------------|-------------------------|",
    ]
    for label, pct in sorted(feature_fail_pct.items(), key=lambda x: -x[1]):
        lines.append(f"| {label} | **{pct}%** |")

    lines.extend([
        "",
        "## Reject tier (rule stage)",
        "- **Near Pass**: ≤1 failed condition, gap ≤10%",
        "- **Medium Reject**: ≤2 failed, gap ≤40%",
        "- **Impossible Reject**: 3+ failed or large gap",
        "",
        "## Portfolio reject breakdown",
        "",
        "| Reason | Count |",
        "|--------|-------|",
    ])
    for reason, cnt in portfolio_fail.most_common():
        lines.append(f"| {reason} | {cnt} |")

    lines.extend([
        "",
        "## Bottleneck (PASS increase potential)",
        "",
        f"- **Primary stage:** `{bottleneck.get('stage')}`",
        f"- **Candidates lost at stage:** {bottleneck.get('candidates_lost')} ({bottleneck.get('severity_pct')}%)",
    ])
    if bottleneck.get("top_feature_blocker"):
        lines.append(
            f"- **Top rule blocker:** {bottleneck.get('top_feature_blocker')} "
            f"({bottleneck.get('top_feature_block_pct')}% of condition failures)"
        )
    if bottleneck.get("top_blocker"):
        lines.append(f"- **Top portfolio blocker:** {bottleneck.get('top_blocker')}")
    lines.append(f"- **Recommendation:** {bottleneck.get('recommendation')}")
    lines.append(f"- **Priority:** {bottleneck.get('priority')}")
    lines.extend([
        "",
        "## Notes",
        "- Rules and thresholds **not modified** — coverage analysis only.",
        "- No prediction / ML.",
        "",
    ])
    lines.extend(mission_summary_lines())
    return "\n".join(lines)
