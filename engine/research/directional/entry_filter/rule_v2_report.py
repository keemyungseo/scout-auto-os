"""LIVE Entry Rule V2 markdown report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from season2_scout_mission import evaluate_convergence, mission_summary_lines

KST = timezone(timedelta(hours=9))


def build_live_entry_rule_v2_report(
    long_best: dict,
    short_best: dict,
    long_v1: dict,
    short_v1: dict,
    meta: dict,
) -> str:
    lines = [
        "# LIVE Entry Rule V2",
        "",
        "Optimized from V1 thresholds — **more recall, precision maintained**.",
        "No new features. No ML. No prediction.",
        "",
        f"Generated: {datetime.now(KST).isoformat()}",
        f"Signals: long={meta.get('long_signals')} short={meta.get('short_signals')}",
        f"Rule trees tested per direction: {meta.get('trees_per_direction')}",
        "",
        "## Long",
        "",
        f"| Metric | V1 (ABCD) | **V2 (selected)** |",
        f"|--------|-----------|-------------------|",
        f"| Rule | `{long_v1.get('rule_expr', '')}` | `{long_best.get('rule_expr', '')}` |",
        f"| Pass count | {long_v1.get('pass_count')} | **{long_best.get('pass_count')}** |",
        f"| Precision | {long_v1.get('precision')} | **{long_best.get('precision')}** |",
        f"| Recall | {long_v1.get('recall')} | **{long_best.get('recall')}** |",
        f"| F1 | {long_v1.get('f1')} | **{long_best.get('f1')}** |",
        f"| Pass/day | {long_v1.get('pass_per_day')} | **{long_best.get('pass_per_day')}** |",
        f"| 2h Avg | {long_v1.get('avg_return_2h')}% | **{long_best.get('avg_return_2h')}%** |",
        f"| 4h Avg | {long_v1.get('avg_return_4h')}% | **{long_best.get('avg_return_4h')}%** |",
        "",
        "## Short",
        "",
        f"| Metric | V1 (ABCD) | **V2 (selected)** |",
        f"|--------|-----------|-------------------|",
        f"| Rule | `{short_v1.get('rule_expr', '')}` | `{short_best.get('rule_expr', '')}` |",
        f"| Pass count | {short_v1.get('pass_count')} | **{short_best.get('pass_count')}** |",
        f"| Precision | {short_v1.get('precision')} | **{short_best.get('precision')}** |",
        f"| Recall | {short_v1.get('recall')} | **{short_best.get('recall')}** |",
        f"| F1 | {short_v1.get('f1')} | **{short_best.get('f1')}** |",
        f"| Pass/day | {short_v1.get('pass_per_day')} | **{short_best.get('pass_per_day')}** |",
        f"| 2h Avg | {short_v1.get('avg_return_2h')}% | **{short_best.get('avg_return_2h')}%** |",
        f"| 4h Avg | {short_v1.get('avg_return_4h')}% | **{short_best.get('avg_return_4h')}%** |",
        "",
        "## Selection criteria",
        "- Precision ≥ 90% of V1 full-AND baseline",
        "- Maximize recall + pass frequency (target ~3 passes/day)",
        "- LIVE score = 0.45×recall + 0.30×freq + 0.25×precision_ratio",
        "",
    ]
    conv = evaluate_convergence(
        "situation_archetype",
        improves=["real_vs_fake_trend_discrimination", "relative_ranking_between_candidates"],
        sample_size=int(meta.get("long_signals", 0)) + int(meta.get("short_signals", 0)),
        confidence="medium",
    )
    lines.append(f"**Convergence tier:** {conv['tier']}")
    lines.extend(mission_summary_lines())
    return "\n".join(lines)
