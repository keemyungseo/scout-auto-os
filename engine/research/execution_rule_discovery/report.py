"""Execution rule discovery report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_execution_rule_report(
    meta: dict,
    blind_comparison: list[dict],
    top_rules: list[dict],
    recommendation: dict,
) -> str:
    lines = [
        "# Execution Rule Discovery V1",
        "",
        "Data-driven execution rules — **execution layer only**.",
        "Search, Entry Rule V2, Entry Score unchanged.",
        "",
        "## Method",
        "",
        "- Features: search-time (entry score, margins, top5 rank) + **first observation bar only**",
        "- Rule mining: threshold, ratio, diff, top5 rank, AND/OR/NOT",
        f"- Train groups: {meta.get('train_groups')} | Blind groups: {meta.get('blind_groups')}",
        "- Selection: blind avg 2h return vs Execution Score Top2 baseline",
        "",
        "## Blind comparison",
        "",
        "| Strategy | Direction | Trades | Avg 2h | Win% | Lift vs Exec Score | Lift vs Entry Top2 |",
        "|----------|-----------|--------|--------|------|--------------------|--------------------|",
    ]

    for r in blind_comparison:
        lines.append(
            f"| {r['strategy']} | {r.get('direction', 'combined')} | {r['trade_count']} | "
            f"{r['avg_return_2h']} | {r['win_rate_pct']} | {r.get('lift_vs_exec_score_pct', '')} | "
            f"{r.get('lift_vs_entry_top2_pct', '')} |",
        )

    lines.extend(["", "## Top discovered rules (blind)", ""])
    for r in top_rules[:10]:
        lines.append(
            f"- **{r.get('direction')}** `{r.get('rule_expr', '')[:70]}` — "
            f"blind avg={r.get('avg_return_2h')} lift={r.get('lift_vs_exec_score_pct')}%",
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"**Decision:** {recommendation.get('decision')}",
            "",
            f"- {recommendation.get('reason')}",
            "",
            f"Baseline Execution Score blind avg: **{recommendation.get('baseline_exec_avg')}**",
            f"Best rule blind avg: **{recommendation.get('best_rule_avg')}**",
            "",
            f"_Generated {datetime.now(KST).isoformat()}_",
        ],
    )

    try:
        from season2_scout_mission import evaluate_convergence, mission_summary_lines

        conv = evaluate_convergence(
            "functional_role",
            improves=["relative_ranking_between_candidates"],
            sample_size=int(meta.get("blind_trades", 0)),
            confidence="medium" if meta.get("blind_groups", 0) >= 10 else "hypothesis",
        )
        lines.append(
            f"**Convergence tier:** {conv['tier']} | {', '.join(conv['convergence_criteria_met']) or 'background'}",
        )
        lines.extend(mission_summary_lines())
    except ImportError:
        pass

    return "\n".join(lines)
