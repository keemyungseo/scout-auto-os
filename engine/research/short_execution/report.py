"""Short Execution Research V1 final report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_decision(
    exit_top5: list[dict],
    hold_summary: dict,
    hold_compare: list[dict],
    live_audit: dict,
    best_checkpoint: dict,
) -> dict:
    top1 = exit_top5[0] if exit_top5 else {}
    rec_hold = int(best_checkpoint.get("checkpoint_min") or hold_summary.get("median_peak_at_minutes") or 90)

    exit_rec = {
        "primary": top1.get("rule_id", "hold_2h"),
        "secondary": exit_top5[1].get("rule_id") if len(exit_top5) > 1 else "",
        "avoid": "hold_4h if avg_return lower than 2h on blind",
    }

    pre_live = [
        i["issue"] for i in live_audit.get("top10_issues", [])
        if i.get("severity") in ("CRITICAL", "HIGH") and i.get("fix_priority", 99) <= 2
    ]

    return {
        "q1_top5_exit_strategies": exit_top5[:5],
        "q2_recommended_holding_minutes": rec_hold,
        "q2_peak_median_minutes": hold_summary.get("median_peak_at_minutes"),
        "q3_exit_constitution_recommendation": exit_rec,
        "q4_live_issues_top10": live_audit.get("top10_issues", [])[:10],
        "q5_code_fix_priority": [
            i for i in live_audit.get("top10_issues", []) if i.get("fix_priority", 0) > 0
        ][:5],
        "q6_pre_live_blockers": pre_live or [
            "Short-side StateExitEngine direction bug",
            "Export live position_review.csv for MET/HEI validation",
            "90d blind calendar for execution generalization",
        ],
        "best_checkpoint": best_checkpoint,
        "hold_compare_winner": max(hold_compare, key=lambda x: float(x.get("avg_return_pct", 0)))
        if hold_compare else {},
    }


def build_report(
    meta: dict,
    exit_top5: list[dict],
    hold_summary: dict,
    hold_compare: list[dict],
    portfolio_summary: dict,
    live_audit: dict,
    decision: dict,
) -> str:
    lines = [
        "# Short Execution Research V1",
        "",
        "Execution / Exit / Portfolio — frozen Long + Short constitutions.",
        "",
        f"- Short blind picks: **{meta.get('short_pick_count')}** | Calendar: **{meta.get('calendar_days')}d**",
        f"- Frozen short label: **{meta.get('frozen_short_label')}**",
        "",
        "## 1. Best exit strategies (blind TOP5)",
        "",
        "| Rank | Rule | Avg return | Sharpe | PF | Avg hold |",
        "|------|------|------------|--------|-----|----------|",
    ]
    for r in exit_top5[:5]:
        lines.append(
            f"| {r.get('exit_rank')} | {r.get('rule_id')} | {r.get('avg_return_pct')}% | "
            f"{r.get('sharpe')} | {r.get('profit_factor')} | {r.get('avg_hold_minutes')}m |",
        )

    lines.extend([
        "",
        "## 2. Recommended holding time",
        "",
        f"- Median peak ROI at **{hold_summary.get('median_peak_at_minutes')}m**",
        f"- {hold_summary.get('pct_peak_before_1h')}% peaks before 1h | "
        f"{hold_summary.get('pct_peak_before_2h')}% before 2h",
        f"- Avg profit at 20%/50%/80% of hold: "
        f"{hold_summary.get('avg_profit_at_20pct_time')}% / "
        f"{hold_summary.get('avg_profit_at_50pct_time')}% / "
        f"{hold_summary.get('avg_profit_at_80pct_time')}%",
        "",
        "## 3. Exit constitution recommendation",
        "",
        f"- **Primary:** `{decision['q3_exit_constitution_recommendation']['primary']}`",
        f"- **Secondary:** `{decision['q3_exit_constitution_recommendation']['secondary']}`",
        "",
        "## 4. Early vs late vs dynamic",
        "",
        "| Strategy | Avg return | Sharpe | MDD |",
        "|----------|------------|--------|-----|",
    ])
    for h in hold_compare:
        lines.append(
            f"| {h.get('strategy')} | {h.get('avg_return_pct')}% | {h.get('sharpe')} | {h.get('mdd')} |",
        )

    lines.extend([
        "",
        "## 5. Portfolio Long3 + Short3",
        "",
        f"- Combined avg 2h: **{portfolio_summary.get('combined_avg_return_2h')}%** | "
        f"Sharpe **{portfolio_summary.get('combined_sharpe')}**",
        f"- Scan-level long/short corr: **{portfolio_summary.get('scan_level_long_short_corr')}**",
        f"- Simultaneous loss scans: **{portfolio_summary.get('simultaneous_loss_pct')}%**",
        "",
        "## 6. Live trading issues TOP10",
        "",
    ])
    for i in decision.get("q4_live_issues_top10", [])[:10]:
        lines.append(
            f"- **[{i.get('severity')}]** {i.get('issue')} — {i.get('detail', '')[:120]}",
        )

    lines.extend([
        "",
        "## Pre-LIVE blockers",
        "",
    ])
    for b in decision.get("q6_pre_live_blockers", []):
        lines.append(f"- {b}")

    lines.extend([
        "",
        "Probabilistic — 15d calendar; no price targets.",
        "",
        f"_Generated {datetime.now(KST).isoformat()}_",
    ])
    return "\n".join(lines)
