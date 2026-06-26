"""Portfolio backtest report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from season2_scout_mission import mission_summary_lines

KST = timezone(timedelta(hours=9))


def build_portfolio_report(stats: dict, rules) -> str:
    lines = [
        "# Portfolio Engine V1 Report",
        "",
        f"Generated: {datetime.now(KST).isoformat()}",
        "",
        "## Pipeline",
        "Direction Champion → Entry Rule V2 → Entry Score → Diversify → Long3 / Short3",
        "",
        f"- Long rule: `{rules.long_meta.get('rule_expr', '')}`",
        f"- Short rule: `{rules.short_meta.get('rule_expr', '')}`",
        "",
        "## Backtest summary",
        f"- Scans (2h interval): **{stats.get('validation_scans')}**",
        f"- Total closed trades: **{stats.get('total_trades')}**",
        f"- Cumulative return (2h sum): **{stats.get('cumulative_return_2h')}%**",
        f"- Max drawdown: **{stats.get('max_drawdown')}%**",
        f"- Win rate (≥3%): **{stats.get('win_rate_pct')}%**",
        f"- Avg return 2h: **{stats.get('avg_return_2h')}%**",
        f"- Replacements: **{stats.get('replacement_count')}**",
        f"- Avg hold hours: **{stats.get('avg_hold_hours')}**",
        f"- Avg slot occupancy: **{stats.get('avg_slot_occupancy', 0) * 100:.1f}%**",
        "",
        "## Long / Short",
        f"- Long trades: {stats.get('long_trades')} | avg2h: {stats.get('long_avg_2h')}%",
        f"- Short trades: {stats.get('short_trades')} | avg2h: {stats.get('short_avg_2h')}%",
        f"- Avg PASS/scan long: {stats.get('pass_per_scan_long')} | short: {stats.get('pass_per_scan_short')}",
        "",
        "## LIVE slots",
        "- Long slots: **3**",
        "- Short slots: **3**",
        "- Replacement margin: **8%** score gap required",
        "",
    ]
    lines.extend(mission_summary_lines())
    return "\n".join(lines)
