"""Generalization test reports."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_generalization_report(
    meta: dict,
    fold_rows: list[dict],
    decision: dict,
) -> str:
    lines = [
        "# Execution Rule Generalization Test V1",
        "",
        "**Frozen rule — no tuning, no new discovery.**",
        "",
        f"- Rule: `{meta.get('rule_expr')}`",
        f"- Direction: **{meta.get('direction')}**",
        f"- Scans: {meta.get('scan_count')} | Groups: {meta.get('group_count')}",
        "",
        "## Overall (all periods)",
        "",
        f"| Metric | Discovered Rule | Execution Score |",
        f"|--------|-----------------|-----------------|",
        f"| Avg 2h | {meta.get('overall_rule_avg')} | {meta.get('overall_base_avg')} |",
        f"| Win% | {meta.get('overall_rule_win')} | {meta.get('overall_base_win')} |",
        f"| Trades | {meta.get('overall_rule_trades')} | {meta.get('overall_base_trades')} |",
        f"| MDD | {meta.get('overall_rule_mdd')} | {meta.get('overall_base_mdd')} |",
        f"| Sharpe | {meta.get('overall_rule_sharpe')} | {meta.get('overall_base_sharpe')} |",
        f"| Return/day | {meta.get('overall_rule_rpd')} | {meta.get('overall_base_rpd')} |",
        "",
        "## Split stability",
        "",
        f"- Folds beating Execution Score: **{decision.get('folds_beat_baseline')}/{decision.get('fold_count')}** "
        f"({decision.get('fold_win_rate')})",
        f"- Monthly stability (std): {meta.get('monthly_stability')}",
        "",
        "### Fold summary",
        "",
        "| Split | Fold | Rule avg | Base avg | Rule win% | Beats base | Trades |",
        "|-------|------|----------|----------|-----------|------------|--------|",
    ]
    for f in fold_rows[:30]:
        lines.append(
            f"| {f.get('split_type')} | {f.get('fold_id')} | {f.get('rule_avg_return_2h')} | "
            f"{f.get('baseline_avg_return_2h')} | {f.get('rule_win_rate_pct')} | "
            f"{f.get('rule_beats_baseline')} | {f.get('rule_trade_count')} |",
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{decision.get('decision')}**",
            "",
            decision.get("summary", ""),
            "",
        ],
    )
    if decision.get("reasons"):
        lines.append("Rejection / caution reasons:")
        for r in decision["reasons"]:
            lines.append(f"- {r}")

    lines.extend(
        [
            "",
            "Probabilistic — small calendar coverage limits regime conclusions.",
            "",
            f"_Generated {datetime.now(KST).isoformat()}_",
        ],
    )
    return "\n".join(lines)


def build_regime_report(regime_rows: list[dict], vol_rows: list[dict]) -> str:
    lines = [
        "# Execution Rule — Regime Report",
        "",
        "## Trend regime",
        "",
        "| Regime | Trades | Rule avg 2h | Base avg | Win% | MDD | Survives |",
        "|--------|--------|-------------|----------|------|-----|----------|",
    ]
    for r in regime_rows:
        survives = float(r.get("rule_avg_return_2h", 0)) >= float(r.get("baseline_avg_return_2h", 0))
        lines.append(
            f"| {r.get('regime')} | {r.get('rule_trade_count')} | {r.get('rule_avg_return_2h')} | "
            f"{r.get('baseline_avg_return_2h')} | {r.get('rule_win_rate_pct')} | "
            f"{r.get('rule_mdd_pct')} | {survives} |",
        )
    lines.extend(
        [
            "",
            "## Volatility band",
            "",
            "| Band | Trades | Rule avg | Base avg | Beats base |",
            "|------|--------|----------|----------|------------|",
        ],
    )
    for r in vol_rows:
        lines.append(
            f"| {r.get('volatility_band')} | {r.get('rule_trade_count')} | "
            f"{r.get('rule_avg_return_2h')} | {r.get('baseline_avg_return_2h')} | "
            f"{r.get('rule_beats_baseline')} |",
        )
    lines.extend(
        [
            "",
            "Collapse = rule avg materially below baseline with n>=3.",
            "",
        ],
    )
    return "\n".join(lines)
