"""KEEP vs REJECT decision for frozen execution rule."""

from __future__ import annotations

from scout_auto_os.engine.research.execution_generalization.constants import MIN_FOLD_WIN_RATE, MIN_TRADES_DECISION


def decide_keep_or_reject(
    fold_rows: list[dict],
    regime_rows: list[dict],
    overall_rule_avg: float,
    overall_base_avg: float,
    total_trades: int,
) -> dict:
    reasons: list[str] = []
    n_folds = len(fold_rows)
    wins = sum(1 for f in fold_rows if f.get("rule_beats_baseline"))
    fold_win_rate = wins / n_folds if n_folds else 0.0

    if total_trades < MIN_TRADES_DECISION:
        reasons.append(f"total trades {total_trades} < {MIN_TRADES_DECISION}")

    if overall_rule_avg <= overall_base_avg:
        reasons.append(
            f"overall avg {overall_rule_avg} <= execution score baseline {overall_base_avg}",
        )

    if fold_win_rate < MIN_FOLD_WIN_RATE:
        reasons.append(f"fold win rate {fold_win_rate:.2f} < {MIN_FOLD_WIN_RATE}")

    negative_regimes = [
        r for r in regime_rows
        if float(r.get("rule_avg_return_2h", 0)) < 0 and int(r.get("rule_trade_count", 0)) >= 3
    ]
    if len(negative_regimes) >= 2:
        reasons.append(f"negative in {len(negative_regimes)} regimes: {[r['regime'] for r in negative_regimes]}")

    if not reasons and overall_rule_avg > overall_base_avg and fold_win_rate >= MIN_FOLD_WIN_RATE:
        decision = "KEEP"
        summary = "Rule generalizes across held-out folds with acceptable regime stability"
    elif overall_rule_avg > overall_base_avg and total_trades >= MIN_TRADES_DECISION:
        decision = "REJECT"
        summary = "Rule beats baseline in aggregate but fails fold/regime robustness gates"
    else:
        decision = "REJECT"
        summary = "Insufficient evidence of generalization — keep manual Execution Score"

    return {
        "decision": decision,
        "summary": summary,
        "fold_win_rate": round(fold_win_rate, 4),
        "folds_beat_baseline": wins,
        "fold_count": n_folds,
        "reasons": reasons,
    }
