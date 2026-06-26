"""Rule metadata for runtime regime routing."""

from __future__ import annotations

from scout_auto_os.engine.research.rule_portfolio.constants import MIN_CONFIDENCE_TRADES, MIN_REGIME_TRADES


def _confidence_tier(trade_count: int, avg_return: float) -> str:
    if trade_count >= MIN_CONFIDENCE_TRADES and avg_return > 0:
        return "high"
    if trade_count >= MIN_REGIME_TRADES:
        return "medium"
    if trade_count > 0:
        return "hypothesis"
    return "unknown"


def _activation_conditions(rule_expr: str) -> str:
    if "execution_score" in rule_expr and "(" not in rule_expr:
        return "universal_execution_score"
    return rule_expr.replace("  ", " ").strip()


def _preferred_avoid(profile: dict) -> tuple[str, str]:
    regime_avg = profile.get("regime_avg_json") or {}
    preferred = profile.get("best_regime", "unknown")
    avoid = profile.get("worst_regime", "unknown")
    negatives = [k for k, v in regime_avg.items() if v < 0 and k != preferred]
    if negatives:
        avoid = min(negatives, key=lambda k: regime_avg[k])
    return preferred, avoid


def build_metadata(profile: dict, cluster_row: dict) -> dict:
    preferred, avoid = _preferred_avoid(profile)
    trades = int(profile.get("trade_count", 0))
    avg = float(profile.get("avg_return_2h", 0))

    holding = "2h_fixed"
    if "rank_obs_return_top5" in profile.get("rule_expr", ""):
        holding = "2h_breakout_follow"
    elif "obs_low_pct" in profile.get("rule_expr", ""):
        holding = "2h_reversal_bounce"

    return {
        "rule_id": profile["rule_id"],
        "rule_expr": profile["rule_expr"],
        "direction": profile["direction"],
        "cluster_id": cluster_row.get("cluster_id", "mixed"),
        "activation_conditions": _activation_conditions(profile.get("rule_expr", "")),
        "preferred_regime": preferred,
        "avoid_regime": avoid,
        "volatility_preference": profile.get("volatility_preference", "unknown"),
        "trend_preference": profile.get("trend_preference", "unknown"),
        "time_preference": profile.get("time_preference", "unknown"),
        "expected_holding_profile": holding,
        "confidence": _confidence_tier(trades, avg),
        "sample_size": trades,
        "avg_return_2h": avg,
        "precision_pct": profile.get("precision_pct", 0),
        "status_tags": profile.get("status_tags", ""),
        "discovery_decision": profile.get("discovery_decision", ""),
    }
