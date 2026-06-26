"""Per-rule performance profiling across regimes and time."""

from __future__ import annotations

import statistics
from collections import defaultdict

from scout_auto_os.engine.research.execution_generalization.metrics import equity_max_drawdown, sharpe_approx
from scout_auto_os.engine.research.execution_rule_discovery.baselines import pick_top2_execution_score
from scout_auto_os.engine.research.execution_rule_discovery.constants import TOP2_SIZE
from scout_auto_os.engine.research.execution_rule_discovery.generator import pick_top2_by_rule
from scout_auto_os.engine.research.rule_portfolio.collectors import PortfolioRule
from scout_auto_os.engine.research.rule_portfolio.constants import BASELINE_RULE_ID, MIN_REGIME_TRADES, SUCCESS_RETURN_PCT


def _hour_bucket(scan_time: str) -> str:
    try:
        h = int(scan_time[11:13])
        if h < 8:
            return "asia"
        if h < 16:
            return "europe"
        return "us"
    except (ValueError, IndexError):
        return "unknown"


def _pick_top2(group: list[dict], pr: PortfolioRule) -> list[dict]:
    if pr.rule is None:
        return pick_top2_execution_score(group)
    return pick_top2_by_rule(group, pr.rule)


def profile_rule(
    pr: PortfolioRule,
    groups: list[list[dict]],
) -> dict:
    dir_groups = [g for g in groups if g[0].get("direction") == pr.direction and len(g) >= TOP2_SIZE]

    all_returns: list[float] = []
    pass_flags: list[bool] = []
    regime_returns: dict[str, list[float]] = defaultdict(list)
    vol_returns: dict[str, list[float]] = defaultdict(list)
    hour_returns: dict[str, list[float]] = defaultdict(list)
    week_returns: dict[str, list[float]] = defaultdict(list)
    activation_by_regime: dict[str, list[bool]] = defaultdict(list)

    for g in dir_groups:
        picks = _pick_top2(g, pr)
        regime = g[0].get("regime", "unknown")
        vol = g[0].get("volatility_band", "unknown")
        hour = _hour_bucket(g[0]["scan_time_kst"])
        week = g[0]["scan_time_kst"][:10]

        for p in picks:
            ret = float(p["return_2h"])
            all_returns.append(ret)
            passed = True
            if pr.rule is not None:
                passed = pr.rule.evaluate(p["features"], p.get("ctx"))
            pass_flags.append(passed)
            regime_returns[regime].append(ret)
            vol_returns[vol].append(ret)
            hour_returns[hour].append(ret)
            week_returns[week].append(ret)
            activation_by_regime[regime].append(passed)

    days = len({g[0]["scan_time_kst"][:10] for g in dir_groups}) or 1
    n_trades = len(all_returns)
    n_pass = sum(1 for x in pass_flags if x)
    wins = sum(1 for r in all_returns if r >= SUCCESS_RETURN_PCT)

    def _best_key(bucket: dict[str, list[float]]) -> str:
        scored = {
            k: statistics.mean(v)
            for k, v in bucket.items()
            if len(v) >= MIN_REGIME_TRADES
        }
        return max(scored, key=scored.get) if scored else "unknown"

    def _worst_key(bucket: dict[str, list[float]]) -> str:
        scored = {
            k: statistics.mean(v)
            for k, v in bucket.items()
            if len(v) >= MIN_REGIME_TRADES
        }
        return min(scored, key=scored.get) if scored else "unknown"

    regime_avg = {k: round(statistics.mean(v), 4) for k, v in regime_returns.items() if v}
    vol_avg = {k: round(statistics.mean(v), 4) for k, v in vol_returns.items() if v}
    hour_avg = {k: round(statistics.mean(v), 4) for k, v in hour_returns.items() if v}

    activation_rates = {
        k: round(sum(v) / len(v) * 100, 2) if v else 0.0
        for k, v in activation_by_regime.items()
    }

    return {
        "rule_id": pr.rule_id,
        "rule_expr": pr.rule_expr,
        "direction": pr.direction,
        "source": pr.source,
        "status_tags": "|".join(pr.status_tags),
        "discovery_decision": pr.discovery_decision,
        "trade_count": n_trades,
        "scan_groups": len(dir_groups),
        "coverage_pct": round(len(dir_groups) / max(len(groups), 1) * 100, 2),
        "precision_pct": round(n_pass / n_trades * 100, 2) if n_trades else 0.0,
        "avg_return_2h": round(statistics.mean(all_returns), 4) if all_returns else 0.0,
        "win_rate_pct": round(wins / n_trades * 100, 2) if n_trades else 0.0,
        "max_drawdown": equity_max_drawdown(all_returns),
        "return_per_day": round(sum(all_returns) / days, 4) if all_returns else 0.0,
        "sharpe": sharpe_approx(all_returns),
        "best_regime": _best_key(regime_returns),
        "worst_regime": _worst_key(regime_returns),
        "volatility_preference": _best_key(vol_returns),
        "trend_preference": _best_key(regime_returns),
        "time_preference": _best_key(hour_returns),
        "regime_avg_json": regime_avg,
        "vol_avg_json": vol_avg,
        "hour_avg_json": hour_avg,
        "activation_by_regime_json": activation_rates,
        "blind_avg": pr.blind_avg,
        "train_avg": pr.train_avg,
    }
