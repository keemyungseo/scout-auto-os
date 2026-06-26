"""Position size simulation — full vs value-score dynamic."""

from __future__ import annotations

import statistics

from scout_auto_os.engine.research.ranking_engine.metrics import equity_mdd, sharpe
from scout_auto_os.engine.research.trade_dna.value_estimator.value_score import compute_value_score, size_multiplier


def simulate_sizing(rows: list[dict], oof_preds: dict) -> dict:
    n = len(rows)
    full_returns = [float(r["y"]["expected_roi"]) for r in rows]
    dynamic_returns: list[float] = []
    dynamic_weights: list[float] = []
    skipped = 0

    norms = {
        "roi_p95": max(abs(r["y"]["expected_roi"]) for r in rows) or 1,
        "dd_p95": max(abs(r["y"]["expected_drawdown"]) for r in rows) or 1,
        "sharpe_p95": max(abs(r["y"]["expected_sharpe_contrib"]) for r in rows) or 1,
    }

    scores: list[float] = []
    for i, r in enumerate(rows):
        pred = {t: float(oof_preds[t][i]) for t in (
            "expected_roi", "expected_peak_roi", "expected_hold_time",
            "expected_drawdown", "expected_win_prob", "expected_sharpe_contrib",
        )}
        score = compute_value_score(pred, norms)
        scores.append(score)
        mult = size_multiplier(score)
        if mult <= 0:
            skipped += 1
            dynamic_returns.append(0.0)
            dynamic_weights.append(0.0)
        else:
            dynamic_returns.append(float(r["y"]["expected_roi"]) * mult)
            dynamic_weights.append(mult)

    def _stats(rets: list[float], taken_only: bool = False) -> dict:
        active = [r for r in rets if r != 0] if taken_only else rets
        if not active:
            return {"total_roi": 0, "avg_roi": 0, "sharpe": 0, "mdd": 0, "win_rate": 0, "trade_count": 0}
        wins = sum(1 for r in active if r >= 3.0)
        return {
            "total_roi": round(sum(rets), 4),
            "avg_roi": round(sum(active) / len(active), 4),
            "sharpe": sharpe(active),
            "mdd": equity_mdd(active),
            "win_rate": round(wins / len(active) * 100, 2),
            "trade_count": len(active),
        }

    full = _stats(full_returns)
    dyn = _stats(dynamic_returns, taken_only=True)
    full["strategy"] = "full_size"
    dyn["strategy"] = "dynamic_value_score"
    dyn["skipped_count"] = skipped
    dyn["avg_value_score"] = round(statistics.mean(scores), 2)

    improvement = {
        "total_roi_delta": round(dyn["total_roi"] - full["total_roi"], 4),
        "avg_roi_delta": round(dyn["avg_roi"] - full["avg_roi"], 4),
        "sharpe_delta": round(dyn["sharpe"] - full["sharpe"], 4),
        "mdd_improvement": round(full["mdd"] - dyn["mdd"], 4),
        "win_rate_delta": round(dyn["win_rate"] - full["win_rate"], 2),
    }
    return {"full": full, "dynamic": dyn, "improvement": improvement, "scores": scores}
