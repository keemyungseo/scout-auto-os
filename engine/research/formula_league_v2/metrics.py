"""Formula evaluation metrics across horizons."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from scout_auto_os.engine.research.formula_league_v2.constants import HORIZON_RETURN_KEY, HORIZONS, SUCCESS_RETURN_PCT


def return_12h_from_klines(klines: list, entry_px: float) -> float:
    if not klines or entry_px <= 0:
        return 0.0
    idx = min(47, len(klines) - 1)  # 12h @ 15m bars
    px = float(klines[idx][4])
    return round((px - entry_px) / entry_px * 100, 4)


def enrich_forward_metrics(metrics: dict, klines: list) -> dict:
    if not metrics:
        return {}
    entry = float(metrics.get("price_at_scan") or klines[0][1] if klines else 0)
    metrics = dict(metrics)
    metrics["return_12h"] = return_12h_from_klines(klines, entry)
    return metrics


def sharpe_like(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mu = statistics.mean(returns)
    sd = statistics.pstdev(returns)
    if sd < 1e-9:
        return 0.0
    return round(mu / sd * math.sqrt(len(returns)), 4)


def equity_max_drawdown(returns: list[float]) -> float:
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    for r in returns:
        cum += r
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return round(mdd, 4)


def return_stability(weekly_avgs: list[float]) -> float:
    if len(weekly_avgs) < 2:
        return 0.0
    return round(statistics.pstdev(weekly_avgs), 4)


def aggregate_formula_metrics(
    samples: list[dict],
    scans_used: set[str],
    total_scans: int,
) -> dict:
    if not samples:
        return {
            "trade_count": 0,
            "coverage_pct": 0.0,
            "hit_rate_top3_pct": 0.0,
            "stability": 0.0,
        }

    row: dict = {
        "trade_count": len(samples),
        "scan_count": len(scans_used),
        "coverage_pct": round(len(scans_used) / max(total_scans, 1) * 100, 2),
        "hit_rate_top3_pct": round(
            sum(1 for s in samples if s.get("in_top3_outcome")) / len(samples) * 100, 2,
        ),
    }

    for h in HORIZONS:
        key = HORIZON_RETURN_KEY[h]
        rets = [float(s.get(key, 0)) for s in samples]
        wins = sum(1 for r in rets if r >= SUCCESS_RETURN_PCT)
        row[f"avg_return_{h}"] = round(statistics.mean(rets), 4) if rets else 0.0
        row[f"median_return_{h}"] = round(statistics.median(rets), 4) if rets else 0.0
        row[f"win_rate_{h}"] = round(wins / len(rets) * 100, 2) if rets else 0.0

    rets_2h = [float(s.get("return_2h", 0)) for s in samples]
    days = len({s.get("scan_kst", "")[:10] for s in samples}) or 1
    row["return_per_day"] = round(sum(rets_2h) / days, 4)
    row["max_drawdown"] = equity_max_drawdown(rets_2h)
    row["sharpe_like"] = sharpe_like(rets_2h)
    row["return_per_coverage"] = round(
        row.get("avg_return_2h", 0) / max(row["coverage_pct"], 0.01) * 100, 4,
    )

    weekly: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        weekly[s.get("scan_kst", "")[:10]].append(float(s.get("return_2h", 0)))
    weekly_avgs = [statistics.mean(v) for v in weekly.values() if v]
    row["stability"] = return_stability(weekly_avgs)

    row["generalization_score"] = round(
        row.get("avg_return_2h", 0) * 0.3
        + row.get("hit_rate_top3_pct", 0) * 0.2
        + row.get("win_rate_2h", 0) * 0.2
        + row.get("stability", 0) * (-5.0)
        - abs(row.get("max_drawdown", 0)) * 0.1,
        4,
    )
    return row
