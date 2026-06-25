"""Forward outcome metrics from 15m kline bars."""

from __future__ import annotations

BAR_MINUTES = 15


def _bar_idx(minutes: int) -> int:
    return max(0, minutes // BAR_MINUTES - 1)


def compute_forward_metrics(klines: list, entry_px: float | None = None) -> dict:
    if not klines or len(klines) < 2:
        return {}
    entry = entry_px or float(klines[0][1])
    if entry <= 0:
        return {}

    def px_at(bar_i: int) -> float:
        i = min(bar_i, len(klines) - 1)
        return float(klines[i][4])

    def ret_at(bar_i: int) -> float:
        return round((px_at(bar_i) - entry) / entry * 100, 4)

    idx_2h = _bar_idx(120)
    highs_2h = [float(k[2]) for k in klines[: idx_2h + 1]]
    lows_2h = [float(k[3]) for k in klines[: idx_2h + 1]]
    max_h = max(highs_2h) if highs_2h else entry
    min_l = min(lows_2h) if lows_2h else entry
    max_ret_2h = round((max_h - entry) / entry * 100, 4)
    min_ret_2h = round((min_l - entry) / entry * 100, 4)
    mdd = min_ret_2h

    rets_2h = [ret_at(i) for i in range(min(idx_2h + 1, len(klines)))]
    peak_i = rets_2h.index(max(rets_2h)) if rets_2h else 0
    time_to_peak = peak_i * BAR_MINUTES

    def time_to_target(pct: float) -> int | None:
        for i, r in enumerate(rets_2h):
            if r >= pct:
                return i * BAR_MINUTES
        return None

    r2h = ret_at(idx_2h)
    trap = max_ret_2h >= 4.0 and r2h <= -2.0
    big_winner = max_ret_2h >= 4.0 or r2h >= 4.0
    success = r2h >= 3.0

    return {
        "price_at_scan": entry,
        "return_30m": ret_at(_bar_idx(30)),
        "return_1h": ret_at(_bar_idx(60)),
        "return_2h": r2h,
        "return_4h": ret_at(_bar_idx(240)),
        "return_6h": ret_at(_bar_idx(360)),
        "max_return_2h": max_ret_2h,
        "min_return_2h": min_ret_2h,
        "max_drawdown_2h": mdd,
        "time_to_peak": time_to_peak,
        "time_to_3pct": time_to_target(3.0),
        "time_to_5pct": time_to_target(5.0),
        "label_trap": trap,
        "label_big_winner": big_winner,
        "label_success_2h": success,
        "downside_capture": min_ret_2h,
    }
