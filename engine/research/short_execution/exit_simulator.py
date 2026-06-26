"""Research 2 — Short exit rule simulation (realtime-computable only)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from scout_auto_os.engine.research.ranking_engine.metrics import equity_mdd, sharpe, sortino
from scout_auto_os.engine.research.short_execution.lifecycle import (
    short_close_roi,
    short_mae_to_bar,
    short_mfe_to_bar,
)
from scout_auto_os.engine.research.short_execution.constants import BAR_MINUTES, MAX_FORWARD_MINUTES


@dataclass
class ExitResult:
    rule_id: str
    return_pct: float
    hold_minutes: int
    exit_reason: str
    mfe_pct: float
    mae_pct: float
    peak_capture_pct: float


def _bar_range_pct(bar: list) -> float:
    o = float(bar[1]) or 1.0
    return (float(bar[2]) - float(bar[3])) / o * 100


def _vol_ratio(klines: list, i: int, lookback: int = 8) -> float:
    if i < 1:
        return 1.0
    start = max(0, i - lookback)
    cur = float(klines[i][5])
    hist = [float(klines[j][5]) for j in range(start, i)]
    avg = statistics.mean(hist) if hist else cur or 1.0
    return cur / avg if avg else 1.0


def _momentum_short(klines: list, i: int, bars: int = 3) -> float:
    if i < bars:
        return 0.0
    s = 0.0
    for j in range(i - bars + 1, i + 1):
        o, c = float(klines[j][1]), float(klines[j][4])
        s += (c - o) / o * 100 if o else 0
    return s


def _recent_high(klines: list, i: int, lookback: int) -> float:
    start = max(0, i - lookback + 1)
    return max(float(klines[j][2]) for j in range(start, i + 1))


def simulate_short_exit(klines: list, rule: dict) -> ExitResult | None:
    if not klines or len(klines) < 2:
        return None
    entry = float(klines[0][1])
    if entry <= 0:
        return None

    rule_id = rule["rule_id"]
    rtype = rule["type"]
    max_bar = min(len(klines) - 1, MAX_FORWARD_MINUTES // BAR_MINUTES)
    end_i = max_bar
    if rtype == "time":
        end_i = min(max_bar, max(0, int(rule["minutes"]) // BAR_MINUTES))

    best_fav = 0.0
    peak_roi = 0.0

    for i in range(end_i + 1):
        bar = klines[i]
        low, high, close = float(bar[3]), float(bar[2]), float(bar[4])
        hold_min = i * BAR_MINUTES
        fav = short_mfe_to_bar(klines, entry, i)
        adv = short_mae_to_bar(klines, entry, i)
        roi = short_close_roi(entry, close)
        best_fav = max(best_fav, fav)
        peak_roi = max(peak_roi, roi)

        if rtype == "tp_sl":
            tp, sl = float(rule["tp"]), float(rule["sl"])
            if (entry - low) / entry * 100 >= tp:
                return _result(rule_id, tp, hold_min, "tp", fav, adv, tp, best_fav)
            if (high - entry) / entry * 100 >= sl:
                return _result(rule_id, -sl, hold_min, "sl", fav, adv, 0, best_fav)
            continue

        if rtype == "tp" and (entry - low) / entry * 100 >= float(rule["tp"]):
            tp = float(rule["tp"])
            return _result(rule_id, tp, hold_min, "tp", fav, adv, tp, best_fav)

        if rtype == "sl" and (high - entry) / entry * 100 >= float(rule["sl"]):
            sl = float(rule["sl"])
            return _result(rule_id, -sl, hold_min, "sl", fav, adv, 0, best_fav)

        if rtype == "trail" and hold_min >= int(rule.get("min_hold", 15)):
            activate = float(rule.get("activate", 5))
            gap = float(rule["gap"])
            if best_fav >= activate:
                trough_px = entry * (1 - best_fav / 100)
                trail_px = trough_px * (1 + gap / 100)
                if high >= trail_px:
                    exit_px = trail_px
                    ret = short_close_roi(entry, exit_px)
                    return _result(rule_id, ret, hold_min, "trail", fav, adv, ret, best_fav)

        if rtype == "roi_trail" and peak_roi >= float(rule.get("min_profit", 3)):
            drop = float(rule["drop"])
            if peak_roi - roi >= drop:
                return _result(rule_id, roi, hold_min, "roi_trail", fav, adv, roi, best_fav)

        if rtype == "high_break" and i >= int(rule.get("lookback", 4)):
            pct = float(rule["pct"])
            rh = _recent_high(klines, i - 1, int(rule.get("lookback", 4)))
            if high >= rh * (1 + pct / 100):
                return _result(rule_id, roi, hold_min, "high_break", fav, adv, roi, best_fav)

        if rtype == "momentum" and hold_min >= 30:
            mom = _momentum_short(klines, i, int(rule.get("bars", 3)))
            if mom > 0.5:
                return _result(rule_id, roi, hold_min, "momentum_weak", fav, adv, roi, best_fav)

        if rtype == "volume" and hold_min >= 30:
            if _vol_ratio(klines, i) < float(rule.get("ratio", 0.7)):
                return _result(rule_id, roi, hold_min, "volume_weak", fav, adv, roi, best_fav)

        if rtype == "peak" and hold_min >= int(rule.get("min_hold", 30)):
            drop = float(rule["drop"])
            if peak_roi >= 3 and peak_roi - roi >= drop:
                return _result(rule_id, roi, hold_min, "peak_drop", fav, adv, roi, best_fav)

        if rtype == "atr_trail" and hold_min >= 15:
            atr = _bar_range_pct(bar)
            gap = max(float(rule.get("atr_mult", 1.5)) * atr, float(rule.get("min_gap", 2)))
            if best_fav >= float(rule.get("activate", 4)):
                trough_px = entry * (1 - best_fav / 100)
                if high >= trough_px * (1 + gap / 100):
                    exit_px = trough_px * (1 + gap / 100)
                    ret = short_close_roi(entry, exit_px)
                    return _result(rule_id, ret, hold_min, "atr_trail", fav, adv, ret, best_fav)

        if rtype == "state_proxy" and hold_min >= int(rule.get("min_hold", 30)):
            mom = _momentum_short(klines, i, 6)
            vol_ok = _vol_ratio(klines, i) >= 0.75
            if mom > 0.8 or (not vol_ok and roi > 5):
                return _result(rule_id, roi, hold_min, "state_proxy_exit", fav, adv, roi, best_fav)

    close = float(klines[end_i][4])
    final = short_close_roi(entry, close)
    fav = short_mfe_to_bar(klines, entry, end_i)
    adv = short_mae_to_bar(klines, entry, end_i)
    reason = "time_expiry" if rtype == "time" else "eod"
    return _result(rule_id, final, end_i * BAR_MINUTES, reason, fav, adv, final, best_fav)


def _result(rule_id, ret, hold_min, reason, mfe, mae, capture, peak) -> ExitResult:
    pc = round(capture / peak * 100, 2) if peak > 1e-6 else 0.0
    return ExitResult(rule_id, round(ret, 4), hold_min, reason, mfe, mae, pc)


def exit_rule_catalog() -> list[dict]:
    return [
        {"rule_id": "hold_2h", "type": "time", "minutes": 120, "category": "baseline"},
        {"rule_id": "hold_4h", "type": "time", "minutes": 240, "category": "baseline"},
        {"rule_id": "hold_1h", "type": "time", "minutes": 60, "category": "baseline"},
        {"rule_id": "hold_90m", "type": "time", "minutes": 90, "category": "time"},
        {"rule_id": "tp10", "type": "tp", "tp": 10, "category": "fixed_tp"},
        {"rule_id": "tp15", "type": "tp", "tp": 15, "category": "fixed_tp"},
        {"rule_id": "sl8", "type": "sl", "sl": 8, "category": "fixed_sl"},
        {"rule_id": "sl10", "type": "sl", "sl": 10, "category": "fixed_sl"},
        {"rule_id": "tp10_sl8", "type": "tp_sl", "tp": 10, "sl": 8, "category": "combo"},
        {"rule_id": "tp12_sl6", "type": "tp_sl", "tp": 12, "sl": 6, "category": "combo"},
        {"rule_id": "trail_gap3", "type": "trail", "gap": 3, "activate": 5, "category": "trailing"},
        {"rule_id": "trail_gap5", "type": "trail", "gap": 5, "activate": 6, "category": "trailing"},
        {"rule_id": "roi_trail5", "type": "roi_trail", "drop": 5, "min_profit": 3, "category": "roi_trail"},
        {"rule_id": "roi_trail3", "type": "roi_trail", "drop": 3, "min_profit": 4, "category": "roi_trail"},
        {"rule_id": "high_break2", "type": "high_break", "pct": 2, "lookback": 4, "category": "structure"},
        {"rule_id": "momentum_weak", "type": "momentum", "bars": 3, "category": "momentum"},
        {"rule_id": "volume_weak", "type": "volume", "ratio": 0.7, "category": "volume"},
        {"rule_id": "peak_drop3", "type": "peak", "drop": 3, "min_hold": 30, "category": "peak"},
        {"rule_id": "atr_trail15", "type": "atr_trail", "atr_mult": 1.5, "min_gap": 2, "activate": 4, "category": "atr"},
        {"rule_id": "state_proxy", "type": "state_proxy", "min_hold": 30, "category": "state"},
    ]


def profit_factor(returns: list[float]) -> float:
    gains = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    if losses < 1e-9:
        return round(gains, 4) if gains > 0 else 0.0
    return round(gains / losses, 4)


def blind_exit_comparison(picks: list[dict], rules: list[dict] | None = None) -> list[dict]:
    rules = rules or exit_rule_catalog()
    rows: list[dict] = []
    for rule in rules:
        results: list[ExitResult] = []
        for pick in picks:
            klines = pick.get("klines") or []
            r = simulate_short_exit(klines, rule)
            if r:
                results.append(r)
        if not results:
            continue
        rets = [r.return_pct for r in results]
        rows.append({
            "rule_id": rule["rule_id"],
            "category": rule.get("category", ""),
            "trade_count": len(results),
            "avg_return_pct": round(statistics.mean(rets), 4),
            "median_return_pct": round(statistics.median(rets), 4),
            "win_rate": round(sum(1 for r in rets if r >= 3) / len(rets) * 100, 2),
            "sharpe": sharpe(rets),
            "sortino": sortino(rets),
            "mdd": equity_mdd(rets),
            "profit_factor": profit_factor(rets),
            "avg_hold_minutes": round(statistics.mean(r.hold_minutes for r in results), 1),
            "avg_peak_capture_pct": round(statistics.mean(r.peak_capture_pct for r in results), 2),
            "avg_mfe_pct": round(statistics.mean(r.mfe_pct for r in results), 4),
            "avg_mae_pct": round(statistics.mean(r.mae_pct for r in results), 4),
        })
    rows.sort(key=lambda x: (-float(x["avg_return_pct"]), -float(x["sharpe"])))
    for i, r in enumerate(rows, 1):
        r["exit_rank"] = i
    return rows


def simulate_picks_with_rule(picks: list[dict], rule: dict) -> list[dict]:
    out: list[dict] = []
    for pick in picks:
        klines = pick.get("klines") or []
        r = simulate_short_exit(klines, rule)
        if not r:
            continue
        out.append({
            "scan_kst": pick["scan_kst"],
            "symbol": pick["symbol"],
            "rank": pick.get("rank"),
            **r.__dict__,
        })
    return out
