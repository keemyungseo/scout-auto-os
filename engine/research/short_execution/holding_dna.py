"""Research 3 — Holding time DNA for short picks."""

from __future__ import annotations

import statistics

from scout_auto_os.engine.research.short_execution.constants import BAR_MINUTES, MAX_FORWARD_MINUTES
from scout_auto_os.engine.research.short_execution.lifecycle import short_close_roi, short_mfe_to_bar


def holding_path(pick: dict) -> list[dict]:
    klines = pick.get("klines") or []
    if len(klines) < 2:
        return []
    entry = float(klines[0][1])
    max_bar = min(len(klines) - 1, MAX_FORWARD_MINUTES // BAR_MINUTES)
    path: list[dict] = []
    peak_roi = float("-inf")
    peak_min = 0
    for i in range(max_bar + 1):
        bar = klines[i]
        mins = i * BAR_MINUTES
        roi = short_close_roi(entry, float(bar[4]))
        mfe = short_mfe_to_bar(klines, entry, i)
        if roi > peak_roi:
            peak_roi = roi
            peak_min = mins
        path.append({"minutes": mins, "roi_pct": roi, "mfe_pct": mfe})
    return path


def holding_dna_rows(picks: list[dict]) -> tuple[list[dict], dict]:
    detail: list[dict] = []
    peak_mins: list[int] = []
    profit_at_20: list[float] = []
    profit_at_50: list[float] = []
    profit_at_80: list[float] = []

    for pick in picks:
        path = holding_path(pick)
        if len(path) < 2:
            continue
        final_roi = path[-1]["roi_pct"]
        peak = max(path, key=lambda x: x["roi_pct"])
        peak_mins.append(int(peak["minutes"]))

        n = len(path)
        i20 = max(0, int(n * 0.2) - 1)
        i50 = max(0, int(n * 0.5) - 1)
        i80 = max(0, int(n * 0.8) - 1)
        r20, r50, r80 = path[i20]["roi_pct"], path[i50]["roi_pct"], path[i80]["roi_pct"]
        profit_at_20.append(r20)
        profit_at_50.append(r50)
        profit_at_80.append(r80)

        detail.append({
            "scan_kst": pick["scan_kst"],
            "symbol": pick["symbol"],
            "rank": pick.get("rank"),
            "final_roi_pct": final_roi,
            "peak_roi_pct": peak["roi_pct"],
            "peak_at_minutes": peak["minutes"],
            "hold_bars": len(path),
            "hold_minutes_max": path[-1]["minutes"],
            "profit_at_20pct_time": r20,
            "profit_at_50pct_time": r50,
            "profit_at_80pct_time": r80,
            "pct_profit_in_first_20pct_time": round(
                r20 / peak["roi_pct"] * 100, 2,
            ) if abs(peak["roi_pct"]) > 0.01 else 0,
            "pct_profit_in_first_50pct_time": round(
                r50 / peak["roi_pct"] * 100, 2,
            ) if abs(peak["roi_pct"]) > 0.01 else 0,
        })

    summary = {
        "sample_count": len(detail),
        "avg_peak_at_minutes": round(statistics.mean(peak_mins), 1) if peak_mins else 0,
        "median_peak_at_minutes": round(statistics.median(peak_mins), 1) if peak_mins else 0,
        "avg_final_roi_pct": round(statistics.mean(d["final_roi_pct"] for d in detail), 4) if detail else 0,
        "avg_profit_at_20pct_time": round(statistics.mean(profit_at_20), 4) if profit_at_20 else 0,
        "avg_profit_at_50pct_time": round(statistics.mean(profit_at_50), 4) if profit_at_50 else 0,
        "avg_profit_at_80pct_time": round(statistics.mean(profit_at_80), 4) if profit_at_80 else 0,
        "pct_peak_before_1h": round(
            sum(1 for m in peak_mins if m <= 60) / len(peak_mins) * 100, 2,
        ) if peak_mins else 0,
        "pct_peak_before_2h": round(
            sum(1 for m in peak_mins if m <= 120) / len(peak_mins) * 100, 2,
        ) if peak_mins else 0,
    }
    return detail, summary


def holding_distribution(detail: list[dict]) -> list[dict]:
    buckets = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 180), (180, 999)]
    rows: list[dict] = []
    for lo, hi in buckets:
        chunk = [d for d in detail if lo <= int(d["peak_at_minutes"]) < hi]
        if not chunk:
            continue
        label = f"{lo}-{hi}m" if hi < 999 else f"{lo}m+"
        rows.append({
            "peak_bucket": label,
            "count": len(chunk),
            "pct_of_samples": round(len(chunk) / len(detail) * 100, 2),
            "avg_peak_roi_pct": round(statistics.mean(float(c["peak_roi_pct"]) for c in chunk), 4),
            "avg_final_roi_pct": round(statistics.mean(float(c["final_roi_pct"]) for c in chunk), 4),
        })
    return rows
