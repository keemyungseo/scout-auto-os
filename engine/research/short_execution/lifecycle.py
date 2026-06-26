"""Research 1 — Short position lifecycle at fixed checkpoints."""

from __future__ import annotations

import statistics

from scout_auto_os.engine.research.short_execution.constants import BAR_MINUTES, CHECKPOINT_MINUTES


def _bar_idx(minutes: int) -> int:
    return max(0, minutes // BAR_MINUTES - 1)


def short_close_roi(entry: float, close: float) -> float:
    if entry <= 0:
        return 0.0
    return round((entry - close) / entry * 100, 4)


def short_mfe_to_bar(klines: list, entry: float, bar_i: int) -> float:
    if not klines or entry <= 0:
        return 0.0
    lows = [float(k[3]) for k in klines[: bar_i + 1]]
    return round((entry - min(lows)) / entry * 100, 4)


def short_mae_to_bar(klines: list, entry: float, bar_i: int) -> float:
    if not klines or entry <= 0:
        return 0.0
    highs = [float(k[2]) for k in klines[: bar_i + 1]]
    return round((max(highs) - entry) / entry * 100, 4)


def trailing_roi_from_peak(klines: list, entry: float, bar_i: int) -> float:
    """Current ROI minus peak favorable ROI up to bar_i."""
    peak = 0.0
    for j in range(bar_i + 1):
        fav = short_mfe_to_bar(klines, entry, j)
        peak = max(peak, fav)
    cur = short_close_roi(entry, float(klines[bar_i][4]))
    return round(cur - peak, 4)


def analyze_pick_lifecycle(pick: dict) -> list[dict]:
    klines = pick.get("klines") or []
    if len(klines) < 2:
        return []
    entry = float(klines[0][1])
    rows: list[dict] = []
    for mins in CHECKPOINT_MINUTES:
        bi = min(_bar_idx(mins), len(klines) - 1)
        close = float(klines[bi][4])
        rows.append({
            "scan_kst": pick["scan_kst"],
            "symbol": pick["symbol"],
            "rank": pick.get("rank"),
            "checkpoint_min": mins,
            "roi_pct": short_close_roi(entry, close),
            "mfe_pct": short_mfe_to_bar(klines, entry, bi),
            "mae_pct": short_mae_to_bar(klines, entry, bi),
            "drawdown_from_peak_pct": trailing_roi_from_peak(klines, entry, bi),
            "trailing_gap_pct": round(
                short_mfe_to_bar(klines, entry, bi) - short_close_roi(entry, close), 4,
            ),
        })
    return rows


def aggregate_lifecycle(pick_rows: list[dict]) -> list[dict]:
    by_cp: dict[int, list[dict]] = {}
    for r in pick_rows:
        by_cp.setdefault(int(r["checkpoint_min"]), []).append(r)

    out: list[dict] = []
    for mins in CHECKPOINT_MINUTES:
        chunk = by_cp.get(mins, [])
        if not chunk:
            continue
        rois = [float(c["roi_pct"]) for c in chunk]
        mfes = [float(c["mfe_pct"]) for c in chunk]
        maes = [float(c["mae_pct"]) for c in chunk]
        out.append({
            "checkpoint_min": mins,
            "label": _checkpoint_label(mins),
            "sample_count": len(chunk),
            "avg_roi_pct": round(statistics.mean(rois), 4),
            "median_roi_pct": round(statistics.median(rois), 4),
            "avg_mfe_pct": round(statistics.mean(mfes), 4),
            "avg_mae_pct": round(statistics.mean(maes), 4),
            "avg_trailing_gap_pct": round(
                statistics.mean(float(c["trailing_gap_pct"]) for c in chunk), 4,
            ),
            "pct_positive_roi": round(sum(1 for r in rois if r > 0) / len(rois) * 100, 2),
        })
    return out


def best_checkpoint_by_roi(agg: list[dict]) -> dict:
    if not agg:
        return {}
    return max(agg, key=lambda x: float(x.get("avg_roi_pct", 0)))


def _checkpoint_label(mins: int) -> str:
    mapping = {5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h"}
    return mapping.get(mins, f"{mins}m")
