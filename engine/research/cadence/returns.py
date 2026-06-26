"""Forward return between synthetic entry/exit times (evaluation only)."""

from __future__ import annotations

from datetime import datetime

BAR_MINUTES = 15


def _parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def return_between_times(
    klines: list,
    feature_scan_kst: str,
    entry_kst: str,
    exit_kst: str,
    direction: str,
) -> float:
    """Causal return from entry to exit using 15m forward bars anchored at feature snapshot."""
    if not klines or len(klines) < 2:
        return 0.0

    feat_t = _parse(feature_scan_kst)
    entry_t = _parse(entry_kst)
    exit_t = _parse(exit_kst)
    if exit_t <= entry_t:
        return 0.0

    entry_off_min = max(0, int((entry_t - feat_t).total_seconds() // 60))
    exit_off_min = max(entry_off_min, int((exit_t - feat_t).total_seconds() // 60))

    entry_bar = min(len(klines) - 1, entry_off_min // BAR_MINUTES)
    exit_bar = min(len(klines) - 1, max(entry_bar, exit_off_min // BAR_MINUTES))

    entry_px = float(klines[entry_bar][1]) or float(klines[entry_bar][4])
    exit_px = float(klines[exit_bar][4])
    if entry_px <= 0:
        return 0.0

    raw = (exit_px - entry_px) / entry_px * 100
    return round(-raw if direction == "short" else raw, 4)
