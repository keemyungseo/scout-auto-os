"""Build causal scan schedules per cadence interval."""

from __future__ import annotations

from datetime import datetime, timedelta


def _parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def build_cadence_schedule(base_scans: list[str], interval_min: int) -> list[tuple[str, str]]:
    """
    Return list of (synthetic_scan_kst, feature_source_scan_kst).
    Feature source is latest base snapshot <= synthetic time (no lookahead).
    """
    if not base_scans:
        return []

    base_scans = sorted(base_scans)
    if interval_min >= 120:
        step = max(1, interval_min // 120)
        return [(s, s) for s in base_scans[::step]]

    schedule: list[tuple[str, str]] = []
    for i, src in enumerate(base_scans):
        t0 = _parse(src)
        if i + 1 < len(base_scans):
            t_end = _parse(base_scans[i + 1])
        else:
            t_end = t0 + timedelta(minutes=120)
        t = t0
        while t < t_end:
            schedule.append((_fmt(t), src))
            t += timedelta(minutes=interval_min)
    return schedule
