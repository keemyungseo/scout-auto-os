"""Temporal split generators — evaluation only, no tuning."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta


def _parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _week_key(s: str) -> str:
    return _parse(s).strftime("%Y-W%W")


def _month_key(s: str) -> str:
    return s[:7]


def monthly_splits(scans: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for s in scans:
        out[_month_key(s)].append(s)
    return dict(out)


def weekly_splits(scans: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for s in scans:
        out[_week_key(s)].append(s)
    return dict(out)


def rolling_walk_forward(scans: list[str], window_days: int = 7) -> dict[str, list[str]]:
    scans = sorted(scans)
    if not scans:
        return {}
    t0 = _parse(scans[0])
    t1 = _parse(scans[-1])
    out: dict[str, list[str]] = {}
    cur = t0
    idx = 0
    while cur <= t1:
        end = cur + timedelta(days=window_days)
        key = cur.strftime("%Y-%m-%d")
        chunk = [s for s in scans if cur <= _parse(s) < end]
        if chunk:
            out[f"wf_{key}"] = chunk
        cur = end
        idx += 1
    return out


def expanding_window_splits(scans: list[str], n_folds: int = 4) -> dict[str, list[str]]:
    scans = sorted(scans)
    if len(scans) < n_folds + 1:
        return {"exp_full": scans}
    out: dict[str, list[str]] = {}
    step = max(1, len(scans) // (n_folds + 1))
    for i in range(1, n_folds + 1):
        cut = min(len(scans), step * (i + 1))
        out[f"exp_{i}"] = scans[:cut]
    return out


def leave_one_period_out(scans: list[str], period: str = "week") -> dict[str, list[str]]:
    splits = weekly_splits(scans) if period == "week" else monthly_splits(scans)
    all_set = set(scans)
    out: dict[str, list[str]] = {}
    for key, held_out in splits.items():
        hold = set(held_out)
        out[f"loo_{key}"] = sorted(all_set - hold)
    return out


def temporal_blind_split(scans: list[str], train_ratio: float = 0.7) -> dict[str, list[str]]:
    scans = sorted(scans)
    cut = max(1, int(len(scans) * train_ratio))
    return {
        "blind_holdout": scans[cut:],
        "train_reference": scans[:cut],
    }
