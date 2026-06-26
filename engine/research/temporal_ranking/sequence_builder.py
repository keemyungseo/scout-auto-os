"""Build per-symbol scan history sequences — strict past-only, no leak."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime


def _parse_scan(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def build_symbol_timelines(rows: list[dict]) -> dict[str, list[dict]]:
    by_sym: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sym[r["symbol"]].append(r)
    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: _parse_scan(x["scan_kst"]))
    return by_sym


def extract_history_vector(row: dict, keys: tuple[str, ...]) -> dict[str, float]:
    x = row.get("x") or {}
    return {k: float(x.get(k, 0.0)) for k in keys}


def attach_sequences(
    rows: list[dict],
    seq_len: int,
    base_keys: tuple[str, ...],
) -> list[dict]:
    """Attach leak-safe history of length seq_len (current + seq_len-1 past scans)."""
    timelines = build_symbol_timelines(rows)
    out: list[dict] = []
    for sym, timeline in timelines.items():
        for i, row in enumerate(timeline):
            past = timeline[max(0, i - seq_len + 1): i + 1]
            past = list(reversed(past))  # [current, t-1, t-2, ...]
            history = [
                {**extract_history_vector(p, base_keys), "_scan_kst": p["scan_kst"]}
                for p in past
            ]
            out.append({
                **row,
                "history": history,
                "history_len": len(history),
                "seq_len_target": seq_len,
            })
    return out


def leak_check_row(row: dict) -> list[str]:
    violations: list[str] = []
    scan_t = _parse_scan(row["scan_kst"])
    for h in row.get("history") or []:
        pass  # history vectors have no scan time embedded
    x = row.get("x_temporal") or row.get("x") or {}
    for k in x:
        for bad in ("return_2h", "label_", "outcome_", "max_up"):
            if bad in k:
                violations.append(f"forbidden key {k}")
    hist = row.get("history") or []
    if hist and row.get("history_len", 0) > row.get("seq_len_target", 99):
        violations.append("history longer than target")
    return violations


def leak_check_dataset(rows: list[dict], timelines: dict[str, list[dict]]) -> dict:
    from scout_auto_os.engine.research.temporal_ranking.constants import LEAK_FORBIDDEN_PREFIXES

    forbidden_obs = 0
    forbidden_label_keys = 0
    future_scan_leaks = 0
    history_overflow = 0

    for r in rows:
        x = r.get("x_temporal") or r.get("x") or {}
        for k in x:
            if k.startswith("exec_obs_"):
                forbidden_obs += 1
            elif any(k.startswith(bad) for bad in LEAK_FORBIDDEN_PREFIXES):
                forbidden_label_keys += 1

        scan_t = _parse_scan(r["scan_kst"])
        hist = r.get("history") or []
        if len(hist) > int(r.get("seq_len_target", 99)):
            history_overflow += 1
        for step in hist:
            step_scan = step.get("_scan_kst")
            if not step_scan:
                continue
            if _parse_scan(step_scan) > scan_t:
                future_scan_leaks += 1

    return {
        "rows_checked": len(rows),
        "forbidden_obs_features": forbidden_obs,
        "forbidden_label_keys": forbidden_label_keys,
        "future_scan_leaks": future_scan_leaks,
        "history_overflow": history_overflow,
        "passed": (
            forbidden_obs == 0
            and forbidden_label_keys == 0
            and future_scan_leaks == 0
            and history_overflow == 0
        ),
    }
