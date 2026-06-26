"""Temporal feature engineering from leak-safe scan history."""

from __future__ import annotations

import statistics


def _slope(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = statistics.mean(vals)
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
    den = sum((i - x_mean) ** 2 for i in range(n)) or 1e-9
    return round(num / den, 6)


def _duration_above_median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    med = statistics.median(vals)
    count = 0
    for v in vals:
        if v >= med:
            count += 1
        else:
            break
    return float(count)


def build_temporal_features(
    sequence: list[dict[str, float]],
    base_keys: tuple[str, ...],
) -> dict[str, float]:
    """sequence[0]=current scan, sequence[1]=2h ago, ... (leak-safe past only)."""
    out: dict[str, float] = {}
    if not sequence:
        return out

    cur = sequence[0]
    for key in base_keys:
        vals = [float(step.get(key, 0.0)) for step in sequence]
        c, p = vals[0], vals[1] if len(vals) > 1 else vals[0]
        out[f"ts_{key}_current"] = round(c, 6)
        out[f"ts_{key}_prev"] = round(p, 6)
        out[f"ts_{key}_delta"] = round(c - p, 6)
        if len(vals) > 2:
            prev_delta = vals[1] - vals[2]
            out[f"ts_{key}_accel"] = round((c - p) - prev_delta, 6)
        else:
            out[f"ts_{key}_accel"] = 0.0
        out[f"ts_{key}_mean"] = round(statistics.mean(vals), 6)
        out[f"ts_{key}_max"] = round(max(vals), 6)
        out[f"ts_{key}_min"] = round(min(vals), 6)
        out[f"ts_{key}_slope"] = _slope(vals)
        out[f"ts_{key}_volatility"] = round(statistics.pstdev(vals), 6) if len(vals) > 1 else 0.0
        out[f"ts_{key}_duration"] = _duration_above_median(vals)

    out["ts_seq_len"] = float(len(sequence))
    out["ts_seq_coverage"] = round(len(sequence) / max(len(sequence), 1), 4)
    return out


def merge_snapshot_and_temporal(
    snapshot_x: dict[str, float],
    temporal_x: dict[str, float],
    include_snapshot: bool = True,
) -> dict[str, float]:
    merged: dict[str, float] = {}
    if include_snapshot:
        merged.update(snapshot_x)
    merged.update(temporal_x)
    return merged
