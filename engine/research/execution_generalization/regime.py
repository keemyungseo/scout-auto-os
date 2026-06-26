"""Scan-level regime labels from scan-time universe features only."""

from __future__ import annotations

import statistics

from scout_auto_os.engine.research.zero_base.validation import classify_regime


def scan_regime(rows: list[dict]) -> str:
    return classify_regime(rows)


def scan_volatility_band(rows: list[dict]) -> str:
    ranges = [float(r["features"].get("1h_current_range_pct", 0)) for r in rows]
    if not ranges:
        return "unknown"
    med = statistics.median(ranges)
    if med >= 20.0:
        return "high_volatility"
    if med <= 10.0:
        return "low_volatility"
    return "mid_volatility"


def attach_regime_to_groups(
    groups: list[list[dict]],
    by_scan: dict,
) -> None:
    for g in groups:
        scan = g[0]["scan_time_kst"]
        rows = by_scan.get(scan, [])
        regime = scan_regime(rows)
        vol = scan_volatility_band(rows)
        for r in g:
            r["regime"] = regime
            r["volatility_band"] = vol
