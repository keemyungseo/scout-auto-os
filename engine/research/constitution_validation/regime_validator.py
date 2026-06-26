"""Multi-axis regime tagging for constitution validation."""

from __future__ import annotations

import statistics
from collections import defaultdict

from scout_auto_os.engine.research.regime.classifier import build_scan_snapshot, classify_regime_8
from scout_auto_os.engine.research.zero_base.validation import classify_regime


def _g(row: dict, key: str) -> float:
    return float((row.get("features") or row.get("x") or {}).get(key, 0))


def tag_scan_regimes(scan_rows: list[dict], scan_kst: str = "") -> dict[str, str]:
    """Scan-time regime axes — no forward leak."""
    scan_kst = scan_kst or (scan_rows[0].get("scan_kst", "") if scan_rows else "")
    snap = build_scan_snapshot(scan_kst, scan_rows)
    ecology = classify_regime_8(snap)
    simple = classify_regime(scan_rows)

    vol = snap.median_range_1h
    vol_tag = "high_volatility" if vol >= 2.5 else "low_volatility"

    if snap.median_release >= 0.12 and snap.median_compression <= 1.2:
        structure = "breakout"
    elif snap.median_compression >= 1.4 and snap.median_release < 0.08:
        structure = "compression"
    else:
        structure = "neutral"

    if snap.breadth_positive_1h >= 0.55 and snap.top5_avg_momentum_1h >= 0.8:
        dynamics = "trend"
    elif snap.breadth_positive_1h >= 0.42 and snap.top20_positive_pct < 0.45:
        dynamics = "rotation"
    else:
        dynamics = "mixed"

    market_map = {
        "Strong_Bull": "bull",
        "Bull": "bull",
        "Bear": "bear",
        "Capitulation": "bear",
        "Sideway": "sideways",
        "Bottom": "sideways",
        "Breakout": "breakout",
        "Unknown": "sideways",
    }

    return {
        "market_simple": simple,
        "market_ecology": ecology,
        "market_mapped": market_map.get(ecology, simple),
        "volatility": vol_tag,
        "structure": structure,
        "dynamics": dynamics,
    }


def build_regime_index(
    annotated: dict[str, list[dict]],
) -> dict[str, dict[str, str]]:
    return {scan: tag_scan_regimes(rows, scan) for scan, rows in annotated.items()}


def aggregate_by_regime_axis(
    picks: list[dict],
    regime_index: dict[str, dict[str, str]],
    axis: str,
) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for p in picks:
        tags = regime_index.get(p["scan_kst"], {})
        key = tags.get(axis, "unknown")
        buckets[key].append(p)
    return dict(buckets)
