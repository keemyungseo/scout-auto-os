"""Short regime taxonomy — empirical from scan-universe behaviour."""

from __future__ import annotations

import statistics

from scout_auto_os.engine.research.regime.classifier import build_scan_snapshot


def classify_short_regime(scan_kst: str, rows: list[dict]) -> dict[str, str]:
    snap = build_scan_snapshot(scan_kst, rows)
    m = snap.median_1h
    b = snap.breadth_positive_1h
    btc = snap.btc_1h
    vol = snap.median_volume_ratio
    rel = snap.median_release
    rng = snap.median_range_1h
    comp = snap.median_compression
    top5 = snap.top5_avg_momentum_1h

    if m <= -1.5 and b < 0.30:
        primary = "Capitulation"
    elif m <= -0.55 and b < 0.42 and top5 < 0:
        primary = "Bear_Trend"
    elif m <= -0.2 and rel >= 0.12 and rng >= 2.5:
        primary = "Bear_Breakout"
    elif m <= 0.1 and b < 0.45 and vol >= 1.2:
        primary = "Distribution"
    elif m >= 0.5 and b < 0.40 and top5 < 0.3:
        primary = "Weak_Bounce"
    elif m >= 0.8 and rel >= 0.15 and b >= 0.55:
        primary = "Short_Squeeze_Risk"
    elif m <= -0.3 and rng >= 2.5:
        primary = "High_Vol_Downtrend"
    elif abs(m) < 0.35 and rng < 2.0:
        primary = "Low_Vol_Drift"
    else:
        primary = "Mixed"

    vol_tag = "high_volatility" if rng >= 2.5 else "low_volatility"
    trend = "downtrend" if m < -0.2 else "uptrend" if m > 0.5 else "neutral"

    return {
        "short_regime": primary,
        "volatility": vol_tag,
        "trend_bias": trend,
        "compression": "compression" if comp >= 1.4 else "release" if rel >= 0.1 else "neutral",
    }


def build_short_regime_index(by_scan: dict[str, list[dict]]) -> dict[str, dict[str, str]]:
    return {scan: classify_short_regime(scan, rows) for scan, rows in by_scan.items()}
