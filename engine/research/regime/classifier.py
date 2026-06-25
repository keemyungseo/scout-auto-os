"""Market regime classification from scan-universe features (research only)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

REGIME_STATES = (
    "Strong_Bull",
    "Bull",
    "Sideway",
    "Bottom",
    "Breakout",
    "Bear",
    "Capitulation",
    "Unknown",
)


@dataclass
class MarketSnapshot:
    scan_kst: str
    median_1h: float
    median_2h: float
    breadth_positive_1h: float
    top20_positive_pct: float
    median_volume_ratio: float
    median_compression: float
    median_release: float
    median_range_1h: float
    btc_1h: float
    btc_2h: float
    top5_avg_momentum_1h: float
    volume_expansion: float
    symbol_count: int

    def to_dict(self) -> dict:
        return {
            "scan_kst": self.scan_kst,
            "median_1h": round(self.median_1h, 4),
            "median_2h": round(self.median_2h, 4),
            "breadth_positive_1h": round(self.breadth_positive_1h, 4),
            "top20_positive_pct": round(self.top20_positive_pct, 4),
            "median_volume_ratio": round(self.median_volume_ratio, 4),
            "median_compression": round(self.median_compression, 4),
            "median_release": round(self.median_release, 4),
            "median_range_1h": round(self.median_range_1h, 4),
            "btc_1h": round(self.btc_1h, 4),
            "btc_2h": round(self.btc_2h, 4),
            "top5_avg_momentum_1h": round(self.top5_avg_momentum_1h, 4),
            "volume_expansion": round(self.volume_expansion, 4),
            "symbol_count": self.symbol_count,
        }


def _g(f: dict, key: str, default: float = 0.0) -> float:
    return float(f.get(key, default))


def build_scan_snapshot(scan_kst: str, rows: list[dict]) -> MarketSnapshot:
    if not rows:
        return MarketSnapshot(scan_kst, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    rets_1h = [_g(r["features"], "1h_current_return_pct") for r in rows]
    rets_2h = [_g(r["features"], "2h_current_return_pct") for r in rows]
    vols = [_g(r["features"], "15m_current_volume_ratio") for r in rows]
    comps = [_g(r["features"], "5m_compression") for r in rows]
    rels = [_g(r["features"], "5m_release") for r in rows]
    ranges = [_g(r["features"], "1h_current_range_pct") for r in rows]

    btc = next((r for r in rows if r["symbol"] == "BTCUSDT"), None)
    if btc:
        btc_1h = _g(btc["features"], "1h_current_return_pct")
        btc_2h = _g(btc["features"], "2h_current_return_pct")
    else:
        btc_1h = statistics.median(rets_1h) if rets_1h else 0.0
        btc_2h = statistics.median(rets_2h) if rets_2h else 0.0

    ranked = sorted(rows, key=lambda r: _g(r["features"], "h4_score"), reverse=True)
    top20 = ranked[:20]
    top5 = ranked[:5]
    top20_pos = sum(1 for r in top20 if _g(r["features"], "1h_current_return_pct") > 0) / max(len(top20), 1)
    top5_mom = statistics.mean([_g(r["features"], "1h_current_return_pct") for r in top5]) if top5 else 0.0
    breadth = sum(1 for x in rets_1h if x > 0) / max(len(rets_1h), 1)
    vol_exp = statistics.median(vols) * statistics.median(ranges) / 10.0 if vols else 0.0

    return MarketSnapshot(
        scan_kst=scan_kst,
        median_1h=statistics.median(rets_1h),
        median_2h=statistics.median(rets_2h),
        breadth_positive_1h=breadth,
        top20_positive_pct=top20_pos,
        median_volume_ratio=statistics.median(vols),
        median_compression=statistics.median(comps),
        median_release=statistics.median(rels),
        median_range_1h=statistics.median(ranges),
        btc_1h=btc_1h,
        btc_2h=btc_2h,
        top5_avg_momentum_1h=top5_mom,
        volume_expansion=vol_exp,
        symbol_count=len(rows),
    )


def classify_regime_8(snap: MarketSnapshot) -> str:
    """Empirical regime label from collective market behaviour at scan time."""
    m = snap.median_1h
    b = snap.breadth_positive_1h
    btc = snap.btc_1h
    vol = snap.median_volume_ratio
    rel = snap.median_release
    rng = snap.median_range_1h
    comp = snap.median_compression
    top20 = snap.top20_positive_pct

    if m <= -1.5 and b < 0.28 and btc <= -1.0:
        return "Capitulation"
    if m <= -0.55 and b < 0.38 and btc < 0:
        return "Bear"
    if -1.0 <= m < 0.2 and comp >= 1.5 and rel >= 0.05 and b < 0.48:
        return "Bottom"
    if rel >= 0.15 and vol >= 1.1 and rng >= 2.5 and b >= 0.45:
        return "Breakout"
    if m >= 1.2 and b >= 0.58 and top20 >= 0.55:
        return "Strong_Bull"
    if m >= 0.35 and b >= 0.50:
        return "Bull"
    if abs(m) < 0.45 and 0.38 <= b <= 0.58:
        return "Sideway"
    return "Unknown"
