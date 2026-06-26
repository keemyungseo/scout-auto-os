"""Pattern classification — HEI / MET / WLD."""

from __future__ import annotations

MANUAL_GUARD_SYMBOLS = frozenset({"WLDUSDT"})


def classify_patterns(trade: dict) -> dict[str, int]:
    """Return pattern flags for a closed trade row."""
    peak = float(trade.get("peak_roi", 0))
    actual = float(trade.get("actual_return", 0))
    expected = float(trade.get("expected_return", 3.0))
    hold = int(trade.get("hold_minutes", 0))
    expected_horizon = int(trade.get("expected_horizon_min", 120))
    symbol = trade.get("symbol", "")
    entry_score = float(trade.get("entry_score", 0))

    hei = int(
        peak >= expected * 1.3
        and actual >= expected * 0.8
        and hold >= expected_horizon * 0.8
    )
    met = int(
        hold >= max(120, expected_horizon)
        and peak - actual >= 5.0
        and actual < expected * 0.5
    )
    wld = int(symbol in MANUAL_GUARD_SYMBOLS)

    return {"pattern_hei": hei, "pattern_met": met, "pattern_wld": wld}


def pattern_summary(trades: list[dict]) -> dict:
    if not trades:
        return {"hei_count": 0, "met_count": 0, "wld_count": 0}
    hei = sum(t.get("pattern_hei", 0) for t in trades)
    met = sum(t.get("pattern_met", 0) for t in trades)
    wld = sum(t.get("pattern_wld", 0) for t in trades)
    return {"hei_count": hei, "met_count": met, "wld_count": wld}
