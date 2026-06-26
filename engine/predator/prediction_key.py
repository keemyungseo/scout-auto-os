"""Prediction identity keys for Value Gate shadow — trade_key / scan_id only."""

from __future__ import annotations


def make_scan_id(scan_time: str, symbol: str, side: str) -> str:
    return f"{scan_time}|{symbol.upper()}|{side.lower()}"


def make_prediction_key(
    *,
    trade_key: str = "",
    scan_id: str = "",
    scan_time: str = "",
    symbol: str = "",
    side: str = "",
) -> str:
    """Resolve prediction identity: trade_key > scan_id > timestamp|symbol|side."""
    if trade_key and trade_key.count("|") >= 2:
        return trade_key.strip()
    if scan_id and scan_id.count("|") >= 2:
        return scan_id.strip()
    if scan_time and symbol and side:
        return make_scan_id(scan_time, symbol, side)
    return ""


def prediction_key_from_row(row: dict) -> str:
    return make_prediction_key(
        trade_key=row.get("trade_key", ""),
        scan_id=row.get("scan_id", ""),
        scan_time=row.get("timestamp", "") or row.get("scan_kst", ""),
        symbol=row.get("symbol", ""),
        side=row.get("side", "") or row.get("direction", ""),
    )


def symbol_side_key(symbol: str, side: str) -> str:
    """Legacy key — must not be used for prediction join in replay."""
    return f"{symbol.upper()}|{side.lower()}"
