"""Binance Futures open position snapshot (V1.1)."""

from __future__ import annotations

from scout_auto_os.engine.binance_client import BinanceClient


class PositionReportService:
    def __init__(self, client: BinanceClient) -> None:
        self.client = client

    def fetch_open_positions(self) -> list[dict]:
        """Return open positions with mark price and unrealized PnL."""
        if not self.client.configured:
            print("[POSITION] open positions fetched count=0 (no API keys)")
            return []
        out = self.client.get_open_positions_detailed()
        print(f"[POSITION] open positions fetched count={len(out)}")
        return out
