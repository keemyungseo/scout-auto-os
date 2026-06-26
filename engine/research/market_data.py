"""Public market data only — no signed order endpoints."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request


def fetch_price(rest_base: str, symbol: str, timeout: float = 10.0) -> float:
    url = f"{rest_base.rstrip('/')}/fapi/v1/ticker/price?{urllib.parse.urlencode({'symbol': symbol})}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return float(data.get("price", 0))


def fetch_klines(
    rest_base: str,
    symbol: str,
    interval: str,
    limit: int = 10,
    end_ms: int | None = None,
    timeout: float = 15.0,
) -> list[list]:
    params: dict = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_ms is not None:
        params["endTime"] = end_ms
    url = f"{rest_base.rstrip('/')}/fapi/v1/klines?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_klines_range(
    rest_base: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    timeout: float = 20.0,
) -> list[list]:
    """Paginated forward klines between start_ms (inclusive) and end_ms."""
    interval_ms = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    }.get(interval, 900_000)
    all_klines: list[list] = []
    current = start_ms
    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1500,
        }
        url = f"{rest_base.rstrip('/')}/fapi/v1/klines?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
            batch = json.loads(resp.read().decode())
        if not batch:
            break
        all_klines.extend(batch)
        last_open = int(batch[-1][0])
        next_start = last_open + interval_ms
        if next_start <= current:
            break
        current = next_start
    return all_klines


def btc_returns(rest_base: str) -> tuple[float, float]:
    kl = fetch_klines(rest_base, "BTCUSDT", "1h", limit=5)
    if len(kl) < 5:
        return 0.0, 0.0
    c_now = float(kl[-1][4])
    c_1h = float(kl[-2][4])
    c_4h = float(kl[-5][4])
    r1 = (c_now - c_1h) / c_1h * 100 if c_1h else 0.0
    r4 = (c_now - c_4h) / c_4h * 100 if c_4h else 0.0
    return round(r1, 4), round(r4, 4)


def classify_regime(btc_1h: float, btc_4h: float) -> str:
    if btc_4h <= -3.0:
        return "Crash"
    if btc_4h <= -1.0:
        return "Bear"
    if btc_4h >= 3.0:
        return "Bull"
    if abs(btc_1h) < 0.3 and abs(btc_4h) < 0.8:
        return "Sideway"
    if btc_1h >= 1.0 and btc_4h >= 0:
        return "Recovery"
    return "Unknown"
