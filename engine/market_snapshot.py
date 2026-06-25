"""Public market snapshots for entry quality guard (unsigned REST)."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request


def _get(rest_base: str, endpoint: str, params: dict | None = None, timeout: float = 12.0):
    url = f"{rest_base.rstrip('/')}{endpoint}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_ticker_24h(rest_base: str, symbol: str) -> dict:
    return _get(rest_base, "/fapi/v1/ticker/24hr", {"symbol": symbol.upper()})


def fetch_book_ticker(rest_base: str, symbol: str) -> dict:
    return _get(rest_base, "/fapi/v1/ticker/bookTicker", {"symbol": symbol.upper()})


def fetch_klines(rest_base: str, symbol: str, interval: str, limit: int = 25) -> list[list]:
    return _get(
        rest_base,
        "/fapi/v1/klines",
        {"symbol": symbol.upper(), "interval": interval, "limit": limit},
    )


def quote_volume_from_kline(k: list) -> float:
    # Binance futures kline index 7 = quote asset volume
    if len(k) > 7:
        return float(k[7])
    return float(k[5]) * float(k[4])


def build_entry_market_snapshot(rest_base: str, symbol: str, current_price: float) -> dict:
    sym = symbol.upper()
    ticker = fetch_ticker_24h(rest_base, sym)
    book = fetch_book_ticker(rest_base, sym)
    k5 = fetch_klines(rest_base, sym, "5m", 22)
    k15 = fetch_klines(rest_base, sym, "15m", 5)

    bid = float(book.get("bidPrice") or 0)
    ask = float(book.get("askPrice") or 0)
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else current_price
    slippage_pct = (ask - bid) / mid * 100 if mid > 0 else 0.0

    qv_24h = float(ticker.get("quoteVolume") or 0)
    qv_1h = sum(quote_volume_from_kline(k) for k in k15[-4:]) if k15 else 0.0

    last_5m = k5[-1] if k5 else None
    prior_5m = k5[-21:-1] if len(k5) >= 21 else k5[:-1]
    qv_5m_last = quote_volume_from_kline(last_5m) if last_5m else 0.0
    qv_5m_avg = (
        sum(quote_volume_from_kline(k) for k in prior_5m) / len(prior_5m)
        if prior_5m else qv_5m_last
    )
    ratio_5m = qv_5m_last / qv_5m_avg if qv_5m_avg > 0 else 0.0

    high_15m = max(float(k[2]) for k in k15) if k15 else current_price
    pullback_pct = (high_15m - current_price) / high_15m * 100 if high_15m > 0 else 0.0

    if last_5m:
        o, c = float(last_5m[1]), float(last_5m[4])
        last_dir = "red" if c < o else ("green" if c > o else "doji")
    else:
        last_dir = "unknown"

    return {
        "quote_volume_24h": round(qv_24h, 2),
        "quote_volume_1h": round(qv_1h, 2),
        "quote_volume_5m": round(qv_5m_last, 2),
        "quote_volume_5m_ratio": round(ratio_5m, 4),
        "pullback_from_15m_high_pct": round(pullback_pct, 4),
        "last_5m_candle_direction": last_dir,
        "estimated_slippage_pct": round(slippage_pct, 4),
        "bid": bid,
        "ask": ask,
    }
