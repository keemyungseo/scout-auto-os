"""
Enrich 2026-06-14 TOP3 rows with forward +2h/+4h returns.
Research only. Does not modify existing scripts.
Reason: state transition learning lacked early checkpoint resolution on 06-14.
"""

import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST_TZ = timezone(timedelta(hours=9))
FUTURES_BASE_URL = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"
INTERVAL = "2h"
INTERVAL_MS = 2 * 60 * 60 * 1000
FORWARD_HOURS = 24
API_SLEEP_SEC = 0.03

INPUT_CSV = Path("logs/top3_gainers_20260614.csv")
OUTPUT_CSV = Path("logs/top3_gainers_20260614_enriched.csv")


def parse_kst(kst_str: str) -> datetime:
    return datetime.strptime(kst_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST_TZ).astimezone(timezone.utc)


def kline_close_dt(kline: list) -> datetime:
    open_dt = datetime.fromtimestamp(int(kline[0]) / 1000, tz=timezone.utc)
    return open_dt + timedelta(hours=2)


def fetch_forward(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    params = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1500,
        }
    )
    url = f"{FUTURES_BASE_URL}{KLINES_ENDPOINT}?{params}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def close_at(candles: list[list], target: datetime) -> float | None:
    close_price: float | None = None
    for candle in candles:
        if kline_close_dt(candle) <= target:
            close_price = float(candle[4])
        else:
            break
    return close_price


def measure(symbol: str, scan_dt: datetime, entry: float) -> dict:
    start_ms = int(scan_dt.timestamp() * 1000)
    end_ms = start_ms + FORWARD_HOURS * 60 * 60 * 1000
    candles = fetch_forward(symbol, start_ms, end_ms)

    def ret(hours: int) -> float | None:
        close_price = close_at(candles, scan_dt + timedelta(hours=hours))
        if close_price is None:
            return None
        return (close_price - entry) / entry * 100

    return {
        "forward_return_2h": ret(2),
        "forward_return_4h": ret(4),
    }


def main() -> None:
    rows = list(csv.DictReader(INPUT_CSV.open(encoding="utf-8")))
    fieldnames = list(rows[0].keys()) if rows else []
    for key in ("forward_return_2h", "forward_return_4h"):
        if key not in fieldnames:
            fieldnames.append(key)

    print(f"Enriching {len(rows)} rows from {INPUT_CSV.name}...")
    for index, row in enumerate(rows, start=1):
        symbol = row["symbol"]
        scan_dt = parse_kst(row["scan_time_kst"])
        entry = float(row["current_price"])
        try:
            forward = measure(symbol, scan_dt, entry)
            row["forward_return_2h"] = (
                f"{forward['forward_return_2h']:.4f}" if forward["forward_return_2h"] is not None else ""
            )
            row["forward_return_4h"] = (
                f"{forward['forward_return_4h']:.4f}" if forward["forward_return_4h"] is not None else ""
            )
            print(f"  {index}/{len(rows)} {symbol} @{row['scan_time_kst'][11:16]} f2={row['forward_return_2h']} f4={row['forward_return_4h']}")
        except Exception as exc:
            print(f"  {index}/{len(rows)} {symbol} failed: {exc}")
            row["forward_return_2h"] = ""
            row["forward_return_4h"] = ""
        time.sleep(API_SLEEP_SEC)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
