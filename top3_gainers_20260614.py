import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCAN_TIMES_KST = [
    "2026-06-14 09:00:00",
    "2026-06-14 11:00:00",
    "2026-06-14 13:00:00",
    "2026-06-14 15:00:00",
    "2026-06-14 17:00:00",
    "2026-06-14 19:00:00",
    "2026-06-14 21:00:00",
    "2026-06-14 23:00:00",
]

KST_TZ = timezone(timedelta(hours=9))

FUTURES_BASE_URL = "https://fapi.binance.com"
EXCHANGE_INFO_ENDPOINT = "/fapi/v1/exchangeInfo"
KLINES_ENDPOINT = "/fapi/v1/klines"

EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "XRPUSDT", "USDCUSDT"}
MIN_PRICE = 0.05
MAX_PRICE = 400.0

CANDLES_24H = 12
LOOKBACK_7D = 84
LOOKBACK_24 = 24
TOP_N = 3

RANKING_KLINES = CANDLES_24H + 1
METRICS_KLINES = LOOKBACK_7D + LOOKBACK_24 + 1
FORWARD_HOURS = 24

INTERVAL = "2h"
INTERVAL_MS = 2 * 60 * 60 * 1000
MAX_LIMIT = 1500
API_SLEEP_SEC = 0.03

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "top3_gainers_20260614.csv"


@dataclass
class Top3Record:
    scan_time_kst: str
    scan_time_utc: str
    scan_dt: datetime
    symbol: str
    market_rank: int
    return_24h_percent: float
    current_price: float
    forward_return_6h: float | None
    forward_return_12h: float | None
    forward_return_24h: float | None
    max_profit_24h: float
    max_drawdown_24h: float
    position_7d_percent: float
    body_expansion_ratio: float
    volume_ratio_ma24: float
    return_prev_24h_percent: float
    ma24_slope_percent: float
    distance_from_ma24_percent: float


def parse_error_message(body: str) -> str:
    try:
        data = json.loads(body)
        if isinstance(data, dict) and data.get("msg"):
            return str(data["msg"])
    except json.JSONDecodeError:
        pass
    return body.strip() or "unknown error"


def format_time_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_kst_to_utc(kst_str: str) -> datetime:
    kst_dt = datetime.strptime(kst_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST_TZ)
    return kst_dt.astimezone(timezone.utc)


def get_scan_times() -> list[tuple[str, str, datetime]]:
    return [
        (kst_str, format_time_utc(parse_kst_to_utc(kst_str)), parse_kst_to_utc(kst_str))
        for kst_str in SCAN_TIMES_KST
    ]


def public_get(endpoint: str, params: dict | None = None) -> dict | list:
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    url = f"{FUTURES_BASE_URL}{endpoint}{query}"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def get_eligible_symbols() -> set[str]:
    exchange_info = public_get(EXCHANGE_INFO_ENDPOINT)
    eligible: set[str] = set()

    for symbol_info in exchange_info.get("symbols", []):
        symbol = symbol_info.get("symbol", "")
        order_types = symbol_info.get("orderTypes", [])

        if symbol_info.get("status") != "TRADING":
            continue
        if symbol_info.get("contractType") != "PERPETUAL":
            continue
        if symbol_info.get("quoteAsset") != "USDT":
            continue
        if "MARKET" not in order_types:
            continue
        if symbol in EXCLUDED_SYMBOLS:
            continue

        eligible.add(symbol)

    return eligible


def fetch_klines_before(symbol: str, end_ms: int, limit: int) -> list[list]:
    params = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": INTERVAL,
            "endTime": end_ms,
            "limit": limit,
        }
    )
    url = f"{FUTURES_BASE_URL}{KLINES_ENDPOINT}?{params}"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_klines_forward(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    all_klines: list[list] = []
    current_start = start_ms

    while current_start < end_ms:
        params = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "interval": INTERVAL,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": MAX_LIMIT,
            }
        )
        url = f"{FUTURES_BASE_URL}{KLINES_ENDPOINT}?{params}"
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=20) as response:
            batch = json.loads(response.read().decode("utf-8"))

        if not batch:
            break

        all_klines.extend(batch)
        last_open = int(batch[-1][0])
        next_start = last_open + INTERVAL_MS
        if next_start <= current_start:
            break
        current_start = next_start

    return all_klines


def kline_close_dt(kline: list) -> datetime:
    open_dt = datetime.fromtimestamp(int(kline[0]) / 1000, tz=timezone.utc)
    return open_dt + timedelta(hours=2)


def candle_close(kline: list) -> float:
    return float(kline[4])


def body_percent(kline: list) -> float:
    open_price = float(kline[1])
    close_price = float(kline[4])
    if open_price == 0:
        return 0.0
    return abs(close_price - open_price) / open_price * 100


def get_close_at_or_before(candles: list[list], target: datetime) -> float | None:
    close_price: float | None = None
    for kline in candles:
        if kline_close_dt(kline) <= target:
            close_price = candle_close(kline)
        else:
            break
    return close_price


def compute_24h_ranking(klines: list[list]) -> dict | None:
    if len(klines) < RANKING_KLINES:
        return None

    current_close = candle_close(klines[-1])
    if not (MIN_PRICE <= current_close <= MAX_PRICE):
        return None

    close_24h_ago = candle_close(klines[-(CANDLES_24H + 1)])
    if close_24h_ago == 0:
        return None

    return_24h_percent = (current_close - close_24h_ago) / close_24h_ago * 100
    return {
        "current_price": current_close,
        "return_24h_percent": return_24h_percent,
    }


def compute_learning_metrics(klines: list[list]) -> dict | None:
    if len(klines) < METRICS_KLINES:
        return None

    signal = klines[-1]
    prev_84 = klines[-(LOOKBACK_7D + 1) : -1]
    prev_24 = klines[-(LOOKBACK_24 + 1) : -1]
    prior_24 = klines[-(LOOKBACK_24 * 2 + 1) : -(LOOKBACK_24 + 1)]

    close_price = candle_close(signal)

    low_7d = min(float(candle[3]) for candle in prev_84)
    high_7d = max(float(candle[2]) for candle in prev_84)
    if high_7d == low_7d:
        position_7d = 50.0
    else:
        position_7d = (close_price - low_7d) / (high_7d - low_7d) * 100

    current_body = body_percent(signal)
    avg_body_24 = sum(body_percent(candle) for candle in prev_24) / LOOKBACK_24
    body_expansion = current_body / avg_body_24 if avg_body_24 > 0 else 0.0

    current_volume = float(signal[5])
    volume_ma24 = sum(float(candle[5]) for candle in prev_24) / LOOKBACK_24
    volume_ratio = current_volume / volume_ma24 if volume_ma24 > 0 else 0.0

    close_24h_ago = candle_close(klines[-(CANDLES_24H + 1)])
    if close_24h_ago == 0:
        return None
    return_prev_24h = (close_price - close_24h_ago) / close_24h_ago * 100

    ma24_now = sum(candle_close(candle) for candle in prev_24) / LOOKBACK_24
    ma24_prior = sum(candle_close(candle) for candle in prior_24) / LOOKBACK_24
    if ma24_prior == 0:
        ma24_slope = 0.0
    else:
        ma24_slope = (ma24_now - ma24_prior) / ma24_prior * 100

    if ma24_now == 0:
        return None
    distance_from_ma24 = (close_price - ma24_now) / ma24_now * 100

    return {
        "position_7d_percent": position_7d,
        "body_expansion_ratio": body_expansion,
        "volume_ratio_ma24": volume_ratio,
        "return_prev_24h_percent": return_prev_24h,
        "ma24_slope_percent": ma24_slope,
        "distance_from_ma24_percent": distance_from_ma24,
    }


def measure_forward(symbol: str, scan_dt: datetime, entry_price: float) -> dict:
    scan_end_ms = int(scan_dt.timestamp() * 1000)
    forward_end_ms = scan_end_ms + FORWARD_HOURS * 60 * 60 * 1000
    forward_klines = fetch_klines_forward(symbol, scan_end_ms, forward_end_ms)

    max_high = entry_price
    min_low = entry_price

    for kline in forward_klines:
        if kline_close_dt(kline) > scan_dt + timedelta(hours=24):
            break
        max_high = max(max_high, float(kline[2]))
        min_low = min(min_low, float(kline[3]))

    max_profit = (max_high - entry_price) / entry_price * 100
    max_drawdown = (entry_price - min_low) / entry_price * 100

    def return_at(hours: int) -> float | None:
        close_price = get_close_at_or_before(forward_klines, scan_dt + timedelta(hours=hours))
        if close_price is None:
            return None
        return (close_price - entry_price) / entry_price * 100

    return {
        "forward_return_6h": return_at(6),
        "forward_return_12h": return_at(12),
        "forward_return_24h": return_at(24),
        "max_profit_24h": max_profit,
        "max_drawdown_24h": max_drawdown,
    }


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def save_results(records: list[Top3Record]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scan_time_kst",
        "scan_time_utc",
        "symbol",
        "market_rank",
        "return_24h_percent",
        "current_price",
        "forward_return_6h",
        "forward_return_12h",
        "forward_return_24h",
        "max_profit_24h",
        "max_drawdown_24h",
        "position_7d_percent",
        "body_expansion_ratio",
        "volume_ratio_ma24",
        "return_prev_24h_percent",
        "ma24_slope_percent",
        "distance_from_ma24_percent",
    ]

    rows: list[dict[str, str]] = []
    for record in records:
        rows.append(
            {
                "scan_time_kst": record.scan_time_kst,
                "scan_time_utc": record.scan_time_utc,
                "symbol": record.symbol,
                "market_rank": str(record.market_rank),
                "return_24h_percent": f"{record.return_24h_percent:.4f}",
                "current_price": f"{record.current_price:.8f}",
                "forward_return_6h": (
                    f"{record.forward_return_6h:.4f}" if record.forward_return_6h is not None else ""
                ),
                "forward_return_12h": (
                    f"{record.forward_return_12h:.4f}" if record.forward_return_12h is not None else ""
                ),
                "forward_return_24h": (
                    f"{record.forward_return_24h:.4f}" if record.forward_return_24h is not None else ""
                ),
                "max_profit_24h": f"{record.max_profit_24h:.4f}",
                "max_drawdown_24h": f"{record.max_drawdown_24h:.4f}",
                "position_7d_percent": f"{record.position_7d_percent:.4f}",
                "body_expansion_ratio": f"{record.body_expansion_ratio:.4f}",
                "volume_ratio_ma24": f"{record.volume_ratio_ma24:.4f}",
                "return_prev_24h_percent": f"{record.return_prev_24h_percent:.4f}",
                "ma24_slope_percent": f"{record.ma24_slope_percent:.4f}",
                "distance_from_ma24_percent": f"{record.distance_from_ma24_percent:.4f}",
            }
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_scan_results(records: list[Top3Record]) -> None:
    for scan_kst, scan_utc, scan_dt in get_scan_times():
        scan_records = sorted(
            [record for record in records if record.scan_dt == scan_dt],
            key=lambda item: item.market_rank,
        )
        time_label = scan_kst.split(" ")[1][:5]
        print(f"\n===== 2026-06-14 {time_label} KST ({scan_utc}) =====")
        if not scan_records:
            print("  (no TOP3 recorded)")
            continue
        for record in scan_records:
            print(
                f"  #{record.market_rank} {record.symbol} | "
                f"24h {record.return_24h_percent:+.2f}% | "
                f"price {record.current_price:.6f} | "
                f"fwd24h {fmt_pct(record.forward_return_24h)} | "
                f"maxP {record.max_profit_24h:+.1f}% | "
                f"pos7d {record.position_7d_percent:.1f}%"
            )


def print_summary(records: list[Top3Record]) -> None:
    print("\n===== SUMMARY =====")
    print(f"Total TOP3 slots: {len(records)}")

    symbol_counts = Counter(record.symbol for record in records)
    unique_symbols = sorted(symbol_counts)
    print(f"Unique symbols: {len(unique_symbols)}")
    if unique_symbols:
        print(f"  {', '.join(unique_symbols)}")

    repeated = [symbol for symbol, count in symbol_counts.items() if count > 1]
    if repeated:
        repeated.sort(key=lambda symbol: symbol_counts[symbol], reverse=True)
        print("Repeated leaders:")
        for symbol in repeated:
            print(f"  {symbol}: {symbol_counts[symbol]} appearances")
    else:
        print("Repeated leaders: none")

    r6 = [record.forward_return_6h for record in records if record.forward_return_6h is not None]
    r12 = [record.forward_return_12h for record in records if record.forward_return_12h is not None]
    r24 = [record.forward_return_24h for record in records if record.forward_return_24h is not None]
    max_p = [record.max_profit_24h for record in records]
    max_dd = [record.max_drawdown_24h for record in records]

    print(f"Average forward 6h: {fmt_pct(average(r6))}")
    print(f"Average forward 12h: {fmt_pct(average(r12))}")
    print(f"Average forward 24h: {fmt_pct(average(r24))}")
    print(f"Average max profit: {fmt_pct(average(max_p))}")
    print(f"Average max drawdown: {fmt_pct(average(max_dd))}")

    if records:
        best = max(records, key=lambda record: record.forward_return_24h or float("-inf"))
        worst = min(records, key=lambda record: record.forward_return_24h or float("inf"))
        print(
            f"Best performer: {best.symbol} @ {best.scan_time_kst.split(' ')[1][:5]} "
            f"(fwd24h {fmt_pct(best.forward_return_24h)}, maxP {best.max_profit_24h:+.1f}%)"
        )
        print(
            f"Worst performer: {worst.symbol} @ {worst.scan_time_kst.split(' ')[1][:5]} "
            f"(fwd24h {fmt_pct(worst.forward_return_24h)}, maxP {worst.max_profit_24h:+.1f}%)"
        )

    print("===================")


def main() -> None:
    try:
        scan_times = get_scan_times()
        eligible_symbols = get_eligible_symbols()

        if not eligible_symbols:
            print("Error: no eligible USDT perpetual symbols found.")
            return

        print("TOP3 gainers ground-truth study starting.")
        print("Date: 2026-06-14 KST")
        print(f"Universe size: {len(eligible_symbols)} symbols")
        print("Research only. No filters. No scoring.")

        all_records: list[Top3Record] = []
        symbols = sorted(eligible_symbols)

        for scan_kst, scan_utc, scan_dt in scan_times:
            print(f"\nRanking {scan_kst} KST...")
            end_ms = int(scan_dt.timestamp() * 1000)
            rankings: list[tuple[str, dict]] = []

            for index, symbol in enumerate(symbols, start=1):
                if index % 100 == 0 or index == len(symbols):
                    print(f"  progress: {index}/{len(symbols)}")

                try:
                    klines = fetch_klines_before(symbol, end_ms, RANKING_KLINES)
                    ranking = compute_24h_ranking(klines)
                    if ranking is not None:
                        rankings.append((symbol, ranking))
                except urllib.error.HTTPError:
                    continue

                time.sleep(API_SLEEP_SEC)

            rankings.sort(key=lambda item: item[1]["return_24h_percent"], reverse=True)
            top3 = rankings[:TOP_N]
            print(f"  TOP3: {', '.join(f'{symbol} ({data['return_24h_percent']:+.1f}%)' for symbol, data in top3)}")

            for rank, (symbol, ranking) in enumerate(top3, start=1):
                try:
                    klines = fetch_klines_before(symbol, end_ms, METRICS_KLINES)
                    metrics = compute_learning_metrics(klines)
                    if metrics is None:
                        print(f"  Warning: could not compute metrics for {symbol}")
                        continue

                    forward = measure_forward(symbol, scan_dt, ranking["current_price"])
                    all_records.append(
                        Top3Record(
                            scan_time_kst=scan_kst,
                            scan_time_utc=scan_utc,
                            scan_dt=scan_dt,
                            symbol=symbol,
                            market_rank=rank,
                            return_24h_percent=ranking["return_24h_percent"],
                            current_price=ranking["current_price"],
                            forward_return_6h=forward["forward_return_6h"],
                            forward_return_12h=forward["forward_return_12h"],
                            forward_return_24h=forward["forward_return_24h"],
                            max_profit_24h=forward["max_profit_24h"],
                            max_drawdown_24h=forward["max_drawdown_24h"],
                            position_7d_percent=metrics["position_7d_percent"],
                            body_expansion_ratio=metrics["body_expansion_ratio"],
                            volume_ratio_ma24=metrics["volume_ratio_ma24"],
                            return_prev_24h_percent=metrics["return_prev_24h_percent"],
                            ma24_slope_percent=metrics["ma24_slope_percent"],
                            distance_from_ma24_percent=metrics["distance_from_ma24_percent"],
                        )
                    )
                except urllib.error.HTTPError:
                    print(f"  Warning: failed to enrich {symbol}")
                    continue

                time.sleep(API_SLEEP_SEC)

        save_results(all_records)
        print_scan_results(all_records)
        print_summary(all_records)
        print(f"\nFull results saved: {OUTPUT_CSV}")

    except ValueError as exc:
        print(f"Error: {exc}")
    except urllib.error.HTTPError as exc:
        details = parse_error_message(exc.read().decode("utf-8", errors="replace"))
        print(f"Error: Binance API request failed. HTTP {exc.code}: {details}")
    except urllib.error.URLError as exc:
        print(f"Error: cannot connect to Binance. {exc.reason}")


if __name__ == "__main__":
    main()
