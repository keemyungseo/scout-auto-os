import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEARCH_TIMES_KST = [
    "2026-06-16 09:00:00",
    "2026-06-16 11:00:00",
    "2026-06-16 13:00:00",
    "2026-06-16 14:00:00",
    "2026-06-16 16:00:00",
    "2026-06-16 18:00:00",
    "2026-06-16 20:00:00",
    "2026-06-16 22:00:00",
    "2026-06-17 00:00:00",
]

KST_TZ = timezone(timedelta(hours=9))

FUTURES_BASE_URL = "https://fapi.binance.com"
EXCHANGE_INFO_ENDPOINT = "/fapi/v1/exchangeInfo"
KLINES_ENDPOINT = "/fapi/v1/klines"

EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "XRPUSDT", "USDCUSDT"}
MIN_PRICE = 0.05
MAX_PRICE = 400.0

MA24_LOOKBACK = 24
MA60_LOOKBACK = 60
MA24_BAND_PCT = 4.0
MAX_BREAKOUT_MULTIPLIER = 1.05
FORWARD_HOURS = 24

# Research heuristic: signal started a meaningful trend if 24h max profit >= 10%.
MEANINGFUL_TREND_MIN_MAX_PROFIT_PCT = 10.0

INTERVAL = "2h"
INTERVAL_MS = 2 * 60 * 60 * 1000
KLINES_NEEDED = MA60_LOOKBACK + 1
MAX_LIMIT = 1500

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "scout_review_v2.csv"


@dataclass
class ScoutSignal:
    search_time_kst: str
    search_time_utc: str
    search_dt: datetime
    symbol: str
    signal_close: float
    prev_24_box_high: float
    ma24: float
    ma60: float
    distance_from_ma24_percent: float
    distance_from_ma60_percent: float
    breakout_percent: float
    trading_volume_2h: float
    max_profit_percent: float
    max_drawdown_percent: float
    profit_after_6h_percent: float | None
    profit_after_12h_percent: float | None
    profit_after_24h_percent: float | None
    meaningful_trend: bool


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


def format_time_kst(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(KST_TZ).strftime("%Y-%m-%d %H:%M:%S KST")


def parse_kst_to_utc(kst_str: str) -> datetime:
    kst_dt = datetime.strptime(kst_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST_TZ)
    return kst_dt.astimezone(timezone.utc)


def get_search_times() -> list[tuple[str, str, datetime]]:
    return [
        (kst_str, format_time_utc(parse_kst_to_utc(kst_str)), parse_kst_to_utc(kst_str))
        for kst_str in SEARCH_TIMES_KST
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


def kline_open_dt(kline: list) -> datetime:
    return datetime.fromtimestamp(int(kline[0]) / 1000, tz=timezone.utc)


def kline_close_dt(kline: list) -> datetime:
    return kline_open_dt(kline) + timedelta(hours=2)


def get_close_at_or_before(candles: list[list], target: datetime) -> float | None:
    close_price: float | None = None
    for kline in candles:
        if kline_close_dt(kline) <= target:
            close_price = float(kline[4])
        else:
            break
    return close_price


def evaluate_scout_signal(symbol: str, search_dt: datetime) -> dict | None:
    end_ms = int(search_dt.timestamp() * 1000)
    klines = fetch_klines_before(symbol, end_ms, KLINES_NEEDED)

    if len(klines) < KLINES_NEEDED:
        return None

    signal = klines[-1]
    prev_24 = klines[-(MA24_LOOKBACK + 1) : -1]
    prev_60 = klines[-(MA60_LOOKBACK + 1) : -1]

    signal_close = float(signal[4])
    signal_open = float(signal[1])

    if not (MIN_PRICE <= signal_close <= MAX_PRICE):
        return None

    prev_24_closes = [float(candle[4]) for candle in prev_24]
    prev_60_closes = [float(candle[4]) for candle in prev_60]

    prev_24_box_high = max(prev_24_closes)
    ma24 = sum(prev_24_closes) / MA24_LOOKBACK
    ma60 = sum(prev_60_closes) / MA60_LOOKBACK

    if ma24 == 0 or ma60 == 0:
        return None

    if signal_close <= prev_24_box_high:
        return None
    if signal_close >= prev_24_box_high * MAX_BREAKOUT_MULTIPLIER:
        return None

    for close_price in prev_24_closes:
        if abs(close_price - ma24) / ma24 * 100 >= MA24_BAND_PCT:
            return None

    if signal_open <= ma60 or signal_close <= ma60:
        return None

    breakout_percent = (signal_close - prev_24_box_high) / prev_24_box_high * 100

    return {
        "signal_kline": signal,
        "signal_close": signal_close,
        "prev_24_box_high": prev_24_box_high,
        "ma24": ma24,
        "ma60": ma60,
        "distance_from_ma24_percent": (signal_close - ma24) / ma24 * 100,
        "distance_from_ma60_percent": (signal_close - ma60) / ma60 * 100,
        "breakout_percent": breakout_percent,
        "trading_volume_2h": float(signal[7]),
    }


def measure_forward_observation(
    symbol: str,
    search_dt: datetime,
    signal_close: float,
) -> dict:
    signal_end_ms = int(search_dt.timestamp() * 1000)
    forward_start_ms = signal_end_ms
    forward_end_ms = forward_start_ms + FORWARD_HOURS * 60 * 60 * 1000

    forward_klines = fetch_klines_forward(symbol, forward_start_ms, forward_end_ms)

    max_high = signal_close
    min_low = signal_close

    for kline in forward_klines:
        max_high = max(max_high, float(kline[2]))
        min_low = min(min_low, float(kline[3]))

    max_profit_percent = (max_high - signal_close) / signal_close * 100
    max_drawdown_percent = (signal_close - min_low) / signal_close * 100

    def profit_at(hours: int) -> float | None:
        close_price = get_close_at_or_before(
            forward_klines, search_dt + timedelta(hours=hours)
        )
        if close_price is None:
            return None
        return (close_price - signal_close) / signal_close * 100

    profit_6h = profit_at(6)
    profit_12h = profit_at(12)
    profit_24h = profit_at(24)

    meaningful_trend = max_profit_percent >= MEANINGFUL_TREND_MIN_MAX_PROFIT_PCT

    return {
        "max_profit_percent": max_profit_percent,
        "max_drawdown_percent": max_drawdown_percent,
        "profit_after_6h_percent": profit_6h,
        "profit_after_12h_percent": profit_12h,
        "profit_after_24h_percent": profit_24h,
        "meaningful_trend": meaningful_trend,
    }


def build_signal(
    search_kst: str,
    search_utc: str,
    search_dt: datetime,
    symbol: str,
    scout: dict,
    forward: dict,
) -> ScoutSignal:
    return ScoutSignal(
        search_time_kst=search_kst,
        search_time_utc=search_utc,
        search_dt=search_dt,
        symbol=symbol,
        signal_close=scout["signal_close"],
        prev_24_box_high=scout["prev_24_box_high"],
        ma24=scout["ma24"],
        ma60=scout["ma60"],
        distance_from_ma24_percent=scout["distance_from_ma24_percent"],
        distance_from_ma60_percent=scout["distance_from_ma60_percent"],
        breakout_percent=scout["breakout_percent"],
        trading_volume_2h=scout["trading_volume_2h"],
        max_profit_percent=forward["max_profit_percent"],
        max_drawdown_percent=forward["max_drawdown_percent"],
        profit_after_6h_percent=forward["profit_after_6h_percent"],
        profit_after_12h_percent=forward["profit_after_12h_percent"],
        profit_after_24h_percent=forward["profit_after_24h_percent"],
        meaningful_trend=forward["meaningful_trend"],
    )


def save_results(signals: list[ScoutSignal]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "search_time_kst",
        "search_time_utc",
        "symbol",
        "signal_close",
        "prev_24_box_high",
        "ma24",
        "ma60",
        "distance_from_ma24_percent",
        "distance_from_ma60_percent",
        "breakout_percent",
        "trading_volume_2h",
        "max_profit_percent",
        "max_drawdown_percent",
        "profit_after_6h_percent",
        "profit_after_12h_percent",
        "profit_after_24h_percent",
        "meaningful_trend",
    ]

    rows: list[dict[str, str]] = []
    for signal in signals:
        rows.append(
            {
                "search_time_kst": signal.search_time_kst,
                "search_time_utc": signal.search_time_utc,
                "symbol": signal.symbol,
                "signal_close": f"{signal.signal_close:.8f}",
                "prev_24_box_high": f"{signal.prev_24_box_high:.8f}",
                "ma24": f"{signal.ma24:.8f}",
                "ma60": f"{signal.ma60:.8f}",
                "distance_from_ma24_percent": f"{signal.distance_from_ma24_percent:.4f}",
                "distance_from_ma60_percent": f"{signal.distance_from_ma60_percent:.4f}",
                "breakout_percent": f"{signal.breakout_percent:.4f}",
                "trading_volume_2h": f"{signal.trading_volume_2h:.2f}",
                "max_profit_percent": f"{signal.max_profit_percent:.4f}",
                "max_drawdown_percent": f"{signal.max_drawdown_percent:.4f}",
                "profit_after_6h_percent": (
                    f"{signal.profit_after_6h_percent:.4f}"
                    if signal.profit_after_6h_percent is not None
                    else ""
                ),
                "profit_after_12h_percent": (
                    f"{signal.profit_after_12h_percent:.4f}"
                    if signal.profit_after_12h_percent is not None
                    else ""
                ),
                "profit_after_24h_percent": (
                    f"{signal.profit_after_24h_percent:.4f}"
                    if signal.profit_after_24h_percent is not None
                    else ""
                ),
                "meaningful_trend": "YES" if signal.meaningful_trend else "NO",
            }
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def print_search_results(signals: list[ScoutSignal]) -> None:
    search_times = get_search_times()

    for search_kst, search_utc, search_dt in search_times:
        matches = [s for s in signals if s.search_dt == search_dt]
        matches.sort(key=lambda item: item.trading_volume_2h, reverse=True)

        print("\n===== SCOUT SEARCH =====")
        print(f"Search time: {search_kst} KST ({search_utc})")
        print(f"Matched symbols: {len(matches)}")

        if not matches:
            print("  (none)")
            continue

        for match in matches:
            print(f"\n  Symbol: {match.symbol}")
            print(f"  Signal candle close: {match.signal_close:.8f}")
            print(f"  Previous 24 box high: {match.prev_24_box_high:.8f}")
            print(f"  MA24: {match.ma24:.8f}")
            print(f"  MA60: {match.ma60:.8f}")
            print(f"  Distance from MA24: {match.distance_from_ma24_percent:+.2f}%")
            print(f"  Distance from MA60: {match.distance_from_ma60_percent:+.2f}%")
            print(f"  Breakout %: {match.breakout_percent:.2f}%")
            print(f"  2h trading volume: {match.trading_volume_2h:,.2f}")
            print(f"  Max profit after signal: {match.max_profit_percent:+.2f}%")
            print(f"  Max drawdown after signal: {match.max_drawdown_percent:.2f}%")
            print(f"  Profit after 6h: {fmt_pct(match.profit_after_6h_percent)}")
            print(f"  Profit after 12h: {fmt_pct(match.profit_after_12h_percent)}")
            print(f"  Profit after 24h: {fmt_pct(match.profit_after_24h_percent)}")
            trend_label = "YES" if match.meaningful_trend else "NO"
            print(
                f"  Meaningful trend start: {trend_label} "
                f"(max profit >= {MEANINGFUL_TREND_MIN_MAX_PROFIT_PCT:.0f}%)"
            )


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def print_overall_summary(signals: list[ScoutSignal]) -> None:
    print("\n===== SCOUT V0.2 SUMMARY =====")
    print(f"Total Scout signals: {len(signals)}")

    search_times = get_search_times()
    print("\nSignals per search time:")
    for search_kst, _, search_dt in search_times:
        count = sum(1 for s in signals if s.search_dt == search_dt)
        print(f"  {search_kst} KST: {count}")

    profits_6h = [s.profit_after_6h_percent for s in signals if s.profit_after_6h_percent is not None]
    profits_12h = [s.profit_after_12h_percent for s in signals if s.profit_after_12h_percent is not None]
    profits_24h = [s.profit_after_24h_percent for s in signals if s.profit_after_24h_percent is not None]

    avg_6h = average(profits_6h)
    avg_12h = average(profits_12h)
    avg_24h = average(profits_24h)

    print(f"\nAverage 6h return: {fmt_pct(avg_6h)}")
    print(f"Average 12h return: {fmt_pct(avg_12h)}")
    print(f"Average 24h return: {fmt_pct(avg_24h)}")

    if signals:
        best = max(signals, key=lambda s: s.profit_after_24h_percent or float("-inf"))
        worst = min(signals, key=lambda s: s.profit_after_24h_percent or float("inf"))
        print(
            f"\nBest signal: {best.symbol} at {best.search_time_kst} KST "
            f"(24h {fmt_pct(best.profit_after_24h_percent)}, "
            f"max profit {best.max_profit_percent:+.2f}%)"
        )
        print(
            f"Worst signal: {worst.symbol} at {worst.search_time_kst} KST "
            f"(24h {fmt_pct(worst.profit_after_24h_percent)}, "
            f"max profit {worst.max_profit_percent:+.2f}%)"
        )

        meaningful_count = sum(1 for s in signals if s.meaningful_trend)
        print(
            f"\nMeaningful trend signals: {meaningful_count}/{len(signals)} "
            f"(max profit >= {MEANINGFUL_TREND_MIN_MAX_PROFIT_PCT:.0f}% within 24h)"
        )
    else:
        print("\nBest signal: N/A")
        print("Worst signal: N/A")

    print("==================================")


def main() -> None:
    try:
        search_times = get_search_times()
        eligible_symbols = get_eligible_symbols()

        if not eligible_symbols:
            print("Error: no eligible USDT perpetual symbols found.")
            return

        print("Scout v0.2 historical review starting.")
        print(f"Universe size: {len(eligible_symbols)} symbols")
        print(f"Search times: {len(search_times)}")
        print("Research only. No trading simulation.")

        all_signals: list[ScoutSignal] = []
        symbols = sorted(eligible_symbols)

        for search_kst, search_utc, search_dt in search_times:
            print(f"\nScanning {search_kst} KST...")
            time_matches = 0

            for index, symbol in enumerate(symbols, start=1):
                if index % 100 == 0 or index == len(symbols):
                    print(f"  progress: {index}/{len(symbols)}")

                try:
                    scout_data = evaluate_scout_signal(symbol, search_dt)
                    if scout_data is None:
                        continue

                    forward_data = measure_forward_observation(
                        symbol,
                        search_dt,
                        scout_data["signal_close"],
                    )

                    all_signals.append(
                        build_signal(
                            search_kst,
                            search_utc,
                            search_dt,
                            symbol,
                            scout_data,
                            forward_data,
                        )
                    )
                    time_matches += 1
                except urllib.error.HTTPError:
                    continue

                time.sleep(0.03)

            print(f"  matches: {time_matches}")

        save_results(all_signals)
        print_search_results(all_signals)
        print_overall_summary(all_signals)
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
