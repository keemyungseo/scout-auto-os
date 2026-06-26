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
    "2026-06-17 09:00:00",
    "2026-06-17 11:00:00",
    "2026-06-17 13:00:00",
    "2026-06-17 15:00:00",
    "2026-06-17 17:00:00",
    "2026-06-17 19:00:00",
    "2026-06-17 21:00:00",
    "2026-06-17 23:00:00",
    "2026-06-18 01:00:00",
    "2026-06-18 03:00:00",
]

KST_TZ = timezone(timedelta(hours=9))

FUTURES_BASE_URL = "https://fapi.binance.com"
EXCHANGE_INFO_ENDPOINT = "/fapi/v1/exchangeInfo"
KLINES_ENDPOINT = "/fapi/v1/klines"

EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "XRPUSDT", "USDCUSDT"}
MIN_PRICE = 0.05
MAX_PRICE = 400.0

LOOKBACK = 24
IMPULSE_LOOKBACK_CANDLES = 7 * 24 // 2
MAX_BREAKOUT_MULTIPLIER = 1.05
MAX_MA24_DISTANCE_MULTIPLIER = 1.05
MAX_MA24_CHANGE_PCT = 3.0
MAX_BOX_RANGE_PCT = 8.0
FORWARD_HOURS = 24

MEANINGFUL_TREND_MIN_MAX_PROFIT_PCT = 10.0

INTERVAL = "2h"
INTERVAL_MS = 2 * 60 * 60 * 1000
KLINES_NEEDED = max(LOOKBACK * 2 + 1, IMPULSE_LOOKBACK_CANDLES + 1)
MAX_LIMIT = 1500

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "scout_review_v3.csv"


@dataclass
class ScoutSignal:
    search_time_kst: str
    search_time_utc: str
    search_dt: datetime
    symbol: str
    signal_close: float
    prev_24_highest_close: float
    ma24: float
    ma24_change_rate_percent: float
    box_high: float
    box_low: float
    box_range_percent: float
    breakout_percent: float
    distance_from_ma24_percent: float
    prior_7d_low: float
    prior_7d_high: float
    impulse_start_price: float
    impulse_percent: float
    reset_depth: float | None
    max_profit_percent: float
    max_drawdown_percent: float
    return_after_6h_percent: float | None
    return_after_12h_percent: float | None
    return_after_24h_percent: float | None
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


def kline_close_dt(kline: list) -> datetime:
    open_dt = datetime.fromtimestamp(int(kline[0]) / 1000, tz=timezone.utc)
    return open_dt + timedelta(hours=2)


def get_close_at_or_before(candles: list[list], target: datetime) -> float | None:
    close_price: float | None = None
    for kline in candles:
        if kline_close_dt(kline) <= target:
            close_price = float(kline[4])
        else:
            break
    return close_price


def compute_impulse_metrics(prev_7d: list[list], box_low: float) -> dict | None:
    prior_7d_low = min(float(candle[3]) for candle in prev_7d)
    prior_7d_high = max(float(candle[2]) for candle in prev_7d)

    if prior_7d_low == 0:
        return None

    impulse_percent = (prior_7d_high - prior_7d_low) / prior_7d_low * 100
    impulse_range = prior_7d_high - prior_7d_low
    reset_depth = (
        (prior_7d_high - box_low) / impulse_range if impulse_range > 0 else None
    )

    return {
        "prior_7d_low": prior_7d_low,
        "prior_7d_high": prior_7d_high,
        "impulse_start_price": prior_7d_low,
        "impulse_percent": impulse_percent,
        "reset_depth": reset_depth,
    }


def evaluate_scout_signal(symbol: str, search_dt: datetime) -> dict | None:
    end_ms = int(search_dt.timestamp() * 1000)
    klines = fetch_klines_before(symbol, end_ms, KLINES_NEEDED)

    if len(klines) < KLINES_NEEDED:
        return None

    signal = klines[-1]
    prev_24 = klines[-(LOOKBACK + 1) : -1]
    prior_24 = klines[-(LOOKBACK * 2 + 1) : -(LOOKBACK + 1)]
    prev_7d = klines[-(IMPULSE_LOOKBACK_CANDLES + 1) : -1]

    signal_close = float(signal[4])

    if not (MIN_PRICE <= signal_close <= MAX_PRICE):
        return None

    prev_24_closes = [float(candle[4]) for candle in prev_24]
    prior_24_closes = [float(candle[4]) for candle in prior_24]

    prev_24_highest_close = max(prev_24_closes)
    ma24 = sum(prev_24_closes) / LOOKBACK
    ma24_prior = sum(prior_24_closes) / LOOKBACK

    if ma24 == 0 or ma24_prior == 0:
        return None

    if signal_close <= prev_24_highest_close:
        return None
    if signal_close >= prev_24_highest_close * MAX_BREAKOUT_MULTIPLIER:
        return None

    if signal_close >= ma24 * MAX_MA24_DISTANCE_MULTIPLIER:
        return None

    ma24_change_rate = abs(ma24 - ma24_prior) / ma24_prior * 100
    if ma24_change_rate >= MAX_MA24_CHANGE_PCT:
        return None

    box_high = max(float(candle[2]) for candle in prev_24)
    box_low = min(float(candle[3]) for candle in prev_24)
    box_range_percent = (box_high - box_low) / ma24 * 100

    if box_range_percent >= MAX_BOX_RANGE_PCT:
        return None

    breakout_percent = (
        (signal_close - prev_24_highest_close) / prev_24_highest_close * 100
    )

    impulse = compute_impulse_metrics(prev_7d, box_low)
    if impulse is None:
        return None

    return {
        "signal_close": signal_close,
        "prev_24_highest_close": prev_24_highest_close,
        "ma24": ma24,
        "ma24_change_rate_percent": ma24_change_rate,
        "box_high": box_high,
        "box_low": box_low,
        "box_range_percent": box_range_percent,
        "breakout_percent": breakout_percent,
        "distance_from_ma24_percent": (signal_close - ma24) / ma24 * 100,
        **impulse,
    }


def measure_forward_observation(
    symbol: str,
    search_dt: datetime,
    signal_close: float,
) -> dict:
    signal_end_ms = int(search_dt.timestamp() * 1000)
    forward_end_ms = signal_end_ms + FORWARD_HOURS * 60 * 60 * 1000

    forward_klines = fetch_klines_forward(symbol, signal_end_ms, forward_end_ms)

    max_high = signal_close
    min_low = signal_close

    for kline in forward_klines:
        max_high = max(max_high, float(kline[2]))
        min_low = min(min_low, float(kline[3]))

    max_profit_percent = (max_high - signal_close) / signal_close * 100
    max_drawdown_percent = (signal_close - min_low) / signal_close * 100

    def return_at(hours: int) -> float | None:
        close_price = get_close_at_or_before(
            forward_klines, search_dt + timedelta(hours=hours)
        )
        if close_price is None:
            return None
        return (close_price - signal_close) / signal_close * 100

    return_6h = return_at(6)
    return_12h = return_at(12)
    return_24h = return_at(24)

    meaningful_trend = max_profit_percent >= MEANINGFUL_TREND_MIN_MAX_PROFIT_PCT

    return {
        "max_profit_percent": max_profit_percent,
        "max_drawdown_percent": max_drawdown_percent,
        "return_after_6h_percent": return_6h,
        "return_after_12h_percent": return_12h,
        "return_after_24h_percent": return_24h,
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
        prev_24_highest_close=scout["prev_24_highest_close"],
        ma24=scout["ma24"],
        ma24_change_rate_percent=scout["ma24_change_rate_percent"],
        box_high=scout["box_high"],
        box_low=scout["box_low"],
        box_range_percent=scout["box_range_percent"],
        breakout_percent=scout["breakout_percent"],
        distance_from_ma24_percent=scout["distance_from_ma24_percent"],
        prior_7d_low=scout["prior_7d_low"],
        prior_7d_high=scout["prior_7d_high"],
        impulse_start_price=scout["impulse_start_price"],
        impulse_percent=scout["impulse_percent"],
        reset_depth=scout["reset_depth"],
        max_profit_percent=forward["max_profit_percent"],
        max_drawdown_percent=forward["max_drawdown_percent"],
        return_after_6h_percent=forward["return_after_6h_percent"],
        return_after_12h_percent=forward["return_after_12h_percent"],
        return_after_24h_percent=forward["return_after_24h_percent"],
        meaningful_trend=forward["meaningful_trend"],
    )


def save_results(signals: list[ScoutSignal]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "search_time_kst",
        "search_time_utc",
        "symbol",
        "signal_close",
        "prev_24_highest_close",
        "ma24",
        "ma24_change_rate_percent",
        "box_high",
        "box_low",
        "box_range_percent",
        "breakout_percent",
        "distance_from_ma24_percent",
        "prior_7d_low",
        "prior_7d_high",
        "impulse_start_price",
        "impulse_percent",
        "reset_depth",
        "max_profit_percent",
        "max_drawdown_percent",
        "return_after_6h_percent",
        "return_after_12h_percent",
        "return_after_24h_percent",
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
                "prev_24_highest_close": f"{signal.prev_24_highest_close:.8f}",
                "ma24": f"{signal.ma24:.8f}",
                "ma24_change_rate_percent": f"{signal.ma24_change_rate_percent:.4f}",
                "box_high": f"{signal.box_high:.8f}",
                "box_low": f"{signal.box_low:.8f}",
                "box_range_percent": f"{signal.box_range_percent:.4f}",
                "breakout_percent": f"{signal.breakout_percent:.4f}",
                "distance_from_ma24_percent": f"{signal.distance_from_ma24_percent:.4f}",
                "prior_7d_low": f"{signal.prior_7d_low:.8f}",
                "prior_7d_high": f"{signal.prior_7d_high:.8f}",
                "impulse_start_price": f"{signal.impulse_start_price:.8f}",
                "impulse_percent": f"{signal.impulse_percent:.4f}",
                "reset_depth": (
                    f"{signal.reset_depth:.4f}"
                    if signal.reset_depth is not None
                    else ""
                ),
                "max_profit_percent": f"{signal.max_profit_percent:.4f}",
                "max_drawdown_percent": f"{signal.max_drawdown_percent:.4f}",
                "return_after_6h_percent": (
                    f"{signal.return_after_6h_percent:.4f}"
                    if signal.return_after_6h_percent is not None
                    else ""
                ),
                "return_after_12h_percent": (
                    f"{signal.return_after_12h_percent:.4f}"
                    if signal.return_after_12h_percent is not None
                    else ""
                ),
                "return_after_24h_percent": (
                    f"{signal.return_after_24h_percent:.4f}"
                    if signal.return_after_24h_percent is not None
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
    for search_kst, search_utc, search_dt in get_search_times():
        matches = [s for s in signals if s.search_dt == search_dt]
        matches.sort(key=lambda item: item.symbol)

        print("\n===== SCOUT SEARCH =====")
        print(f"Search time: {search_kst} KST ({search_utc})")
        print(f"Matched symbols: {len(matches)}")

        if not matches:
            print("  (none)")
            continue

        for match in matches:
            print(f"\n  Symbol: {match.symbol}")
            print(f"  Signal candle close: {match.signal_close:.8f}")
            print(f"  Previous 24 highest close: {match.prev_24_highest_close:.8f}")
            print(f"  MA24: {match.ma24:.8f}")
            print(f"  MA24 change rate: {match.ma24_change_rate_percent:.2f}%")
            print(f"  BoxHigh: {match.box_high:.8f}")
            print(f"  BoxLow: {match.box_low:.8f}")
            print(f"  Box range %: {match.box_range_percent:.2f}%")
            print(f"  Breakout %: {match.breakout_percent:.2f}%")
            print(f"  Distance from MA24: {match.distance_from_ma24_percent:+.2f}%")
            print(f"  Prior 7d low (impulse start proxy): {match.prior_7d_low:.8f}")
            print(f"  Prior 7d high: {match.prior_7d_high:.8f}")
            print(f"  Impulse %: {match.impulse_percent:.2f}%")
            reset_depth_label = (
                f"{match.reset_depth:.4f}" if match.reset_depth is not None else "N/A"
            )
            print(f"  Reset depth: {reset_depth_label}")
            print(f"  Max profit after signal: {match.max_profit_percent:+.2f}%")
            print(f"  Max drawdown after signal: {match.max_drawdown_percent:.2f}%")
            print(f"  Return after 6h: {fmt_pct(match.return_after_6h_percent)}")
            print(f"  Return after 12h: {fmt_pct(match.return_after_12h_percent)}")
            print(f"  Return after 24h: {fmt_pct(match.return_after_24h_percent)}")
            trend_label = "YES" if match.meaningful_trend else "NO"
            print(
                f"  Meaningful trend: {trend_label} "
                f"(max profit >= {MEANINGFUL_TREND_MIN_MAX_PROFIT_PCT:.0f}%)"
            )


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def print_overall_summary(signals: list[ScoutSignal]) -> None:
    print("\n===== SCOUT V0.3 SUMMARY =====")
    print(f"Total Scout signals: {len(signals)}")

    print("\nSignals by search time:")
    for search_kst, _, search_dt in get_search_times():
        count = sum(1 for s in signals if s.search_dt == search_dt)
        print(f"  {search_kst} KST: {count}")

    returns_6h = [
        s.return_after_6h_percent
        for s in signals
        if s.return_after_6h_percent is not None
    ]
    returns_12h = [
        s.return_after_12h_percent
        for s in signals
        if s.return_after_12h_percent is not None
    ]
    returns_24h = [
        s.return_after_24h_percent
        for s in signals
        if s.return_after_24h_percent is not None
    ]

    print(f"\nAverage 6h return: {fmt_pct(average(returns_6h))}")
    print(f"Average 12h return: {fmt_pct(average(returns_12h))}")
    print(f"Average 24h return: {fmt_pct(average(returns_24h))}")

    if signals:
        best = max(signals, key=lambda s: s.return_after_24h_percent or float("-inf"))
        worst = min(signals, key=lambda s: s.return_after_24h_percent or float("inf"))
        print(
            f"\nBest Scout signal: {best.symbol} at {best.search_time_kst} KST "
            f"(24h {fmt_pct(best.return_after_24h_percent)}, "
            f"max profit {best.max_profit_percent:+.2f}%)"
        )
        print(
            f"Worst Scout signal: {worst.symbol} at {worst.search_time_kst} KST "
            f"(24h {fmt_pct(worst.return_after_24h_percent)}, "
            f"max profit {worst.max_profit_percent:+.2f}%)"
        )

        meaningful_count = sum(1 for s in signals if s.meaningful_trend)
        print(
            f"\nMeaningful trend signals: {meaningful_count}/{len(signals)} "
            f"(max profit >= {MEANINGFUL_TREND_MIN_MAX_PROFIT_PCT:.0f}% within 24h)"
        )
    else:
        print("\nBest Scout signal: N/A")
        print("Worst Scout signal: N/A")

    print("==================================")


def main() -> None:
    try:
        search_times = get_search_times()
        eligible_symbols = get_eligible_symbols()

        if not eligible_symbols:
            print("Error: no eligible USDT perpetual symbols found.")
            return

        print("Scout v0.3 historical review starting.")
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
