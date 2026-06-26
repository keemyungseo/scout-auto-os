import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEARCH_TIMES = [
    ("2026-06-16 09:00:00", "2026-06-16 00:00:00"),
    ("2026-06-16 11:00:00", "2026-06-16 02:00:00"),
    ("2026-06-16 13:00:00", "2026-06-16 04:00:00"),
    ("2026-06-16 15:00:00", "2026-06-16 06:00:00"),
    ("2026-06-16 17:00:00", "2026-06-16 08:00:00"),
    ("2026-06-16 19:00:00", "2026-06-16 10:00:00"),
    ("2026-06-16 21:00:00", "2026-06-16 12:00:00"),
]

KST_OFFSET = timedelta(hours=9)
KST_TZ = timezone(KST_OFFSET)

FUTURES_BASE_URL = "https://fapi.binance.com"
EXCHANGE_INFO_ENDPOINT = "/fapi/v1/exchangeInfo"
KLINES_ENDPOINT = "/fapi/v1/klines"

EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "XRPUSDT", "USDCUSDT"}
MIN_PRICE = 0.05
MAX_PRICE = 400.0
BOX_LOOKBACK = 24
MAX_BOX_RANGE_PCT = 15.0
VOLUME_MULTIPLIER = 3.0
MAX_BREAKOUT_PCT = 5.0
EMERGENCY_STOP_MULTIPLIER = 0.975
FORWARD_HOURS = 24

INTERVAL = "2h"
INTERVAL_MS = 2 * 60 * 60 * 1000
KLINES_NEEDED = BOX_LOOKBACK + 1
MAX_LIMIT = 1500

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "scout_review.csv"


@dataclass
class ScoutMatch:
    search_time_kst: str
    search_time_utc: str
    search_dt: datetime
    symbol: str
    price_at_signal_close: float
    box_high: float
    box_low: float
    box_range_percent: float
    signal_2h_change_percent: float
    signal_volume: float
    average_volume_previous_24: float
    volume_ratio: float
    breakout_percent: float
    trading_value_2h: float
    entry_time_kst: str
    entry_time_utc: str
    entry_price: float
    max_high_24h: float
    max_profit_percent: float
    max_profit_before_stop_percent: float
    time_to_max_profit_kst: str
    time_to_max_profit_utc: str
    stop_occurred: bool
    stop_time_kst: str
    stop_time_utc: str
    lowest_price_after_entry: float
    max_adverse_percent: float
    close_after_6h: float | None
    return_after_6h_percent: float | None
    close_after_12h: float | None
    return_after_12h_percent: float | None
    close_after_24h: float | None
    return_after_24h_percent: float | None


def parse_error_message(body: str) -> str:
    try:
        data = json.loads(body)
        if isinstance(data, dict) and data.get("msg"):
            return str(data["msg"])
    except json.JSONDecodeError:
        pass
    return body.strip() or "알 수 없는 오류"


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


def get_search_times_utc() -> list[tuple[str, str, datetime]]:
    times: list[tuple[str, str, datetime]] = []
    for kst_str, utc_str in SEARCH_TIMES:
        search_dt = parse_kst_to_utc(kst_str)
        expected = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
        if search_dt != expected:
            raise ValueError(
                f"Search time conversion mismatch for {kst_str}: "
                f"got {format_time_utc(search_dt)}, expected {utc_str}"
            )
        times.append((kst_str, utc_str, search_dt))
    return times


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


def evaluate_scout_at_search(
    symbol: str,
    search_dt: datetime,
) -> dict | None:
    end_ms = int(search_dt.timestamp() * 1000)
    klines = fetch_klines_before(symbol, end_ms, KLINES_NEEDED)

    if len(klines) < KLINES_NEEDED:
        return None

    signal = klines[-1]
    box_candles = klines[-(BOX_LOOKBACK + 1) : -1]

    signal_close = float(signal[4])
    if not (MIN_PRICE <= signal_close <= MAX_PRICE):
        return None

    box_high = max(float(candle[2]) for candle in box_candles)
    box_low = min(float(candle[3]) for candle in box_candles)
    if box_low == 0:
        return None

    box_range_percent = (box_high - box_low) / box_low * 100
    if box_range_percent > MAX_BOX_RANGE_PCT:
        return None

    if signal_close <= box_high:
        return None

    breakout_percent = (signal_close - box_high) / box_high * 100
    if breakout_percent > MAX_BREAKOUT_PCT:
        return None

    prior_volumes = [float(candle[5]) for candle in box_candles]
    average_volume = sum(prior_volumes) / BOX_LOOKBACK
    signal_volume = float(signal[5])

    if average_volume == 0 or signal_volume < VOLUME_MULTIPLIER * average_volume:
        return None

    signal_open = float(signal[1])
    if signal_open == 0:
        return None

    signal_2h_change_percent = (signal_close - signal_open) / signal_open * 100

    return {
        "signal_kline": signal,
        "price_at_signal_close": signal_close,
        "box_high": box_high,
        "box_low": box_low,
        "box_range_percent": box_range_percent,
        "signal_2h_change_percent": signal_2h_change_percent,
        "signal_volume": signal_volume,
        "average_volume_previous_24": average_volume,
        "volume_ratio": signal_volume / average_volume,
        "breakout_percent": breakout_percent,
        "trading_value_2h": float(signal[7]),
    }


def measure_forward_trend(
    symbol: str,
    search_dt: datetime,
    scout_data: dict,
) -> dict | None:
    signal_kline = scout_data["signal_kline"]
    signal_open_ms = int(signal_kline[0])
    entry_start_ms = signal_open_ms + INTERVAL_MS
    entry_end_ms = entry_start_ms + FORWARD_HOURS * 60 * 60 * 1000

    forward_klines = fetch_klines_forward(symbol, entry_start_ms, entry_end_ms)
    if not forward_klines:
        return None

    entry_kline = forward_klines[0]
    entry_price = float(entry_kline[1])
    entry_dt = kline_open_dt(entry_kline)
    stop_price = entry_price * EMERGENCY_STOP_MULTIPLIER

    max_high = entry_price
    max_profit_percent = 0.0
    max_profit_before_stop_percent = 0.0
    max_profit_dt = entry_dt
    stop_occurred = False
    stop_dt: datetime | None = None
    lowest_price = entry_price
    max_adverse_percent = 0.0

    for kline in forward_klines:
        high_price = float(kline[2])
        low_price = float(kline[3])
        open_dt = kline_open_dt(kline)

        lowest_price = min(lowest_price, low_price)
        max_adverse_percent = max(
            max_adverse_percent,
            (entry_price - low_price) / entry_price * 100,
        )

        if high_price > max_high:
            max_high = high_price
            max_profit_percent = (max_high - entry_price) / entry_price * 100
            max_profit_dt = open_dt

        if not stop_occurred:
            if high_price > entry_price:
                candidate = (high_price - entry_price) / entry_price * 100
                max_profit_before_stop_percent = max(
                    max_profit_before_stop_percent,
                    candidate,
                )

            if low_price <= stop_price:
                stop_occurred = True
                stop_dt = open_dt

    close_6h = get_close_at_or_before(
        forward_klines, entry_dt + timedelta(hours=6)
    )
    close_12h = get_close_at_or_before(
        forward_klines, entry_dt + timedelta(hours=12)
    )
    close_24h = get_close_at_or_before(
        forward_klines, entry_dt + timedelta(hours=24)
    )

    def return_pct(close: float | None) -> float | None:
        if close is None or entry_price == 0:
            return None
        return (close - entry_price) / entry_price * 100

    return {
        "entry_time_kst": format_time_kst(entry_dt),
        "entry_time_utc": format_time_utc(entry_dt),
        "entry_price": entry_price,
        "max_high_24h": max_high,
        "max_profit_percent": max_profit_percent,
        "max_profit_before_stop_percent": max_profit_before_stop_percent,
        "time_to_max_profit_kst": format_time_kst(max_profit_dt),
        "time_to_max_profit_utc": format_time_utc(max_profit_dt),
        "stop_occurred": stop_occurred,
        "stop_time_kst": format_time_kst(stop_dt) if stop_dt else "",
        "stop_time_utc": format_time_utc(stop_dt) if stop_dt else "",
        "lowest_price_after_entry": lowest_price,
        "max_adverse_percent": max_adverse_percent,
        "close_after_6h": close_6h,
        "return_after_6h_percent": return_pct(close_6h),
        "close_after_12h": close_12h,
        "return_after_12h_percent": return_pct(close_12h),
        "close_after_24h": close_24h,
        "return_after_24h_percent": return_pct(close_24h),
    }


def build_scout_match(
    search_kst: str,
    search_utc: str,
    search_dt: datetime,
    symbol: str,
    scout_data: dict,
    forward_data: dict,
) -> ScoutMatch:
    return ScoutMatch(
        search_time_kst=search_kst,
        search_time_utc=search_utc,
        search_dt=search_dt,
        symbol=symbol,
        price_at_signal_close=scout_data["price_at_signal_close"],
        box_high=scout_data["box_high"],
        box_low=scout_data["box_low"],
        box_range_percent=scout_data["box_range_percent"],
        signal_2h_change_percent=scout_data["signal_2h_change_percent"],
        signal_volume=scout_data["signal_volume"],
        average_volume_previous_24=scout_data["average_volume_previous_24"],
        volume_ratio=scout_data["volume_ratio"],
        breakout_percent=scout_data["breakout_percent"],
        trading_value_2h=scout_data["trading_value_2h"],
        entry_time_kst=forward_data["entry_time_kst"],
        entry_time_utc=forward_data["entry_time_utc"],
        entry_price=forward_data["entry_price"],
        max_high_24h=forward_data["max_high_24h"],
        max_profit_percent=forward_data["max_profit_percent"],
        max_profit_before_stop_percent=forward_data["max_profit_before_stop_percent"],
        time_to_max_profit_kst=forward_data["time_to_max_profit_kst"],
        time_to_max_profit_utc=forward_data["time_to_max_profit_utc"],
        stop_occurred=forward_data["stop_occurred"],
        stop_time_kst=forward_data["stop_time_kst"],
        stop_time_utc=forward_data["stop_time_utc"],
        lowest_price_after_entry=forward_data["lowest_price_after_entry"],
        max_adverse_percent=forward_data["max_adverse_percent"],
        close_after_6h=forward_data["close_after_6h"],
        return_after_6h_percent=forward_data["return_after_6h_percent"],
        close_after_12h=forward_data["close_after_12h"],
        return_after_12h_percent=forward_data["return_after_12h_percent"],
        close_after_24h=forward_data["close_after_24h"],
        return_after_24h_percent=forward_data["return_after_24h_percent"],
    )


def save_results(matches: list[ScoutMatch]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "search_time_kst",
        "search_time_utc",
        "symbol",
        "price_at_signal_close",
        "box_high",
        "box_low",
        "box_range_percent",
        "signal_2h_change_percent",
        "signal_volume",
        "average_volume_previous_24",
        "volume_ratio",
        "breakout_percent",
        "trading_value_2h",
        "entry_time_kst",
        "entry_time_utc",
        "entry_price",
        "max_high_24h",
        "max_profit_percent",
        "max_profit_before_stop_percent",
        "time_to_max_profit_kst",
        "time_to_max_profit_utc",
        "stop_occurred",
        "stop_time_kst",
        "stop_time_utc",
        "lowest_price_after_entry",
        "max_adverse_percent",
        "close_after_6h",
        "return_after_6h_percent",
        "close_after_12h",
        "return_after_12h_percent",
        "close_after_24h",
        "return_after_24h_percent",
    ]

    rows: list[dict[str, str]] = []
    for match in matches:
        rows.append(
            {
                "search_time_kst": match.search_time_kst,
                "search_time_utc": match.search_time_utc,
                "symbol": match.symbol,
                "price_at_signal_close": f"{match.price_at_signal_close:.8f}",
                "box_high": f"{match.box_high:.8f}",
                "box_low": f"{match.box_low:.8f}",
                "box_range_percent": f"{match.box_range_percent:.4f}",
                "signal_2h_change_percent": f"{match.signal_2h_change_percent:.4f}",
                "signal_volume": f"{match.signal_volume:.4f}",
                "average_volume_previous_24": f"{match.average_volume_previous_24:.4f}",
                "volume_ratio": f"{match.volume_ratio:.4f}",
                "breakout_percent": f"{match.breakout_percent:.4f}",
                "trading_value_2h": f"{match.trading_value_2h:.2f}",
                "entry_time_kst": match.entry_time_kst,
                "entry_time_utc": match.entry_time_utc,
                "entry_price": f"{match.entry_price:.8f}",
                "max_high_24h": f"{match.max_high_24h:.8f}",
                "max_profit_percent": f"{match.max_profit_percent:.4f}",
                "max_profit_before_stop_percent": (
                    f"{match.max_profit_before_stop_percent:.4f}"
                ),
                "time_to_max_profit_kst": match.time_to_max_profit_kst,
                "time_to_max_profit_utc": match.time_to_max_profit_utc,
                "stop_occurred": "YES" if match.stop_occurred else "NO",
                "stop_time_kst": match.stop_time_kst,
                "stop_time_utc": match.stop_time_utc,
                "lowest_price_after_entry": f"{match.lowest_price_after_entry:.8f}",
                "max_adverse_percent": f"{match.max_adverse_percent:.4f}",
                "close_after_6h": (
                    f"{match.close_after_6h:.8f}" if match.close_after_6h is not None else ""
                ),
                "return_after_6h_percent": (
                    f"{match.return_after_6h_percent:.4f}"
                    if match.return_after_6h_percent is not None
                    else ""
                ),
                "close_after_12h": (
                    f"{match.close_after_12h:.8f}"
                    if match.close_after_12h is not None
                    else ""
                ),
                "return_after_12h_percent": (
                    f"{match.return_after_12h_percent:.4f}"
                    if match.return_after_12h_percent is not None
                    else ""
                ),
                "close_after_24h": (
                    f"{match.close_after_24h:.8f}"
                    if match.close_after_24h is not None
                    else ""
                ),
                "return_after_24h_percent": (
                    f"{match.return_after_24h_percent:.4f}"
                    if match.return_after_24h_percent is not None
                    else ""
                ),
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


def print_scout_report(all_matches: list[ScoutMatch]) -> None:
    print("\n===== SCOUT REVIEW =====")

    search_times = get_search_times_utc()
    for search_kst, search_utc, search_dt in search_times:
        matches = [
            match
            for match in all_matches
            if match.search_dt == search_dt
        ]
        matches.sort(key=lambda item: item.trading_value_2h, reverse=True)

        print(f"\nSearch time KST: {search_kst}")
        print(f"Search time UTC: {search_utc}")
        print(f"Number of matches: {len(matches)}")

        if not matches:
            print("No matches.")
            continue

        header = (
            f"{'Rank':>4} | {'Symbol':<12} | {'Box%':>6} | {'VolR':>6} | "
            f"{'Brk%':>6} | {'2h Value':>12} | {'MaxP%':>7} | {'Stop?':>5} | "
            f"{'Ret6h':>8} | {'Ret12h':>8} | {'Ret24h':>8}"
        )
        print(header)
        print("-" * len(header))

        for rank, match in enumerate(matches, start=1):
            print(
                f"{rank:>4} | {match.symbol:<12} | "
                f"{match.box_range_percent:>6.2f} | "
                f"{match.volume_ratio:>6.2f} | "
                f"{match.breakout_percent:>6.2f} | "
                f"{match.trading_value_2h:>12,.0f} | "
                f"{match.max_profit_percent:>+7.2f} | "
                f"{'YES' if match.stop_occurred else 'NO':>5} | "
                f"{fmt_pct(match.return_after_6h_percent):>8} | "
                f"{fmt_pct(match.return_after_12h_percent):>8} | "
                f"{fmt_pct(match.return_after_24h_percent):>8}"
            )

    print("\n==================================")


def main() -> None:
    try:
        search_times = get_search_times_utc()
        eligible_symbols = get_eligible_symbols()

        if not eligible_symbols:
            print("오류: TRADING 상태의 USDT 무기한 선물 심볼을 찾지 못했습니다.")
            return

        print("Scout historical review를 시작합니다.")
        print(f"Symbols in universe: {len(eligible_symbols)}")
        print(f"Search times: {len(search_times)}")
        print("Read-only public market data only. No orders.")

        all_matches: list[ScoutMatch] = []
        symbols = sorted(eligible_symbols)

        for search_kst, search_utc, search_dt in search_times:
            print(
                f"\nScanning search time {search_kst} KST "
                f"({search_utc} UTC)..."
            )
            time_matches = 0

            for index, symbol in enumerate(symbols, start=1):
                if index % 100 == 0 or index == len(symbols):
                    print(f"  progress: {index}/{len(symbols)}")

                try:
                    scout_data = evaluate_scout_at_search(symbol, search_dt)
                    if scout_data is None:
                        continue

                    forward_data = measure_forward_trend(
                        symbol, search_dt, scout_data
                    )
                    if forward_data is None:
                        continue

                    all_matches.append(
                        build_scout_match(
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

            print(f"  matches at this search time: {time_matches}")

        save_results(all_matches)
        print_scout_report(all_matches)
        print(f"\nDetailed CSV saved: {OUTPUT_CSV}")
        print(f"Total matches across all search times: {len(all_matches)}")

    except ValueError as exc:
        print(f"오류: {exc}")
    except urllib.error.HTTPError as exc:
        details = parse_error_message(exc.read().decode("utf-8", errors="replace"))
        print(
            "오류: Binance API 요청에 실패했습니다. "
            f"HTTP {exc.code}: {details}"
        )
    except urllib.error.URLError as exc:
        print(f"오류: Binance 서버에 연결할 수 없습니다. {exc.reason}")


if __name__ == "__main__":
    main()
