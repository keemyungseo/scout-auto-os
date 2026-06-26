import csv
import json
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCAN_TIMES_KST = [
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
    "2026-06-18 05:00:00",
    "2026-06-18 07:00:00",
    "2026-06-18 09:00:00",
]

KST_TZ = timezone(timedelta(hours=9))

FUTURES_BASE_URL = "https://fapi.binance.com"
EXCHANGE_INFO_ENDPOINT = "/fapi/v1/exchangeInfo"
KLINES_ENDPOINT = "/fapi/v1/klines"

EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "XRPUSDT", "USDCUSDT"}
MIN_PRICE = 0.05
MAX_PRICE = 400.0

CANDLES_24H = 12
CANDLES_7D = 84
LOOKBACK_24 = 24
LOOKBACK_60 = 60
TOP_N = 3

RANKING_KLINES = CANDLES_24H + 1
ANALYSIS_KLINES = CANDLES_7D + LOOKBACK_60 + 5
FORWARD_HOURS = 24

STRONG_MAX_PROFIT_24H_PCT = 10.0
FAILED_RETURN_24H_PCT = 0.0

INTERVAL = "2h"
INTERVAL_MS = 2 * 60 * 60 * 1000
MAX_LIMIT = 1500
API_SLEEP_SEC = 0.03

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "top3_gainer_pattern_study.csv"


@dataclass
class TopGainerRecord:
    scan_time_kst: str
    scan_time_utc: str
    scan_dt: datetime
    rank: int
    symbol: str
    price_at_scan: float
    change_24h_percent: float
    trading_value_24h: float
    low_7d: float
    high_7d: float
    range_7d_percent: float
    distance_from_7d_low_percent: float
    distance_from_7d_high_percent: float
    position_in_7d_range_percent: float
    ma24: float
    ma24_change_rate_percent: float
    ma60: float
    ma60_change_rate_percent: float
    box_high_24: float
    box_low_24: float
    box_range_24_percent: float
    box_high_84: float
    box_low_84: float
    box_range_84_percent: float
    atr24_percent: float
    atr84_percent: float
    atr_compression_ratio: float
    current_2h_volume: float
    avg_volume_24: float
    avg_volume_84: float
    volume_ratio_24: float
    volume_ratio_84: float
    recent_6_candle_avg_volume: float
    previous_24_candle_avg_volume: float
    volume_acceleration_ratio: float
    current_candle_body_percent: float
    avg_body_24_percent: float
    body_expansion_ratio: float
    close_position_in_candle: float
    upper_wick_percent: float
    lower_wick_percent: float
    major_drawdown_before_rise: bool
    recent_7d_impulse_percent: float
    scan_price_vs_prior_7d_high_percent: float
    did_scan_break_7d_high_close: bool
    did_scan_break_24_high_close: bool
    green_candles_last_6: int
    green_candles_last_12: int
    cumulative_return_last_6_candles: float
    cumulative_return_previous_24_candles: float
    max_profit_after_6h: float | None
    max_profit_after_12h: float | None
    max_profit_after_24h: float | None
    return_after_6h: float | None
    return_after_12h: float | None
    return_after_24h: float | None
    max_drawdown_after_24h: float | None


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


def candle_open(kline: list) -> float:
    return float(kline[1])


def candle_high(kline: list) -> float:
    return float(kline[2])


def candle_low(kline: list) -> float:
    return float(kline[3])


def candle_close(kline: list) -> float:
    return float(kline[4])


def candle_volume(kline: list) -> float:
    return float(kline[5])


def candle_quote_volume(kline: list) -> float:
    return float(kline[7])


def body_percent(kline: list) -> float:
    open_price = candle_open(kline)
    if open_price == 0:
        return 0.0
    return abs(candle_close(kline) - open_price) / open_price * 100


def is_green(kline: list) -> bool:
    return candle_close(kline) > candle_open(kline)


def cumulative_return_percent(candles: list[list]) -> float:
    total = 0.0
    for kline in candles:
        open_price = candle_open(kline)
        if open_price == 0:
            continue
        total += (candle_close(kline) - open_price) / open_price * 100
    return total


def true_range(kline: list, prev_close: float) -> float:
    high_price = candle_high(kline)
    low_price = candle_low(kline)
    return max(
        high_price - low_price,
        abs(high_price - prev_close),
        abs(low_price - prev_close),
    )


def average_true_range_percent(candles: list[list], reference_price: float) -> float:
    if len(candles) < 2 or reference_price == 0:
        return 0.0

    ranges: list[float] = []
    for index, kline in enumerate(candles):
        prev_close = candle_close(candles[index - 1]) if index > 0 else candle_open(kline)
        ranges.append(true_range(kline, prev_close))

    atr = sum(ranges) / len(ranges)
    return atr / reference_price * 100


def wick_percents(kline: list) -> tuple[float, float, float]:
    open_price = candle_open(kline)
    close_price = candle_close(kline)
    high_price = candle_high(kline)
    low_price = candle_low(kline)

    if open_price == 0:
        return 0.0, 0.0, 0.5

    body_top = max(open_price, close_price)
    body_bottom = min(open_price, close_price)
    upper_wick = (high_price - body_top) / open_price * 100
    lower_wick = (body_bottom - low_price) / open_price * 100

    if high_price == low_price:
        close_position = 0.5
    else:
        close_position = (close_price - low_price) / (high_price - low_price)

    return upper_wick, lower_wick, close_position


def ma_change_rate_percent(recent_closes: list[float], prior_closes: list[float]) -> float:
    recent_ma = sum(recent_closes) / len(recent_closes)
    prior_ma = sum(prior_closes) / len(prior_closes)
    if prior_ma == 0:
        return 0.0
    return abs(recent_ma - prior_ma) / prior_ma * 100


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

    scan = klines[-1]
    price = candle_close(scan)

    if not (MIN_PRICE <= price <= MAX_PRICE):
        return None

    close_24h_ago = candle_close(klines[-(CANDLES_24H + 1)])
    if close_24h_ago == 0:
        return None

    change_24h = (price - close_24h_ago) / close_24h_ago * 100
    trading_value_24h = sum(candle_quote_volume(k) for k in klines[-CANDLES_24H:])

    return {
        "price_at_scan": price,
        "change_24h_percent": change_24h,
        "trading_value_24h": trading_value_24h,
    }


def measure_forward_outcomes(symbol: str, scan_dt: datetime, price_at_scan: float) -> dict:
    scan_end_ms = int(scan_dt.timestamp() * 1000)
    forward_end_ms = scan_end_ms + FORWARD_HOURS * 60 * 60 * 1000
    forward_klines = fetch_klines_forward(symbol, scan_end_ms, forward_end_ms)

    def window_stats(hours: int) -> tuple[float | None, float | None]:
        target = scan_dt + timedelta(hours=hours)
        max_high = price_at_scan
        min_low = price_at_scan

        for kline in forward_klines:
            if kline_close_dt(kline) > target:
                break
            max_high = max(max_high, candle_high(kline))
            min_low = min(min_low, candle_low(kline))

        max_profit = (max_high - price_at_scan) / price_at_scan * 100
        close_price = get_close_at_or_before(forward_klines, target)
        if close_price is None:
            return max_profit, None
        return max_profit, (close_price - price_at_scan) / price_at_scan * 100

    max_profit_6h, return_6h = window_stats(6)
    max_profit_12h, return_12h = window_stats(12)
    max_profit_24h, return_24h = window_stats(24)

    min_low_24h = price_at_scan
    for kline in forward_klines:
        if kline_close_dt(kline) > scan_dt + timedelta(hours=24):
            break
        min_low_24h = min(min_low_24h, candle_low(kline))

    max_drawdown_24h = (price_at_scan - min_low_24h) / price_at_scan * 100

    return {
        "max_profit_after_6h": max_profit_6h,
        "max_profit_after_12h": max_profit_12h,
        "max_profit_after_24h": max_profit_24h,
        "return_after_6h": return_6h,
        "return_after_12h": return_12h,
        "return_after_24h": return_24h,
        "max_drawdown_after_24h": max_drawdown_24h,
    }


def analyze_top_gainer(
    symbol: str,
    scan_kst: str,
    scan_utc: str,
    scan_dt: datetime,
    rank: int,
    ranking: dict,
    klines: list[list],
) -> TopGainerRecord | None:
    if len(klines) < ANALYSIS_KLINES:
        return None

    scan = klines[-1]
    prev_24 = klines[-(LOOKBACK_24 + 1) : -1]
    prior_24 = klines[-(LOOKBACK_24 * 2 + 1) : -(LOOKBACK_24 + 1)]
    prev_60 = klines[-(LOOKBACK_60 + 1) : -1]
    prior_60 = klines[-(LOOKBACK_60 * 2 + 1) : -(LOOKBACK_60 + 1)]
    prev_84 = klines[-(CANDLES_7D + 1) : -1]
    last_6 = klines[-6:]
    last_12 = klines[-12:]
    prev_24_before_6 = klines[-30:-6]

    price = ranking["price_at_scan"]
    prev_24_closes = [candle_close(k) for k in prev_24]
    prior_24_closes = [candle_close(k) for k in prior_24]
    prev_60_closes = [candle_close(k) for k in prev_60]
    prior_60_closes = [candle_close(k) for k in prior_60]

    ma24 = sum(prev_24_closes) / LOOKBACK_24
    ma60 = sum(prev_60_closes) / LOOKBACK_60
    if ma24 == 0 or ma60 == 0:
        return None

    low_7d = min(candle_low(k) for k in prev_84)
    high_7d = max(candle_high(k) for k in prev_84)
    high_7d_close = max(candle_close(k) for k in prev_84)
    high_24_close = max(prev_24_closes)

    if low_7d == 0:
        return None

    range_7d = (high_7d - low_7d) / low_7d * 100
    distance_from_low = (price - low_7d) / low_7d * 100
    distance_from_high = (price - high_7d) / high_7d * 100

    if high_7d == low_7d:
        position_in_range = 50.0
    else:
        position_in_range = (price - low_7d) / (high_7d - low_7d) * 100

    box_high_24 = max(candle_high(k) for k in prev_24)
    box_low_24 = min(candle_low(k) for k in prev_24)
    box_high_84 = max(candle_high(k) for k in prev_84)
    box_low_84 = min(candle_low(k) for k in prev_84)

    box_range_24 = (box_high_24 - box_low_24) / ma24 * 100
    box_range_84 = (box_high_84 - box_low_84) / ma60 * 100

    atr24 = average_true_range_percent(klines[-LOOKBACK_24:], price)
    atr84 = average_true_range_percent(klines[-CANDLES_7D:], price)
    atr_compression = atr24 / atr84 if atr84 > 0 else 0.0

    current_volume = candle_volume(scan)
    avg_volume_24 = sum(candle_volume(k) for k in prev_24) / LOOKBACK_24
    avg_volume_84 = sum(candle_volume(k) for k in prev_84) / CANDLES_7D
    recent_6_avg_volume = sum(candle_volume(k) for k in last_6) / len(last_6)
    previous_24_avg_volume = (
        sum(candle_volume(k) for k in prev_24_before_6) / len(prev_24_before_6)
        if prev_24_before_6
        else 0.0
    )

    volume_ratio_24 = current_volume / avg_volume_24 if avg_volume_24 > 0 else 0.0
    volume_ratio_84 = current_volume / avg_volume_84 if avg_volume_84 > 0 else 0.0
    volume_accel = (
        recent_6_avg_volume / previous_24_avg_volume
        if previous_24_avg_volume > 0
        else 0.0
    )

    current_body = body_percent(scan)
    avg_body_24 = sum(body_percent(k) for k in prev_24) / LOOKBACK_24
    body_expansion = current_body / avg_body_24 if avg_body_24 > 0 else 0.0
    upper_wick, lower_wick, close_position = wick_percents(scan)

    impulse_percent = range_7d
    scan_vs_7d_high = (price - high_7d_close) / high_7d_close * 100
    break_7d_high_close = price > high_7d_close
    break_24_high_close = price > high_24_close

    major_drawdown = (
        range_7d >= 12.0
        and distance_from_low >= 20.0
        and position_in_range >= 40.0
    )

    forward = measure_forward_outcomes(symbol, scan_dt, price)

    return TopGainerRecord(
        scan_time_kst=scan_kst,
        scan_time_utc=scan_utc,
        scan_dt=scan_dt,
        rank=rank,
        symbol=symbol,
        price_at_scan=price,
        change_24h_percent=ranking["change_24h_percent"],
        trading_value_24h=ranking["trading_value_24h"],
        low_7d=low_7d,
        high_7d=high_7d,
        range_7d_percent=range_7d,
        distance_from_7d_low_percent=distance_from_low,
        distance_from_7d_high_percent=distance_from_high,
        position_in_7d_range_percent=position_in_range,
        ma24=ma24,
        ma24_change_rate_percent=ma_change_rate_percent(prev_24_closes, prior_24_closes),
        ma60=ma60,
        ma60_change_rate_percent=ma_change_rate_percent(prev_60_closes, prior_60_closes),
        box_high_24=box_high_24,
        box_low_24=box_low_24,
        box_range_24_percent=box_range_24,
        box_high_84=box_high_84,
        box_low_84=box_low_84,
        box_range_84_percent=box_range_84,
        atr24_percent=atr24,
        atr84_percent=atr84,
        atr_compression_ratio=atr_compression,
        current_2h_volume=current_volume,
        avg_volume_24=avg_volume_24,
        avg_volume_84=avg_volume_84,
        volume_ratio_24=volume_ratio_24,
        volume_ratio_84=volume_ratio_84,
        recent_6_candle_avg_volume=recent_6_avg_volume,
        previous_24_candle_avg_volume=previous_24_avg_volume,
        volume_acceleration_ratio=volume_accel,
        current_candle_body_percent=current_body,
        avg_body_24_percent=avg_body_24,
        body_expansion_ratio=body_expansion,
        close_position_in_candle=close_position,
        upper_wick_percent=upper_wick,
        lower_wick_percent=lower_wick,
        major_drawdown_before_rise=major_drawdown,
        recent_7d_impulse_percent=impulse_percent,
        scan_price_vs_prior_7d_high_percent=scan_vs_7d_high,
        did_scan_break_7d_high_close=break_7d_high_close,
        did_scan_break_24_high_close=break_24_high_close,
        green_candles_last_6=sum(1 for k in last_6 if is_green(k)),
        green_candles_last_12=sum(1 for k in last_12 if is_green(k)),
        cumulative_return_last_6_candles=cumulative_return_percent(last_6),
        cumulative_return_previous_24_candles=cumulative_return_percent(prev_24),
        max_profit_after_6h=forward["max_profit_after_6h"],
        max_profit_after_12h=forward["max_profit_after_12h"],
        max_profit_after_24h=forward["max_profit_after_24h"],
        return_after_6h=forward["return_after_6h"],
        return_after_12h=forward["return_after_12h"],
        return_after_24h=forward["return_after_24h"],
        max_drawdown_after_24h=forward["max_drawdown_after_24h"],
    )


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def fmt_bool(value: bool) -> str:
    return "YES" if value else "NO"


def record_to_csv_row(record: TopGainerRecord) -> dict[str, str]:
    row: dict[str, str] = {}
    for field in fields(record):
        value = getattr(record, field.name)
        if field.name == "scan_dt":
            continue
        if isinstance(value, bool):
            row[field.name] = fmt_bool(value)
        elif isinstance(value, float):
            row[field.name] = f"{value:.8f}" if "price" in field.name or field.name.endswith("_low") or field.name.endswith("_high") or field.name.startswith("box_") or field.name.startswith("ma") else f"{value:.4f}"
        elif value is None:
            row[field.name] = ""
        else:
            row[field.name] = str(value)
    return row


def save_results(records: list[TopGainerRecord]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in fields(TopGainerRecord) if f.name != "scan_dt"]
    rows = [record_to_csv_row(record) for record in records]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_scan_summary(records: list[TopGainerRecord]) -> None:
    print("\n===== TOP3 GAINER PATTERN STUDY =====")

    for scan_kst, scan_utc, scan_dt in get_scan_times():
        matches = [r for r in records if r.scan_dt == scan_dt]
        matches.sort(key=lambda item: item.rank)

        print(f"\nScan time: {scan_kst} KST ({scan_utc})")
        if not matches:
            print("  No top gainers recorded.")
            continue

        for match in matches:
            print(
                f"  #{match.rank} {match.symbol} | 24h {match.change_24h_percent:+.2f}% | "
                f"pos7d {match.position_in_7d_range_percent:.1f}% | "
                f"brk24 {fmt_bool(match.did_scan_break_24_high_close)} | "
                f"vol24x {match.volume_ratio_24:.2f} | "
                f"maxP24h {fmt_pct(match.max_profit_after_24h)} | "
                f"ret24h {fmt_pct(match.return_after_24h)}"
            )


def numeric_metric_names() -> list[str]:
    skip = {
        "scan_time_kst",
        "scan_time_utc",
        "scan_dt",
        "rank",
        "symbol",
        "major_drawdown_before_rise",
        "did_scan_break_7d_high_close",
        "did_scan_break_24_high_close",
        "green_candles_last_6",
        "green_candles_last_12",
    }
    numeric_types = (int, float)
    names: list[str] = []
    for field in fields(TopGainerRecord):
        if field.name in skip:
            continue
        field_type = field.type
        if field_type in numeric_types:
            names.append(field.name)
        elif isinstance(field_type, str) and "float" in field_type:
            names.append(field.name)
    return names


def bool_metric_names() -> list[str]:
    return [
        "major_drawdown_before_rise",
        "did_scan_break_7d_high_close",
        "did_scan_break_24_high_close",
    ]


def group_records(records: list[TopGainerRecord]) -> tuple[list[TopGainerRecord], list[TopGainerRecord], list[TopGainerRecord]]:
    strong: list[TopGainerRecord] = []
    failed: list[TopGainerRecord] = []
    middle: list[TopGainerRecord] = []

    for record in records:
        max_profit_24h = record.max_profit_after_24h or 0.0
        return_24h = record.return_after_24h

        if max_profit_24h >= STRONG_MAX_PROFIT_24H_PCT:
            strong.append(record)
        elif return_24h is not None and return_24h < FAILED_RETURN_24H_PCT:
            failed.append(record)
        else:
            middle.append(record)

    return strong, failed, middle


def mean_metric(records: list[TopGainerRecord], metric: str) -> float | None:
    values = [getattr(record, metric) for record in records]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return statistics.mean(values)


def bool_rate(records: list[TopGainerRecord], metric: str) -> float | None:
    if not records:
        return None
    return sum(1 for record in records if getattr(record, metric)) / len(records) * 100


def print_learning_summary(records: list[TopGainerRecord]) -> None:
    print("\n===== LEARNING SUMMARY (DATA-DRIVEN) =====")

    if not records:
        print("No records to analyze.")
        return

    strong, failed, middle = group_records(records)

    print(f"\nSample split (this run only):")
    print(f"  Strong follow-through: {len(strong)} (max profit 24h >= {STRONG_MAX_PROFIT_24H_PCT:.0f}%)")
    print(f"  Failed top gainers: {len(failed)} (return 24h < {FAILED_RETURN_24H_PCT:.0f}%)")
    print(f"  Middle / mixed: {len(middle)}")

    if not strong or not failed:
        print("\nNot enough strong vs failed cases in this sample for metric comparison.")
        print("Re-run with more scan times or relax grouping thresholds for exploration.")
        return

    metric_gaps: list[tuple[str, float, float, float]] = []

    for metric in numeric_metric_names():
        strong_mean = mean_metric(strong, metric)
        failed_mean = mean_metric(failed, metric)
        if strong_mean is None or failed_mean is None:
            continue
        gap = strong_mean - failed_mean
        denominator = abs(failed_mean) if failed_mean != 0 else 1.0
        relative_gap = abs(gap) / denominator
        metric_gaps.append((metric, strong_mean, failed_mean, relative_gap))

    metric_gaps.sort(key=lambda item: item[3], reverse=True)

    print("\nLargest numeric gaps (strong mean vs failed mean):")
    for metric, strong_mean, failed_mean, relative_gap in metric_gaps[:10]:
        print(
            f"  {metric}: strong {strong_mean:.4f} | failed {failed_mean:.4f} | "
            f"gap {strong_mean - failed_mean:+.4f} | rel {relative_gap:.2f}x"
        )

    print("\nSmallest numeric gaps (possibly weak or misleading signals):")
    for metric, strong_mean, failed_mean, relative_gap in metric_gaps[-5:]:
        print(
            f"  {metric}: strong {strong_mean:.4f} | failed {failed_mean:.4f} | "
            f"gap {strong_mean - failed_mean:+.4f} | rel {relative_gap:.2f}x"
        )

    print("\nBoolean feature rates:")
    for metric in bool_metric_names():
        strong_rate = bool_rate(strong, metric)
        failed_rate = bool_rate(failed, metric)
        if strong_rate is None or failed_rate is None:
            continue
        print(
            f"  {metric}: strong {strong_rate:.1f}% YES | failed {failed_rate:.1f}% YES"
        )

    print("\nObserved patterns (descriptive only, not trading rules):")
    strong_break_24 = bool_rate(strong, "did_scan_break_24_high_close")
    failed_break_24 = bool_rate(failed, "did_scan_break_24_high_close")
    if strong_break_24 is not None and failed_break_24 is not None:
        print(
            f"  24h high-close breakout rate: strong {strong_break_24:.1f}% vs failed {failed_break_24:.1f}%"
        )

    strong_vol = mean_metric(strong, "volume_ratio_24")
    failed_vol = mean_metric(failed, "volume_ratio_24")
    if strong_vol is not None and failed_vol is not None:
        print(f"  Volume ratio 24 mean: strong {strong_vol:.2f}x vs failed {failed_vol:.2f}x")

    strong_pos = mean_metric(strong, "position_in_7d_range_percent")
    failed_pos = mean_metric(failed, "position_in_7d_range_percent")
    if strong_pos is not None and failed_pos is not None:
        print(f"  Position in 7d range mean: strong {strong_pos:.1f}% vs failed {failed_pos:.1f}%")

    print("\nSuggested hypotheses to test next (from this sample):")
    if metric_gaps:
        top_metrics = [item[0] for item in metric_gaps[:5]]
        print(
            "  - Test whether top gainers with higher "
            + ", ".join(top_metrics[:3])
            + " continue better after the scan."
        )
    print("  - Separate 'already extended' top gainers from 'early breakout' top gainers using position_in_7d_range_percent and scan_price_vs_prior_7d_high_percent.")
    print("  - Compare volume_acceleration_ratio vs volume_ratio_24 as an early participation signal.")
    print("  - Check whether ATR_compression_ratio helps filter late-stage top gainers.")
    print("  - Re-run across more dates before converting any pattern into Scout conditions.")

    print("==========================================")


def main() -> None:
    try:
        scan_times = get_scan_times()
        eligible_symbols = get_eligible_symbols()

        if not eligible_symbols:
            print("Error: no eligible USDT perpetual symbols found.")
            return

        print("Top3 gainer pattern study starting.")
        print(f"Universe size: {len(eligible_symbols)} symbols")
        print(f"Scan times: {len(scan_times)}")
        print("Research only. No trading simulation.")

        all_records: list[TopGainerRecord] = []
        symbols = sorted(eligible_symbols)

        for scan_kst, scan_utc, scan_dt in scan_times:
            print(f"\nScanning {scan_kst} KST for top gainers...")
            end_ms = int(scan_dt.timestamp() * 1000)
            candidates: list[tuple[str, dict]] = []

            for index, symbol in enumerate(symbols, start=1):
                if index % 100 == 0 or index == len(symbols):
                    print(f"  ranking progress: {index}/{len(symbols)}")

                try:
                    klines = fetch_klines_before(symbol, end_ms, RANKING_KLINES)
                    ranking = compute_24h_ranking(klines)
                    if ranking is None:
                        continue
                    candidates.append((symbol, ranking))
                except urllib.error.HTTPError:
                    continue

                time.sleep(API_SLEEP_SEC)

            candidates.sort(key=lambda item: item[1]["change_24h_percent"], reverse=True)
            top_candidates = candidates[:TOP_N]

            print(f"  top {TOP_N} by 24h change:")
            for rank, (symbol, ranking) in enumerate(top_candidates, start=1):
                print(
                    f"    #{rank} {symbol} {ranking['change_24h_percent']:+.2f}% "
                    f"(24h value {ranking['trading_value_24h']:,.0f})"
                )

            for rank, (symbol, ranking) in enumerate(top_candidates, start=1):
                try:
                    klines = fetch_klines_before(symbol, end_ms, ANALYSIS_KLINES)
                    record = analyze_top_gainer(
                        symbol,
                        scan_kst,
                        scan_utc,
                        scan_dt,
                        rank,
                        ranking,
                        klines,
                    )
                    if record is not None:
                        all_records.append(record)
                except urllib.error.HTTPError:
                    continue

                time.sleep(API_SLEEP_SEC)

        save_results(all_records)
        print_scan_summary(all_records)
        print_learning_summary(all_records)
        print(f"\nFull results saved: {OUTPUT_CSV}")
        print(f"Total top-gainer records: {len(all_records)}")

    except ValueError as exc:
        print(f"Error: {exc}")
    except urllib.error.HTTPError as exc:
        details = parse_error_message(exc.read().decode("utf-8", errors="replace"))
        print(f"Error: Binance API request failed. HTTP {exc.code}: {details}")
    except urllib.error.URLError as exc:
        print(f"Error: cannot connect to Binance. {exc.reason}")


if __name__ == "__main__":
    main()
