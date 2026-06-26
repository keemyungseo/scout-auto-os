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
    "2026-06-15 09:00:00",
    "2026-06-15 11:00:00",
    "2026-06-15 13:00:00",
    "2026-06-15 15:00:00",
    "2026-06-15 17:00:00",
    "2026-06-15 19:00:00",
    "2026-06-15 21:00:00",
    "2026-06-15 23:00:00",
]

KST_TZ = timezone(timedelta(hours=9))

FUTURES_BASE_URL = "https://fapi.binance.com"
EXCHANGE_INFO_ENDPOINT = "/fapi/v1/exchangeInfo"
KLINES_ENDPOINT = "/fapi/v1/klines"

EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "XRPUSDT", "USDCUSDT"}
MIN_PRICE = 0.05
MAX_PRICE = 400.0

TOP_N = 10
CANDLES_24H_2H = 12
CANDLES_48H_2H = 24
CANDLES_7D_2H = 84
CANDLES_7D_1H = 168

RANKING_KLINES_2H = CANDLES_24H_2H + 1
ANALYSIS_KLINES_2H = CANDLES_7D_2H + 24
ANALYSIS_KLINES_1H = CANDLES_7D_1H + 24
FORWARD_HOURS = 24

INTERVAL_2H = "2h"
INTERVAL_1H = "1h"
INTERVAL_2H_MS = 2 * 60 * 60 * 1000
MAX_LIMIT = 1500
API_SLEEP_SEC = 0.03

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "top10_gainer_learning_20260615.csv"


@dataclass
class Top10Record:
    scan_time_kst: str
    scan_time_utc: str
    scan_dt: datetime
    symbol: str
    rank_24h: int
    return_24h_percent: float
    group: str
    forward_2h: float | None
    forward_4h: float | None
    forward_6h: float | None
    forward_12h: float | None
    forward_24h: float | None
    max_profit: float
    max_drawdown: float
    position_24h_percent: float
    position_48h_percent: float
    position_7d_percent: float
    break_24h_highest_close: bool
    break_48h_highest_close: bool
    break_7d_highest_close: bool
    distance_ma6_percent: float
    distance_ma12_percent: float
    distance_ma24_percent: float
    distance_ma48_percent: float
    distance_ma84_percent: float
    ma24_slope_percent: float
    ma48_slope_percent: float
    ma84_slope_percent: float
    current_body_percent: float
    avg_body_24_percent: float
    body_expansion_ratio: float
    atr_percent: float
    atr_ratio: float
    range_expansion_ratio: float
    volume_current: float
    volume_ma6: float
    volume_ma12: float
    volume_ma24: float
    volume_ratio_ma24: float
    volume_acceleration_ratio: float
    volume_before_price: bool
    return_prev_2h_percent: float
    return_prev_4h_percent: float
    return_prev_6h_percent: float
    return_prev_12h_percent: float
    return_prev_24h_percent: float
    return_prev_48h_percent: float
    return_prev_7d_percent: float
    pre6_tight_range: bool
    pre6_body_compression: bool
    pre6_volatility_compression: bool
    pre6_volume_contraction: bool
    pre12_tight_range: bool
    pre12_body_compression: bool
    pre12_volatility_compression: bool
    pre12_volume_contraction: bool
    pre6_1h_tight_range: bool
    pre6_1h_volume_contraction: bool


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


def fetch_klines_before(symbol: str, interval: str, end_ms: int, limit: int) -> list[list]:
    params = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": interval,
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
                "interval": INTERVAL_2H,
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
        next_start = last_open + INTERVAL_2H_MS
        if next_start <= current_start:
            break
        current_start = next_start

    return all_klines


def kline_close_dt(kline: list) -> datetime:
    open_dt = datetime.fromtimestamp(int(kline[0]) / 1000, tz=timezone.utc)
    return open_dt + timedelta(hours=2)


def ohlcv(kline: list) -> tuple[float, float, float, float, float]:
    return (
        float(kline[1]),
        float(kline[2]),
        float(kline[3]),
        float(kline[4]),
        float(kline[5]),
    )


def body_percent(open_price: float, close_price: float) -> float:
    if open_price == 0:
        return 0.0
    return abs(close_price - open_price) / open_price * 100


def range_percent(open_price: float, high_price: float, low_price: float) -> float:
    if open_price == 0:
        return 0.0
    return (high_price - low_price) / open_price * 100


def position_in_range(price: float, low_price: float, high_price: float) -> float:
    if high_price == low_price:
        return 50.0
    return (price - low_price) / (high_price - low_price) * 100


def ma_from_values(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def distance_from_ma_percent(price: float, ma_value: float) -> float:
    if ma_value == 0:
        return 0.0
    return (price - ma_value) / ma_value * 100


def ma_slope_percent(klines: list[list], period: int, shift: int = 6) -> float:
    if len(klines) < period + shift:
        return 0.0

    recent = [float(candle[4]) for candle in klines[-period:]]
    prior = [float(candle[4]) for candle in klines[-(period + shift) : -shift]]
    recent_ma = ma_from_values(recent)
    prior_ma = ma_from_values(prior)
    if prior_ma == 0:
        return 0.0
    return (recent_ma - prior_ma) / prior_ma * 100


def close_return_percent(current_close: float, prior_close: float) -> float:
    if prior_close == 0:
        return 0.0
    return (current_close - prior_close) / prior_close * 100


def cumulative_return_percent(candles: list[list]) -> float:
    total = 0.0
    for candle in candles:
        open_price, _, _, close_price, _ = ohlcv(candle)
        if open_price == 0:
            continue
        total += (close_price - open_price) / open_price * 100
    return total


def avg_metric(candles: list[list], metric_fn) -> float:
    if not candles:
        return 0.0
    return sum(metric_fn(candle) for candle in candles) / len(candles)


def candle_body_metric(candle: list) -> float:
    open_price, _, _, close_price, _ = ohlcv(candle)
    return body_percent(open_price, close_price)


def candle_range_metric(candle: list) -> float:
    open_price, high_price, low_price, _, _ = ohlcv(candle)
    return range_percent(open_price, high_price, low_price)


def true_range(kline: list, prev_close: float) -> float:
    _, high_price, low_price, close_price, _ = ohlcv(kline)
    return max(
        high_price - low_price,
        abs(high_price - prev_close),
        abs(low_price - prev_close),
    )


def average_true_range_percent(candles: list[list], reference_price: float) -> float:
    if len(candles) < 2 or reference_price == 0:
        return 0.0

    ranges: list[float] = []
    for index, candle in enumerate(candles):
        prev_close = float(candles[index - 1][4]) if index > 0 else float(candle[1])
        ranges.append(true_range(candle, prev_close))

    return (sum(ranges) / len(ranges)) / reference_price * 100


def compression_flags(candles: list[list], window: int) -> dict[str, bool]:
    needed = window * 2
    if len(candles) < needed:
        return {
            "tight_range": False,
            "body_compression": False,
            "volatility_compression": False,
            "volume_contraction": False,
        }

    recent = candles[-window:]
    prior = candles[-(window * 2) : -window]

    range_recent = avg_metric(recent, candle_range_metric)
    range_prior = avg_metric(prior, candle_range_metric)
    body_recent = avg_metric(recent, candle_body_metric)
    body_prior = avg_metric(prior, candle_body_metric)
    vol_recent = avg_metric(recent, lambda c: ohlcv(c)[4])
    vol_prior = avg_metric(prior, lambda c: ohlcv(c)[4])
    atr_recent = average_true_range_percent(recent, float(recent[-1][4]))
    atr_prior = average_true_range_percent(prior, float(prior[-1][4]))

    return {
        "tight_range": range_prior > 0 and range_recent < range_prior * 0.9,
        "body_compression": body_prior > 0 and body_recent < body_prior * 0.9,
        "volatility_compression": atr_prior > 0 and atr_recent < atr_prior * 0.9,
        "volume_contraction": vol_prior > 0 and vol_recent < vol_prior * 0.9,
    }


def volume_before_price(candles: list[list]) -> bool:
    if len(candles) < 6:
        return False

    first_half = candles[-6:-3]
    second_half = candles[-3:]
    price_move_first = abs(cumulative_return_percent(first_half))
    price_move_second = abs(cumulative_return_percent(second_half))
    vol_first = avg_metric(first_half, lambda c: ohlcv(c)[4])
    vol_second = avg_metric(second_half, lambda c: ohlcv(c)[4])

    return vol_first > 0 and vol_second / vol_first >= 1.15 and price_move_second >= price_move_first


def get_close_at_or_before(candles: list[list], target: datetime) -> float | None:
    close_price: float | None = None
    for candle in candles:
        if kline_close_dt(candle) <= target:
            close_price = float(candle[4])
        else:
            break
    return close_price


def compute_24h_ranking(klines: list[list]) -> dict | None:
    if len(klines) < RANKING_KLINES_2H:
        return None

    scan = klines[-1]
    _, _, _, close_price, _ = ohlcv(scan)
    if not (MIN_PRICE <= close_price <= MAX_PRICE):
        return None

    close_24h_ago = float(klines[-(CANDLES_24H_2H + 1)][4])
    if close_24h_ago == 0:
        return None

    return {
        "price_at_scan": close_price,
        "return_24h_percent": (close_price - close_24h_ago) / close_24h_ago * 100,
    }


def classify_group(max_profit: float, forward_24h: float | None) -> str:
    if forward_24h is not None and forward_24h < 0:
        return "D_FAILED"
    if max_profit >= 20:
        return "A_STRONG"
    if max_profit >= 10:
        return "B_MODERATE"
    return "C_WEAK"


def measure_forward(symbol: str, scan_dt: datetime, entry_price: float) -> dict:
    scan_end_ms = int(scan_dt.timestamp() * 1000)
    forward_end_ms = scan_end_ms + FORWARD_HOURS * 60 * 60 * 1000
    forward_klines = fetch_klines_forward(symbol, scan_end_ms, forward_end_ms)

    max_high = entry_price
    min_low = entry_price

    for candle in forward_klines:
        if kline_close_dt(candle) > scan_dt + timedelta(hours=24):
            break
        _, high_price, low_price, _, _ = ohlcv(candle)
        max_high = max(max_high, high_price)
        min_low = min(min_low, low_price)

    max_profit = (max_high - entry_price) / entry_price * 100
    max_drawdown = (entry_price - min_low) / entry_price * 100

    def forward_return(hours: int) -> float | None:
        close_price = get_close_at_or_before(forward_klines, scan_dt + timedelta(hours=hours))
        if close_price is None:
            return None
        return (close_price - entry_price) / entry_price * 100

    return {
        "forward_2h": forward_return(2),
        "forward_4h": forward_return(4),
        "forward_6h": forward_return(6),
        "forward_12h": forward_return(12),
        "forward_24h": forward_return(24),
        "max_profit": max_profit,
        "max_drawdown": max_drawdown,
    }


def build_record(
    scan_kst: str,
    scan_utc: str,
    scan_dt: datetime,
    rank: int,
    symbol: str,
    ranking: dict,
    klines_2h: list[list],
    klines_1h: list[list],
    forward: dict,
) -> Top10Record | None:
    if len(klines_2h) < ANALYSIS_KLINES_2H:
        return None

    signal = klines_2h[-1]
    price = ranking["price_at_scan"]
    open_price, high_price, low_price, close_price, volume = ohlcv(signal)

    prev_84 = klines_2h[-(CANDLES_7D_2H + 1) : -1]
    prev_48 = klines_2h[-(CANDLES_48H_2H + 1) : -1]
    prev_24 = klines_2h[-(CANDLES_24H_2H + 1) : -1]
    closes_before = [float(candle[4]) for candle in klines_2h[:-1]]

    low_7d = min(ohlcv(candle)[2] for candle in prev_84)
    high_7d = max(ohlcv(candle)[1] for candle in prev_84)
    low_48h = min(ohlcv(candle)[2] for candle in prev_48)
    low_24h = min(ohlcv(candle)[2] for candle in prev_24)
    high_48h = max(ohlcv(candle)[1] for candle in prev_48)
    high_24h = max(ohlcv(candle)[1] for candle in prev_24)

    def ma_distance(period: int) -> float:
        if len(closes_before) < period:
            return 0.0
        return distance_from_ma_percent(price, ma_from_values(closes_before[-period:]))

    def volume_ma(period: int) -> float:
        if len(klines_2h) < period + 1:
            return 0.0
        volumes = [ohlcv(candle)[4] for candle in klines_2h[-(period + 1) : -1]]
        return ma_from_values(volumes)

    vol_ma6 = volume_ma(6)
    vol_ma12 = volume_ma(12)
    vol_ma24 = volume_ma(24)

    avg_body_24 = avg_metric(prev_24, candle_body_metric)
    current_body = body_percent(open_price, close_price)
    current_range = range_percent(open_price, high_price, low_price)
    avg_range_24 = avg_metric(prev_24, candle_range_metric)

    prev_close_2h = float(klines_2h[-2][4])
    prev_close_4h = float(klines_2h[-3][4])
    prev_close_6h = float(klines_2h[-4][4])
    prev_close_12h = float(klines_2h[-7][4])
    prev_close_24h = float(klines_2h[-13][4])
    prev_close_48h = float(klines_2h[-25][4])
    prev_close_7d = float(klines_2h[-85][4])

    atr_24 = average_true_range_percent(prev_24, price)
    current_tr = true_range(signal, float(klines_2h[-2][4]))
    current_atr_percent = current_tr / price * 100 if price > 0 else 0.0
    atr_ratio = current_atr_percent / atr_24 if atr_24 > 0 else 0.0

    recent_3_vol = avg_metric(klines_2h[-4:-1], lambda c: ohlcv(c)[4])
    prior_3_vol = avg_metric(klines_2h[-7:-4], lambda c: ohlcv(c)[4])
    vol_accel = recent_3_vol / prior_3_vol if prior_3_vol > 0 else 0.0

    pre6 = compression_flags(klines_2h[:-1], 6)
    pre12 = compression_flags(klines_2h[:-1], 12)
    pre6_1h = compression_flags(klines_1h[:-1], 6) if len(klines_1h) >= 13 else {
        "tight_range": False,
        "volume_contraction": False,
    }

    highest_close_24 = max(float(candle[4]) for candle in prev_24)
    highest_close_48 = max(float(candle[4]) for candle in prev_48)
    highest_close_7d = max(float(candle[4]) for candle in prev_84)

    group = classify_group(forward["max_profit"], forward["forward_24h"])

    return Top10Record(
        scan_time_kst=scan_kst,
        scan_time_utc=scan_utc,
        scan_dt=scan_dt,
        symbol=symbol,
        rank_24h=rank,
        return_24h_percent=ranking["return_24h_percent"],
        group=group,
        forward_2h=forward["forward_2h"],
        forward_4h=forward["forward_4h"],
        forward_6h=forward["forward_6h"],
        forward_12h=forward["forward_12h"],
        forward_24h=forward["forward_24h"],
        max_profit=forward["max_profit"],
        max_drawdown=forward["max_drawdown"],
        position_24h_percent=position_in_range(price, low_24h, high_24h),
        position_48h_percent=position_in_range(price, low_48h, high_48h),
        position_7d_percent=position_in_range(price, low_7d, high_7d),
        break_24h_highest_close=close_price > highest_close_24,
        break_48h_highest_close=close_price > highest_close_48,
        break_7d_highest_close=close_price > highest_close_7d,
        distance_ma6_percent=ma_distance(6),
        distance_ma12_percent=ma_distance(12),
        distance_ma24_percent=ma_distance(24),
        distance_ma48_percent=ma_distance(48),
        distance_ma84_percent=ma_distance(84),
        ma24_slope_percent=ma_slope_percent(klines_2h[:-1], 24),
        ma48_slope_percent=ma_slope_percent(klines_2h[:-1], 48),
        ma84_slope_percent=ma_slope_percent(klines_2h[:-1], 84),
        current_body_percent=current_body,
        avg_body_24_percent=avg_body_24,
        body_expansion_ratio=current_body / avg_body_24 if avg_body_24 > 0 else 0.0,
        atr_percent=current_atr_percent,
        atr_ratio=atr_ratio,
        range_expansion_ratio=current_range / avg_range_24 if avg_range_24 > 0 else 0.0,
        volume_current=volume,
        volume_ma6=vol_ma6,
        volume_ma12=vol_ma12,
        volume_ma24=vol_ma24,
        volume_ratio_ma24=volume / vol_ma24 if vol_ma24 > 0 else 0.0,
        volume_acceleration_ratio=vol_accel,
        volume_before_price=volume_before_price(klines_2h[:-1]),
        return_prev_2h_percent=close_return_percent(close_price, prev_close_2h),
        return_prev_4h_percent=close_return_percent(close_price, prev_close_4h),
        return_prev_6h_percent=close_return_percent(close_price, prev_close_6h),
        return_prev_12h_percent=close_return_percent(close_price, prev_close_12h),
        return_prev_24h_percent=close_return_percent(close_price, prev_close_24h),
        return_prev_48h_percent=close_return_percent(close_price, prev_close_48h),
        return_prev_7d_percent=close_return_percent(close_price, prev_close_7d),
        pre6_tight_range=pre6["tight_range"],
        pre6_body_compression=pre6["body_compression"],
        pre6_volatility_compression=pre6["volatility_compression"],
        pre6_volume_contraction=pre6["volume_contraction"],
        pre12_tight_range=pre12["tight_range"],
        pre12_body_compression=pre12["body_compression"],
        pre12_volatility_compression=pre12["volatility_compression"],
        pre12_volume_contraction=pre12["volume_contraction"],
        pre6_1h_tight_range=pre6_1h["tight_range"],
        pre6_1h_volume_contraction=pre6_1h["volume_contraction"],
    )


def numeric_metric_names() -> list[str]:
    skip = {
        "scan_time_kst",
        "scan_time_utc",
        "scan_dt",
        "symbol",
        "rank_24h",
        "group",
        "return_24h_percent",
        "forward_2h",
        "forward_4h",
        "forward_6h",
        "forward_12h",
        "forward_24h",
        "max_profit",
        "max_drawdown",
        "volume_current",
    }
    bool_names = bool_metric_names()
    names: list[str] = []
    for field in fields(Top10Record):
        if field.name in skip or field.name in bool_names:
            continue
        names.append(field.name)
    return names


def bool_metric_names() -> list[str]:
    return [
        "break_24h_highest_close",
        "break_48h_highest_close",
        "break_7d_highest_close",
        "volume_before_price",
        "pre6_tight_range",
        "pre6_body_compression",
        "pre6_volatility_compression",
        "pre6_volume_contraction",
        "pre12_tight_range",
        "pre12_body_compression",
        "pre12_volatility_compression",
        "pre12_volume_contraction",
        "pre6_1h_tight_range",
        "pre6_1h_volume_contraction",
    ]


def group_mean(records: list[Top10Record], metric: str) -> float | None:
    values = [getattr(record, metric) for record in records]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return statistics.mean(values)


def group_bool_rate(records: list[Top10Record], metric: str) -> float | None:
    if not records:
        return None
    return sum(1 for record in records if getattr(record, metric)) / len(records) * 100


def record_to_row(record: Top10Record) -> dict[str, str]:
    row: dict[str, str] = {}
    for field in fields(Top10Record):
        if field.name == "scan_dt":
            continue
        value = getattr(record, field.name)
        if isinstance(value, bool):
            row[field.name] = "YES" if value else "NO"
        elif isinstance(value, float):
            row[field.name] = f"{value:.4f}"
        elif value is None:
            row[field.name] = ""
        else:
            row[field.name] = str(value)
    return row


def save_results(records: list[Top10Record]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(Top10Record) if field.name != "scan_dt"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record_to_row(record) for record in records)


def compare_groups(
    records: list[Top10Record],
    group_a: str,
    group_b: str,
) -> list[tuple[str, float, float, float, str]]:
    left = [record for record in records if record.group == group_a]
    right = [record for record in records if record.group == group_b]
    comparisons: list[tuple[str, float, float, float, str]] = []
    skip_absolute = {"volume_ma6", "volume_ma12", "volume_ma24", "volume_current"}

    for metric in numeric_metric_names():
        if metric in skip_absolute:
            continue
        left_mean = group_mean(left, metric)
        right_mean = group_mean(right, metric)
        if left_mean is None or right_mean is None:
            continue
        gap = left_mean - right_mean
        denom = abs(right_mean) if right_mean != 0 else 1.0
        comparisons.append((metric, left_mean, right_mean, abs(gap) / denom, "numeric"))

    for metric in bool_metric_names():
        left_rate = group_bool_rate(left, metric)
        right_rate = group_bool_rate(right, metric)
        if left_rate is None or right_rate is None:
            continue
        comparisons.append((metric, left_rate, right_rate, abs(left_rate - right_rate) / 100.0, "bool"))

    comparisons.sort(key=lambda item: item[3], reverse=True)
    return comparisons


def print_group_averages(records: list[Top10Record]) -> None:
    groups = ["A_STRONG", "B_MODERATE", "C_WEAK", "D_FAILED"]
    key_metrics = [
        "position_7d_percent",
        "position_24h_percent",
        "distance_ma24_percent",
        "distance_ma48_percent",
        "ma24_slope_percent",
        "body_expansion_ratio",
        "volume_ratio_ma24",
        "volume_acceleration_ratio",
        "return_prev_24h_percent",
        "atr_ratio",
        "range_expansion_ratio",
    ]

    print("\n===== GROUP AVERAGES (pre-trend features) =====")
    for group_name in groups:
        group_records = [record for record in records if record.group == group_name]
        print(f"\n{group_name} (n={len(group_records)}):")
        if not group_records:
            print("  (no records)")
            continue
        for metric in key_metrics:
            value = group_mean(group_records, metric)
            if value is not None:
                print(f"  {metric}: {value:.4f}")


def print_learning_report(records: list[Top10Record]) -> None:
    group_a = [record for record in records if record.group == "A_STRONG"]
    group_b = [record for record in records if record.group == "B_MODERATE"]
    group_c = [record for record in records if record.group == "C_WEAK"]
    group_d = [record for record in records if record.group == "D_FAILED"]

    print("\n===== SUMMARY STATISTICS =====")
    print(f"Total TOP10 records: {len(records)}")
    print(f"Group A (max profit >=20%): {len(group_a)}")
    print(f"Group B (max profit 10-20%): {len(group_b)}")
    print(f"Group C (max profit <10%): {len(group_c)}")
    print(f"Group D (forward 24h <0): {len(group_d)}")

    print_group_averages(records)

    comparisons_ad = compare_groups(records, "A_STRONG", "D_FAILED")
    comparisons_ac = compare_groups(records, "A_STRONG", "C_WEAK")

    print("\n===== LEARNING ANSWERS =====")

    print("\n1. Features repeated in STRONG trends (Group A):")
    if group_a:
        for metric in [
            "position_7d_percent",
            "position_24h_percent",
            "distance_ma24_percent",
            "body_expansion_ratio",
            "volume_ratio_ma24",
            "return_prev_24h_percent",
        ]:
            value = group_mean(group_a, metric)
            if value is not None:
                print(f"   avg {metric}: {value:.2f}")
        for metric in [
            "break_24h_highest_close",
            "break_7d_highest_close",
            "volume_before_price",
            "pre6_tight_range",
            "pre6_volume_contraction",
        ]:
            rate = group_bool_rate(group_a, metric)
            if rate is not None:
                print(f"   {metric} YES rate: {rate:.0f}%")
    else:
        print("   No Group A records in this sample.")

    print("\n2. Features repeated in FAILED trends (Group D):")
    if group_d:
        for metric in [
            "position_7d_percent",
            "position_24h_percent",
            "distance_ma24_percent",
            "body_expansion_ratio",
            "volume_ratio_ma24",
            "return_prev_24h_percent",
        ]:
            value = group_mean(group_d, metric)
            if value is not None:
                print(f"   avg {metric}: {value:.2f}")
        for metric in [
            "break_24h_highest_close",
            "break_7d_highest_close",
            "pre6_tight_range",
            "pre12_body_compression",
        ]:
            rate = group_bool_rate(group_d, metric)
            if rate is not None:
                print(f"   {metric} YES rate: {rate:.0f}%")
    else:
        print("   No Group D records in this sample.")

    print("\n3. Largest differences (Group A vs Group D):")
    if comparisons_ad:
        for index, (metric, a_val, d_val, _, kind) in enumerate(comparisons_ad[:8], start=1):
            if kind == "bool":
                print(f"   {index}. {metric}: A {a_val:.0f}% YES | D {d_val:.0f}% YES")
            else:
                print(f"   {index}. {metric}: A {a_val:.2f} | D {d_val:.2f} | gap {a_val - d_val:+.2f}")
    else:
        print("   Not enough A vs D separation to compare.")

    print("\n4. Features that seem useless (small A vs D gap):")
    if comparisons_ad:
        weak = sorted(comparisons_ad, key=lambda item: item[3])[:6]
        names = ", ".join(item[0] for item in weak)
        print(f"   {names}")
    else:
        print("   N/A")

    print("\n5. Features deserving further study:")
    study_candidates: list[str] = []
    if comparisons_ad:
        for metric, a_val, d_val, rel_gap, kind in comparisons_ad[:5]:
            if rel_gap >= 0.15:
                study_candidates.append(metric)
    if group_a and group_d:
        a_pos = group_mean(group_a, "position_7d_percent")
        d_pos = group_mean(group_d, "position_7d_percent")
        if a_pos is not None and d_pos is not None and abs(a_pos - d_pos) >= 5:
            study_candidates.append("position_7d_percent")
    if study_candidates:
        for metric in dict.fromkeys(study_candidates):
            print(f"   - {metric}")
    else:
        print("   - compression flags across more dates")
        print("   - volume_before_price sequence timing")

    print("\n6. Research hypotheses for Scout Learning vNext (not trading rules):")
    hypotheses: list[str] = []

    if group_a and group_d:
        a_pos7d = group_mean(group_a, "position_7d_percent")
        d_pos7d = group_mean(group_d, "position_7d_percent")
        if a_pos7d is not None and d_pos7d is not None and a_pos7d < d_pos7d:
            hypotheses.append(
                "H1: Early-phase movers (lower 7d range position) among top gainers "
                "may continue better than already-extended leaders."
            )

    if comparisons_ad:
        top_metric = comparisons_ad[0][0]
        hypotheses.append(
            f"H2: {top_metric} may separate continuation from exhaustion "
            "when studied across additional unseen dates."
        )
    vol_before_a = group_bool_rate(group_a, "volume_before_price")
    vol_before_d = group_bool_rate(group_d, "volume_before_price")
    if vol_before_a is not None and vol_before_d is not None and vol_before_a > vol_before_d:
        hypotheses.append(
            "H3: Volume rising before price expansion in the prior 6 candles "
            "may mark healthier continuation than volume chasing a late spike."
        )

    pre6_tight_a = group_bool_rate(group_a, "pre6_tight_range")
    pre6_tight_d = group_bool_rate(group_d, "pre6_tight_range")
    if pre6_tight_a is not None and pre6_tight_d is not None:
        hypotheses.append(
            "H4: Pre-scan compression (tight range / body / volatility) "
            f"appeared in A {pre6_tight_a:.0f}% vs D {pre6_tight_d:.0f}% of cases; "
            "test whether compression-before-breakout predicts follow-through."
        )

    hypotheses.append(
        "H5: Combine 2H ranking with 1H compression/volume timing "
        "to study whether multi-timeframe pretrend alignment improves signal quality."
    )

    for index, hypothesis in enumerate(hypotheses[:5], start=1):
        print(f"   {index}. {hypothesis}")

    print("\n===== REPEATED PATTERN OBSERVATIONS =====")
    print_strong_failure_patterns(records, comparisons_ad, comparisons_ac)
    print("==========================================")


def print_strong_failure_patterns(
    records: list[Top10Record],
    comparisons_ad: list[tuple[str, float, float, float, str]],
    comparisons_ac: list[tuple[str, float, float, float, str]],
) -> None:
    group_a = [record for record in records if record.group == "A_STRONG"]
    group_d = [record for record in records if record.group == "D_FAILED"]

    print("\nStrong-pattern observations:")
    observations: list[str] = []
    if group_a:
        pos7d = group_mean(group_a, "position_7d_percent")
        if pos7d is not None:
            observations.append(
                f"Group A average 7d position was {pos7d:.0f}% "
                "(where in the prior week range price sat at scan)."
            )
        ret24 = group_mean(group_a, "return_prev_24h_percent")
        if ret24 is not None:
            observations.append(f"Group A already had {ret24:.1f}% prior 24h momentum at scan.")
        break7d = group_bool_rate(group_a, "break_7d_highest_close")
        if break7d is not None:
            observations.append(f"{break7d:.0f}% of Group A had broken the 7d highest close.")

    if not observations:
        observations.append("No Group A records; strong-pattern notes deferred.")
    for index, text in enumerate(observations, start=1):
        print(f"  {index}. {text}")

    print("\nFailure-pattern observations:")
    fail_notes: list[str] = []
    if group_d:
        pos7d = group_mean(group_d, "position_7d_percent")
        if pos7d is not None:
            fail_notes.append(
                f"Group D averaged {pos7d:.0f}% 7d position, "
                "often more extended than strong continuations."
            )
        dist_ma24 = group_mean(group_d, "distance_ma24_percent")
        if dist_ma24 is not None:
            fail_notes.append(f"Group D sat {dist_ma24:.1f}% above MA24 on average at scan.")
        break24 = group_bool_rate(group_d, "break_24h_highest_close")
        if break24 is not None:
            fail_notes.append(
                f"{break24:.0f}% of failures had already broken 24h highest close "
                "(late chase risk)."
            )

    if comparisons_ac:
        top = comparisons_ac[0]
        if top[4] == "numeric":
            fail_notes.append(
                f"Largest A vs C gap: {top[0]} ({top[1]:.2f} vs {top[2]:.2f})."
            )

    if not fail_notes:
        fail_notes.append("No Group D records; failure-pattern notes deferred.")
    for index, text in enumerate(fail_notes, start=1):
        print(f"  {index}. {text}")


def main() -> None:
    try:
        scan_times = get_scan_times()
        eligible_symbols = get_eligible_symbols()

        if not eligible_symbols:
            print("Error: no eligible USDT perpetual symbols found.")
            return

        print("TOP10 gainer pre-trend learning starting.")
        print("Study date: 2026-06-15 KST")
        print(f"Universe size: {len(eligible_symbols)} symbols")
        print("Research only. No Scout filters. No trading logic.")

        all_records: list[Top10Record] = []
        symbols = sorted(eligible_symbols)

        for scan_kst, scan_utc, scan_dt in scan_times:
            print(f"\nRanking {scan_kst} KST...")
            end_ms = int(scan_dt.timestamp() * 1000)
            candidates: list[tuple[str, dict]] = []

            for index, symbol in enumerate(symbols, start=1):
                if index % 100 == 0 or index == len(symbols):
                    print(f"  progress: {index}/{len(symbols)}")

                try:
                    klines = fetch_klines_before(symbol, INTERVAL_2H, end_ms, RANKING_KLINES_2H)
                    ranking = compute_24h_ranking(klines)
                    if ranking is not None:
                        candidates.append((symbol, ranking))
                except urllib.error.HTTPError:
                    continue

                time.sleep(API_SLEEP_SEC)

            candidates.sort(key=lambda item: item[1]["return_24h_percent"], reverse=True)
            top_candidates = candidates[:TOP_N]
            print(
                "  TOP10: "
                + ", ".join(f"{symbol} ({data['return_24h_percent']:+.1f}%)" for symbol, data in top_candidates[:5])
                + (" ..." if len(top_candidates) > 5 else "")
            )

            for rank, (symbol, ranking) in enumerate(top_candidates, start=1):
                try:
                    klines_2h = fetch_klines_before(symbol, INTERVAL_2H, end_ms, ANALYSIS_KLINES_2H)
                    klines_1h = fetch_klines_before(symbol, INTERVAL_1H, end_ms, ANALYSIS_KLINES_1H)
                    forward = measure_forward(symbol, scan_dt, ranking["price_at_scan"])
                    record = build_record(
                        scan_kst,
                        scan_utc,
                        scan_dt,
                        rank,
                        symbol,
                        ranking,
                        klines_2h,
                        klines_1h,
                        forward,
                    )
                    if record is not None:
                        all_records.append(record)
                except urllib.error.HTTPError:
                    continue

                time.sleep(API_SLEEP_SEC)

        save_results(all_records)
        print_learning_report(all_records)
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
