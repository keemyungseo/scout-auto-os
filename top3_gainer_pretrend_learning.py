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

TOP_N = 3
CANDLES_24H_2H = 12
CANDLES_48H_2H = 24
CANDLES_7D_2H = 84
CANDLES_7D_1H = 168

RANKING_KLINES_2H = CANDLES_24H_2H + 1
ANALYSIS_KLINES_2H = CANDLES_7D_2H + 10
ANALYSIS_KLINES_1H = CANDLES_7D_1H + 10
FORWARD_HOURS = 24

STRONG_MAX_PROFIT_24H_PCT = 10.0
FAILED_RETURN_24H_PCT = 0.0
MA_FLAT_THRESHOLD_PCT = 2.0

INTERVAL_2H = "2h"
INTERVAL_1H = "1h"
INTERVAL_2H_MS = 2 * 60 * 60 * 1000
INTERVAL_1H_MS = 60 * 60 * 1000
MAX_LIMIT = 1500
API_SLEEP_SEC = 0.03

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "top3_pretrend_learning.csv"


@dataclass
class PretrendRecord:
    scan_time_kst: str
    scan_time_utc: str
    scan_dt: datetime
    rank: int
    symbol: str
    group: str
    price_at_scan: float
    change_24h_percent: float
    position_7d_percent: float
    position_48h_percent: float
    position_24h_percent: float
    distance_ma6_percent: float
    distance_ma12_percent: float
    distance_ma24_percent: float
    distance_ma48_percent: float
    distance_ma84_percent: float
    ma6_slope_percent: float
    ma12_slope_percent: float
    ma24_slope_percent: float
    ma48_slope_percent: float
    ma84_slope_percent: float
    body_vs_avg_percent: float
    range_vs_avg_ratio: float
    break_24h_highest_close: bool
    break_48h_highest_close: bool
    break_7d_highest_close: bool
    volume_current: float
    volume_ma6: float
    volume_ma12: float
    volume_ma24: float
    volume_ma48: float
    volume_ma84: float
    volume_ratio_ma24: float
    volume_ratio_ma6: float
    volume_acceleration_ratio: float
    volume_expansion_before_price: bool
    return_prev_6h_percent: float
    return_prev_12h_percent: float
    return_prev_24h_percent: float
    return_prev_48h_percent: float
    return_prev_7d_percent: float
    pre6_price_compressing: bool
    pre6_ma24_flattening: bool
    pre6_volume_increasing: bool
    pre6_body_expanding: bool
    pre6_volatility_shrink_then_expand: bool
    pre6_1h_price_compressing: bool
    pre6_1h_volume_increasing: bool
    max_profit_after_24h: float | None
    return_after_24h: float | None


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


def fetch_klines_forward_2h(symbol: str, start_ms: int, end_ms: int) -> list[list]:
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


def kline_close_dt_2h(kline: list) -> datetime:
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


def range_ratio(open_price: float, high_price: float, low_price: float) -> float:
    if open_price == 0:
        return 0.0
    return (high_price - low_price) / open_price * 100


def position_in_range(price: float, low_price: float, high_price: float) -> float:
    if high_price == low_price:
        return 50.0
    return (price - low_price) / (high_price - low_price) * 100


def ma_from_closes(closes: list[float]) -> float:
    if not closes:
        return 0.0
    return sum(closes) / len(closes)


def distance_from_ma_percent(price: float, ma_value: float) -> float:
    if ma_value == 0:
        return 0.0
    return (price - ma_value) / ma_value * 100


def ma_slope_percent(closes: list[list], period: int, shift: int = 6) -> float:
    if len(closes) < period + shift:
        return 0.0

    recent = [float(candle[4]) for candle in closes[-period:]]
    prior_slice = closes[-(period + shift) : -shift]
    prior = [float(candle[4]) for candle in prior_slice]

    recent_ma = ma_from_closes(recent)
    prior_ma = ma_from_closes(prior)
    if prior_ma == 0:
        return 0.0
    return (recent_ma - prior_ma) / prior_ma * 100


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
    return range_ratio(open_price, high_price, low_price)


def volume_expansion_before_price(candles: list[list]) -> bool:
    if len(candles) < 6:
        return False

    first_half = candles[-6:-3]
    second_half = candles[-3:]
    price_move_first = abs(cumulative_return_percent(first_half))
    price_move_second = abs(cumulative_return_percent(second_half))
    vol_first = avg_metric(first_half, lambda c: ohlcv(c)[4])
    vol_second = avg_metric(second_half, lambda c: ohlcv(c)[4])

    return vol_first > 0 and vol_second / vol_first >= 1.15 and price_move_second >= price_move_first


def analyze_pre6_candles(candles: list[list], ma24_now: float, ma24_6_ago: float) -> dict[str, bool]:
    pre6 = candles[-7:-1]
    if len(pre6) < 6:
        return {
            "price_compressing": False,
            "ma24_flattening": False,
            "volume_increasing": False,
            "body_expanding": False,
            "volatility_shrink_then_expand": False,
        }

    first3 = pre6[:3]
    last3 = pre6[3:]
    range_first = avg_metric(first3, candle_range_metric)
    range_last = avg_metric(last3, candle_range_metric)
    vol_first = avg_metric(first3, lambda c: ohlcv(c)[4])
    vol_last = avg_metric(last3, lambda c: ohlcv(c)[4])
    body_first = avg_metric(first3, candle_body_metric)
    body_last = avg_metric(last3, candle_body_metric)

    ma24_change = abs(ma24_now - ma24_6_ago) / ma24_6_ago * 100 if ma24_6_ago else 0.0

    return {
        "price_compressing": range_last < range_first * 0.9,
        "ma24_flattening": ma24_change < MA_FLAT_THRESHOLD_PCT,
        "volume_increasing": vol_last > vol_first * 1.1,
        "body_expanding": body_last > body_first * 1.1,
        "volatility_shrink_then_expand": range_first < range_last and range_first < body_first,
    }


def analyze_pre6_1h(candles_1h: list[list]) -> dict[str, bool]:
    pre6 = candles_1h[-7:-1]
    if len(pre6) < 6:
        return {"price_compressing": False, "volume_increasing": False}

    first3 = pre6[:3]
    last3 = pre6[3:]
    range_first = avg_metric(first3, candle_range_metric)
    range_last = avg_metric(last3, candle_range_metric)
    vol_first = avg_metric(first3, lambda c: ohlcv(c)[4])
    vol_last = avg_metric(last3, lambda c: ohlcv(c)[4])

    return {
        "price_compressing": range_last < range_first * 0.9,
        "volume_increasing": vol_last > vol_first * 1.1,
    }


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

    change_24h = (close_price - close_24h_ago) / close_24h_ago * 100
    return {
        "price_at_scan": close_price,
        "change_24h_percent": change_24h,
    }


def classify_group(max_profit_24h: float | None, return_24h: float | None) -> str:
    if max_profit_24h is not None and max_profit_24h >= STRONG_MAX_PROFIT_24H_PCT:
        return "STRONG"
    if return_24h is not None and return_24h < FAILED_RETURN_24H_PCT:
        return "FAILED"
    return "MIDDLE"


def measure_forward(symbol: str, scan_dt: datetime, price_at_scan: float) -> dict:
    scan_end_ms = int(scan_dt.timestamp() * 1000)
    forward_end_ms = scan_end_ms + FORWARD_HOURS * 60 * 60 * 1000
    forward = fetch_klines_forward_2h(symbol, scan_end_ms, forward_end_ms)

    max_high = price_at_scan
    close_24h: float | None = None

    for candle in forward:
        if kline_close_dt_2h(candle) > scan_dt + timedelta(hours=24):
            break
        _, high_price, _, close_price, _ = ohlcv(candle)
        max_high = max(max_high, high_price)
        close_24h = close_price

    max_profit = (max_high - price_at_scan) / price_at_scan * 100
    return_24h = None
    if close_24h is not None:
        return_24h = (close_24h - price_at_scan) / price_at_scan * 100

    return {
        "max_profit_after_24h": max_profit,
        "return_after_24h": return_24h,
    }


def build_pretrend_record(
    scan_kst: str,
    scan_utc: str,
    scan_dt: datetime,
    rank: int,
    symbol: str,
    ranking: dict,
    klines_2h: list[list],
    klines_1h: list[list],
    forward: dict,
) -> PretrendRecord | None:
    if len(klines_2h) < ANALYSIS_KLINES_2H:
        return None

    signal = klines_2h[-1]
    price = ranking["price_at_scan"]
    open_price, high_price, low_price, close_price, volume = ohlcv(signal)

    prev_84 = klines_2h[-(CANDLES_7D_2H + 1) : -1]
    prev_48 = klines_2h[-(CANDLES_48H_2H + 1) : -1]
    prev_24 = klines_2h[-(CANDLES_24H_2H + 1) : -1]
    prev_6h = klines_2h[-4:-1]
    prev_12h = klines_2h[-7:-1]
    prev_24h = klines_2h[-13:-1]
    prev_48h = klines_2h[-25:-1]

    low_7d = min(ohlcv(c)[2] for c in prev_84)
    high_7d = max(ohlcv(c)[1] for c in prev_84)
    low_48h = min(ohlcv(c)[2] for c in prev_48)
    high_48h = max(ohlcv(c)[1] for c in prev_48)
    low_24h = min(ohlcv(c)[2] for c in prev_24)
    high_24h = max(ohlcv(c)[1] for c in prev_24)

    closes_before = [float(c[4]) for c in klines_2h[:-1]]

    def ma_distance(period: int) -> float:
        if len(closes_before) < period:
            return 0.0
        ma_value = ma_from_closes(closes_before[-period:])
        return distance_from_ma_percent(price, ma_value)

    def volume_ma(period: int) -> float:
        if len(klines_2h) < period + 1:
            return 0.0
        volumes = [ohlcv(c)[4] for c in klines_2h[-(period + 1) : -1]]
        return ma_from_closes(volumes)

    vol_ma6 = volume_ma(6)
    vol_ma12 = volume_ma(12)
    vol_ma24 = volume_ma(24)
    vol_ma48 = volume_ma(48)
    vol_ma84 = volume_ma(84)

    avg_body_24 = avg_metric(prev_24, candle_body_metric)
    avg_range_24 = avg_metric(prev_24, candle_range_metric)
    current_body = body_percent(open_price, close_price)
    current_range = range_ratio(open_price, high_price, low_price)

    highest_close_24 = max(float(c[4]) for c in prev_24)
    highest_close_48 = max(float(c[4]) for c in prev_48)
    highest_close_7d = max(float(c[4]) for c in prev_84)

    ma24_now = ma_from_closes([float(c[4]) for c in klines_2h[-25:-1]])
    ma24_6_ago = ma_from_closes([float(c[4]) for c in klines_2h[-31:-7]])
    pre6 = analyze_pre6_candles(klines_2h, ma24_now, ma24_6_ago)
    pre6_1h = analyze_pre6_1h(klines_1h)

    recent_3_vol = avg_metric(klines_2h[-4:-1], lambda c: ohlcv(c)[4])
    prior_3_vol = avg_metric(klines_2h[-7:-4], lambda c: ohlcv(c)[4])
    vol_accel = recent_3_vol / prior_3_vol if prior_3_vol > 0 else 0.0

    group = classify_group(forward["max_profit_after_24h"], forward["return_after_24h"])

    return PretrendRecord(
        scan_time_kst=scan_kst,
        scan_time_utc=scan_utc,
        scan_dt=scan_dt,
        rank=rank,
        symbol=symbol,
        group=group,
        price_at_scan=price,
        change_24h_percent=ranking["change_24h_percent"],
        position_7d_percent=position_in_range(price, low_7d, high_7d),
        position_48h_percent=position_in_range(price, low_48h, high_48h),
        position_24h_percent=position_in_range(price, low_24h, high_24h),
        distance_ma6_percent=ma_distance(6),
        distance_ma12_percent=ma_distance(12),
        distance_ma24_percent=ma_distance(24),
        distance_ma48_percent=ma_distance(48),
        distance_ma84_percent=ma_distance(84),
        ma6_slope_percent=ma_slope_percent(klines_2h[:-1], 6),
        ma12_slope_percent=ma_slope_percent(klines_2h[:-1], 12),
        ma24_slope_percent=ma_slope_percent(klines_2h[:-1], 24),
        ma48_slope_percent=ma_slope_percent(klines_2h[:-1], 48),
        ma84_slope_percent=ma_slope_percent(klines_2h[:-1], 84),
        body_vs_avg_percent=current_body / avg_body_24 * 100 if avg_body_24 > 0 else 0.0,
        range_vs_avg_ratio=current_range / avg_range_24 if avg_range_24 > 0 else 0.0,
        break_24h_highest_close=close_price > highest_close_24,
        break_48h_highest_close=close_price > highest_close_48,
        break_7d_highest_close=close_price > highest_close_7d,
        volume_current=volume,
        volume_ma6=vol_ma6,
        volume_ma12=vol_ma12,
        volume_ma24=vol_ma24,
        volume_ma48=vol_ma48,
        volume_ma84=vol_ma84,
        volume_ratio_ma24=volume / vol_ma24 if vol_ma24 > 0 else 0.0,
        volume_ratio_ma6=volume / vol_ma6 if vol_ma6 > 0 else 0.0,
        volume_acceleration_ratio=vol_accel,
        volume_expansion_before_price=volume_expansion_before_price(klines_2h[:-1]),
        return_prev_6h_percent=cumulative_return_percent(prev_6h),
        return_prev_12h_percent=cumulative_return_percent(prev_12h),
        return_prev_24h_percent=cumulative_return_percent(prev_24h),
        return_prev_48h_percent=cumulative_return_percent(prev_48h),
        return_prev_7d_percent=cumulative_return_percent(prev_84),
        pre6_price_compressing=pre6["price_compressing"],
        pre6_ma24_flattening=pre6["ma24_flattening"],
        pre6_volume_increasing=pre6["volume_increasing"],
        pre6_body_expanding=pre6["body_expanding"],
        pre6_volatility_shrink_then_expand=pre6["volatility_shrink_then_expand"],
        pre6_1h_price_compressing=pre6_1h["price_compressing"],
        pre6_1h_volume_increasing=pre6_1h["volume_increasing"],
        max_profit_after_24h=forward["max_profit_after_24h"],
        return_after_24h=forward["return_after_24h"],
    )


def numeric_fields() -> list[str]:
    skip = {
        "scan_time_kst",
        "scan_time_utc",
        "scan_dt",
        "rank",
        "symbol",
        "group",
        "max_profit_after_24h",
        "return_after_24h",
    }
    bool_fields = {
        "break_24h_highest_close",
        "break_48h_highest_close",
        "break_7d_highest_close",
        "volume_expansion_before_price",
        "pre6_price_compressing",
        "pre6_ma24_flattening",
        "pre6_volume_increasing",
        "pre6_body_expanding",
        "pre6_volatility_shrink_then_expand",
        "pre6_1h_price_compressing",
        "pre6_1h_volume_increasing",
    }
    names: list[str] = []
    for field in fields(PretrendRecord):
        if field.name in skip or field.name in bool_fields:
            continue
        names.append(field.name)
    return names


def bool_fields_list() -> list[str]:
    return [
        "break_24h_highest_close",
        "break_48h_highest_close",
        "break_7d_highest_close",
        "volume_expansion_before_price",
        "pre6_price_compressing",
        "pre6_ma24_flattening",
        "pre6_volume_increasing",
        "pre6_body_expanding",
        "pre6_volatility_shrink_then_expand",
        "pre6_1h_price_compressing",
        "pre6_1h_volume_increasing",
    ]


def group_mean(records: list[PretrendRecord], metric: str) -> float | None:
    values = [getattr(record, metric) for record in records]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return statistics.mean(values)


def group_bool_rate(records: list[PretrendRecord], metric: str) -> float | None:
    if not records:
        return None
    return sum(1 for record in records if getattr(record, metric)) / len(records) * 100


def record_to_row(record: PretrendRecord) -> dict[str, str]:
    row: dict[str, str] = {}
    for field in fields(PretrendRecord):
        if field.name == "scan_dt":
            continue
        value = getattr(record, field.name)
        if isinstance(value, bool):
            row[field.name] = "YES" if value else "NO"
        elif isinstance(value, float):
            if field.name == "price_at_scan":
                row[field.name] = f"{value:.8f}"
            else:
                row[field.name] = f"{value:.4f}"
        elif value is None:
            row[field.name] = ""
        else:
            row[field.name] = str(value)
    return row


def save_results(records: list[PretrendRecord]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in fields(PretrendRecord) if f.name != "scan_dt"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record_to_row(record) for record in records)


def compare_groups(records: list[PretrendRecord]) -> list[tuple[str, float, float, float, str]]:
    strong = [r for r in records if r.group == "STRONG"]
    failed = [r for r in records if r.group == "FAILED"]
    comparisons: list[tuple[str, float, float, float, str]] = []

    for metric in numeric_fields():
        strong_mean = group_mean(strong, metric)
        failed_mean = group_mean(failed, metric)
        if strong_mean is None or failed_mean is None:
            continue
        gap = strong_mean - failed_mean
        denom = abs(failed_mean) if failed_mean != 0 else 1.0
        comparisons.append((metric, strong_mean, failed_mean, abs(gap) / denom, "numeric"))

    for metric in bool_fields_list():
        strong_rate = group_bool_rate(strong, metric)
        failed_rate = group_bool_rate(failed, metric)
        if strong_rate is None or failed_rate is None:
            continue
        gap = strong_rate - failed_rate
        comparisons.append((metric, strong_rate, failed_rate, abs(gap) / 100.0, "bool"))

    comparisons.sort(key=lambda item: item[3], reverse=True)
    return comparisons


def print_learning_report(records: list[PretrendRecord]) -> None:
    strong = [r for r in records if r.group == "STRONG"]
    failed = [r for r in records if r.group == "FAILED"]
    middle = [r for r in records if r.group == "MIDDLE"]

    print("\n======== PRE-TREND LEARNING ========")
    print(f"Records: {len(records)} | STRONG {len(strong)} | FAILED {len(failed)} | MIDDLE {len(middle)}")
    print("All features use prior data only. Forward fields are for grouping only.")

    key_metrics = [
        "position_7d_percent",
        "position_24h_percent",
        "distance_ma24_percent",
        "ma24_slope_percent",
        "body_vs_avg_percent",
        "volume_ratio_ma24",
        "volume_acceleration_ratio",
        "return_prev_24h_percent",
    ]

    print("\nStrong average:")
    for metric in key_metrics:
        value = group_mean(strong, metric)
        if value is not None:
            print(f"  {metric}: {value:.4f}")

    print("\nFailed average:")
    for metric in key_metrics:
        value = group_mean(failed, metric)
        if value is not None:
            print(f"  {metric}: {value:.4f}")

    comparisons = compare_groups(records)
    pretrend_comparisons = [
        item for item in comparisons if item[0] not in {"max_profit_after_24h", "return_after_24h"}
    ]
    print("\nBiggest differences (STRONG vs FAILED, pre-trend features only):")
    for index, (metric, strong_value, failed_value, _, kind) in enumerate(
        pretrend_comparisons[:5], start=1
    ):
        if kind == "bool":
            print(
                f"  {index}. {metric}: STRONG {strong_value:.1f}% YES | "
                f"FAILED {failed_value:.1f}% YES"
            )
        else:
            print(
                f"  {index}. {metric}: STRONG {strong_value:.4f} | "
                f"FAILED {failed_value:.4f} | gap {strong_value - failed_value:+.4f}"
            )

    unexpected: list[str] = []
    if strong and failed:
        strong_break_24 = group_bool_rate(strong, "break_24h_highest_close")
        failed_break_24 = group_bool_rate(failed, "break_24h_highest_close")
        if strong_break_24 is not None and failed_break_24 is not None:
            if strong_break_24 < failed_break_24:
                unexpected.append(
                    "24h highest-close breakout was more common in FAILED than STRONG top gainers."
                )

        strong_pos7d = group_mean(strong, "position_7d_percent")
        failed_pos7d = group_mean(failed, "position_7d_percent")
        if strong_pos7d is not None and failed_pos7d is not None and strong_pos7d < failed_pos7d:
            unexpected.append(
                "STRONG cases started lower inside the 7-day range than FAILED cases."
            )

        strong_pre6_vol = group_bool_rate(strong, "pre6_volume_increasing")
        failed_pre6_vol = group_bool_rate(failed, "pre6_volume_increasing")
        if strong_pre6_vol is not None and failed_pre6_vol is not None:
            if strong_pre6_vol > failed_pre6_vol + 10:
                unexpected.append(
                    "Volume often rose in the 6 candles before the scan among STRONG cases."
                )

    print("\nUnexpected findings:")
    if unexpected:
        for item in unexpected:
            print(f"  - {item}")
    else:
        print("  - No major surprises in this sample split.")

    print_top_observations(records, pretrend_comparisons)


def print_top_observations(
    records: list[PretrendRecord],
    comparisons: list[tuple[str, float, float, float, str]],
) -> None:
    strong = [r for r in records if r.group == "STRONG"]
    failed = [r for r in records if r.group == "FAILED"]

    print("\nTOP 10 observations (data-driven, not trading rules):")

    observations: list[str] = []

    if strong:
        pos7d = group_mean(strong, "position_7d_percent")
        if pos7d is not None:
            observations.append(
                f"STRONG trends in this sample often began around {pos7d:.0f}% "
                "of the prior 7-day range, suggesting many were still early-phase movers."
            )

    if strong and failed:
        s_pos = group_mean(strong, "position_7d_percent")
        f_pos = group_mean(failed, "position_7d_percent")
        if s_pos is not None and f_pos is not None:
            observations.append(
                f"Average 7-day range position was lower for STRONG ({s_pos:.1f}%) "
                f"than FAILED ({f_pos:.1f}%), so top gainers were not always the most extended."
            )

    pre6_vol_strong = group_bool_rate(strong, "pre6_volume_increasing")
    if pre6_vol_strong is not None:
        observations.append(
            f"Among STRONG cases, {pre6_vol_strong:.0f}% showed rising volume "
            "across the 6 prior 2H candles."
        )

    pre6_flat = group_bool_rate(strong, "pre6_ma24_flattening")
    if pre6_flat is not None:
        observations.append(
            f"MA24 flattening in the 6 candles before scan appeared in "
            f"{pre6_flat:.0f}% of STRONG cases."
        )

    vol_before_price = group_bool_rate(strong, "volume_expansion_before_price")
    if vol_before_price is not None:
        observations.append(
            f"Volume expansion before price expansion was seen in "
            f"{vol_before_price:.0f}% of STRONG cases in this run."
        )

    if comparisons:
        pretrend_comparisons = [
            item for item in comparisons if item[0] not in {"max_profit_after_24h", "return_after_24h"}
        ]
        top = pretrend_comparisons[0] if pretrend_comparisons else comparisons[0]
        if top[4] == "numeric":
            observations.append(
                f"Largest STRONG vs FAILED pre-trend gap in this sample: {top[0]} "
                f"({top[1]:.2f} vs {top[2]:.2f})."
            )

    weak = sorted(comparisons, key=lambda item: item[3])[:3]
    if weak:
        names = ", ".join(item[0] for item in weak)
        observations.append(
            f"Several metrics showed little separation between groups, including: {names}."
        )

    s_ret24 = group_mean(strong, "return_prev_24h_percent")
    f_ret24 = group_mean(failed, "return_prev_24h_percent")
    if s_ret24 is not None and f_ret24 is not None:
        observations.append(
            f"Prior 24h price momentum before scan: STRONG avg {s_ret24:.1f}% "
            f"vs FAILED avg {f_ret24:.1f}%."
        )

    s_vol = group_mean(strong, "volume_ratio_ma24")
    f_vol = group_mean(failed, "volume_ratio_ma24")
    if s_vol is not None and f_vol is not None:
        observations.append(
            f"Current volume vs 24-period average: STRONG {s_vol:.2f}x "
            f"vs FAILED {f_vol:.2f}x."
        )

    s_break7d = group_bool_rate(strong, "break_7d_highest_close")
    f_break7d = group_bool_rate(failed, "break_7d_highest_close")
    if s_break7d is not None and f_break7d is not None:
        observations.append(
            f"7-day highest-close breakout at scan: STRONG {s_break7d:.0f}% "
            f"vs FAILED {f_break7d:.0f}%."
        )

    observations.append(
        "Re-run across more dates before translating any pattern into Scout conditions."
    )

    for index, text in enumerate(observations[:10], start=1):
        print(f"  {index}. {text}")

    print("====================================")


def main() -> None:
    try:
        scan_times = get_scan_times()
        eligible_symbols = get_eligible_symbols()

        if not eligible_symbols:
            print("Error: no eligible USDT perpetual symbols found.")
            return

        print("Top3 gainer pre-trend learning starting.")
        print(f"Universe size: {len(eligible_symbols)} symbols")
        print(f"Scan times: {len(scan_times)}")
        print("Research only. Prior-data features. No trading simulation.")

        all_records: list[PretrendRecord] = []
        symbols = sorted(eligible_symbols)

        for scan_kst, scan_utc, scan_dt in scan_times:
            print(f"\nScanning {scan_kst} KST for top gainers...")
            end_ms = int(scan_dt.timestamp() * 1000)
            candidates: list[tuple[str, dict]] = []

            for index, symbol in enumerate(symbols, start=1):
                if index % 100 == 0 or index == len(symbols):
                    print(f"  ranking progress: {index}/{len(symbols)}")

                try:
                    klines = fetch_klines_before(symbol, INTERVAL_2H, end_ms, RANKING_KLINES_2H)
                    ranking = compute_24h_ranking(klines)
                    if ranking is None:
                        continue
                    candidates.append((symbol, ranking))
                except urllib.error.HTTPError:
                    continue

                time.sleep(API_SLEEP_SEC)

            candidates.sort(key=lambda item: item[1]["change_24h_percent"], reverse=True)
            top_candidates = candidates[:TOP_N]

            print(f"  top {TOP_N}:")
            for rank, (symbol, ranking) in enumerate(top_candidates, start=1):
                print(f"    #{rank} {symbol} {ranking['change_24h_percent']:+.2f}%")

            for rank, (symbol, ranking) in enumerate(top_candidates, start=1):
                try:
                    klines_2h = fetch_klines_before(
                        symbol, INTERVAL_2H, end_ms, ANALYSIS_KLINES_2H
                    )
                    klines_1h = fetch_klines_before(
                        symbol, INTERVAL_1H, end_ms, ANALYSIS_KLINES_1H
                    )
                    forward = measure_forward(symbol, scan_dt, ranking["price_at_scan"])
                    record = build_pretrend_record(
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
        print(f"Total records: {len(all_records)}")

    except ValueError as exc:
        print(f"Error: {exc}")
    except urllib.error.HTTPError as exc:
        details = parse_error_message(exc.read().decode("utf-8", errors="replace"))
        print(f"Error: Binance API request failed. HTTP {exc.code}: {details}")
    except urllib.error.URLError as exc:
        print(f"Error: cannot connect to Binance. {exc.reason}")


if __name__ == "__main__":
    main()
