import csv
import json
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
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

LOOKBACK_7D = 84
LOOKBACK_24 = 24
CANDLES_24H = 12
MA_SLOPE_SHIFT = 6

MAX_POSITION_7D = 70.0
MIN_BODY_EXPANSION = 1.5
MIN_VOLUME_RATIO_MA24 = 0.8
MIN_RETURN_PREV_24H = 10.0
MAX_RETURN_PREV_24H = 80.0

MEANINGFUL_MAX_PROFIT_PCT = 10.0
PROFIT_TARGET_10_PCT = 10.0
PROFIT_TARGET_20_PCT = 20.0

INTERVAL = "2h"
INTERVAL_MS = 2 * 60 * 60 * 1000
KLINES_NEEDED = LOOKBACK_7D + 1
FORWARD_HOURS = 24
MAX_LIMIT = 1500
API_SLEEP_SEC = 0.03

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "scout_learning_v2_backtest.csv"


@dataclass
class ScoutMatch:
    search_time_kst: str
    search_time_utc: str
    search_dt: datetime
    symbol: str
    signal_close: float
    position_7d_percent: float
    body_expansion_ratio: float
    volume_ratio_ma24: float
    prior_24h_return: float
    ma24: float
    ma24_slope_percent: float
    return_after_6h: float | None
    return_after_12h: float | None
    return_after_24h: float | None
    max_profit_24h: float
    max_drawdown_24h: float
    time_to_max_profit_kst: str
    time_to_max_profit_utc: str
    profit_10_occurred: bool
    profit_20_occurred: bool


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


def body_percent(kline: list) -> float:
    open_price = float(kline[1])
    close_price = float(kline[4])
    if open_price == 0:
        return 0.0
    return abs(close_price - open_price) / open_price * 100


def ma_from_closes(closes: list[float]) -> float:
    if not closes:
        return 0.0
    return sum(closes) / len(closes)


def ma24_slope_percent(klines: list[list]) -> float:
    if len(klines) < LOOKBACK_24 + MA_SLOPE_SHIFT + 1:
        return 0.0

    recent = [float(candle[4]) for candle in klines[-(LOOKBACK_24 + 1) : -1]]
    prior = [float(candle[4]) for candle in klines[-(LOOKBACK_24 + MA_SLOPE_SHIFT + 1) : -(MA_SLOPE_SHIFT + 1)]]

    recent_ma = ma_from_closes(recent)
    prior_ma = ma_from_closes(prior)
    if prior_ma == 0:
        return 0.0
    return (recent_ma - prior_ma) / prior_ma * 100


def evaluate_hypothesis(klines: list[list]) -> dict | None:
    if len(klines) < KLINES_NEEDED:
        return None

    signal = klines[-1]
    prev_84 = klines[-(LOOKBACK_7D + 1) : -1]
    prev_24 = klines[-(LOOKBACK_24 + 1) : -1]

    close_price = float(signal[4])
    if not (MIN_PRICE <= close_price <= MAX_PRICE):
        return None

    low_7d = min(float(candle[3]) for candle in prev_84)
    high_7d = max(float(candle[2]) for candle in prev_84)
    if high_7d == low_7d:
        position_7d = 50.0
    else:
        position_7d = (close_price - low_7d) / (high_7d - low_7d) * 100

    if position_7d > MAX_POSITION_7D:
        return None

    current_body = body_percent(signal)
    avg_body_24 = sum(body_percent(candle) for candle in prev_24) / LOOKBACK_24
    if avg_body_24 == 0:
        return None

    body_expansion = current_body / avg_body_24
    if body_expansion < MIN_BODY_EXPANSION:
        return None

    current_volume = float(signal[5])
    volume_ma24 = sum(float(candle[5]) for candle in prev_24) / LOOKBACK_24
    if volume_ma24 == 0:
        return None

    volume_ratio = current_volume / volume_ma24
    if volume_ratio < MIN_VOLUME_RATIO_MA24:
        return None

    close_24h_ago = float(klines[-(CANDLES_24H + 1)][4])
    if close_24h_ago == 0:
        return None

    prior_24h_return = (close_price - close_24h_ago) / close_24h_ago * 100
    if prior_24h_return < MIN_RETURN_PREV_24H:
        return None
    if prior_24h_return > MAX_RETURN_PREV_24H:
        return None

    prev_24_closes = [float(candle[4]) for candle in prev_24]
    ma24 = ma_from_closes(prev_24_closes)

    return {
        "signal_close": close_price,
        "position_7d_percent": position_7d,
        "body_expansion_ratio": body_expansion,
        "volume_ratio_ma24": volume_ratio,
        "prior_24h_return": prior_24h_return,
        "ma24": ma24,
        "ma24_slope_percent": ma24_slope_percent(klines),
    }


def measure_forward_outcomes(
    symbol: str,
    search_dt: datetime,
    signal_close: float,
) -> dict:
    search_end_ms = int(search_dt.timestamp() * 1000)
    forward_end_ms = search_end_ms + FORWARD_HOURS * 60 * 60 * 1000
    forward_klines = fetch_klines_forward(symbol, search_end_ms, forward_end_ms)

    max_high = signal_close
    min_low = signal_close
    max_profit_dt = search_dt

    for kline in forward_klines:
        if kline_close_dt(kline) > search_dt + timedelta(hours=24):
            break

        high_price = float(kline[2])
        low_price = float(kline[3])
        open_dt = kline_open_dt(kline)

        if high_price > max_high:
            max_high = high_price
            max_profit_dt = open_dt

        min_low = min(min_low, low_price)

    max_profit = (max_high - signal_close) / signal_close * 100
    max_drawdown = (signal_close - min_low) / signal_close * 100

    def return_at(hours: int) -> float | None:
        close_price = get_close_at_or_before(forward_klines, search_dt + timedelta(hours=hours))
        if close_price is None:
            return None
        return (close_price - signal_close) / signal_close * 100

    return {
        "return_after_6h": return_at(6),
        "return_after_12h": return_at(12),
        "return_after_24h": return_at(24),
        "max_profit_24h": max_profit,
        "max_drawdown_24h": max_drawdown,
        "time_to_max_profit_kst": format_time_kst(max_profit_dt),
        "time_to_max_profit_utc": format_time_utc(max_profit_dt),
        "profit_10_occurred": max_profit >= PROFIT_TARGET_10_PCT,
        "profit_20_occurred": max_profit >= PROFIT_TARGET_20_PCT,
    }


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def save_results(matches: list[ScoutMatch]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "search_time_kst",
        "search_time_utc",
        "symbol",
        "signal_close",
        "position_7d_percent",
        "body_expansion_ratio",
        "volume_ratio_ma24",
        "prior_24h_return",
        "ma24",
        "ma24_slope_percent",
        "return_after_6h",
        "return_after_12h",
        "return_after_24h",
        "max_profit_24h",
        "max_drawdown_24h",
        "time_to_max_profit_kst",
        "time_to_max_profit_utc",
        "profit_10_occurred",
        "profit_20_occurred",
    ]

    rows: list[dict[str, str]] = []
    for match in matches:
        rows.append(
            {
                "search_time_kst": match.search_time_kst,
                "search_time_utc": match.search_time_utc,
                "symbol": match.symbol,
                "signal_close": f"{match.signal_close:.8f}",
                "position_7d_percent": f"{match.position_7d_percent:.4f}",
                "body_expansion_ratio": f"{match.body_expansion_ratio:.4f}",
                "volume_ratio_ma24": f"{match.volume_ratio_ma24:.4f}",
                "prior_24h_return": f"{match.prior_24h_return:.4f}",
                "ma24": f"{match.ma24:.8f}",
                "ma24_slope_percent": f"{match.ma24_slope_percent:.4f}",
                "return_after_6h": (
                    f"{match.return_after_6h:.4f}" if match.return_after_6h is not None else ""
                ),
                "return_after_12h": (
                    f"{match.return_after_12h:.4f}" if match.return_after_12h is not None else ""
                ),
                "return_after_24h": (
                    f"{match.return_after_24h:.4f}" if match.return_after_24h is not None else ""
                ),
                "max_profit_24h": f"{match.max_profit_24h:.4f}",
                "max_drawdown_24h": f"{match.max_drawdown_24h:.4f}",
                "time_to_max_profit_kst": match.time_to_max_profit_kst,
                "time_to_max_profit_utc": match.time_to_max_profit_utc,
                "profit_10_occurred": "YES" if match.profit_10_occurred else "NO",
                "profit_20_occurred": "YES" if match.profit_20_occurred else "NO",
            }
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_scan_results(matches: list[ScoutMatch]) -> None:
    for search_kst, _, search_dt in get_scan_times():
        time_label = search_kst.split(" ")[1][:5]
        print(f"\n===== 2026-06-15 {time_label} KST =====")

        time_matches = [m for m in matches if m.search_dt == search_dt]
        if not time_matches:
            print("  (no matches)")
            continue

        for match in sorted(time_matches, key=lambda item: item.prior_24h_return, reverse=True):
            print(f"\n  {match.symbol}")
            print(f"    signal_close: {match.signal_close:.8f}")
            print(f"    position_7d_percent: {match.position_7d_percent:.2f}%")
            print(f"    body_expansion_ratio: {match.body_expansion_ratio:.2f}")
            print(f"    volume_ratio_ma24: {match.volume_ratio_ma24:.2f}")
            print(f"    prior_24h_return: {match.prior_24h_return:.2f}%")
            print(f"    MA24: {match.ma24:.8f} | MA24 slope: {match.ma24_slope_percent:.2f}%")
            print(f"    forward 6H: {fmt_pct(match.return_after_6h)}")
            print(f"    forward 12H: {fmt_pct(match.return_after_12h)}")
            print(f"    forward 24H: {fmt_pct(match.return_after_24h)}")
            print(f"    max profit 24H: {match.max_profit_24h:+.2f}%")
            print(f"    max drawdown 24H: {match.max_drawdown_24h:.2f}%")
            print(f"    time to max profit: {match.time_to_max_profit_kst}")
            print(
                f"    10% profit: {'YES' if match.profit_10_occurred else 'NO'} | "
                f"20% profit: {'YES' if match.profit_20_occurred else 'NO'}"
            )


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def group_mean(records: list[ScoutMatch], attr: str) -> float | None:
    values = [getattr(record, attr) for record in records]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return statistics.mean(values)


def classify_success(match: ScoutMatch) -> bool:
    return match.max_profit_24h >= MEANINGFUL_MAX_PROFIT_PCT


def classify_failed(match: ScoutMatch) -> bool:
    return match.return_after_24h is not None and match.return_after_24h < 0


def print_summary(matches: list[ScoutMatch]) -> None:
    print("\n===== SUMMARY =====")
    print(f"Total matches: {len(matches)}")

    unique_symbols = sorted({match.symbol for match in matches})
    print(f"Unique symbols ({len(unique_symbols)}): {', '.join(unique_symbols)}")

    returns_6h = [m.return_after_6h for m in matches if m.return_after_6h is not None]
    returns_12h = [m.return_after_12h for m in matches if m.return_after_12h is not None]
    returns_24h = [m.return_after_24h for m in matches if m.return_after_24h is not None]
    max_profits = [m.max_profit_24h for m in matches]
    max_drawdowns = [m.max_drawdown_24h for m in matches]

    print(f"Average 6H return: {fmt_pct(average(returns_6h))}")
    print(f"Average 12H return: {fmt_pct(average(returns_12h))}")
    print(f"Average 24H return: {fmt_pct(average(returns_24h))}")
    print(f"Average max profit: {fmt_pct(average(max_profits))}")
    print(f"Average max drawdown: {fmt_pct(average(max_drawdowns))}")

    winners_10 = sum(1 for m in matches if m.profit_10_occurred)
    winners_20 = sum(1 for m in matches if m.profit_20_occurred)
    print(f"10% winners: {winners_10}/{len(matches)}")
    print(f"20% winners: {winners_20}/{len(matches)}")

    if matches:
        best = max(matches, key=lambda m: m.return_after_24h or float("-inf"))
        worst = min(matches, key=lambda m: m.return_after_24h or float("inf"))
        print(
            f"Best symbol: {best.symbol} at {best.search_time_kst} "
            f"(24H {fmt_pct(best.return_after_24h)}, max {best.max_profit_24h:+.2f}%)"
        )
        print(
            f"Worst symbol: {worst.symbol} at {worst.search_time_kst} "
            f"(24H {fmt_pct(worst.return_after_24h)}, max {worst.max_profit_24h:+.2f}%)"
        )
    else:
        print("Best symbol: N/A")
        print("Worst symbol: N/A")

    print("===================")


def print_learning_section(matches: list[ScoutMatch]) -> None:
    print("\n===== LEARNING SECTION =====")

    if not matches:
        print("No matches to analyze on this unseen date.")
        return

    successful = [m for m in matches if classify_success(m)]
    failed = [m for m in matches if classify_failed(m)]
    middle = [m for m in matches if m not in successful and m not in failed]

    print(f"\n1. Meaningful trends (max profit >= {MEANINGFUL_MAX_PROFIT_PCT:.0f}%): {len(successful)}")
    for match in successful:
        print(
            f"   {match.symbol} @ {match.search_time_kst.split(' ')[1][:5]} | "
            f"pos7d {match.position_7d_percent:.1f}% | max {match.max_profit_24h:+.1f}% | "
            f"24H close {fmt_pct(match.return_after_24h)}"
        )

    print(f"\n2. Failed matches (24H return < 0%): {len(failed)}")
    for match in failed:
        print(
            f"   {match.symbol} @ {match.search_time_kst.split(' ')[1][:5]} | "
            f"pos7d {match.position_7d_percent:.1f}% | max {match.max_profit_24h:+.1f}% | "
            f"24H close {fmt_pct(match.return_after_24h)}"
        )

    print(f"\n3. Middle / mixed outcomes: {len(middle)}")
    for match in middle:
        print(
            f"   {match.symbol} @ {match.search_time_kst.split(' ')[1][:5]} | "
            f"max {match.max_profit_24h:+.1f}% | 24H {fmt_pct(match.return_after_24h)}"
        )

    if successful and failed:
        print("\n4. Successful match averages:")
        for metric in [
            "position_7d_percent",
            "body_expansion_ratio",
            "volume_ratio_ma24",
            "prior_24h_return",
            "ma24_slope_percent",
            "max_drawdown_24h",
        ]:
            value = group_mean(successful, metric)
            if value is not None:
                print(f"   {metric}: {value:.4f}")

        print("\n5. Failed match averages:")
        for metric in [
            "position_7d_percent",
            "body_expansion_ratio",
            "volume_ratio_ma24",
            "prior_24h_return",
            "ma24_slope_percent",
            "max_drawdown_24h",
        ]:
            value = group_mean(failed, metric)
            if value is not None:
                print(f"   {metric}: {value:.4f}")

    print("\n6. Candidate filters for Scout Learning v2 (research proposals only):")

    proposals: list[str] = []
    if successful and failed:
        s_pos = group_mean(successful, "position_7d_percent")
        f_pos = group_mean(failed, "position_7d_percent")
        if s_pos is not None and f_pos is not None and s_pos < f_pos:
            proposals.append(
                f"Consider tightening position_7d_percent below ~{max(s_pos, 50):.0f}% "
                f"(successful avg {s_pos:.1f}% vs failed avg {f_pos:.1f}%)."
            )

        s_vol = group_mean(successful, "volume_ratio_ma24")
        f_vol = group_mean(failed, "volume_ratio_ma24")
        if s_vol is not None and f_vol is not None and s_vol > f_vol:
            proposals.append(
                f"Consider requiring volume_ratio_ma24 >= {max(1.0, f_vol):.1f} "
                f"(successful avg {s_vol:.2f}x vs failed avg {f_vol:.2f}x)."
            )

        s_slope = group_mean(successful, "ma24_slope_percent")
        f_slope = group_mean(failed, "ma24_slope_percent")
        if s_slope is not None and f_slope is not None:
            proposals.append(
                f"Consider capping MA24 slope near {max(abs(s_slope), 3):.1f}% "
                f"to prefer flatter bases (successful avg {s_slope:.2f}% "
                f"vs failed avg {f_slope:.2f}%)."
            )

    if not proposals:
        proposals.append(
            "Sample size is small on this date; collect more unseen backtests before changing filters."
        )

    for index, proposal in enumerate(proposals[:2], start=1):
        print(f"   Candidate {index}: {proposal}")

    print("============================")


def main() -> None:
    try:
        scan_times = get_scan_times()
        eligible_symbols = get_eligible_symbols()

        if not eligible_symbols:
            print("Error: no eligible USDT perpetual symbols found.")
            return

        print("Scout learning v2 backtest starting.")
        print("Unseen date: 2026-06-15 KST")
        print(f"Universe size: {len(eligible_symbols)} symbols")
        print(f"Scan times: {len(scan_times)}")
        print("Research only. Scout Learning v1 conditions. No trading simulation.")

        all_matches: list[ScoutMatch] = []
        symbols = sorted(eligible_symbols)

        for search_kst, search_utc, search_dt in scan_times:
            print(f"\nScanning {search_kst} KST...")
            end_ms = int(search_dt.timestamp() * 1000)
            time_matches = 0

            for index, symbol in enumerate(symbols, start=1):
                if index % 100 == 0 or index == len(symbols):
                    print(f"  progress: {index}/{len(symbols)}")

                try:
                    klines = fetch_klines_before(symbol, end_ms, KLINES_NEEDED)
                    result = evaluate_hypothesis(klines)
                    if result is None:
                        continue

                    forward = measure_forward_outcomes(
                        symbol, search_dt, result["signal_close"]
                    )

                    all_matches.append(
                        ScoutMatch(
                            search_time_kst=search_kst,
                            search_time_utc=search_utc,
                            search_dt=search_dt,
                            symbol=symbol,
                            signal_close=result["signal_close"],
                            position_7d_percent=result["position_7d_percent"],
                            body_expansion_ratio=result["body_expansion_ratio"],
                            volume_ratio_ma24=result["volume_ratio_ma24"],
                            prior_24h_return=result["prior_24h_return"],
                            ma24=result["ma24"],
                            ma24_slope_percent=result["ma24_slope_percent"],
                            return_after_6h=forward["return_after_6h"],
                            return_after_12h=forward["return_after_12h"],
                            return_after_24h=forward["return_after_24h"],
                            max_profit_24h=forward["max_profit_24h"],
                            max_drawdown_24h=forward["max_drawdown_24h"],
                            time_to_max_profit_kst=forward["time_to_max_profit_kst"],
                            time_to_max_profit_utc=forward["time_to_max_profit_utc"],
                            profit_10_occurred=forward["profit_10_occurred"],
                            profit_20_occurred=forward["profit_20_occurred"],
                        )
                    )
                    time_matches += 1
                except urllib.error.HTTPError:
                    continue

                time.sleep(API_SLEEP_SEC)

            print(f"  matches: {time_matches}")

        save_results(all_matches)
        print_scan_results(all_matches)
        print_summary(all_matches)
        print_learning_section(all_matches)
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
