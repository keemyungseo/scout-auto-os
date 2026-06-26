import csv
import json
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
TOP_N = 3

V2_MAX_POSITION_7D = 60.0
V2_MIN_BODY_EXPANSION = 1.5
V2_MAX_BODY_EXPANSION = 3.5
V2_MIN_VOLUME_RATIO = 1.2
V2_MAX_VOLUME_RATIO = 3.0
V2_MIN_RETURN_24H = 10.0
V2_MAX_RETURN_24H = 80.0
V2_MIN_MA24_SLOPE = 0.0
V2_MAX_MA24_SLOPE = 5.0

INTERVAL = "2h"
INTERVAL_MS = 2 * 60 * 60 * 1000
KLINES_NEEDED = LOOKBACK_7D + LOOKBACK_24 + 1
FORWARD_HOURS = 24
MAX_LIMIT = 1500
API_SLEEP_SEC = 0.03

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "top3_vs_scout_learning_v2_check.csv"


@dataclass
class SymbolSnapshot:
    scan_time_kst: str
    scan_time_utc: str
    scan_dt: datetime
    symbol: str
    current_close: float
    return_prev_24h_percent: float
    position_7d_percent: float
    body_expansion_ratio: float
    volume_ratio_ma24: float
    ma24_slope_percent: float
    scout_v2_match: bool
    scout_v2_fail_reasons: str
    top3_rank: int | None
    is_top3: bool
    return_after_6h: float | None
    return_after_12h: float | None
    return_after_24h: float | None
    max_profit_24h: float | None
    max_drawdown_24h: float | None


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


def compute_symbol_metrics(klines: list[list]) -> dict | None:
    if len(klines) < KLINES_NEEDED:
        return None

    signal = klines[-1]
    prev_84 = klines[-(LOOKBACK_7D + 1) : -1]
    prev_24 = klines[-(LOOKBACK_24 + 1) : -1]
    prior_24 = klines[-(LOOKBACK_24 * 2 + 1) : -(LOOKBACK_24 + 1)]

    close_price = float(signal[4])
    if not (MIN_PRICE <= close_price <= MAX_PRICE):
        return None

    low_7d = min(float(candle[3]) for candle in prev_84)
    high_7d = max(float(candle[2]) for candle in prev_84)
    if high_7d == low_7d:
        position_7d = 50.0
    else:
        position_7d = (close_price - low_7d) / (high_7d - low_7d) * 100

    current_body = body_percent(signal)
    avg_body_24 = sum(body_percent(candle) for candle in prev_24) / LOOKBACK_24
    if avg_body_24 == 0:
        body_expansion = 0.0
    else:
        body_expansion = current_body / avg_body_24

    current_volume = float(signal[5])
    volume_ma24 = sum(float(candle[5]) for candle in prev_24) / LOOKBACK_24
    if volume_ma24 == 0:
        volume_ratio = 0.0
    else:
        volume_ratio = current_volume / volume_ma24

    close_24h_ago = float(klines[-(CANDLES_24H + 1)][4])
    if close_24h_ago == 0:
        return None
    return_prev_24h = (close_price - close_24h_ago) / close_24h_ago * 100

    ma24_now = ma_from_closes([float(candle[4]) for candle in prev_24])
    ma24_prior = ma_from_closes([float(candle[4]) for candle in prior_24])
    if ma24_prior == 0:
        ma24_slope = 0.0
    else:
        ma24_slope = (ma24_now - ma24_prior) / ma24_prior * 100

    return {
        "current_close": close_price,
        "return_prev_24h_percent": return_prev_24h,
        "position_7d_percent": position_7d,
        "body_expansion_ratio": body_expansion,
        "volume_ratio_ma24": volume_ratio,
        "ma24_slope_percent": ma24_slope,
    }


def evaluate_scout_v2(metrics: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []

    if metrics["position_7d_percent"] > V2_MAX_POSITION_7D:
        failures.append(f"position_7d>{V2_MAX_POSITION_7D:.0f}")
    if metrics["body_expansion_ratio"] < V2_MIN_BODY_EXPANSION:
        failures.append(f"body_expansion<{V2_MIN_BODY_EXPANSION}")
    if metrics["body_expansion_ratio"] > V2_MAX_BODY_EXPANSION:
        failures.append(f"body_expansion>{V2_MAX_BODY_EXPANSION}")
    if metrics["volume_ratio_ma24"] < V2_MIN_VOLUME_RATIO:
        failures.append(f"volume_ratio<{V2_MIN_VOLUME_RATIO}")
    if metrics["volume_ratio_ma24"] > V2_MAX_VOLUME_RATIO:
        failures.append(f"volume_ratio>{V2_MAX_VOLUME_RATIO}")
    if metrics["return_prev_24h_percent"] < V2_MIN_RETURN_24H:
        failures.append(f"return_24h<{V2_MIN_RETURN_24H:.0f}")
    if metrics["return_prev_24h_percent"] > V2_MAX_RETURN_24H:
        failures.append(f"return_24h>{V2_MAX_RETURN_24H:.0f}")
    if metrics["ma24_slope_percent"] < V2_MIN_MA24_SLOPE:
        failures.append(f"ma24_slope<{V2_MIN_MA24_SLOPE:.0f}")
    if metrics["ma24_slope_percent"] > V2_MAX_MA24_SLOPE:
        failures.append(f"ma24_slope>{V2_MAX_MA24_SLOPE:.0f}")

    return not failures, failures


def measure_forward_outcomes(
    symbol: str,
    scan_dt: datetime,
    signal_close: float,
) -> dict[str, float | None]:
    scan_end_ms = int(scan_dt.timestamp() * 1000)
    forward_end_ms = scan_end_ms + FORWARD_HOURS * 60 * 60 * 1000
    forward_klines = fetch_klines_forward(symbol, scan_end_ms, forward_end_ms)

    max_high = signal_close
    min_low = signal_close

    for kline in forward_klines:
        if kline_close_dt(kline) > scan_dt + timedelta(hours=24):
            break
        max_high = max(max_high, float(kline[2]))
        min_low = min(min_low, float(kline[3]))

    max_profit = (max_high - signal_close) / signal_close * 100
    max_drawdown = (signal_close - min_low) / signal_close * 100

    def return_at(hours: int) -> float | None:
        close_price = get_close_at_or_before(forward_klines, scan_dt + timedelta(hours=hours))
        if close_price is None:
            return None
        return (close_price - signal_close) / signal_close * 100

    return {
        "return_after_6h": return_at(6),
        "return_after_12h": return_at(12),
        "return_after_24h": return_at(24),
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


def save_csv(rows: list[SymbolSnapshot]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scan_time_kst",
        "scan_time_utc",
        "symbol",
        "is_top3",
        "top3_rank",
        "scout_v2_match",
        "scout_v2_fail_reasons",
        "current_close",
        "return_prev_24h_percent",
        "position_7d_percent",
        "body_expansion_ratio",
        "volume_ratio_ma24",
        "ma24_slope_percent",
        "return_after_6h",
        "return_after_12h",
        "return_after_24h",
        "max_profit_24h",
        "max_drawdown_24h",
    ]

    csv_rows: list[dict[str, str]] = []
    for row in rows:
        csv_rows.append(
            {
                "scan_time_kst": row.scan_time_kst,
                "scan_time_utc": row.scan_time_utc,
                "symbol": row.symbol,
                "is_top3": "YES" if row.is_top3 else "NO",
                "top3_rank": str(row.top3_rank) if row.top3_rank is not None else "",
                "scout_v2_match": "YES" if row.scout_v2_match else "NO",
                "scout_v2_fail_reasons": row.scout_v2_fail_reasons,
                "current_close": f"{row.current_close:.8f}",
                "return_prev_24h_percent": f"{row.return_prev_24h_percent:.4f}",
                "position_7d_percent": f"{row.position_7d_percent:.4f}",
                "body_expansion_ratio": f"{row.body_expansion_ratio:.4f}",
                "volume_ratio_ma24": f"{row.volume_ratio_ma24:.4f}",
                "ma24_slope_percent": f"{row.ma24_slope_percent:.4f}",
                "return_after_6h": (
                    f"{row.return_after_6h:.4f}" if row.return_after_6h is not None else ""
                ),
                "return_after_12h": (
                    f"{row.return_after_12h:.4f}" if row.return_after_12h is not None else ""
                ),
                "return_after_24h": (
                    f"{row.return_after_24h:.4f}" if row.return_after_24h is not None else ""
                ),
                "max_profit_24h": (
                    f"{row.max_profit_24h:.4f}" if row.max_profit_24h is not None else ""
                ),
                "max_drawdown_24h": (
                    f"{row.max_drawdown_24h:.4f}" if row.max_drawdown_24h is not None else ""
                ),
            }
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def print_scan_comparison(snapshots: list[SymbolSnapshot], scan_dt: datetime) -> None:
    scan_rows = [row for row in snapshots if row.scan_dt == scan_dt]
    if not scan_rows:
        return

    time_label = scan_rows[0].scan_time_kst.split(" ")[1][:5]
    print(f"\n===== 2026-06-15 {time_label} KST =====")

    top3_rows = sorted(
        [row for row in scan_rows if row.is_top3],
        key=lambda item: item.top3_rank or 99,
    )
    print("\nTOP3 gainers:")
    for row in top3_rows:
        captured = "YES" if row.scout_v2_match else "NO"
        print(
            f"  #{row.top3_rank} {row.symbol} | ret24h {row.return_prev_24h_percent:+.2f}% | "
            f"scout_v2_captured {captured}"
        )
        if not row.scout_v2_match:
            print(f"    missed because: {row.scout_v2_fail_reasons or 'n/a'}")

    scout_rows = [row for row in scan_rows if row.scout_v2_match]
    print("\nScout Learning v2 matches:")
    if not scout_rows:
        print("  (none)")
        return

    for row in sorted(scout_rows, key=lambda item: item.return_prev_24h_percent, reverse=True):
        print(
            f"  {row.symbol} | is_top3 {'YES' if row.is_top3 else 'NO'} | "
            f"ret24h {row.return_prev_24h_percent:.2f}% | "
            f"pos7d {row.position_7d_percent:.1f}% | "
            f"bodyExp {row.body_expansion_ratio:.2f} | "
            f"vol24x {row.volume_ratio_ma24:.2f} | "
            f"ma24slope {row.ma24_slope_percent:.2f}% | "
            f"fwd24h {fmt_pct(row.return_after_24h)} | "
            f"maxP {row.max_profit_24h:+.2f}%"
        )


def print_summary(all_rows: list[SymbolSnapshot]) -> None:
    total_top3_slots = len(SCAN_TIMES_KST) * TOP_N
    top3_rows = [row for row in all_rows if row.is_top3]
    captured_top3 = [row for row in top3_rows if row.scout_v2_match]
    missed_top3 = [row for row in top3_rows if not row.scout_v2_match]
    scout_rows = [row for row in all_rows if row.scout_v2_match]
    scout_top3 = [row for row in scout_rows if row.is_top3]
    scout_non_top3 = [row for row in scout_rows if not row.is_top3]

    print("\n===== SUMMARY =====")
    print(f"1. Total TOP3 slots: {total_top3_slots}")
    print(f"2. TOP3 slots captured by Scout v2: {len(captured_top3)}")
    print(
        f"3. Capture rate: {len(captured_top3) / total_top3_slots * 100:.1f}%"
        if total_top3_slots
        else "3. Capture rate: N/A"
    )
    print(f"4. Scout v2 total matches: {len(scout_rows)}")
    print(f"5. Scout v2 matches that were TOP3: {len(scout_top3)}")
    print(f"6. Non-TOP3 Scout matches: {len(scout_non_top3)}")

    def print_avg_group(label: str, rows: list[SymbolSnapshot]) -> None:
        r6 = [row.return_after_6h for row in rows if row.return_after_6h is not None]
        r12 = [row.return_after_12h for row in rows if row.return_after_12h is not None]
        r24 = [row.return_after_24h for row in rows if row.return_after_24h is not None]
        print(
            f"   {label}: 6H {fmt_pct(average(r6))} | "
            f"12H {fmt_pct(average(r12))} | 24H {fmt_pct(average(r24))}"
        )

    print("\n7. Average forward returns:")
    print_avg_group("captured TOP3", captured_top3)
    print_avg_group("missed TOP3", missed_top3)
    print_avg_group("non-TOP3 Scout matches", scout_non_top3)

    if scout_rows:
        best = max(scout_rows, key=lambda row: row.return_after_24h or float("-inf"))
        worst = min(scout_rows, key=lambda row: row.return_after_24h or float("inf"))
        print(
            f"\n8. Best match: {best.symbol} @ {best.scan_time_kst.split(' ')[1][:5]} "
            f"(24H {fmt_pct(best.return_after_24h)}, max {best.max_profit_24h:+.2f}%)"
        )
        print(
            f"9. Worst match: {worst.symbol} @ {worst.scan_time_kst.split(' ')[1][:5]} "
            f"(24H {fmt_pct(worst.return_after_24h)}, max {worst.max_profit_24h:+.2f}%)"
        )
    else:
        print("\n8. Best match: N/A")
        print("9. Worst match: N/A")

    print("\n10. Learning notes:")

    if missed_top3:
        print("   Missed TOP3:")
        miss_reason_counts: dict[str, int] = {}
        for row in missed_top3:
            for reason in row.scout_v2_fail_reasons.split("; "):
                if reason:
                    miss_reason_counts[reason] = miss_reason_counts.get(reason, 0) + 1
            print(
                f"     {row.symbol} @ {row.scan_time_kst.split(' ')[1][:5]} "
                f"(ret24h {row.return_prev_24h_percent:+.1f}%) -> {row.scout_v2_fail_reasons}"
            )
        if miss_reason_counts:
            top_reason = max(miss_reason_counts.items(), key=lambda item: item[1])[0]
            print(f"   Most common miss reason: {top_reason}")
            print(
                f"   Possibly too strict: {top_reason} "
                f"({miss_reason_counts[top_reason]} misses)"
            )

    if scout_non_top3:
        print("   Non-TOP3 Scout captures:")
        for row in scout_non_top3:
            quality = "good" if (row.return_after_24h or 0) > 0 else "bad"
            print(
                f"     {row.symbol} @ {row.scan_time_kst.split(' ')[1][:5]} | "
                f"ret24h {row.return_prev_24h_percent:.1f}% | "
                f"fwd24h {fmt_pct(row.return_after_24h)} ({quality})"
            )

    loose_notes = _loose_condition_notes(scout_non_top3, scout_top3)
    if loose_notes:
        print(f"   Possibly too loose: {loose_notes}")

    print("====================")


def _loose_condition_notes(
    non_top3: list[SymbolSnapshot],
    top3_captured: list[SymbolSnapshot],
) -> str:
    if not non_top3:
        return "non-TOP3 capture count is low in this sample"
    bad_non_top3 = sum(1 for row in non_top3 if (row.return_after_24h or 0) < 0)
    if bad_non_top3 >= len(non_top3) / 2:
        return "many non-TOP3 captures had negative 24H forward returns"
    if len(non_top3) > len(top3_captured):
        return "Scout v2 still captures more non-TOP3 than captured TOP3"
    return ""


def main() -> None:
    try:
        scan_times = get_scan_times()
        eligible_symbols = get_eligible_symbols()

        if not eligible_symbols:
            print("Error: no eligible USDT perpetual symbols found.")
            return

        print("TOP3 vs Scout Learning v2 check starting.")
        print("Date: 2026-06-15 KST")
        print(f"Universe size: {len(eligible_symbols)} symbols")
        print("Research only. No trading simulation.")

        all_rows: list[SymbolSnapshot] = []
        symbols = sorted(eligible_symbols)

        for scan_kst, scan_utc, scan_dt in scan_times:
            print(f"\nProcessing {scan_kst} KST...")
            end_ms = int(scan_dt.timestamp() * 1000)

            candidates: list[tuple[str, dict]] = []

            for index, symbol in enumerate(symbols, start=1):
                if index % 100 == 0 or index == len(symbols):
                    print(f"  progress: {index}/{len(symbols)}")

                try:
                    klines = fetch_klines_before(symbol, end_ms, KLINES_NEEDED)
                    metrics = compute_symbol_metrics(klines)
                    if metrics is None:
                        continue
                    candidates.append((symbol, metrics))
                except urllib.error.HTTPError:
                    continue

                time.sleep(API_SLEEP_SEC)

            candidates.sort(
                key=lambda item: item[1]["return_prev_24h_percent"],
                reverse=True,
            )
            top3_symbols = {symbol: rank for rank, (symbol, _) in enumerate(candidates[:TOP_N], start=1)}

            scan_snapshots: list[SymbolSnapshot] = []
            symbols_needing_forward: set[str] = set(top3_symbols)
            for symbol, metrics in candidates:
                scout_match, failures = evaluate_scout_v2(metrics)
                if scout_match:
                    symbols_needing_forward.add(symbol)

            forward_cache: dict[str, dict[str, float | None]] = {}
            for symbol in symbols_needing_forward:
                metrics = next(item[1] for item in candidates if item[0] == symbol)
                forward_cache[symbol] = measure_forward_outcomes(
                    symbol, scan_dt, metrics["current_close"]
                )
                time.sleep(API_SLEEP_SEC)

            for symbol, metrics in candidates:
                is_top3 = symbol in top3_symbols
                scout_match, failures = evaluate_scout_v2(metrics)
                if not is_top3 and not scout_match:
                    continue

                forward = forward_cache.get(symbol, {})
                if symbol in symbols_needing_forward and not forward:
                    forward = measure_forward_outcomes(
                        symbol, scan_dt, metrics["current_close"]
                    )

                scan_snapshots.append(
                    SymbolSnapshot(
                        scan_time_kst=scan_kst,
                        scan_time_utc=scan_utc,
                        scan_dt=scan_dt,
                        symbol=symbol,
                        current_close=metrics["current_close"],
                        return_prev_24h_percent=metrics["return_prev_24h_percent"],
                        position_7d_percent=metrics["position_7d_percent"],
                        body_expansion_ratio=metrics["body_expansion_ratio"],
                        volume_ratio_ma24=metrics["volume_ratio_ma24"],
                        ma24_slope_percent=metrics["ma24_slope_percent"],
                        scout_v2_match=scout_match,
                        scout_v2_fail_reasons="; ".join(failures),
                        top3_rank=top3_symbols.get(symbol),
                        is_top3=is_top3,
                        return_after_6h=forward.get("return_after_6h"),
                        return_after_12h=forward.get("return_after_12h"),
                        return_after_24h=forward.get("return_after_24h"),
                        max_profit_24h=forward.get("max_profit_24h"),
                        max_drawdown_24h=forward.get("max_drawdown_24h"),
                    )
                )

            all_rows.extend(scan_snapshots)
            print(
                f"  TOP3: {', '.join(f'#{rank} {sym}' for sym, rank in sorted(top3_symbols.items(), key=lambda x: x[1]))}"
            )
            print(
                f"  Scout v2 matches: {sum(1 for row in scan_snapshots if row.scout_v2_match)} | "
                f"TOP3 captured: {sum(1 for row in scan_snapshots if row.is_top3 and row.scout_v2_match)}"
            )

        for _, _, scan_dt in scan_times:
            print_scan_comparison(all_rows, scan_dt)

        save_csv(all_rows)
        print_summary(all_rows)
        print(f"\nDetailed CSV saved: {OUTPUT_CSV}")

    except ValueError as exc:
        print(f"Error: {exc}")
    except urllib.error.HTTPError as exc:
        details = parse_error_message(exc.read().decode("utf-8", errors="replace"))
        print(f"Error: Binance API request failed. HTTP {exc.code}: {details}")
    except urllib.error.URLError as exc:
        print(f"Error: cannot connect to Binance. {exc.reason}")


if __name__ == "__main__":
    main()
