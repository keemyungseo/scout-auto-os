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

LOOKBACK_7D = 84
LOOKBACK_24 = 24
CANDLES_24H = 12
TOP_N = 3

MAX_POSITION_7D = 70.0
MIN_BODY_EXPANSION = 1.5
MIN_VOLUME_RATIO_MA24 = 0.8
MIN_RETURN_PREV_24H = 10.0
MAX_RETURN_PREV_24H = 80.0

INTERVAL = "2h"
INTERVAL_MS = 2 * 60 * 60 * 1000
KLINES_NEEDED = LOOKBACK_7D + 1
RANKING_KLINES = CANDLES_24H + 1
FORWARD_HOURS = 24
MAX_LIMIT = 1500
API_SLEEP_SEC = 0.03

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "scout_learning_v1.csv"


@dataclass
class ScoutMatch:
    scan_time_kst: str
    scan_time_utc: str
    scan_dt: datetime
    symbol: str
    position_7d_percent: float
    body_expansion_ratio: float
    volume_ratio_ma24: float
    return_prev_24h_percent: float
    is_top3_gainer: bool
    top3_rank: int | None
    return_after_6h: float | None
    return_after_12h: float | None
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


def compute_24h_change(klines: list[list]) -> float | None:
    if len(klines) < RANKING_KLINES:
        return None
    close_now = float(klines[-1][4])
    close_24h_ago = float(klines[-(CANDLES_24H + 1)][4])
    if close_24h_ago == 0:
        return None
    return (close_now - close_24h_ago) / close_24h_ago * 100


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

    return_prev_24h = (close_price - close_24h_ago) / close_24h_ago * 100
    if return_prev_24h < MIN_RETURN_PREV_24H:
        return None
    if return_prev_24h > MAX_RETURN_PREV_24H:
        return None

    return {
        "price_at_scan": close_price,
        "position_7d_percent": position_7d,
        "body_expansion_ratio": body_expansion,
        "volume_ratio_ma24": volume_ratio,
        "return_prev_24h_percent": return_prev_24h,
    }


def measure_forward_returns(
    symbol: str,
    scan_dt: datetime,
    price_at_scan: float,
) -> dict[str, float | None]:
    scan_end_ms = int(scan_dt.timestamp() * 1000)
    forward_end_ms = scan_end_ms + FORWARD_HOURS * 60 * 60 * 1000
    forward_klines = fetch_klines_forward(symbol, scan_end_ms, forward_end_ms)

    def return_at(hours: int) -> float | None:
        close_price = get_close_at_or_before(forward_klines, scan_dt + timedelta(hours=hours))
        if close_price is None:
            return None
        return (close_price - price_at_scan) / price_at_scan * 100

    return {
        "return_after_6h": return_at(6),
        "return_after_12h": return_at(12),
        "return_after_24h": return_at(24),
    }


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def save_results(matches: list[ScoutMatch]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scan_time_kst",
        "scan_time_utc",
        "symbol",
        "position_7d_percent",
        "body_expansion_ratio",
        "volume_ratio_ma24",
        "return_prev_24h_percent",
        "is_top3_gainer",
        "top3_rank",
        "return_after_6h",
        "return_after_12h",
        "return_after_24h",
    ]

    rows: list[dict[str, str]] = []
    for match in matches:
        rows.append(
            {
                "scan_time_kst": match.scan_time_kst,
                "scan_time_utc": match.scan_time_utc,
                "symbol": match.symbol,
                "position_7d_percent": f"{match.position_7d_percent:.4f}",
                "body_expansion_ratio": f"{match.body_expansion_ratio:.4f}",
                "volume_ratio_ma24": f"{match.volume_ratio_ma24:.4f}",
                "return_prev_24h_percent": f"{match.return_prev_24h_percent:.4f}",
                "is_top3_gainer": "YES" if match.is_top3_gainer else "NO",
                "top3_rank": str(match.top3_rank) if match.top3_rank is not None else "",
                "return_after_6h": (
                    f"{match.return_after_6h:.4f}" if match.return_after_6h is not None else ""
                ),
                "return_after_12h": (
                    f"{match.return_after_12h:.4f}" if match.return_after_12h is not None else ""
                ),
                "return_after_24h": (
                    f"{match.return_after_24h:.4f}" if match.return_after_24h is not None else ""
                ),
            }
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_scan_matches(matches: list[ScoutMatch]) -> None:
    for scan_kst, scan_utc, scan_dt in get_scan_times():
        time_matches = [m for m in matches if m.scan_dt == scan_dt]
        time_matches.sort(key=lambda item: item.return_prev_24h_percent, reverse=True)

        print(f"\nTime: {scan_kst} KST ({scan_utc})")
        if not time_matches:
            print("  (no matches)")
            continue

        for match in time_matches:
            top3_label = f"TOP3 #{match.top3_rank}" if match.is_top3_gainer else "non-TOP3"
            print(
                f"  {match.symbol} [{top3_label}] | "
                f"pos7d {match.position_7d_percent:.2f}% | "
                f"bodyExp {match.body_expansion_ratio:.2f} | "
                f"vol24x {match.volume_ratio_ma24:.2f} | "
                f"ret24h {match.return_prev_24h_percent:.2f}%"
            )


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def print_research_summary(matches: list[ScoutMatch]) -> None:
    print("\n===== SCOUT LEARNING V1 SUMMARY =====")
    print(f"Total matches: {len(matches)}")

    unique_symbols = sorted({match.symbol for match in matches})
    print(f"Symbols found ({len(unique_symbols)}): {', '.join(unique_symbols)}")

    top3_captured = sum(1 for match in matches if match.is_top3_gainer)
    total_top3_slots = len(SCAN_TIMES_KST) * TOP_N
    print(
        f"TOP3 gainers captured: {top3_captured} / {total_top3_slots} "
        f"(scout matches that were also TOP3 at that scan)"
    )

    non_top3 = sorted(
        {
            f"{match.symbol} @ {match.scan_time_kst}"
            for match in matches
            if not match.is_top3_gainer
        }
    )
    print(f"\nNon-TOP3 scout matches ({len(non_top3)}):")
    if non_top3:
        for item in non_top3:
            print(f"  {item}")
    else:
        print("  (none)")

    returns_6h = [m.return_after_6h for m in matches if m.return_after_6h is not None]
    returns_12h = [m.return_after_12h for m in matches if m.return_after_12h is not None]
    returns_24h = [m.return_after_24h for m in matches if m.return_after_24h is not None]

    print(f"\nAverage forward 6H: {fmt_pct(average(returns_6h))}")
    print(f"Average forward 12H: {fmt_pct(average(returns_12h))}")
    print(f"Average forward 24H: {fmt_pct(average(returns_24h))}")

    if matches:
        best = max(matches, key=lambda m: m.return_after_24h or float("-inf"))
        worst = min(matches, key=lambda m: m.return_after_24h or float("inf"))
        print(
            f"\nBest symbol: {best.symbol} at {best.scan_time_kst} KST "
            f"(24h forward {fmt_pct(best.return_after_24h)})"
        )
        print(
            f"Worst symbol: {worst.symbol} at {worst.scan_time_kst} KST "
            f"(24h forward {fmt_pct(worst.return_after_24h)})"
        )
    else:
        print("\nBest symbol: N/A")
        print("Worst symbol: N/A")

    print("=====================================")


def main() -> None:
    try:
        scan_times = get_scan_times()
        eligible_symbols = get_eligible_symbols()

        if not eligible_symbols:
            print("Error: no eligible USDT perpetual symbols found.")
            return

        print("Scout learning v1 starting.")
        print(f"Universe size: {len(eligible_symbols)} symbols")
        print(f"Scan times: {len(scan_times)}")
        print("Research only. No trading simulation.")

        all_matches: list[ScoutMatch] = []
        symbols = sorted(eligible_symbols)

        for scan_kst, scan_utc, scan_dt in scan_times:
            print(f"\nScanning {scan_kst} KST...")
            end_ms = int(scan_dt.timestamp() * 1000)

            ranking_candidates: list[tuple[str, float, dict]] = []
            scan_matches: list[ScoutMatch] = []

            for index, symbol in enumerate(symbols, start=1):
                if index % 100 == 0 or index == len(symbols):
                    print(f"  progress: {index}/{len(symbols)}")

                try:
                    klines = fetch_klines_before(symbol, end_ms, KLINES_NEEDED)
                    change_24h = compute_24h_change(klines)
                    if change_24h is not None:
                        price = float(klines[-1][4])
                        if MIN_PRICE <= price <= MAX_PRICE:
                            ranking_candidates.append((symbol, change_24h))

                    result = evaluate_hypothesis(klines)
                    if result is None:
                        continue

                    forward = measure_forward_returns(
                        symbol, scan_dt, result["price_at_scan"]
                    )
                    scan_matches.append(
                        ScoutMatch(
                            scan_time_kst=scan_kst,
                            scan_time_utc=scan_utc,
                            scan_dt=scan_dt,
                            symbol=symbol,
                            position_7d_percent=result["position_7d_percent"],
                            body_expansion_ratio=result["body_expansion_ratio"],
                            volume_ratio_ma24=result["volume_ratio_ma24"],
                            return_prev_24h_percent=result["return_prev_24h_percent"],
                            is_top3_gainer=False,
                            top3_rank=None,
                            return_after_6h=forward["return_after_6h"],
                            return_after_12h=forward["return_after_12h"],
                            return_after_24h=forward["return_after_24h"],
                        )
                    )
                except urllib.error.HTTPError:
                    continue

                time.sleep(API_SLEEP_SEC)

            ranking_candidates.sort(key=lambda item: item[1], reverse=True)
            top3_map = {
                symbol: rank
                for rank, (symbol, _) in enumerate(ranking_candidates[:TOP_N], start=1)
            }
            top3_list = ", ".join(
                f"#{rank} {sym}" for sym, rank in sorted(top3_map.items(), key=lambda x: x[1])
            )
            print(f"  TOP3: {top3_list}")

            for match in scan_matches:
                if match.symbol in top3_map:
                    match.is_top3_gainer = True
                    match.top3_rank = top3_map[match.symbol]

            all_matches.extend(scan_matches)
            print(f"  hypothesis matches: {len(scan_matches)}")

        save_results(all_matches)
        print_scan_matches(all_matches)
        print_research_summary(all_matches)
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
