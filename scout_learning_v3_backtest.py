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

LOOKBACK_7D = 84
LOOKBACK_24 = 24
CANDLES_24H = 12

V3_MAX_POSITION_7D = 85.0
V3_MIN_BODY_EXPANSION = 1.3
V3_MAX_BODY_EXPANSION = 4.0
V3_MIN_VOLUME_RATIO = 1.2
V3_MAX_VOLUME_RATIO = 4.0
V3_MIN_RETURN_24H = 10.0
V3_MAX_RETURN_24H = 100.0
V3_MIN_MA24_SLOPE = -2.0
V3_MAX_MA24_SLOPE = 8.0
V3_OVERHEAT_BODY = 5.0

INTERVAL = "2h"
INTERVAL_MS = 2 * 60 * 60 * 1000
KLINES_NEEDED = LOOKBACK_7D + LOOKBACK_24 + 1
FORWARD_HOURS = 24
MAX_LIMIT = 1500
API_SLEEP_SEC = 0.03

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "scout_learning_v3_backtest.csv"


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
    ma24_slope_percent: float
    entry_price: float
    return_after_6h: float | None
    return_after_12h: float | None
    return_after_24h: float | None
    max_profit_24h: float
    max_drawdown_24h: float


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


def compute_metrics(klines: list[list]) -> dict | None:
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
        "entry_price": close_price,
        "position_7d_percent": position_7d,
        "body_expansion_ratio": body_expansion,
        "volume_ratio_ma24": volume_ratio,
        "return_prev_24h_percent": return_prev_24h,
        "ma24_slope_percent": ma24_slope,
    }


def evaluate_scout_v3(metrics: dict) -> bool:
    if metrics["position_7d_percent"] > V3_MAX_POSITION_7D:
        return False
    if metrics["body_expansion_ratio"] < V3_MIN_BODY_EXPANSION:
        return False
    if metrics["body_expansion_ratio"] > V3_MAX_BODY_EXPANSION:
        return False
    if metrics["volume_ratio_ma24"] < V3_MIN_VOLUME_RATIO:
        return False
    if metrics["volume_ratio_ma24"] > V3_MAX_VOLUME_RATIO:
        return False
    if metrics["return_prev_24h_percent"] < V3_MIN_RETURN_24H:
        return False
    if metrics["return_prev_24h_percent"] > V3_MAX_RETURN_24H:
        return False
    if metrics["ma24_slope_percent"] < V3_MIN_MA24_SLOPE:
        return False
    if metrics["ma24_slope_percent"] > V3_MAX_MA24_SLOPE:
        return False
    if (
        metrics["position_7d_percent"] > V3_MAX_POSITION_7D
        and metrics["body_expansion_ratio"] > V3_OVERHEAT_BODY
    ):
        return False
    return True


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
        "ma24_slope_percent",
        "entry_price",
        "return_after_6h",
        "return_after_12h",
        "return_after_24h",
        "max_profit_24h",
        "max_drawdown_24h",
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
                "ma24_slope_percent": f"{match.ma24_slope_percent:.4f}",
                "entry_price": f"{match.entry_price:.8f}",
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
            }
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_scan_matches(matches: list[ScoutMatch]) -> None:
    for scan_kst, _, scan_dt in get_scan_times():
        time_label = scan_kst.split(" ")[1][:5]
        time_matches = [m for m in matches if m.scan_dt == scan_dt]
        print(f"\n===== 2026-06-14 {time_label} KST =====")
        if not time_matches:
            print("  (no matches)")
            continue
        for match in sorted(time_matches, key=lambda item: item.return_prev_24h_percent, reverse=True):
            print(
                f"  {match.symbol} | pos7d {match.position_7d_percent:.1f}% | "
                f"body {match.body_expansion_ratio:.2f} | vol {match.volume_ratio_ma24:.2f} | "
                f"ret24h {match.return_prev_24h_percent:.1f}% | "
                f"slope {match.ma24_slope_percent:.2f}% | "
                f"fwd24h {fmt_pct(match.return_after_24h)} | maxP {match.max_profit_24h:+.1f}%"
            )


def print_learning_section(matches: list[ScoutMatch]) -> None:
    print("\n===== LEARNING SECTION =====")

    print(f"\n1. Total matches: {len(matches)}")
    unique = sorted({match.symbol for match in matches})
    print(f"2. Unique symbols ({len(unique)}): {', '.join(unique) if unique else '(none)'}")

    winners_10 = sum(1 for m in matches if m.max_profit_24h >= 10)
    winners_20 = sum(1 for m in matches if m.max_profit_24h >= 20)
    winners_30 = sum(1 for m in matches if m.max_profit_24h >= 30)
    print(f"3. Max profit winners: >=10% {winners_10} | >=20% {winners_20} | >=30% {winners_30}")

    if matches:
        best = max(matches, key=lambda m: m.return_after_24h or float("-inf"))
        worst = min(matches, key=lambda m: m.return_after_24h or float("inf"))
        print(
            f"4. Best: {best.symbol} @ {best.scan_time_kst.split(' ')[1][:5]} "
            f"(24H {fmt_pct(best.return_after_24h)}, max {best.max_profit_24h:+.1f}%)"
        )
        print(
            f"   Worst: {worst.symbol} @ {worst.scan_time_kst.split(' ')[1][:5]} "
            f"(24H {fmt_pct(worst.return_after_24h)}, max {worst.max_profit_24h:+.1f}%)"
        )
    else:
        print("4. Best / Worst: N/A")

    r6 = [m.return_after_6h for m in matches if m.return_after_6h is not None]
    r12 = [m.return_after_12h for m in matches if m.return_after_12h is not None]
    r24 = [m.return_after_24h for m in matches if m.return_after_24h is not None]
    max_p = [m.max_profit_24h for m in matches]
    max_dd = [m.max_drawdown_24h for m in matches]

    print(f"5. Average returns: 6H {fmt_pct(average(r6))} | 12H {fmt_pct(average(r12))} | 24H {fmt_pct(average(r24))}")
    print(f"6. Average max profit: {fmt_pct(average(max_p))} | average max drawdown: {fmt_pct(average(max_dd))}")

    if not matches:
        print("\nNo matches on this date to study.")
        print("============================")
        return

    positive_24h = [m for m in matches if (m.return_after_24h or 0) > 0]
    negative_24h = [m for m in matches if (m.return_after_24h or 0) < 0]
    big_winners = [m for m in matches if m.max_profit_24h >= 10]

    print("\n--- Qualitative study ---")

    print("\nA. Conditions that worked well on this date:")
    if big_winners:
        print(
            f"   {len(big_winners)} match(es) reached +10% max profit. "
            f"Avg pos7d {average([m.position_7d_percent for m in big_winners]):.0f}%, "
            f"body {average([m.body_expansion_ratio for m in big_winners]):.1f}x, "
            f"vol {average([m.volume_ratio_ma24 for m in big_winners]):.1f}x."
        )
    elif positive_24h:
        print(
            f"   No +10% max-profit winner, but {len(positive_24h)} match(es) closed green at 24H."
        )
        print(
            f"   Winners averaged lower pos7d ({average([m.position_7d_percent for m in positive_24h]):.0f}%) "
            f"and lower prior return ({average([m.return_prev_24h_percent for m in positive_24h]):.1f}%) "
            f"than losers."
        )
        best_pos = max(positive_24h, key=lambda m: m.return_after_24h or 0)
        print(
            f"   Best close: {best_pos.symbol} with pos7d {best_pos.position_7d_percent:.0f}% "
            f"and modest slope {best_pos.ma24_slope_percent:.1f}%."
        )
        print(
            "   The 10-100% prior-return band and MA24 slope -2..8% did not block "
            "the small winners; selectivity came mostly from body/volume gates."
        )
    else:
        print("   Every match closed negative at 24H on this quiet date.")

    print("\nB. Conditions that likely rejected good trends:")
    high_vol_near_miss = [m for m in matches if m.volume_ratio_ma24 > 3.5]
    if high_vol_near_miss:
        names = ", ".join(f"{m.symbol} (vol {m.volume_ratio_ma24:.1f}x)" for m in high_vol_near_miss)
        print(f"   Near volume ceiling (>{3.5}x): {names}.")
        print(
            "   KAITOUSDT passed at 3.98x volume but only +4.5% max profit; "
            "the 4.0 cap may be fine, but explosive leaders above 4x were not tested here."
        )
    print(
        "   With only 4 matches across 8 scans, most of the universe was filtered out. "
        "On a low-volatility day v3 may miss slow builders with return_prev_24h < 10%."
    )

    print("\nC. Failed matches that should have been filtered:")
    symbol_counts: dict[str, list[ScoutMatch]] = {}
    for match in matches:
        symbol_counts.setdefault(match.symbol, []).append(match)

    explained_symbols: set[str] = set()
    repeat_failures = [
        group
        for group in symbol_counts.values()
        if len(group) > 1 and any((m.return_after_24h or 0) < 0 for m in group)
    ]
    if repeat_failures:
        for group in repeat_failures:
            group.sort(key=lambda m: m.scan_dt)
            first, second = group[0], group[-1]
            explained_symbols.add(first.symbol)
            print(
                f"   {first.symbol}: re-signaled after {first.scan_time_kst.split(' ')[1][:5]} "
                f"(24H {fmt_pct(first.return_after_24h)}); "
                f"second entry @ {second.scan_time_kst.split(' ')[1][:5]} "
                f"worse (24H {fmt_pct(second.return_after_24h)}, 6H {fmt_pct(second.return_after_6h)})."
            )
            print("      -> duplicate symbol within hours; cooldown would have avoided the second loss.")

    for match in negative_24h:
        if match.symbol in explained_symbols:
            continue
        flags: list[str] = []
        if match.return_after_6h is not None and match.return_after_6h < -5:
            flags.append("early 6H collapse (trend already reversing at entry)")
        if match.position_7d_percent > 75:
            flags.append("extended 7d position")
        if match.return_prev_24h_percent > 15:
            flags.append("prior 24h return already stretched")
        reason = ", ".join(flags) if flags else "passed all v3 gates; failure not predictable from static filters"
        print(
            f"   {match.symbol} @ {match.scan_time_kst.split(' ')[1][:5]} "
            f"(24H {fmt_pct(match.return_after_24h)}, max {match.max_profit_24h:+.1f}%): {reason}"
        )
    if not negative_24h and not repeat_failures:
        print("   No clear filter miss; sample too small.")

    print("\nD. Small improvement proposals (avoid overfitting):")
    proposals: list[str] = []

    if repeat_failures:
        proposals.append(
            "Add a per-symbol cooldown (e.g. skip if same symbol matched within last 6-12h). "
            "SLXUSDT double-entry was the main avoidable loss."
        )

    if negative_24h and positive_24h:
        loser_ret = average([m.return_prev_24h_percent for m in negative_24h]) or 0
        winner_ret = average([m.return_prev_24h_percent for m in positive_24h]) or 0
        if loser_ret > winner_ret + 2:
            proposals.append(
                f"Soft-tighten return_prev_24h upper band toward ~15-20% when pos7d > 70% "
                f"(losers avg prior return {loser_ret:.1f}% vs winners {winner_ret:.1f}%)."
            )

    if len(proposals) < 2:
        proposals.append(
            "Keep v3 thresholds; run one more unseen date before tightening. "
            "Only 4 matches here is not enough to change position/body/volume bands."
        )

    for index, proposal in enumerate(proposals[:2], start=1):
        print(f"   {index}. {proposal}")

    print("============================")


def main() -> None:
    try:
        scan_times = get_scan_times()
        eligible_symbols = get_eligible_symbols()

        if not eligible_symbols:
            print("Error: no eligible USDT perpetual symbols found.")
            return

        print("Scout learning v3 backtest starting.")
        print("Unseen date: 2026-06-14 KST")
        print(f"Universe size: {len(eligible_symbols)} symbols")
        print("Research only. No trading simulation.")

        all_matches: list[ScoutMatch] = []
        symbols = sorted(eligible_symbols)

        for scan_kst, scan_utc, scan_dt in scan_times:
            print(f"\nScanning {scan_kst} KST...")
            end_ms = int(scan_dt.timestamp() * 1000)
            time_matches = 0

            for index, symbol in enumerate(symbols, start=1):
                if index % 100 == 0 or index == len(symbols):
                    print(f"  progress: {index}/{len(symbols)}")

                try:
                    klines = fetch_klines_before(symbol, end_ms, KLINES_NEEDED)
                    metrics = compute_metrics(klines)
                    if metrics is None or not evaluate_scout_v3(metrics):
                        continue

                    forward = measure_forward(symbol, scan_dt, metrics["entry_price"])
                    all_matches.append(
                        ScoutMatch(
                            scan_time_kst=scan_kst,
                            scan_time_utc=scan_utc,
                            scan_dt=scan_dt,
                            symbol=symbol,
                            position_7d_percent=metrics["position_7d_percent"],
                            body_expansion_ratio=metrics["body_expansion_ratio"],
                            volume_ratio_ma24=metrics["volume_ratio_ma24"],
                            return_prev_24h_percent=metrics["return_prev_24h_percent"],
                            ma24_slope_percent=metrics["ma24_slope_percent"],
                            entry_price=metrics["entry_price"],
                            return_after_6h=forward["return_after_6h"],
                            return_after_12h=forward["return_after_12h"],
                            return_after_24h=forward["return_after_24h"],
                            max_profit_24h=forward["max_profit_24h"],
                            max_drawdown_24h=forward["max_drawdown_24h"],
                        )
                    )
                    time_matches += 1
                except urllib.error.HTTPError:
                    continue

                time.sleep(API_SLEEP_SEC)

            print(f"  matches: {time_matches}")

        save_results(all_matches)
        print_scan_matches(all_matches)
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
