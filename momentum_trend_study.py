import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

SEARCH_TIME_KST = "2026-06-13 11:00:00"
SEARCH_TIME_UTC = "2026-06-13 02:00:00"

KST_OFFSET = timedelta(hours=9)
KST_TZ = timezone(KST_OFFSET)

FUTURES_BASE_URL = "https://fapi.binance.com"
EXCHANGE_INFO_ENDPOINT = "/fapi/v1/exchangeInfo"
KLINES_ENDPOINT = "/fapi/v1/klines"

EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "XRPUSDT", "USDCUSDT"}
MIN_PRICE = 0.05
MAX_PRICE = 400.0
MIN_PRICE_INCREASE_PCT = 3.0
MAX_PRICE_INCREASE_PCT = 8.0
VOLUME_MA_PERIOD = 20
VOLUME_MA_MULTIPLIER = 3.0
EMERGENCY_STOP_MULTIPLIER = 0.975
INTERVAL = "2h"
INTERVAL_MS = 2 * 60 * 60 * 1000
KLINES_BEFORE_LIMIT = VOLUME_MA_PERIOD + 2
MAX_LIMIT = 1500
FORWARD_FETCH_DAYS = 30

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "momentum_trend_study.csv"


@dataclass
class QualifyingSymbol:
    symbol: str
    price_increase_pct: float
    volume: float
    volume_ma20: float
    trading_value_2h: float
    signal_close: float


@dataclass
class TrendStudyResult:
    symbol: str
    entry_time_kst: str
    entry_time_utc: str
    entry_price: float
    max_price: float
    max_profit_pct: float
    max_profit_time_kst: str
    max_profit_time_utc: str
    stop_occurred: bool
    stop_time_kst: str
    stop_time_utc: str
    max_adverse_excursion_pct: float
    max_favourable_excursion_pct: float
    trend_duration_hours: float
    still_holding: bool
    signal_price_increase_pct: float
    signal_trading_value_2h: float


def get_credentials() -> tuple[str, str]:
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    secret_key = os.getenv("BINANCE_SECRET_KEY", "").strip()
    return api_key, secret_key


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


def get_search_time_utc() -> datetime:
    search_dt = parse_kst_to_utc(SEARCH_TIME_KST)
    expected = datetime.strptime(SEARCH_TIME_UTC, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    if search_dt != expected:
        raise ValueError(
            f"Search time conversion mismatch: got {format_time_utc(search_dt)}, "
            f"expected {SEARCH_TIME_UTC}"
        )
    return search_dt


def public_get(api_key: str, endpoint: str, params: dict | None = None) -> dict:
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    url = f"{FUTURES_BASE_URL}{endpoint}{query}"
    request = urllib.request.Request(
        url,
        headers={"X-MBX-APIKEY": api_key},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def get_eligible_symbols(api_key: str) -> set[str]:
    exchange_info = public_get(api_key, EXCHANGE_INFO_ENDPOINT)
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


def fetch_klines_before(
    api_key: str,
    symbol: str,
    end_ms: int,
    limit: int,
) -> list[list]:
    params = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": INTERVAL,
            "endTime": end_ms,
            "limit": limit,
        }
    )
    url = f"{FUTURES_BASE_URL}{KLINES_ENDPOINT}?{params}"
    request = urllib.request.Request(
        url,
        headers={"X-MBX-APIKEY": api_key},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_klines_forward(
    api_key: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
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
        request = urllib.request.Request(
            url,
            headers={"X-MBX-APIKEY": api_key},
            method="GET",
        )
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


def evaluate_symbol_at_search(
    api_key: str,
    symbol: str,
    search_dt: datetime,
) -> QualifyingSymbol | None:
    end_ms = int(search_dt.timestamp() * 1000)
    klines = fetch_klines_before(api_key, symbol, end_ms, KLINES_BEFORE_LIMIT)

    if len(klines) < VOLUME_MA_PERIOD + 1:
        return None

    signal = klines[-1]
    close_price = float(signal[4])

    if not (MIN_PRICE <= close_price <= MAX_PRICE):
        return None

    open_price = float(signal[1])
    if open_price == 0:
        return None

    price_increase_pct = (close_price - open_price) / open_price * 100
    if not (MIN_PRICE_INCREASE_PCT <= price_increase_pct <= MAX_PRICE_INCREASE_PCT):
        return None

    prior_volumes = [float(kline[5]) for kline in klines[-(VOLUME_MA_PERIOD + 1) : -1]]
    volume_ma20 = sum(prior_volumes) / VOLUME_MA_PERIOD
    current_volume = float(signal[5])

    if volume_ma20 == 0 or current_volume < VOLUME_MA_MULTIPLIER * volume_ma20:
        return None

    return QualifyingSymbol(
        symbol=symbol,
        price_increase_pct=price_increase_pct,
        volume=current_volume,
        volume_ma20=volume_ma20,
        trading_value_2h=float(signal[7]),
        signal_close=close_price,
    )


def simulate_trend_hold(
    api_key: str,
    qualifying: QualifyingSymbol,
    search_dt: datetime,
    data_end_dt: datetime,
) -> TrendStudyResult | None:
    end_ms = int(search_dt.timestamp() * 1000)
    prior_klines = fetch_klines_before(api_key, qualifying.symbol, end_ms, 2)
    if len(prior_klines) < 1:
        return None

    signal_kline = prior_klines[-1]
    signal_open_ms = int(signal_kline[0])
    entry_start_ms = signal_open_ms + INTERVAL_MS

    forward_end_ms = int(data_end_dt.timestamp() * 1000)
    forward_klines = fetch_klines_forward(
        api_key,
        qualifying.symbol,
        entry_start_ms,
        forward_end_ms,
    )

    if not forward_klines:
        return None

    entry_kline = forward_klines[0]
    entry_price = float(entry_kline[1])
    entry_dt = kline_open_dt(entry_kline)
    stop_price = entry_price * EMERGENCY_STOP_MULTIPLIER

    max_price = entry_price
    max_profit_pct = 0.0
    max_profit_dt = entry_dt
    max_adverse_excursion_pct = 0.0
    max_favourable_excursion_pct = 0.0
    stop_occurred = False
    stop_dt: datetime | None = None
    last_dt = entry_dt

    for kline in forward_klines:
        open_dt = kline_open_dt(kline)
        high_price = float(kline[2])
        low_price = float(kline[3])
        last_dt = open_dt + timedelta(hours=2)

        adverse_pct = (entry_price - low_price) / entry_price * 100
        favourable_pct = (high_price - entry_price) / entry_price * 100
        max_adverse_excursion_pct = max(max_adverse_excursion_pct, adverse_pct)
        max_favourable_excursion_pct = max(max_favourable_excursion_pct, favourable_pct)

        if high_price > max_price:
            max_price = high_price
            max_profit_pct = (max_price - entry_price) / entry_price * 100
            max_profit_dt = open_dt

        if low_price <= stop_price:
            stop_occurred = True
            stop_dt = open_dt
            break

    trend_end_dt = stop_dt if stop_occurred else last_dt
    trend_duration_hours = (trend_end_dt - entry_dt).total_seconds() / 3600
    still_holding = not stop_occurred

    return TrendStudyResult(
        symbol=qualifying.symbol,
        entry_time_kst=format_time_kst(entry_dt),
        entry_time_utc=format_time_utc(entry_dt),
        entry_price=entry_price,
        max_price=max_price,
        max_profit_pct=max_profit_pct,
        max_profit_time_kst=format_time_kst(max_profit_dt),
        max_profit_time_utc=format_time_utc(max_profit_dt),
        stop_occurred=stop_occurred,
        stop_time_kst=format_time_kst(stop_dt) if stop_dt else "",
        stop_time_utc=format_time_utc(stop_dt) if stop_dt else "",
        max_adverse_excursion_pct=max_adverse_excursion_pct,
        max_favourable_excursion_pct=max_favourable_excursion_pct,
        trend_duration_hours=trend_duration_hours,
        still_holding=still_holding,
        signal_price_increase_pct=qualifying.price_increase_pct,
        signal_trading_value_2h=qualifying.trading_value_2h,
    )


def save_results(results: list[TrendStudyResult], search_dt: datetime) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for result in results:
        rows.append(
            {
                "search_time_kst": format_time_kst(search_dt),
                "search_time_utc": format_time_utc(search_dt),
                "symbol": result.symbol,
                "signal_price_increase_pct": f"{result.signal_price_increase_pct:.4f}",
                "signal_trading_value_2h": f"{result.signal_trading_value_2h:.2f}",
                "entry_time_kst": result.entry_time_kst,
                "entry_time_utc": result.entry_time_utc,
                "entry_price": f"{result.entry_price:.8f}",
                "max_price": f"{result.max_price:.8f}",
                "max_profit_pct": f"{result.max_profit_pct:.4f}",
                "max_profit_time_kst": result.max_profit_time_kst,
                "max_profit_time_utc": result.max_profit_time_utc,
                "stop_occurred": "YES" if result.stop_occurred else "NO",
                "stop_time_kst": result.stop_time_kst,
                "stop_time_utc": result.stop_time_utc,
                "max_adverse_excursion_pct": f"{result.max_adverse_excursion_pct:.4f}",
                "max_favourable_excursion_pct": f"{result.max_favourable_excursion_pct:.4f}",
                "trend_duration_hours": f"{result.trend_duration_hours:.2f}",
                "still_holding": "YES" if result.still_holding else "NO",
            }
        )

    fieldnames = list(rows[0].keys()) if rows else [
        "search_time_kst",
        "search_time_utc",
        "symbol",
        "signal_price_increase_pct",
        "signal_trading_value_2h",
        "entry_time_kst",
        "entry_time_utc",
        "entry_price",
        "max_price",
        "max_profit_pct",
        "max_profit_time_kst",
        "max_profit_time_utc",
        "stop_occurred",
        "stop_time_kst",
        "stop_time_utc",
        "max_adverse_excursion_pct",
        "max_favourable_excursion_pct",
        "trend_duration_hours",
        "still_holding",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary_statistics(results: list[TrendStudyResult]) -> None:
    if not results:
        print("\nNo qualifying symbols found.")
        return

    stopped = [result for result in results if result.stop_occurred]
    holding = [result for result in results if result.still_holding]
    max_profits = [result.max_profit_pct for result in results]
    durations = [result.trend_duration_hours for result in results]

    print("\n===== SUMMARY STATISTICS =====")
    print(f"Qualifying symbols: {len(results)}")
    print(f"Emergency stop occurred: {len(stopped)}")
    print(f"Still holding at data end: {len(holding)}")
    print(f"Average max profit %: {sum(max_profits) / len(max_profits):.2f}")
    print(f"Median max profit %: {sorted(max_profits)[len(max_profits) // 2]:.2f}")
    print(f"Best max profit %: {max(max_profits):.2f}")
    print(f"Worst max profit %: {min(max_profits):.2f}")
    print(f"Average trend duration (hours): {sum(durations) / len(durations):.2f}")

    if stopped:
        stop_durations = [result.trend_duration_hours for result in stopped]
        print(
            f"Average duration before stop (hours): "
            f"{sum(stop_durations) / len(stop_durations):.2f}"
        )


def print_study_report(
    search_dt: datetime,
    qualifying: list[QualifyingSymbol],
    results: list[TrendStudyResult],
) -> None:
    print("\n===== MOMENTUM TREND STUDY =====")
    print(f"Search time KST: {format_time_kst(search_dt)}")
    print(f"Search time UTC: {format_time_utc(search_dt)}")
    print(f"Interval: {INTERVAL}")
    print(
        f"Selection: {MIN_PRICE_INCREASE_PCT:.0f}% <= 2h increase <= "
        f"{MAX_PRICE_INCREASE_PCT:.0f}%, volume >= "
        f"{VOLUME_MA_MULTIPLIER:.0f}x MA{VOLUME_MA_PERIOD}"
    )

    print(f"\nQualifying symbols ({len(qualifying)}):")
    for index, item in enumerate(qualifying, start=1):
        print(
            f"  {index:>2}. {item.symbol:<12} "
            f"2h_change={item.price_increase_pct:.2f}% "
            f"volume={item.volume:,.0f} "
            f"ma20={item.volume_ma20:,.0f} "
            f"trading_value={item.trading_value_2h:,.0f} USDT"
        )

    print("\n----- Trend continuation results -----")
    for result in results:
        print(f"\nSymbol: {result.symbol}")
        print(f"  Entry: {result.entry_time_kst} @ {result.entry_price:.8f}")
        print(
            f"  Max profit: {result.max_profit_pct:+.2f}% "
            f"@ {result.max_profit_time_kst} (high={result.max_price:.8f})"
        )
        print(f"  Stop occurred? {'YES' if result.stop_occurred else 'NO'}")
        if result.stop_occurred:
            print(f"  Stop time: {result.stop_time_kst}")
        print(f"  Still holding? {'YES' if result.still_holding else 'NO'}")
        print(f"  MAE: {result.max_adverse_excursion_pct:.2f}%")
        print(f"  MFE: {result.max_favourable_excursion_pct:.2f}%")
        print(f"  Trend duration: {result.trend_duration_hours:.1f} hours")

    print_summary_statistics(results)
    print("==================================")


def main() -> None:
    api_key, secret_key = get_credentials()

    if not api_key:
        print("오류: BINANCE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    if not secret_key:
        print("오류: BINANCE_SECRET_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    try:
        search_dt = get_search_time_utc()
        data_end_dt = min(
            datetime.now(timezone.utc),
            search_dt + timedelta(days=FORWARD_FETCH_DAYS),
        )

        print("Momentum trend study를 시작합니다.")
        print(f"Search KST: {format_time_kst(search_dt)}")
        print(f"Search UTC: {format_time_utc(search_dt)}")
        print(f"Forward data until: {format_time_utc(data_end_dt)}")

        eligible_symbols = get_eligible_symbols(api_key)
        if not eligible_symbols:
            print("오류: TRADING 상태의 USDT 무기한 선물 심볼을 찾지 못했습니다.")
            return

        qualifying: list[QualifyingSymbol] = []
        symbols = sorted(eligible_symbols)
        total = len(symbols)

        print(f"\n심볼 스크리닝 중... ({total} symbols)")
        for index, symbol in enumerate(symbols, start=1):
            if index % 25 == 0 or index == total:
                print(f"  progress: {index}/{total}")

            try:
                result = evaluate_symbol_at_search(api_key, symbol, search_dt)
                if result is not None:
                    qualifying.append(result)
            except urllib.error.HTTPError:
                continue

            time.sleep(0.05)

        qualifying.sort(key=lambda item: item.trading_value_2h, reverse=True)

        study_results: list[TrendStudyResult] = []
        for index, item in enumerate(qualifying, start=1):
            print(f"Trend simulation {index}/{len(qualifying)}: {item.symbol}")
            try:
                trend_result = simulate_trend_hold(
                    api_key, item, search_dt, data_end_dt
                )
                if trend_result is not None:
                    study_results.append(trend_result)
            except urllib.error.HTTPError:
                continue

            time.sleep(0.05)

        save_results(study_results, search_dt)
        print_study_report(search_dt, qualifying, study_results)
        print(f"\nDetailed CSV saved: {OUTPUT_CSV}")
        print("Research simulation only. No orders placed.")

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
