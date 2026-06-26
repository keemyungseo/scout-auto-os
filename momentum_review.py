import csv
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

SYMBOL = "ESPORTSUSDT"
INTERVAL = "5m"
START_TIME = "2026-06-11 00:00:00"
END_TIME = "2026-06-12 12:00:00"

FUTURES_BASE_URL = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"
INTERVAL_MS = 5 * 60 * 1000
LOOKBACK_HOURS = 6
CHECKPOINT_HOURS = 1
FUTURE_HOURS = (3, 6)
TOP_ROW_COUNT = 5
MAX_LIMIT = 1500

LOGS_DIR = Path("logs")


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


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def format_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S UTC")


def fetch_klines_range(
    api_key: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    all_klines: list[list] = []
    current_start = start_ms

    while current_start < end_ms:
        params = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "interval": interval,
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


def build_candles(klines: list[list]) -> list[dict]:
    candles: list[dict] = []

    for candle in klines:
        open_dt = datetime.fromtimestamp(int(candle[0]) / 1000, tz=timezone.utc)
        candles.append(
            {
                "open_dt": open_dt,
                "open_time": format_time(open_dt),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
            }
        )

    return candles


def calculate_heikin_ashi(candles: list[dict]) -> None:
    prev_ha_open: float | None = None
    prev_ha_close: float | None = None

    for candle in candles:
        open_price = candle["open"]
        high_price = candle["high"]
        low_price = candle["low"]
        close_price = candle["close"]

        ha_close = (open_price + high_price + low_price + close_price) / 4
        ha_open = (
            (open_price + close_price) / 2
            if prev_ha_open is None or prev_ha_close is None
            else (prev_ha_open + prev_ha_close) / 2
        )

        candle["ha_open"] = ha_open
        candle["ha_close"] = ha_close
        candle["direction"] = "GREEN" if ha_close >= ha_open else "RED"

        prev_ha_open = ha_open
        prev_ha_close = ha_close


def generate_checkpoints(start_dt: datetime, end_dt: datetime) -> list[datetime]:
    checkpoints: list[datetime] = []
    current = start_dt

    while current <= end_dt:
        checkpoints.append(current)
        current += timedelta(hours=CHECKPOINT_HOURS)

    return checkpoints


def get_close_at(candles: list[dict], target: datetime) -> float | None:
    close_price: float | None = None

    for candle in candles:
        candle_end = candle["open_dt"] + timedelta(minutes=5)
        if candle_end <= target:
            close_price = candle["close"]
        else:
            break

    return close_price


def candles_in_window(
    candles: list[dict],
    window_start: datetime,
    window_end: datetime,
) -> list[dict]:
    return [
        candle
        for candle in candles
        if window_start <= candle["open_dt"] < window_end
    ]


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def format_ratio(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def format_percent(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def calculate_lookback_stats(
    window_candles: list[dict],
    checkpoint: datetime,
    candles: list[dict],
) -> dict[str, float | None]:
    green_volume_sum = sum(
        candle["volume"] for candle in window_candles if candle["direction"] == "GREEN"
    )
    red_volume_sum = sum(
        candle["volume"] for candle in window_candles if candle["direction"] == "RED"
    )
    total_volume = green_volume_sum + red_volume_sum

    lookback_start = checkpoint - timedelta(hours=LOOKBACK_HOURS)
    start_price = get_close_at(candles, lookback_start)
    end_price = get_close_at(candles, checkpoint)

    if start_price is None or end_price is None or start_price == 0:
        price_change_percent = None
    else:
        price_change_percent = (end_price - start_price) / start_price * 100

    return {
        "green_volume_sum": green_volume_sum,
        "red_volume_sum": red_volume_sum,
        "total_volume": total_volume,
        "green_volume_ratio": safe_ratio(green_volume_sum, total_volume),
        "red_volume_ratio": safe_ratio(red_volume_sum, total_volume),
        "momentum_ratio_long": safe_ratio(green_volume_sum, red_volume_sum),
        "momentum_ratio_short": safe_ratio(red_volume_sum, green_volume_sum),
        "price_change_percent": price_change_percent,
    }


def calculate_future_returns(
    checkpoint: datetime,
    candles: list[dict],
) -> dict[str, float | None]:
    base_price = get_close_at(candles, checkpoint)
    returns: dict[str, float | None] = {}

    if base_price is None or base_price == 0:
        for hours in FUTURE_HOURS:
            returns[f"future_{hours}h_return_percent"] = None
        return returns

    for hours in FUTURE_HOURS:
        future_time = checkpoint + timedelta(hours=hours)
        future_price = get_close_at(candles, future_time)
        if future_price is None:
            returns[f"future_{hours}h_return_percent"] = None
        else:
            returns[f"future_{hours}h_return_percent"] = (
                (future_price - base_price) / base_price * 100
            )

    return returns


def build_results(candles: list[dict], start_dt: datetime, end_dt: datetime) -> list[dict[str, str]]:
    checkpoints = generate_checkpoints(start_dt, end_dt)
    rows: list[dict[str, str]] = []

    for checkpoint in checkpoints:
        window_start = checkpoint - timedelta(hours=LOOKBACK_HOURS)
        window_candles = candles_in_window(candles, window_start, checkpoint)
        stats = calculate_lookback_stats(window_candles, checkpoint, candles)
        future_returns = calculate_future_returns(checkpoint, candles)

        rows.append(
            {
                "checkpoint_time": format_time(checkpoint),
                "green_volume_sum": f"{stats['green_volume_sum']:.4f}",
                "red_volume_sum": f"{stats['red_volume_sum']:.4f}",
                "total_volume": f"{stats['total_volume']:.4f}",
                "green_volume_ratio": format_ratio(stats["green_volume_ratio"]),
                "red_volume_ratio": format_ratio(stats["red_volume_ratio"]),
                "momentum_ratio_long": format_ratio(stats["momentum_ratio_long"]),
                "momentum_ratio_short": format_ratio(stats["momentum_ratio_short"]),
                "price_change_percent": format_percent(stats["price_change_percent"]),
                "future_3h_return_percent": format_percent(
                    future_returns["future_3h_return_percent"]
                ),
                "future_6h_return_percent": format_percent(
                    future_returns["future_6h_return_percent"]
                ),
            }
        )

    return rows


def save_results(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_row(index: int, row: dict[str, str]) -> None:
    print(f"\n[{index}] Checkpoint: {row['checkpoint_time']}")
    print(f"  6h price change: {row['price_change_percent'] or 'N/A'}%")
    print(f"  momentum_ratio_long: {row['momentum_ratio_long'] or 'N/A'}")
    print(f"  future_3h_return: {row['future_3h_return_percent'] or 'N/A'}%")
    print(f"  future_6h_return: {row['future_6h_return_percent'] or 'N/A'}%")


def print_top_rows(rows: list[dict[str, str]]) -> None:
    print(f"\n=== Momentum Review ({SYMBOL}) - 6h Lightweight ===")
    print(f"Interval: {INTERVAL}")
    print(f"Range: {START_TIME} ~ {END_TIME}")
    print(f"Checkpoints: every {CHECKPOINT_HOURS}h | Lookback: {LOOKBACK_HOURS}h")
    print(f"Total rows: {len(rows)}")

    top_momentum = sorted(
        rows,
        key=lambda row: parse_float(row["momentum_ratio_long"]) or float("-inf"),
        reverse=True,
    )[:TOP_ROW_COUNT]

    top_future_6h = sorted(
        rows,
        key=lambda row: parse_float(row["future_6h_return_percent"]) or float("-inf"),
        reverse=True,
    )[:TOP_ROW_COUNT]

    print(f"\n=== Top {TOP_ROW_COUNT} by momentum_ratio_long ===")
    for index, row in enumerate(top_momentum, start=1):
        print_row(index, row)

    print(f"\n=== Top {TOP_ROW_COUNT} by future_6h_return_percent ===")
    for index, row in enumerate(top_future_6h, start=1):
        print_row(index, row)


def main() -> None:
    api_key, secret_key = get_credentials()

    if not api_key:
        print("오류: BINANCE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    if not secret_key:
        print("오류: BINANCE_SECRET_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    try:
        start_dt = parse_time(START_TIME)
        end_dt = parse_time(END_TIME)

        if start_dt >= end_dt:
            print("오류: START_TIME은 END_TIME보다 이전이어야 합니다.")
            return

        fetch_start = start_dt - timedelta(hours=LOOKBACK_HOURS)
        fetch_end = end_dt + timedelta(hours=max(FUTURE_HOURS))

        klines = fetch_klines_range(
            api_key,
            SYMBOL,
            INTERVAL,
            int(fetch_start.timestamp() * 1000),
            int(fetch_end.timestamp() * 1000),
        )

        if not klines:
            print("오류: 선택한 기간에 대한 캔들 데이터를 찾을 수 없습니다.")
            return

        candles = build_candles(klines)
        calculate_heikin_ashi(candles)
        rows = build_results(candles, start_dt, end_dt)

        if not rows:
            print("오류: 모멘텀 결과를 생성하지 못했습니다.")
            return

        output_path = LOGS_DIR / f"momentum_review_{SYMBOL}.csv"
        save_results(rows, output_path)
        print_top_rows(rows)
        print(f"\n결과 저장: {output_path}")

    except ValueError as exc:
        print(f"오류: {exc}")
    except urllib.error.HTTPError as exc:
        details = parse_error_message(exc.read().decode("utf-8", errors="replace"))
        print(
            "오류: Binance 캔들 데이터 요청에 실패했습니다. "
            f"HTTP {exc.code}: {details}"
        )
    except urllib.error.URLError as exc:
        print(f"오류: Binance 서버에 연결할 수 없습니다. {exc.reason}")


if __name__ == "__main__":
    main()
