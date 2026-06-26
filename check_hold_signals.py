import csv
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

MIN_STREAK = 5
CANDLE_LIMIT = 100

FUTURES_BASE_URL = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"

LOGS_DIR = Path("logs")
HOLD_SYMBOLS_CSV = LOGS_DIR / "current_hold_symbols.csv"
OUTPUT_CSV = LOGS_DIR / "current_hold_signals.csv"


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


def read_hold_symbols(source_path: Path) -> list[dict[str, str]]:
    with source_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = [row for row in reader if row.get("hold_role") in {"HOLD1", "HOLD2"}]

    if not rows:
        raise ValueError("HOLD1 또는 HOLD2 심볼을 찾을 수 없습니다.")

    rows.sort(key=lambda row: row["hold_role"])
    return rows


def fetch_klines(api_key: str, symbol: str, limit: int) -> list[dict[str, str]]:
    params = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": "5m",
            "limit": limit,
        }
    )
    url = f"{FUTURES_BASE_URL}{KLINES_ENDPOINT}?{params}"

    request = urllib.request.Request(
        url,
        headers={"X-MBX-APIKEY": api_key},
        method="GET",
    )

    with urllib.request.urlopen(request, timeout=15) as response:
        klines = json.loads(response.read().decode("utf-8"))

    candles: list[dict[str, str]] = []
    for candle in klines:
        open_time_ms = int(candle[0])
        candles.append(
            {
                "open_time": datetime.fromtimestamp(
                    open_time_ms / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5],
            }
        )

    return candles


def calculate_heikin_ashi(candles: list[dict[str, str]]) -> list[dict[str, str]]:
    ha_candles: list[dict[str, str]] = []
    prev_ha_open: float | None = None
    prev_ha_close: float | None = None

    for candle in candles:
        open_price = float(candle["open"])
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        close_price = float(candle["close"])

        ha_close = (open_price + high_price + low_price + close_price) / 4
        ha_open = (
            (open_price + close_price) / 2
            if prev_ha_open is None or prev_ha_close is None
            else (prev_ha_open + prev_ha_close) / 2
        )
        direction = "GREEN" if ha_close >= ha_open else "RED"

        ha_candles.append(
            {
                "open_time": candle["open_time"],
                "direction": direction,
            }
        )

        prev_ha_open = ha_open
        prev_ha_close = ha_close

    return ha_candles


def add_streak_counts(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    prev_direction: str | None = None
    prev_streak_count = 0

    for row in rows:
        direction = row["direction"]
        streak_count = prev_streak_count + 1 if direction == prev_direction else 1

        result.append(
            {
                **row,
                "streak_count": streak_count,
            }
        )

        prev_direction = direction
        prev_streak_count = streak_count

    return result


def detect_signal(
    current: dict[str, str | int],
    previous: dict[str, str | int] | None,
) -> str:
    if previous is None:
        return "NONE"

    prev_direction = previous["direction"]
    prev_streak_count = int(previous["streak_count"])
    current_direction = current["direction"]

    if (
        prev_direction == "RED"
        and prev_streak_count >= MIN_STREAK
        and current_direction == "GREEN"
    ):
        return "LONG"

    if (
        prev_direction == "GREEN"
        and prev_streak_count >= MIN_STREAK
        and current_direction == "RED"
    ):
        return "SHORT"

    return "NONE"


def analyze_symbol(api_key: str, symbol: str, hold_role: str) -> dict[str, str]:
    candles = fetch_klines(api_key, symbol, CANDLE_LIMIT)
    ha_candles = calculate_heikin_ashi(candles)
    ha_with_streaks = add_streak_counts(ha_candles)

    current = ha_with_streaks[-1]
    previous = ha_with_streaks[-2] if len(ha_with_streaks) >= 2 else None
    signal = detect_signal(current, previous)

    return {
        "symbol": symbol,
        "hold_role": hold_role,
        "direction": current["direction"],
        "streak_count": str(current["streak_count"]),
        "signal": signal,
        "open_time": current["open_time"],
    }


def print_result(result: dict[str, str]) -> None:
    print(f"\nSymbol: {result['symbol']} ({result['hold_role']})")
    print(f"Current HA direction: {result['direction']}")
    print(f"Current streak count: {result['streak_count']}")
    print(f"Signal: {result['signal']}")


def save_results(rows: list[dict[str, str]]) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "hold_role",
                "symbol",
                "open_time",
                "direction",
                "streak_count",
                "signal",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return OUTPUT_CSV


def main() -> None:
    api_key, secret_key = get_credentials()

    if not api_key:
        print("오류: BINANCE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    if not secret_key:
        print("오류: BINANCE_SECRET_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    if not HOLD_SYMBOLS_CSV.exists():
        print(
            f"오류: HOLD 심볼 CSV 파일을 찾을 수 없습니다. "
            f"먼저 select_hold_symbols.py를 실행해 주세요: {HOLD_SYMBOLS_CSV}"
        )
        return

    try:
        hold_symbols = read_hold_symbols(HOLD_SYMBOLS_CSV)
        results: list[dict[str, str]] = []

        print("=== HOLD symbol signals ===")
        for row in hold_symbols:
            result = analyze_symbol(api_key, row["symbol"], row["hold_role"])
            results.append(result)
            print_result(result)

        output_path = save_results(results)
        print(f"\n결과를 저장했습니다: {output_path}")

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
