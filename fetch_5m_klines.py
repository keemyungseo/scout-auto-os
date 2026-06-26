import csv
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

SYMBOL = "HUSDT"

FUTURES_BASE_URL = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"
CANDLE_LIMIT = 100
PRINT_COUNT = 10
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


def fetch_klines(api_key: str, symbol: str, limit: int) -> list[list]:
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

    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def format_open_time(open_time_ms: int) -> str:
    return datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def candle_rows(klines: list[list]) -> list[dict[str, str]]:
    rows = []
    for candle in klines:
        rows.append(
            {
                "open_time": format_open_time(int(candle[0])),
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5],
            }
        )
    return rows


def print_candles(rows: list[dict[str, str]], count: int) -> None:
    latest_rows = rows[-count:]
    print(f"=== Latest {count} candles for {SYMBOL} (5m) ===")
    for index, row in enumerate(latest_rows, start=1):
        print(f"\n[{index}]")
        print(f"  Open time: {row['open_time']}")
        print(f"  Open:      {row['open']}")
        print(f"  High:      {row['high']}")
        print(f"  Low:       {row['low']}")
        print(f"  Close:     {row['close']}")
        print(f"  Volume:    {row['volume']}")


def save_csv(rows: list[dict[str, str]], symbol: str) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOGS_DIR / f"{symbol}_5m_klines.csv"

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["open_time", "open", "high", "low", "close", "volume"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main() -> None:
    api_key, secret_key = get_credentials()

    if not api_key:
        print("오류: BINANCE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    if not secret_key:
        print("오류: BINANCE_SECRET_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    try:
        klines = fetch_klines(api_key, SYMBOL, CANDLE_LIMIT)
        rows = candle_rows(klines)
        print_candles(rows, PRINT_COUNT)
        output_path = save_csv(rows, SYMBOL)
        print(f"\n전체 {len(rows)}개 캔들을 저장했습니다: {output_path}")
    except urllib.error.HTTPError as exc:
        details = parse_error_message(exc.read().decode("utf-8", errors="replace"))
        print(
            "오류: 캔들 데이터를 불러오지 못했습니다. "
            f"HTTP {exc.code}: {details}"
        )
    except urllib.error.URLError as exc:
        print(f"오류: Binance 서버에 연결할 수 없습니다. {exc.reason}")


if __name__ == "__main__":
    main()
