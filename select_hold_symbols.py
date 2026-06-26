import csv
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

FUTURES_BASE_URL = "https://fapi.binance.com"
EXCHANGE_INFO_ENDPOINT = "/fapi/v1/exchangeInfo"
TICKER_24HR_ENDPOINT = "/fapi/v1/ticker/24hr"
KLINES_ENDPOINT = "/fapi/v1/klines"

EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "USDCUSDT", "XRPUSDT"}
MIN_PRICE = 0.05
MAX_PRICE = 400.0
TOP_GAINER_COUNT = 10
HOLD_COUNT = 2
KLINES_FOR_1H = 12
LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "current_hold_symbols.csv"


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


def public_get(api_key: str, endpoint: str, params: dict | None = None) -> object:
    query = urllib.parse.urlencode(params or {})
    url = f"{FUTURES_BASE_URL}{endpoint}"
    if query:
        url = f"{url}?{query}"

    request = urllib.request.Request(
        url,
        headers={"X-MBX-APIKEY": api_key},
        method="GET",
    )

    with urllib.request.urlopen(request, timeout=15) as response:
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


def get_top_gainers(api_key: str, eligible_symbols: set[str]) -> list[dict[str, str | float]]:
    tickers = public_get(api_key, TICKER_24HR_ENDPOINT)
    candidates: list[dict[str, str | float]] = []

    for ticker in tickers:
        symbol = ticker.get("symbol", "")
        if symbol not in eligible_symbols:
            continue

        try:
            last_price = float(ticker.get("lastPrice", 0))
            price_change_percent = float(ticker.get("priceChangePercent", 0))
        except (TypeError, ValueError):
            continue

        if not (MIN_PRICE <= last_price <= MAX_PRICE):
            continue

        candidates.append(
            {
                "symbol": symbol,
                "last_price": last_price,
                "price_change_24h_pct": price_change_percent,
            }
        )

    candidates.sort(key=lambda item: item["price_change_24h_pct"], reverse=True)
    return candidates[:TOP_GAINER_COUNT]


def get_trading_value_1h(api_key: str, symbol: str) -> float:
    klines = public_get(
        api_key,
        KLINES_ENDPOINT,
        {
            "symbol": symbol,
            "interval": "5m",
            "limit": KLINES_FOR_1H,
        },
    )
    return sum(float(candle[7]) for candle in klines)


def assign_hold_symbols(
    top_gainers: list[dict[str, str | float]],
) -> tuple[str | None, str | None]:
    ranked = sorted(
        top_gainers,
        key=lambda item: item["trading_value_1h"],
        reverse=True,
    )
    hold1 = ranked[0]["symbol"] if len(ranked) >= 1 else None
    hold2 = ranked[1]["symbol"] if len(ranked) >= 2 else None
    return hold1, hold2


def save_results(rows: list[dict[str, str]], hold1: str | None, hold2: str | None) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "rank",
                "symbol",
                "price_change_24h_pct",
                "last_price",
                "trading_value_1h",
                "hold_role",
            ],
        )
        writer.writeheader()

        for index, row in enumerate(rows, start=1):
            symbol = row["symbol"]
            hold_role = ""
            if symbol == hold1:
                hold_role = "HOLD1"
            elif symbol == hold2:
                hold_role = "HOLD2"

            writer.writerow(
                {
                    "rank": str(index),
                    "symbol": symbol,
                    "price_change_24h_pct": f"{row['price_change_24h_pct']:.4f}",
                    "last_price": f"{row['last_price']:.8f}",
                    "trading_value_1h": f"{row['trading_value_1h']:.2f}",
                    "hold_role": hold_role,
                }
            )

    return OUTPUT_CSV


def print_results(
    rows: list[dict[str, str | float]],
    hold1: str | None,
    hold2: str | None,
) -> None:
    print("=== TOP10 (24hr price change) ===")
    for index, row in enumerate(rows, start=1):
        print(
            f"{index:>2}. {row['symbol']:<12} "
            f"change={row['price_change_24h_pct']:>8.4f}% "
            f"price={row['last_price']:.8f}"
        )

    print("\n=== Trading value (latest 1 hour) ===")
    for index, row in enumerate(rows, start=1):
        print(
            f"{index:>2}. {row['symbol']:<12} "
            f"trading_value={row['trading_value_1h']:,.2f} USDT"
        )

    print("\n=== Final HOLD symbols ===")
    print(f"Final HOLD1: {hold1 or '없음'}")
    print(f"Final HOLD2: {hold2 or '없음'}")


def main() -> None:
    api_key, secret_key = get_credentials()

    if not api_key:
        print("오류: BINANCE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    if not secret_key:
        print("오류: BINANCE_SECRET_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    try:
        eligible_symbols = get_eligible_symbols(api_key)
        if not eligible_symbols:
            print("오류: 조건에 맞는 USDT 무기한 선물 심볼을 찾을 수 없습니다.")
            return

        top_gainers = get_top_gainers(api_key, eligible_symbols)
        if not top_gainers:
            print("오류: TOP10 gainer 후보를 찾을 수 없습니다.")
            return

        for row in top_gainers:
            row["trading_value_1h"] = get_trading_value_1h(api_key, row["symbol"])

        hold1, hold2 = assign_hold_symbols(top_gainers)
        output_path = save_results(top_gainers, hold1, hold2)
        print_results(top_gainers, hold1, hold2)
        print(f"\n결과를 저장했습니다: {output_path}")

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
