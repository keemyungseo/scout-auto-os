import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from dotenv import load_dotenv

from virtual_trader import (
    VIRTUAL_POSITIONS_CSV,
    calculate_unrealized_profit_percent,
    get_open_position,
    now_utc,
    read_all_positions,
    save_positions,
)

FUTURES_BASE_URL = "https://fapi.binance.com"
TICKER_PRICE_ENDPOINT = "/fapi/v1/ticker/price"


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


def fetch_market_price(api_key: str, symbol: str) -> float:
    params = urllib.parse.urlencode({"symbol": symbol})
    url = f"{FUTURES_BASE_URL}{TICKER_PRICE_ENDPOINT}?{params}"

    request = urllib.request.Request(
        url,
        headers={"X-MBX-APIKEY": api_key},
        method="GET",
    )

    with urllib.request.urlopen(request, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))

    return float(data["price"])


def parse_entry_time(entry_time: str) -> datetime:
    return datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S UTC").replace(
        tzinfo=timezone.utc
    )


def calculate_holding_minutes(entry_time: str) -> int:
    entry_dt = parse_entry_time(entry_time)
    now_dt = datetime.now(timezone.utc)
    return max(int((now_dt - entry_dt).total_seconds() // 60), 0)


def update_open_position_with_market_price(
    position: dict[str, str],
    current_price: float,
    all_positions: list[dict[str, str]],
) -> dict[str, str]:
    entry_price = float(position["entry_price"])
    profit = calculate_unrealized_profit_percent(
        position["position_side"],
        entry_price,
        current_price,
    )

    position["run_time"] = now_utc()
    position["current_price"] = f"{current_price:.8f}"
    position["unrealized_profit_percent"] = f"{profit:.4f}"
    position["status"] = "OPEN"

    save_positions(all_positions)
    return position


def print_position_update(position: dict[str, str], holding_minutes: int) -> None:
    print("\n=== 가상 포지션 현황 ===")
    print(f"Symbol: {position['symbol']}")
    print(f"Side: {position['position_side']}")
    print(f"Entry price: {position['entry_price']}")
    print(f"Current price: {position['current_price']}")
    print(f"Unrealized profit: {position['unrealized_profit_percent']}%")
    print(f"Holding time: {holding_minutes} minutes")


def main() -> None:
    api_key, secret_key = get_credentials()

    if not api_key:
        print("오류: BINANCE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    if not secret_key:
        print("오류: BINANCE_SECRET_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    if not VIRTUAL_POSITIONS_CSV.exists():
        print(
            f"오류: 가상 포지션 CSV 파일을 찾을 수 없습니다. "
            f"먼저 virtual_trader.py를 실행해 주세요: {VIRTUAL_POSITIONS_CSV}"
        )
        return

    try:
        all_positions = read_all_positions()
        open_position = get_open_position(all_positions)

        if open_position is None:
            print("열린 가상 포지션이 없습니다.")
            return

        symbol = open_position["symbol"]
        current_price = fetch_market_price(api_key, symbol)
        updated = update_open_position_with_market_price(
            open_position,
            current_price,
            all_positions,
        )
        holding_minutes = calculate_holding_minutes(updated["entry_time"])

        print("가상 포지션 정보를 업데이트했습니다.")
        print_position_update(updated, holding_minutes)
        print(f"\n저장 완료: {VIRTUAL_POSITIONS_CSV}")

    except ValueError as exc:
        print(f"오류: {exc}")
    except urllib.error.HTTPError as exc:
        details = parse_error_message(exc.read().decode("utf-8", errors="replace"))
        print(
            "오류: Binance 시세 조회에 실패했습니다. "
            f"HTTP {exc.code}: {details}"
        )
    except urllib.error.URLError as exc:
        print(f"오류: Binance 서버에 연결할 수 없습니다. {exc.reason}")


if __name__ == "__main__":
    main()
