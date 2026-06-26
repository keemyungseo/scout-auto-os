import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from dotenv import load_dotenv

FUTURES_BASE_URL = "https://fapi.binance.com"
ACCOUNT_ENDPOINT = "/fapi/v2/account"


def get_credentials() -> tuple[str, str]:
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    secret_key = os.getenv("BINANCE_SECRET_KEY", "").strip()
    return api_key, secret_key


def sign_query(query: str, secret_key: str) -> str:
    return hmac.new(
        secret_key.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def parse_error_message(body: str) -> str:
    try:
        data = json.loads(body)
        if isinstance(data, dict) and data.get("msg"):
            return str(data["msg"])
    except json.JSONDecodeError:
        pass
    return body.strip() or "알 수 없는 오류"


def signed_get(api_key: str, secret_key: str, endpoint: str) -> dict:
    timestamp = int(time.time() * 1000)
    query = urllib.parse.urlencode({"timestamp": timestamp})
    signature = sign_query(query, secret_key)
    url = f"{FUTURES_BASE_URL}{endpoint}?{query}&signature={signature}"

    request = urllib.request.Request(
        url,
        headers={"X-MBX-APIKEY": api_key},
        method="GET",
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def get_usdt_balance(assets: list) -> str:
    for asset in assets:
        if asset.get("asset") == "USDT":
            return asset.get("walletBalance", "0")
    return "0"


def get_open_positions(positions: list) -> list[dict]:
    open_positions = []
    for position in positions:
        try:
            size = float(position.get("positionAmt", 0))
        except (TypeError, ValueError):
            continue
        if size != 0:
            open_positions.append(position)
    return open_positions


def print_account_info(account: dict) -> None:
    assets = account.get("assets", [])
    positions = account.get("positions", [])
    open_positions = get_open_positions(positions)

    print("=== Binance Futures Account ===")
    print(f"USDT balance: {get_usdt_balance(assets)}")
    print(f"Available balance: {account.get('availableBalance', '0')}")
    print(f"Total wallet balance: {account.get('totalWalletBalance', '0')}")
    print()

    if not open_positions:
        print("현재 열려 있는 포지션이 없습니다.")
        return

    print(f"Open positions: {len(open_positions)}")
    for index, position in enumerate(open_positions, start=1):
        print(f"\n[{index}] {position.get('symbol', 'N/A')}")
        print(f"  Position size: {position.get('positionAmt', '0')}")
        print(f"  Entry price: {position.get('entryPrice', '0')}")


def main() -> None:
    api_key, secret_key = get_credentials()

    if not api_key:
        print("오류: BINANCE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    if not secret_key:
        print("오류: BINANCE_SECRET_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    try:
        account = signed_get(api_key, secret_key, ACCOUNT_ENDPOINT)
        print_account_info(account)
    except urllib.error.HTTPError as exc:
        details = parse_error_message(exc.read().decode("utf-8", errors="replace"))
        print(
            "오류: 계정 정보를 불러오지 못했습니다. "
            f"HTTP {exc.code}: {details}"
        )
    except urllib.error.URLError as exc:
        print(f"오류: Binance 서버에 연결할 수 없습니다. {exc.reason}")


if __name__ == "__main__":
    main()
