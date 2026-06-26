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


def test_connection(api_key: str, secret_key: str) -> None:
    timestamp = int(time.time() * 1000)
    query = urllib.parse.urlencode({"timestamp": timestamp})
    signature = sign_query(query, secret_key)
    url = f"{FUTURES_BASE_URL}{ACCOUNT_ENDPOINT}?{query}&signature={signature}"

    request = urllib.request.Request(
        url,
        headers={"X-MBX-APIKEY": api_key},
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status == 200:
                print("Binance connection successful")
    except urllib.error.HTTPError as exc:
        details = parse_error_message(exc.read().decode("utf-8", errors="replace"))
        print(
            "오류: Binance API 연결에 실패했습니다. "
            f"HTTP {exc.code}: {details}"
        )
    except urllib.error.URLError as exc:
        print(f"오류: Binance 서버에 연결할 수 없습니다. {exc.reason}")


def main() -> None:
    api_key, secret_key = get_credentials()

    if not api_key:
        print("오류: BINANCE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    if not secret_key:
        print("오류: BINANCE_SECRET_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    test_connection(api_key, secret_key)


if __name__ == "__main__":
    main()
