import os

from dotenv import load_dotenv

load_dotenv()

KEYS = ("BINANCE_API_KEY", "BINANCE_SECRET_KEY")


def preview_key(value: str) -> str:
    value = value.strip()
    if len(value) >= 8:
        return f"{value[:4]}...{value[-4:]}"
    if len(value) >= 4:
        return f"{value[:2]}...{value[-2:]}"
    return "****"


def check_key(name: str) -> None:
    value = os.getenv(name)

    if not value or not value.strip():
        print(f"오류: {name}가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        print(f"{name}: 없음")
        return

    print(f"{name}: 존재함")
    print(f"  확인: {preview_key(value)}")


def main() -> None:
    for name in KEYS:
        check_key(name)


if __name__ == "__main__":
    main()
