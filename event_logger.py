import csv
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")
EVENT_LOG_CSV = LOGS_DIR / "event_log.csv"

COLUMNS = [
    "event_time",
    "event_type",
    "symbol",
    "message",
    "detail",
]


def current_local_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_event_log_file() -> bool:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if EVENT_LOG_CSV.exists():
        return False

    with EVENT_LOG_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
        writer.writeheader()

    return True


def log_event(
    event_type: str,
    symbol: str,
    message: str,
    detail: str = "",
) -> None:
    ensure_event_log_file()

    with EVENT_LOG_CSV.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
        writer.writerow(
            {
                "event_time": current_local_time(),
                "event_type": event_type,
                "symbol": symbol,
                "message": message,
                "detail": detail,
            }
        )


if __name__ == "__main__":
    created = ensure_event_log_file()
    if created:
        print("event_log.csv 생성 완료")

    log_event("TEST", "NONE", "이벤트 로그 테스트", "Five Solo")
    print("테스트 이벤트 기록 완료")
