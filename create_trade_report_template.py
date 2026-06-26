import csv
from pathlib import Path

LOGS_DIR = Path("logs")
TRADE_REPORT_CSV = LOGS_DIR / "trade_report.csv"

COLUMNS = [
    "run_time",
    "symbol",
    "hold_rank",
    "signal",
    "entry_time",
    "entry_side",
    "entry_price",
    "exit_time",
    "exit_price",
    "profit_percent",
    "trade_amount_usdt",
    "exit_reason",
    "holding_minutes",
    "note",
]


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if TRADE_REPORT_CSV.exists():
        print("trade_report.csv 파일이 이미 존재합니다.")
        return

    with TRADE_REPORT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
        writer.writeheader()

    print("trade_report.csv 파일을 생성했습니다.")


if __name__ == "__main__":
    main()
