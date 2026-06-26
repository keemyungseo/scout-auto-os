import csv
import shutil
from pathlib import Path

LOGS_DIR = Path("logs")
HOLD_SIGNALS_CSV = LOGS_DIR / "current_hold_signals.csv"
BACKUP_CSV = LOGS_DIR / "current_hold_signals_backup.csv"


def read_signals(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        raise ValueError("CSV 헤더를 찾을 수 없습니다.")

    if not rows:
        raise ValueError("HOLD 신호 데이터가 없습니다.")

    return fieldnames, rows


def save_signals(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not HOLD_SIGNALS_CSV.exists():
        print(
            f"오류: 신호 CSV 파일을 찾을 수 없습니다. "
            f"먼저 check_hold_signals.py를 실행해 주세요: {HOLD_SIGNALS_CSV}"
        )
        return

    try:
        fieldnames, rows = read_signals(HOLD_SIGNALS_CSV)
        shutil.copy2(HOLD_SIGNALS_CSV, BACKUP_CSV)

        first_row = rows[0]
        first_row["signal"] = "LONG"

        save_signals(HOLD_SIGNALS_CSV, fieldnames, rows)
        print("테스트용 LONG 신호를 생성했습니다.")
        print(f"백업 저장: {BACKUP_CSV}")
        print(
            f"수정 대상: {first_row.get('symbol', 'N/A')} "
            f"({first_row.get('hold_role', 'N/A')})"
        )

    except ValueError as exc:
        print(f"오류: {exc}")


if __name__ == "__main__":
    main()
