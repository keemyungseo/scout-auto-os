import csv
from pathlib import Path

SYMBOL = "HUSDT"

LOGS_DIR = Path("logs")
SOURCE_CSV = LOGS_DIR / f"{SYMBOL}_5m_heikin_ashi.csv"
OUTPUT_CSV = LOGS_DIR / f"{SYMBOL}_5m_heikin_ashi_streaks.csv"
PRINT_COUNT = 20


def read_heikin_ashi(source_path: Path) -> list[dict[str, str]]:
    with source_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV 파일에 Heikin Ashi 데이터가 없습니다.")

    return rows


def add_streak_counts(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    prev_direction: str | None = None
    prev_streak_count = 0

    for row in rows:
        direction = row["direction"]

        if direction == prev_direction:
            streak_count = prev_streak_count + 1
        else:
            streak_count = 1

        result.append(
            {
                **row,
                "streak_count": str(streak_count),
            }
        )

        prev_direction = direction
        prev_streak_count = streak_count

    return result


def save_streaks(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_latest(rows: list[dict[str, str]], count: int) -> None:
    latest_rows = rows[-count:]
    print(f"=== Latest {count} HA streak candles for {SYMBOL} (5m) ===")

    for index, row in enumerate(latest_rows, start=1):
        print(f"\n[{index}]")
        print(f"  Open time:    {row['open_time']}")
        print(f"  Direction:    {row['direction']}")
        print(f"  Streak count: {row['streak_count']}")


def main() -> None:
    if not SOURCE_CSV.exists():
        print(
            f"오류: 원본 CSV 파일을 찾을 수 없습니다. "
            f"먼저 build_heikin_ashi.py를 실행해 주세요: {SOURCE_CSV}"
        )
        return

    try:
        rows = read_heikin_ashi(SOURCE_CSV)
        rows_with_streaks = add_streak_counts(rows)
        save_streaks(rows_with_streaks, OUTPUT_CSV)
        print_latest(rows_with_streaks, PRINT_COUNT)
        print(
            f"\n전체 {len(rows_with_streaks)}개 캔들을 저장했습니다: {OUTPUT_CSV}"
        )
    except ValueError as exc:
        print(f"오류: {exc}")
    except KeyError as exc:
        print(f"오류: CSV 형식이 올바르지 않습니다. 누락된 컬럼: {exc}")


if __name__ == "__main__":
    main()
