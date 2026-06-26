import csv
from pathlib import Path

SYMBOL = "HUSDT"
MIN_STREAK = 5

LOGS_DIR = Path("logs")
SOURCE_CSV = LOGS_DIR / f"{SYMBOL}_5m_heikin_ashi_streaks.csv"
OUTPUT_CSV = LOGS_DIR / f"{SYMBOL}_5m_reversal_candidates.csv"
PRINT_COUNT = 20


def read_streaks(source_path: Path) -> list[dict[str, str]]:
    with source_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV 파일에 streak 데이터가 없습니다.")

    return rows


def detect_reversal_candidate(
    current: dict[str, str],
    previous: dict[str, str] | None,
) -> str:
    if previous is None:
        return "NONE"

    prev_direction = previous["direction"]
    prev_streak_count = int(previous["streak_count"])
    current_direction = current["direction"]

    if (
        prev_direction == "RED"
        and prev_streak_count >= MIN_STREAK
        and current_direction == "GREEN"
    ):
        return "LONG"

    if (
        prev_direction == "GREEN"
        and prev_streak_count >= MIN_STREAK
        and current_direction == "RED"
    ):
        return "SHORT"

    return "NONE"


def add_reversal_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []

    for index, row in enumerate(rows):
        previous = rows[index - 1] if index > 0 else None
        result.append(
            {
                **row,
                "reversal_candidate": detect_reversal_candidate(row, previous),
            }
        )

    return result


def save_candidates(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_latest(rows: list[dict[str, str]], count: int) -> None:
    latest_rows = rows[-count:]
    print(f"=== Latest {count} reversal candidate rows for {SYMBOL} (5m) ===")

    for index, row in enumerate(latest_rows, start=1):
        print(f"\n[{index}]")
        print(f"  Open time:           {row['open_time']}")
        print(f"  Direction:           {row['direction']}")
        print(f"  Streak count:        {row['streak_count']}")
        print(f"  Reversal candidate:  {row['reversal_candidate']}")


def main() -> None:
    if not SOURCE_CSV.exists():
        print(
            f"오류: 원본 CSV 파일을 찾을 수 없습니다. "
            f"먼저 count_ha_streaks.py를 실행해 주세요: {SOURCE_CSV}"
        )
        return

    try:
        rows = read_streaks(SOURCE_CSV)
        rows_with_candidates = add_reversal_candidates(rows)
        save_candidates(rows_with_candidates, OUTPUT_CSV)
        print_latest(rows_with_candidates, PRINT_COUNT)
        print(
            f"\n전체 {len(rows_with_candidates)}개 행을 저장했습니다: {OUTPUT_CSV}"
        )
    except ValueError as exc:
        print(f"오류: {exc}")
    except KeyError as exc:
        print(f"오류: CSV 형식이 올바르지 않습니다. 누락된 컬럼: {exc}")


if __name__ == "__main__":
    main()
