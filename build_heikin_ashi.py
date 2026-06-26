import csv
from pathlib import Path

SYMBOL = "HUSDT"

LOGS_DIR = Path("logs")
SOURCE_CSV = LOGS_DIR / f"{SYMBOL}_5m_klines.csv"
OUTPUT_CSV = LOGS_DIR / f"{SYMBOL}_5m_heikin_ashi.csv"
PRINT_COUNT = 10


def to_float(value: str) -> float:
    return float(value)


def read_candles(source_path: Path) -> list[dict[str, str]]:
    with source_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV 파일에 캔들 데이터가 없습니다.")

    return rows[-100:]


def calculate_heikin_ashi(candles: list[dict[str, str]]) -> list[dict[str, str]]:
    ha_candles: list[dict[str, str]] = []
    prev_ha_open: float | None = None
    prev_ha_close: float | None = None

    for candle in candles:
        open_price = to_float(candle["open"])
        high_price = to_float(candle["high"])
        low_price = to_float(candle["low"])
        close_price = to_float(candle["close"])

        ha_close = (open_price + high_price + low_price + close_price) / 4

        if prev_ha_open is None or prev_ha_close is None:
            ha_open = (open_price + close_price) / 2
        else:
            ha_open = (prev_ha_open + prev_ha_close) / 2

        ha_high = max(high_price, ha_open, ha_close)
        ha_low = min(low_price, ha_open, ha_close)
        direction = "GREEN" if ha_close >= ha_open else "RED"

        ha_candles.append(
            {
                "open_time": candle["open_time"],
                "ha_open": f"{ha_open:.7f}",
                "ha_high": f"{ha_high:.7f}",
                "ha_low": f"{ha_low:.7f}",
                "ha_close": f"{ha_close:.7f}",
                "direction": direction,
            }
        )

        prev_ha_open = ha_open
        prev_ha_close = ha_close

    return ha_candles


def save_heikin_ashi(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "open_time",
                "ha_open",
                "ha_high",
                "ha_low",
                "ha_close",
                "direction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_latest(rows: list[dict[str, str]], count: int) -> None:
    latest_rows = rows[-count:]
    print(f"=== Latest {count} Heikin Ashi candles for {SYMBOL} (5m) ===")

    for index, row in enumerate(latest_rows, start=1):
        print(f"\n[{index}]")
        print(f"  Open time: {row['open_time']}")
        print(f"  HA Open:   {row['ha_open']}")
        print(f"  HA High:   {row['ha_high']}")
        print(f"  HA Low:    {row['ha_low']}")
        print(f"  HA Close:  {row['ha_close']}")
        print(f"  Direction: {row['direction']}")


def main() -> None:
    if not SOURCE_CSV.exists():
        print(
            f"오류: 원본 CSV 파일을 찾을 수 없습니다. "
            f"먼저 fetch_5m_klines.py를 실행해 주세요: {SOURCE_CSV}"
        )
        return

    try:
        candles = read_candles(SOURCE_CSV)
        ha_candles = calculate_heikin_ashi(candles)
        save_heikin_ashi(ha_candles, OUTPUT_CSV)
        print_latest(ha_candles, PRINT_COUNT)
        print(f"\n전체 {len(ha_candles)}개 Heikin Ashi 캔들을 저장했습니다: {OUTPUT_CSV}")
    except ValueError as exc:
        print(f"오류: {exc}")
    except KeyError as exc:
        print(f"오류: CSV 형식이 올바르지 않습니다. 누락된 컬럼: {exc}")


if __name__ == "__main__":
    main()
