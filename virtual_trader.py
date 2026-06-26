import csv
from datetime import datetime, timezone
from pathlib import Path

from check_hold_signals import read_hold_symbols
from event_logger import log_event

LOGS_DIR = Path("logs")
HOLD_SYMBOLS_CSV = LOGS_DIR / "current_hold_symbols.csv"
HOLD_SIGNALS_CSV = LOGS_DIR / "current_hold_signals.csv"
VIRTUAL_POSITIONS_CSV = LOGS_DIR / "virtual_positions.csv"
EVENT_LOG_CSV = LOGS_DIR / "event_log.csv"

POSITION_COLUMNS = [
    "run_time",
    "symbol",
    "hold_rank",
    "position_side",
    "entry_time",
    "entry_price",
    "current_price",
    "unrealized_profit_percent",
    "status",
    "signal",
    "note",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def read_hold_signals(source_path: Path) -> list[dict[str, str]]:
    with source_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError("HOLD 신호 데이터가 없습니다.")

    return rows


def build_price_map(hold_symbols: list[dict[str, str]]) -> dict[str, float]:
    return {row["symbol"]: float(row["last_price"]) for row in hold_symbols}


def ensure_positions_file() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if VIRTUAL_POSITIONS_CSV.exists():
        return

    with VIRTUAL_POSITIONS_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=POSITION_COLUMNS)
        writer.writeheader()


def read_all_positions() -> list[dict[str, str]]:
    ensure_positions_file()

    with VIRTUAL_POSITIONS_CSV.open("r", newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def get_open_position(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in reversed(rows):
        if row.get("status") == "OPEN":
            return row
    return None


def calculate_unrealized_profit_percent(
    position_side: str,
    entry_price: float,
    current_price: float,
) -> float:
    if entry_price == 0:
        return 0.0

    if position_side == "LONG":
        return (current_price - entry_price) / entry_price * 100

    if position_side == "SHORT":
        return (entry_price - current_price) / entry_price * 100

    return 0.0


def save_positions(rows: list[dict[str, str]]) -> None:
    with VIRTUAL_POSITIONS_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=POSITION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_entry_detail(
    hold_rank: str,
    entry_price: str,
    signal: str,
    entry_time: str,
) -> str:
    return (
        f"hold_rank={hold_rank}; "
        f"entry_price={entry_price}; "
        f"signal={signal}; "
        f"entry_time={entry_time}"
    )


def duplicate_skip_already_logged(symbol: str, entry_time: str) -> bool:
    if not EVENT_LOG_CSV.exists():
        return False

    with EVENT_LOG_CSV.open("r", newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("event_type") != "VIRTUAL_SKIP":
                continue
            if row.get("symbol") != symbol:
                continue
            if entry_time in row.get("detail", ""):
                return True

    return False


def log_virtual_entry(position: dict[str, str]) -> None:
    side = position["position_side"]
    message = "가상 LONG 진입" if side == "LONG" else "가상 SHORT 진입"
    detail = build_entry_detail(
        position["hold_rank"],
        position["entry_price"],
        position["signal"],
        position["entry_time"],
    )
    log_event("VIRTUAL_ENTRY", position["symbol"], message, detail)


def print_position_status(position: dict[str, str]) -> None:
    print("\n=== 현재 가상 포지션 ===")
    print(f"Symbol: {position['symbol']}")
    print(f"HOLD rank: {position['hold_rank']}")
    print(f"Position side: {position['position_side']}")
    print(f"Entry time: {position['entry_time']}")
    print(f"Entry price: {position['entry_price']}")
    print(f"Current status: {position['status']}")

    profit = position.get("unrealized_profit_percent", "")
    if profit != "":
        print(f"Unrealized profit: {profit}%")


def find_entry_signal(
    hold_signals: list[dict[str, str]],
) -> dict[str, str] | None:
    hold_signals.sort(key=lambda row: row["hold_role"])

    for row in hold_signals:
        signal = row.get("signal", "NONE")
        if signal in {"LONG", "SHORT"}:
            return row

    return None


def open_virtual_position(
    signal_row: dict[str, str],
    price_map: dict[str, float],
    all_positions: list[dict[str, str]],
) -> dict[str, str]:
    symbol = signal_row["symbol"]
    signal = signal_row["signal"]
    entry_price = price_map.get(symbol)

    if entry_price is None:
        raise ValueError(f"{symbol}의 현재 가격 정보를 찾을 수 없습니다.")

    position = {
        "run_time": now_utc(),
        "symbol": symbol,
        "hold_rank": signal_row["hold_role"],
        "position_side": "LONG" if signal == "LONG" else "SHORT",
        "entry_time": now_utc(),
        "entry_price": f"{entry_price:.8f}",
        "current_price": f"{entry_price:.8f}",
        "unrealized_profit_percent": "0.0000",
        "status": "OPEN",
        "signal": signal,
        "note": f"signal_candle_time={signal_row.get('open_time', '')}",
    }

    all_positions.append(position)
    save_positions(all_positions)
    log_virtual_entry(position)
    return position


def update_open_position(
    position: dict[str, str],
    price_map: dict[str, float],
    all_positions: list[dict[str, str]],
) -> dict[str, str]:
    symbol = position["symbol"]
    current_price = price_map.get(symbol)

    if current_price is None:
        raise ValueError(f"{symbol}의 현재 가격 정보를 찾을 수 없습니다.")

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


def main() -> None:
    if not HOLD_SYMBOLS_CSV.exists():
        print(
            f"오류: HOLD 심볼 CSV 파일을 찾을 수 없습니다. "
            f"먼저 select_hold_symbols.py를 실행해 주세요: {HOLD_SYMBOLS_CSV}"
        )
        return

    if not HOLD_SIGNALS_CSV.exists():
        print(
            f"오류: HOLD 신호 CSV 파일을 찾을 수 없습니다. "
            f"먼저 check_hold_signals.py를 실행해 주세요: {HOLD_SIGNALS_CSV}"
        )
        return

    try:
        hold_symbols = read_hold_symbols(HOLD_SYMBOLS_CSV)
        hold_signals = read_hold_signals(HOLD_SIGNALS_CSV)
        price_map = build_price_map(hold_symbols)
        all_positions = read_all_positions()
        open_position = get_open_position(all_positions)

        if open_position is not None:
            updated = update_open_position(open_position, price_map, all_positions)
            if not duplicate_skip_already_logged(
                updated["symbol"],
                updated["entry_time"],
            ):
                detail = build_entry_detail(
                    updated["hold_rank"],
                    updated["entry_price"],
                    updated.get("signal", ""),
                    updated["entry_time"],
                )
                log_event(
                    "VIRTUAL_SKIP",
                    updated["symbol"],
                    "중복 진입 건너뜀",
                    detail,
                )
            print("이미 열린 가상 포지션이 있습니다.")
            print_position_status(updated)
            print(f"\n포지션 정보를 업데이트했습니다: {VIRTUAL_POSITIONS_CSV}")
            return

        entry_signal = find_entry_signal(hold_signals)
        if entry_signal is None:
            print("진입 신호 없음. 대기 중입니다.")
            for signal_row in hold_signals:
                print(
                    f"- {signal_row['symbol']} ({signal_row['hold_role']}): "
                    f"signal={signal_row.get('signal', 'NONE')}"
                )
            return

        new_position = open_virtual_position(entry_signal, price_map, all_positions)
        side = new_position["position_side"]
        print(f"가상 {side} 포지션을 열었습니다.")
        print_position_status(new_position)
        print(f"\n포지션 정보를 저장했습니다: {VIRTUAL_POSITIONS_CSV}")

    except ValueError as exc:
        print(f"오류: {exc}")


if __name__ == "__main__":
    main()
