import csv
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")

HOLD_SYMBOLS_CSV = LOGS_DIR / "current_hold_symbols.csv"
HOLD_SIGNALS_CSV = LOGS_DIR / "current_hold_signals.csv"
VIRTUAL_POSITIONS_CSV = LOGS_DIR / "virtual_positions.csv"
TRADE_REPORT_CSV = LOGS_DIR / "trade_report.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        return None

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            rows = list(csv.DictReader(csv_file))
        return rows
    except (OSError, csv.Error):
        return None


def parse_utc_datetime(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None

    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return None


def is_today_local(utc_time_value: str) -> bool:
    parsed = parse_utc_datetime(utc_time_value)
    if parsed is None:
        return False

    local_dt = parsed.astimezone()
    today = datetime.now().astimezone().date()
    return local_dt.date() == today


def print_missing_file(section_name: str, path: Path) -> None:
    print(f"  데이터 없음: {path.name} 파일을 찾을 수 없습니다.")


def print_hold_symbols(rows: list[dict[str, str]] | None) -> None:
    print("\nCurrent HOLD symbols")

    if rows is None:
        print_missing_file("HOLD symbols", HOLD_SYMBOLS_CSV)
        return

    hold_rows = [row for row in rows if row.get("hold_role") in {"HOLD1", "HOLD2"}]
    if not hold_rows:
        print("  현재 HOLD 종목이 없습니다.")
        return

    hold_rows.sort(key=lambda row: row.get("hold_role", ""))
    for row in hold_rows:
        print(
            f"  - {row.get('hold_role', 'N/A')}: {row.get('symbol', 'N/A')} "
            f"(price={row.get('last_price', 'N/A')}, "
            f"24h={row.get('price_change_24h_pct', 'N/A')}%)"
        )


def print_current_signals(rows: list[dict[str, str]] | None) -> None:
    print("\nCurrent signals")

    if rows is None:
        print_missing_file("signals", HOLD_SIGNALS_CSV)
        return

    if not rows:
        print("  신호 데이터가 없습니다.")
        return

    for row in rows:
        print(
            f"  - {row.get('symbol', 'N/A')} ({row.get('hold_role', 'N/A')}): "
            f"direction={row.get('direction', 'N/A')}, "
            f"streak={row.get('streak_count', 'N/A')}, "
            f"signal={row.get('signal', 'N/A')}"
        )


def print_open_positions(rows: list[dict[str, str]] | None) -> None:
    print("\nOpen virtual positions")

    if rows is None:
        print_missing_file("virtual positions", VIRTUAL_POSITIONS_CSV)
        return

    open_rows = [row for row in rows if row.get("status") == "OPEN"]
    if not open_rows:
        print("  열린 가상 포지션이 없습니다.")
        return

    for row in open_rows:
        print(
            f"  - {row.get('symbol', 'N/A')} ({row.get('hold_rank', 'N/A')}): "
            f"{row.get('position_side', 'N/A')} "
            f"entry={row.get('entry_price', 'N/A')} "
            f"current={row.get('current_price', 'N/A')} "
            f"pnl={row.get('unrealized_profit_percent', 'N/A')}%"
        )


def get_today_trades(rows: list[dict[str, str]] | None) -> list[dict[str, str]]:
    if not rows:
        return []

    today_trades = []
    for row in rows:
        exit_time = row.get("exit_time", "")
        run_time = row.get("run_time", "")
        if is_today_local(exit_time) or is_today_local(run_time):
            today_trades.append(row)

    return today_trades


def print_today_trades(rows: list[dict[str, str]] | None) -> float:
    print("\nToday's completed trades")

    if rows is None:
        print_missing_file("trade report", TRADE_REPORT_CSV)
        return 0.0

    today_trades = get_today_trades(rows)
    if not today_trades:
        print("  오늘 완료된 거래가 없습니다.")
        return 0.0

    total_profit = 0.0
    for row in today_trades:
        try:
            profit = float(row.get("profit_percent", 0) or 0)
        except ValueError:
            profit = 0.0

        total_profit += profit
        print(
            f"  - {row.get('symbol', 'N/A')} {row.get('entry_side', 'N/A')} "
            f"profit={profit:.4f}% "
            f"reason={row.get('exit_reason', 'N/A')}"
        )

    return total_profit


def print_total_realized_profit(total_profit: float, trade_rows: list[dict[str, str]] | None) -> None:
    print("\nTotal realized profit %")

    if trade_rows is None:
        print("  계산할 거래 데이터가 없습니다.")
        return

    print(f"  {total_profit:.4f}%")


def main() -> None:
    print("=== Five Solo Dashboard ===")

    hold_symbols = read_csv_rows(HOLD_SYMBOLS_CSV)
    hold_signals = read_csv_rows(HOLD_SIGNALS_CSV)
    virtual_positions = read_csv_rows(VIRTUAL_POSITIONS_CSV)
    trade_report = read_csv_rows(TRADE_REPORT_CSV)

    print_hold_symbols(hold_symbols)
    print_current_signals(hold_signals)
    print_open_positions(virtual_positions)
    total_profit = print_today_trades(trade_report)
    print_total_realized_profit(total_profit, trade_report)

    print("\nScheduler status: READY")


if __name__ == "__main__":
    main()
