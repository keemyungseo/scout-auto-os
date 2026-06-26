import csv
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")

FILES = {
    "HOLD symbols": LOGS_DIR / "current_hold_symbols.csv",
    "Signals": LOGS_DIR / "current_hold_signals.csv",
    "Virtual positions": LOGS_DIR / "virtual_positions.csv",
    "Event log": LOGS_DIR / "event_log.csv",
    "Trade report": LOGS_DIR / "trade_report.csv",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            return list(csv.DictReader(csv_file))
    except (OSError, csv.Error):
        return []


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


def count_hold_symbols(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if row.get("hold_role") in {"HOLD1", "HOLD2"})


def count_open_positions(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if row.get("status") == "OPEN")


def count_today_trades(rows: list[dict[str, str]]) -> int:
    count = 0
    for row in rows:
        exit_time = row.get("exit_time", "")
        run_time = row.get("run_time", "")
        if is_today_local(exit_time) or is_today_local(run_time):
            count += 1
    return count


def main() -> None:
    file_status: dict[str, str] = {}

    for label, path in FILES.items():
        status = "OK" if path.exists() else "MISSING"
        file_status[label] = status
        print(f"{path.name}: {status}")

    hold_rows = read_csv_rows(FILES["HOLD symbols"])
    virtual_rows = read_csv_rows(FILES["Virtual positions"])
    event_rows = read_csv_rows(FILES["Event log"])
    trade_rows = read_csv_rows(FILES["Trade report"])

    hold_count = count_hold_symbols(hold_rows)
    open_positions = count_open_positions(virtual_rows)
    event_count = len(event_rows)
    trade_count = len(trade_rows)
    today_trades = count_today_trades(trade_rows)

    print()
    print(f"Current HOLD symbols: {hold_count}")
    print(f"Open virtual positions: {open_positions}")
    print(f"Total event log entries: {event_count}")
    print(f"Total completed trades: {trade_count}")
    print()

    all_ok = all(status == "OK" for status in file_status.values())
    system_status = "HEALTHY" if all_ok else "WARNING"

    print("===== Five Solo Health =====")
    print()
    print(f"HOLD symbols: {file_status['HOLD symbols']}")
    print(f"Signals: {file_status['Signals']}")
    print(f"Virtual positions: {file_status['Virtual positions']}")
    print(f"Event log: {file_status['Event log']}")
    print(f"Trade report: {file_status['Trade report']}")
    print()
    print(f"Open positions: {open_positions}")
    print(f"Today's trades: {today_trades}")
    print()
    print(f"System status: {system_status}")
    print()
    print("============================")


if __name__ == "__main__":
    main()
