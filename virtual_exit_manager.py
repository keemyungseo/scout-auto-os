import csv
import urllib.error

from check_hold_signals import CANDLE_LIMIT, fetch_klines, parse_error_message
from create_trade_report_template import COLUMNS as TRADE_REPORT_COLUMNS
from create_trade_report_template import TRADE_REPORT_CSV
from event_logger import log_event
from virtual_position_manager import (
    calculate_holding_minutes,
    fetch_market_price,
    get_credentials,
)
from virtual_trader import (
    LOGS_DIR,
    VIRTUAL_POSITIONS_CSV,
    calculate_unrealized_profit_percent,
    get_open_position,
    now_utc,
    read_all_positions,
    save_positions,
)

STOP_LOSS_PCT = 2.5
TAKE_PROFIT_PCT = 5.0


def calculate_heikin_ashi(candles: list[dict[str, str]]) -> list[dict[str, float]]:
    ha_candles: list[dict[str, float]] = []
    prev_ha_open: float | None = None
    prev_ha_close: float | None = None

    for candle in candles:
        open_price = float(candle["open"])
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        close_price = float(candle["close"])

        ha_close = (open_price + high_price + low_price + close_price) / 4
        ha_open = (
            (open_price + close_price) / 2
            if prev_ha_open is None or prev_ha_close is None
            else (prev_ha_open + prev_ha_close) / 2
        )

        ha_candles.append({"ha_open": ha_open, "ha_close": ha_close})
        prev_ha_open = ha_open
        prev_ha_close = ha_close

    return ha_candles


def get_completed_ha_closes(ha_candles: list[dict[str, float]]) -> tuple[float, float]:
    if len(ha_candles) < 4:
        raise ValueError("Heikin Ashi 계산을 위한 캔들 수가 부족합니다.")

    completed = ha_candles[:-1]
    latest_completed = completed[-1]["ha_close"]
    previous_completed = completed[-2]["ha_close"]
    return latest_completed, previous_completed


def check_long_exit(
    entry_price: float,
    current_price: float,
    latest_ha_close: float,
    previous_ha_close: float,
) -> tuple[bool, str | None]:
    if current_price <= entry_price * (1 - STOP_LOSS_PCT / 100):
        return True, "STOP_LOSS"

    if current_price >= entry_price * (1 + TAKE_PROFIT_PCT / 100):
        return True, "TAKE_PROFIT"

    if latest_ha_close < previous_ha_close:
        return True, "HA_REVERSAL"

    return False, None


def check_short_exit(
    entry_price: float,
    current_price: float,
    latest_ha_close: float,
    previous_ha_close: float,
) -> tuple[bool, str | None]:
    if current_price >= entry_price * (1 + STOP_LOSS_PCT / 100):
        return True, "STOP_LOSS"

    if current_price <= entry_price * (1 - TAKE_PROFIT_PCT / 100):
        return True, "TAKE_PROFIT"

    if latest_ha_close > previous_ha_close:
        return True, "HA_REVERSAL"

    return False, None


def check_exit_conditions(
    position_side: str,
    entry_price: float,
    current_price: float,
    latest_ha_close: float,
    previous_ha_close: float,
) -> tuple[bool, str | None]:
    if position_side == "LONG":
        return check_long_exit(
            entry_price,
            current_price,
            latest_ha_close,
            previous_ha_close,
        )

    if position_side == "SHORT":
        return check_short_exit(
            entry_price,
            current_price,
            latest_ha_close,
            previous_ha_close,
        )

    return False, None


def ensure_trade_report_file() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if TRADE_REPORT_CSV.exists():
        return

    with TRADE_REPORT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRADE_REPORT_COLUMNS)
        writer.writeheader()


def append_trade_report(
    position: dict[str, str],
    exit_price: float,
    profit_percent: float,
    holding_minutes: int,
    exit_reason: str,
) -> None:
    ensure_trade_report_file()

    with TRADE_REPORT_CSV.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRADE_REPORT_COLUMNS)
        writer.writerow(
            {
                "run_time": now_utc(),
                "symbol": position["symbol"],
                "hold_rank": position["hold_rank"],
                "signal": position.get("signal", ""),
                "entry_time": position["entry_time"],
                "entry_side": position["position_side"],
                "entry_price": position["entry_price"],
                "exit_time": now_utc(),
                "exit_price": f"{exit_price:.8f}",
                "profit_percent": f"{profit_percent:.4f}",
                "trade_amount_usdt": "",
                "exit_reason": exit_reason,
                "holding_minutes": str(holding_minutes),
                "note": position.get("note", ""),
            }
        )


def build_exit_detail(
    side: str,
    entry_price: str,
    exit_price: float,
    profit_percent: float,
    holding_minutes: int,
    exit_reason: str,
) -> str:
    return (
        f"side={side}; "
        f"entry_price={entry_price}; "
        f"exit_price={exit_price:.8f}; "
        f"profit_percent={profit_percent:.4f}; "
        f"holding_minutes={holding_minutes}; "
        f"exit_reason={exit_reason}"
    )


def log_virtual_exit(
    position: dict[str, str],
    exit_price: float,
    profit_percent: float,
    holding_minutes: int,
    exit_reason: str,
) -> None:
    detail = build_exit_detail(
        position["position_side"],
        position["entry_price"],
        exit_price,
        profit_percent,
        holding_minutes,
        exit_reason,
    )
    log_event("VIRTUAL_EXIT", position["symbol"], "가상 포지션 청산", detail)


def close_position(
    position: dict[str, str],
    current_price: float,
    profit_percent: float,
    all_positions: list[dict[str, str]],
) -> dict[str, str]:
    position["run_time"] = now_utc()
    position["current_price"] = f"{current_price:.8f}"
    position["unrealized_profit_percent"] = f"{profit_percent:.4f}"
    position["status"] = "CLOSED"
    save_positions(all_positions)
    return position


def print_exit_summary(
    position: dict[str, str],
    exit_price: float,
    profit_percent: float,
    holding_minutes: int,
    exit_reason: str,
) -> None:
    reason_labels = {
        "STOP_LOSS": "손절",
        "TAKE_PROFIT": "익절",
        "HA_REVERSAL": "Heikin Ashi 반전",
    }

    print("\n=== 가상 포지션 청산 ===")
    print(f"Symbol: {position['symbol']}")
    print(f"Side: {position['position_side']}")
    print(f"Entry price: {position['entry_price']}")
    print(f"Exit price: {exit_price:.8f}")
    print(f"Profit: {profit_percent:.4f}%")
    print(f"Holding time: {holding_minutes} minutes")
    print(f"Exit reason: {reason_labels.get(exit_reason, exit_reason)}")
    print("가상 포지션을 CLOSED 상태로 저장했습니다.")


def print_hold_message(
    position: dict[str, str],
    current_price: float,
    profit_percent: float,
    latest_ha_close: float,
    previous_ha_close: float,
) -> None:
    print("\n청산 조건 미충족. 포지션 유지 중입니다.")
    print(f"Symbol: {position['symbol']}")
    print(f"Side: {position['position_side']}")
    print(f"Current price: {current_price:.8f}")
    print(f"Unrealized profit: {profit_percent:.4f}%")
    print(
        "Latest completed HA close: "
        f"{latest_ha_close:.7f} / Previous: {previous_ha_close:.7f}"
    )


def main() -> None:
    api_key, secret_key = get_credentials()

    if not api_key:
        print("오류: BINANCE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    if not secret_key:
        print("오류: BINANCE_SECRET_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    if not VIRTUAL_POSITIONS_CSV.exists():
        print(
            f"오류: 가상 포지션 CSV 파일을 찾을 수 없습니다. "
            f"먼저 virtual_trader.py를 실행해 주세요: {VIRTUAL_POSITIONS_CSV}"
        )
        return

    try:
        all_positions = read_all_positions()
        open_position = get_open_position(all_positions)

        if open_position is None:
            print("열린 가상 포지션이 없습니다.")
            return

        symbol = open_position["symbol"]
        side = open_position["position_side"]
        entry_price = float(open_position["entry_price"])

        candles = fetch_klines(api_key, symbol, CANDLE_LIMIT)
        ha_candles = calculate_heikin_ashi(candles)
        latest_ha_close, previous_ha_close = get_completed_ha_closes(ha_candles)
        current_price = fetch_market_price(api_key, symbol)

        should_exit, exit_reason = check_exit_conditions(
            side,
            entry_price,
            current_price,
            latest_ha_close,
            previous_ha_close,
        )
        profit_percent = calculate_unrealized_profit_percent(
            side,
            entry_price,
            current_price,
        )

        if not should_exit or exit_reason is None:
            print_hold_message(
                open_position,
                current_price,
                profit_percent,
                latest_ha_close,
                previous_ha_close,
            )
            return

        holding_minutes = calculate_holding_minutes(open_position["entry_time"])
        closed_position = close_position(
            open_position,
            current_price,
            profit_percent,
            all_positions,
        )
        append_trade_report(
            closed_position,
            current_price,
            profit_percent,
            holding_minutes,
            exit_reason,
        )
        log_virtual_exit(
            closed_position,
            current_price,
            profit_percent,
            holding_minutes,
            exit_reason,
        )
        print_exit_summary(
            closed_position,
            current_price,
            profit_percent,
            holding_minutes,
            exit_reason,
        )
        print(f"\n거래 기록 저장: {TRADE_REPORT_CSV}")
        print(f"포지션 저장: {VIRTUAL_POSITIONS_CSV}")

    except ValueError as exc:
        print(f"오류: {exc}")
    except urllib.error.HTTPError as exc:
        details = parse_error_message(exc.read().decode("utf-8", errors="replace"))
        print(
            "오류: Binance API 요청에 실패했습니다. "
            f"HTTP {exc.code}: {details}"
        )
    except urllib.error.URLError as exc:
        print(f"오류: Binance 서버에 연결할 수 없습니다. {exc.reason}")


if __name__ == "__main__":
    main()
