import csv
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

REVIEW_TIMEZONE = "KST"
KST_OFFSET = timedelta(hours=9)
KST_TZ = timezone(KST_OFFSET)

REVIEW_START_KST = "2026-06-15 11:00:00"
REVIEW_END_KST = "2026-06-15 17:00:00"

REVIEW_START_UTC = "2026-06-15 02:00:00"
REVIEW_END_UTC = "2026-06-15 08:00:00"

RESEARCH_FIXED_SLOTS = {
    "Slot1": "JTOUSDT",
    "Slot2": "WIFUSDT",
}

REVIEW_DATE = REVIEW_START_KST.split()[0]

FUTURES_BASE_URL = "https://fapi.binance.com"
EXCHANGE_INFO_ENDPOINT = "/fapi/v1/exchangeInfo"
KLINES_ENDPOINT = "/fapi/v1/klines"

EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "XRPUSDT", "USDCUSDT"}
MIN_PRICE = 0.05
MAX_PRICE = 400.0
PRICE_CHANGE_MIN = 0.0
PRICE_CHANGE_MAX = 30.0
TOP_VOLUME_COUNT = 50
SLOT_COUNT = 2
HA_SIGNAL_LOOKBACK = 6
LOOKBACK_3H_CANDLES = 36
TOP_RANK_COUNT = 10
EMERGENCY_LOSS_PCT = 2.5
EMERGENCY_STOP_MULTIPLIER = 0.975
EMERGENCY_SHORT_STOP_MULTIPLIER = 1.025
LOOKBACK_1H_CANDLES = 12
LOOKBACK_6H_CANDLES = 72
INTERVAL_MS = 5 * 60 * 1000
CANDLE_MINUTES = 5
MAX_LIMIT = 1500
REVIEW_HOURS = 6
EXPECTED_REVIEW_MINUTES = REVIEW_HOURS * 60
EXPECTED_SIGNAL_CANDLES = REVIEW_HOURS * 12

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "five_review.csv"
SLOT_SCORE_OUTPUT_CSV = LOGS_DIR / "slot_score_review.csv"
TRADE_SUMMARY_CSV = LOGS_DIR / "trade_summary.csv"


@dataclass
class TradeEvent:
    event_type: str
    slot: str
    symbol: str
    event_time: str
    price: float
    profit_percent: float = 0.0
    cumulative_profit_percent: float = 0.0
    detail: str = ""


PROFIT_EVENT_TYPES = {
    "EXIT",
    "EMERGENCY_EXIT",
    "FORCED_EXIT",
    "LONG_EXIT",
    "LONG_EMERGENCY_EXIT",
    "SHORT_EXIT",
    "SHORT_EMERGENCY_EXIT",
}


@dataclass
class SlotSelection:
    rank: int
    slot_name: str
    symbol: str
    previous_6h_price_change_percent: float
    ha_body_percent: float
    body_ratio: float
    volume_ratio: float
    momentum: float
    slot_score: float
    recent_1h_trading_value: float


@dataclass
class SlotResult:
    slot_name: str
    symbol: str
    events: list[TradeEvent] = field(default_factory=list)
    total_profit_percent: float = 0.0
    total_trades: int = 0
    emergency_exits: int = 0
    unfilled_limits: int = 0
    forced_exits: int = 0
    long_profit_percent: float = 0.0
    short_profit_percent: float = 0.0
    long_trades: int = 0
    short_trades: int = 0
    long_emergency_exits: int = 0
    short_emergency_exits: int = 0
    processed_candle_count: int = 0
    signal_candle_count: int = 0
    actual_duration_minutes: float = 0.0


LONG_ENTRY_EVENT_TYPES = {"ENTRY", "REENTRY", "LONG_ENTRY", "LONG_REENTRY"}
SHORT_ENTRY_EVENT_TYPES = {"SHORT_ENTRY", "SHORT_REENTRY"}


@dataclass
class CompletedTrade:
    slot: str
    symbol: str
    direction: str
    trade_number: int
    entry_time_kst: str
    entry_time_utc: str
    exit_time_kst: str
    exit_time_utc: str
    entry_price: float
    exit_price: float
    profit_percent: float
    exit_reason: str
    running_profit_after_trade: float


def format_time_kst_short(time_label: str) -> str:
    clock = time_label.replace(" KST", "").split()[1]
    return f"{clock[:5]} KST"


def get_exit_direction(event: TradeEvent) -> str | None:
    if event.event_type in {"EXIT", "EMERGENCY_EXIT", "LONG_EXIT", "LONG_EMERGENCY_EXIT"}:
        return "LONG"
    if event.event_type in {"SHORT_EXIT", "SHORT_EMERGENCY_EXIT"}:
        return "SHORT"
    if event.event_type == "FORCED_EXIT":
        if "side=SHORT" in event.detail:
            return "SHORT"
        return "LONG"
    return None


def get_exit_reason_label(event: TradeEvent) -> str:
    if event.event_type in {
        "EMERGENCY_EXIT",
        "LONG_EMERGENCY_EXIT",
        "SHORT_EMERGENCY_EXIT",
    }:
        return "EMERGENCY STOP"
    if event.event_type == "FORCED_EXIT":
        return "FORCED EXIT"
    return "NORMAL EXIT"


def extract_completed_trades(slot_result: SlotResult) -> list[CompletedTrade]:
    trades: list[CompletedTrade] = []
    open_entry: TradeEvent | None = None
    open_direction: str | None = None
    long_count = 0
    short_count = 0

    for event in slot_result.events:
        if event.event_type in LONG_ENTRY_EVENT_TYPES:
            open_entry = event
            open_direction = "LONG"
            continue

        if event.event_type in SHORT_ENTRY_EVENT_TYPES:
            open_entry = event
            open_direction = "SHORT"
            continue

        exit_direction = get_exit_direction(event)
        if exit_direction is None or open_entry is None or open_direction is None:
            continue

        if exit_direction != open_direction:
            continue

        if exit_direction == "LONG":
            long_count += 1
            trade_number = long_count
        else:
            short_count += 1
            trade_number = short_count

        entry_time_kst, entry_time_utc = format_event_times(open_entry.event_time)
        exit_time_kst, exit_time_utc = format_event_times(event.event_time)

        trades.append(
            CompletedTrade(
                slot=slot_result.slot_name,
                symbol=slot_result.symbol,
                direction=exit_direction,
                trade_number=trade_number,
                entry_time_kst=entry_time_kst,
                entry_time_utc=entry_time_utc,
                exit_time_kst=exit_time_kst,
                exit_time_utc=exit_time_utc,
                entry_price=open_entry.price,
                exit_price=event.price,
                profit_percent=event.profit_percent,
                exit_reason=get_exit_reason_label(event),
                running_profit_after_trade=event.cumulative_profit_percent,
            )
        )
        open_entry = None
        open_direction = None

    return trades


def save_trade_summary(
    slot_results: list[SlotResult],
    long_only: bool = False,
) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for slot_result in slot_results:
        for trade in extract_completed_trades(slot_result):
            if long_only and trade.direction != "LONG":
                continue
            rows.append(
                {
                    "slot": trade.slot,
                    "symbol": trade.symbol,
                    "direction": trade.direction,
                    "trade_number": str(trade.trade_number),
                    "entry_time_kst": trade.entry_time_kst,
                    "entry_time_utc": trade.entry_time_utc,
                    "exit_time_kst": trade.exit_time_kst,
                    "exit_time_utc": trade.exit_time_utc,
                    "entry_price": f"{trade.entry_price:.8f}",
                    "exit_price": f"{trade.exit_price:.8f}",
                    "profit_percent": f"{trade.profit_percent:.4f}",
                    "exit_reason": trade.exit_reason,
                    "running_profit_after_trade": (
                        f"{trade.running_profit_after_trade:.4f}"
                    ),
                }
            )

    fieldnames = [
        "slot",
        "symbol",
        "direction",
        "trade_number",
        "entry_time_kst",
        "entry_time_utc",
        "exit_time_kst",
        "exit_time_utc",
        "entry_price",
        "exit_price",
        "profit_percent",
        "exit_reason",
        "running_profit_after_trade",
    ]

    with TRADE_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_direction_trades(trades: list[CompletedTrade], direction: str) -> None:
    print(f"\n{direction}")
    direction_trades = [trade for trade in trades if trade.direction == direction]
    if not direction_trades:
        print("  No trades")
        print(f"\n{direction} TOTAL:")
        print("Trades: 0")
        print("Profit: 0.00%")
        return

    total_profit = 0.0
    for trade in direction_trades:
        sign = "+" if trade.profit_percent >= 0 else ""
        print(f"\nTrade {trade.trade_number}")
        print(f"Entry : {format_time_kst_short(trade.entry_time_kst)}")
        print(f"Exit  : {format_time_kst_short(trade.exit_time_kst)}")
        print(f"Profit: {sign}{trade.profit_percent:.2f}%")
        print(f"Reason: {trade.exit_reason}")
        total_profit += trade.profit_percent

    print(f"\n{direction} TOTAL:")
    print(f"Trades: {len(direction_trades)}")
    sign = "+" if total_profit >= 0 else ""
    print(f"Profit: {sign}{total_profit:.2f}%")


def print_long_only_review(
    review_start: datetime,
    trading_selections: list[SlotSelection],
    slot_results: list[SlotResult],
    review_valid: bool = True,
) -> None:
    print("\n==========================")
    print("FIVE LONG ONLY REVIEW")
    print("==========================")
    print("\nSEARCH")
    print(f"time_kst={format_time_kst(review_start)}")
    print(f"time_utc={format_time_utc(review_start)}")

    for selection in trading_selections:
        print(f"Selected {selection.slot_name}: {selection.symbol}")

    for slot_result in slot_results:
        long_trades = [
            trade
            for trade in extract_completed_trades(slot_result)
            if trade.direction == "LONG"
        ]
        total_profit = sum(trade.profit_percent for trade in long_trades)

        print(f"\n----- {slot_result.slot_name} : {slot_result.symbol} -----")
        if not long_trades:
            print("\nNo LONG trades")
        else:
            for trade in long_trades:
                sign = "+" if trade.profit_percent >= 0 else ""
                print(f"\nTrade {trade.trade_number}")
                print(f"Entry : {format_time_kst_short(trade.entry_time_kst)}")
                print(f"Exit  : {format_time_kst_short(trade.exit_time_kst)}")
                print(f"Profit: {sign}{trade.profit_percent:.2f}%")
                print(f"Reason: {trade.exit_reason}")

        print("\nLONG TOTAL")
        print(f"Trades: {len(long_trades)}")
        sign = "+" if total_profit >= 0 else ""
        print(f"Profit: {sign}{total_profit:.2f}%")

    combined_profit = sum(slot.total_profit_percent for slot in slot_results)
    total_trades = sum(slot.total_trades for slot in slot_results)
    total_emergency = sum(slot.emergency_exits for slot in slot_results)
    total_forced = sum(slot.forced_exits for slot in slot_results)
    slot1_symbol = trading_selections[0].symbol if trading_selections else "-"
    slot2_symbol = trading_selections[1].symbol if len(trading_selections) > 1 else "-"

    print("\nOverall:")
    print(f"Selected symbols: {slot1_symbol}, {slot2_symbol}")
    sign = "+" if combined_profit >= 0 else ""
    print(f"Combined profit: {sign}{combined_profit:.2f}%")
    print(f"Trade count: {total_trades}")
    print(f"Emergency exits: {total_emergency}")
    print(f"Forced exits: {total_forced}")
    print(f"Review duration: {EXPECTED_REVIEW_MINUTES} minutes")
    print(f"Review status: {'VALID' if review_valid else 'INVALID'}")
    print("==========================")


def print_trade_summary(
    slot_results: list[SlotResult],
    trading_selections: list[SlotSelection],
) -> None:
    print("\n====================================")
    print("===== FIVE TRADE SUMMARY =====")
    print("====================================")

    for slot_result in slot_results:
        trades = extract_completed_trades(slot_result)
        long_profit = sum(
            trade.profit_percent for trade in trades if trade.direction == "LONG"
        )
        short_profit = sum(
            trade.profit_percent for trade in trades if trade.direction == "SHORT"
        )
        combined_profit = long_profit + short_profit

        print(f"\n----- {slot_result.slot_name} : {slot_result.symbol} -----")
        print_direction_trades(trades, "LONG")
        print("\n----------------------------")
        print_direction_trades(trades, "SHORT")
        print("\n----------------------------")
        print("\nCOMBINED:")
        print(f"LONG Profit: {long_profit:+.2f}%")
        print(f"SHORT Profit: {short_profit:+.2f}%")
        print(f"Combined Profit: {combined_profit:+.2f}%")

    total_long = sum(slot.long_profit_percent for slot in slot_results)
    total_short = sum(slot.short_profit_percent for slot in slot_results)
    total_combined = sum(slot.total_profit_percent for slot in slot_results)
    total_long_trades = sum(slot.long_trades for slot in slot_results)
    total_short_trades = sum(slot.short_trades for slot in slot_results)
    total_trades = sum(slot.total_trades for slot in slot_results)
    total_emergency = sum(slot.emergency_exits for slot in slot_results)
    total_forced = sum(slot.forced_exits for slot in slot_results)
    slot1_symbol = trading_selections[0].symbol if trading_selections else "-"
    slot2_symbol = trading_selections[1].symbol if len(trading_selections) > 1 else "-"

    print("\n===== OVERALL SUMMARY =====")
    print(f"\nSelected symbols:")
    print(f"Slot1: {slot1_symbol}")
    print(f"Slot2: {slot2_symbol}")
    print(f"\nLONG total profit: {total_long:+.2f}%")
    print(f"SHORT total profit: {total_short:+.2f}%")
    print(f"Combined total profit: {total_combined:+.2f}%")
    print(f"LONG trades: {total_long_trades}")
    print(f"SHORT trades: {total_short_trades}")
    print(f"Total trades: {total_trades}")
    print(f"Emergency exits: {total_emergency}")
    print(f"Forced exits: {total_forced}")
    print(f"Review duration: {EXPECTED_REVIEW_MINUTES} minutes")
    print("====================================")


def get_credentials() -> tuple[str, str]:
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    secret_key = os.getenv("BINANCE_SECRET_KEY", "").strip()
    return api_key, secret_key


def parse_error_message(body: str) -> str:
    try:
        data = json.loads(body)
        if isinstance(data, dict) and data.get("msg"):
            return str(data["msg"])
    except json.JSONDecodeError:
        pass
    return body.strip() or "알 수 없는 오류"


def format_time_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def format_time_kst(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(KST_TZ).strftime("%Y-%m-%d %H:%M:%S KST")


def format_time(value: datetime) -> str:
    return format_time_utc(value)


def parse_kst_to_utc(kst_str: str) -> datetime:
    kst_dt = datetime.strptime(kst_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST_TZ)
    return kst_dt.astimezone(timezone.utc)


def get_review_window_utc() -> tuple[datetime, datetime]:
    review_start = parse_kst_to_utc(REVIEW_START_KST)
    review_end = parse_kst_to_utc(REVIEW_END_KST)

    expected_start = datetime.strptime(
        REVIEW_START_UTC, "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=timezone.utc)
    expected_end = datetime.strptime(
        REVIEW_END_UTC, "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=timezone.utc)

    if review_start != expected_start or review_end != expected_end:
        raise ValueError(
            "KST review window conversion mismatch. "
            f"expected start={REVIEW_START_UTC}, end={REVIEW_END_UTC}; "
            f"got start={format_time_utc(review_start)}, end={format_time_utc(review_end)}"
        )

    return review_start, review_end


def format_review_window_detail(review_start: datetime, review_end: datetime) -> str:
    return (
        f"review_start_kst={format_time_kst(review_start)}; "
        f"review_end_kst={format_time_kst(review_end)}; "
        f"review_start_utc={format_time_utc(review_start)}; "
        f"review_end_utc={format_time_utc(review_end)}"
    )


def print_review_window_times(review_start: datetime, review_end: datetime) -> None:
    print(f"Review timezone: {REVIEW_TIMEZONE}")
    print(f"Review start KST: {format_time_kst(review_start)}")
    print(f"Review end KST: {format_time_kst(review_end)}")
    print(f"Review start UTC: {format_time_utc(review_start)}")
    print(f"Review end UTC: {format_time_utc(review_end)}")


def get_simulation_candles(
    candles: list[dict],
    review_start: datetime,
    review_end: datetime,
) -> list[dict]:
    spillover_end = review_end + timedelta(minutes=CANDLE_MINUTES)
    return [
        candle
        for candle in candles
        if review_start <= candle["open_dt"] < spillover_end
    ]


def is_signal_candle(candle: dict, review_end: datetime) -> bool:
    candle_end = candle["open_dt"] + timedelta(minutes=CANDLE_MINUTES)
    return candle_end <= review_end


def calculate_actual_duration_minutes(
    simulation_candles: list[dict],
    review_start: datetime,
    review_end: datetime,
) -> float:
    if not simulation_candles:
        return 0.0

    last_candle_end = simulation_candles[-1]["open_dt"] + timedelta(minutes=CANDLE_MINUTES)
    actual_end = min(review_end, last_candle_end)
    return max((actual_end - review_start).total_seconds() / 60, 0.0)


def expected_signal_opens(review_start: datetime, review_end: datetime) -> list[datetime]:
    opens: list[datetime] = []
    current = review_start
    step = timedelta(minutes=CANDLE_MINUTES)

    while current + step <= review_end:
        opens.append(current)
        current += step

    return opens


def missing_signal_opens(
    candles: list[dict],
    review_start: datetime,
    review_end: datetime,
) -> list[datetime]:
    available = {candle["open_dt"] for candle in candles}
    return [
        open_dt
        for open_dt in expected_signal_opens(review_start, review_end)
        if open_dt not in available
    ]


def merge_klines(existing: list[list], new: list[list]) -> list[list]:
    by_open = {int(kline[0]): kline for kline in existing}
    for kline in new:
        by_open[int(kline[0])] = kline
    return [by_open[open_ms] for open_ms in sorted(by_open)]


def group_contiguous_opens(opens: list[datetime]) -> list[tuple[datetime, datetime]]:
    if not opens:
        return []

    sorted_opens = sorted(opens)
    ranges: list[tuple[datetime, datetime]] = []
    range_start = sorted_opens[0]
    previous = sorted_opens[0]
    step = timedelta(minutes=CANDLE_MINUTES)

    for open_dt in sorted_opens[1:]:
        if open_dt - previous == step:
            previous = open_dt
            continue

        ranges.append(
            (range_start, previous + timedelta(minutes=CANDLE_MINUTES))
        )
        range_start = open_dt
        previous = open_dt

    ranges.append((range_start, previous + timedelta(minutes=CANDLE_MINUTES)))
    return ranges


def build_missing_data_reason(
    symbol: str,
    missing_opens: list[datetime],
    review_start: datetime,
    review_end: datetime,
) -> str:
    expected_count = len(expected_signal_opens(review_start, review_end))
    available_count = expected_count - len(missing_opens)
    first_missing = format_time(missing_opens[0])
    last_missing = format_time(missing_opens[-1])
    return (
        f"symbol={symbol}; "
        f"signal_candles={available_count}/{expected_count}; "
        f"missing_count={len(missing_opens)}; "
        f"missing_from={first_missing}; "
        f"missing_to={last_missing}"
    )


def print_kline_diagnostics(
    symbol: str,
    klines: list[list],
    candles: list[dict],
    requested_start: datetime,
    requested_end: datetime,
) -> None:
    print(f"\n  === Kline Diagnostics ({symbol}) ===")
    print(f"  requested start KST: {format_time_kst(requested_start)}")
    print(f"  requested end KST: {format_time_kst(requested_end)}")
    print(f"  requested start UTC: {format_time_utc(requested_start)}")
    print(f"  requested end UTC: {format_time_utc(requested_end)}")
    print(f"  raw kline count: {len(klines)}")

    if klines:
        first_dt = datetime.fromtimestamp(int(klines[0][0]) / 1000, tz=timezone.utc)
        last_dt = datetime.fromtimestamp(int(klines[-1][0]) / 1000, tz=timezone.utc)
        print(f"  first raw candle time: {format_time_utc(first_dt)}")
        print(f"  last raw candle time: {format_time_utc(last_dt)}")
    else:
        print("  first raw candle time: N/A")
        print("  last raw candle time: N/A")

    print(f"  candle count (pre-HA): {len(candles)}")


def fetch_klines_for_review(
    api_key: str,
    symbol: str,
    warmup_start: datetime,
    review_start: datetime,
    review_end: datetime,
) -> tuple[list[list], list[dict], bool, str]:
    fetch_end = review_end + timedelta(minutes=CANDLE_MINUTES * 2)
    start_ms = int(warmup_start.timestamp() * 1000)
    end_ms = int(fetch_end.timestamp() * 1000)

    print(f"  {symbol}: 초기 캔들 데이터 요청 중...")
    klines = fetch_klines_range(api_key, symbol, start_ms, end_ms)
    max_attempts = 12

    for attempt in range(1, max_attempts + 1):
        candles = build_candles(klines)
        missing = missing_signal_opens(candles, review_start, review_end)

        if not missing:
            print(
                f"  {symbol}: 리뷰 구간 캔들 {EXPECTED_SIGNAL_CANDLES}개 확인 완료"
            )
            return klines, candles, True, ""

        print(
            f"  {symbol}: 누락 캔들 {len(missing)}개 -> 추가 요청 ({attempt}/{max_attempts})"
        )
        previous_count = len(klines)
        missing_ranges = group_contiguous_opens(missing)

        for range_start, range_end in missing_ranges:
            range_start_ms = int(
                (range_start - timedelta(minutes=CANDLE_MINUTES)).timestamp() * 1000
            )
            range_end_ms = int(
                (range_end + timedelta(minutes=CANDLE_MINUTES)).timestamp() * 1000
            )
            extra = fetch_klines_range(
                api_key,
                symbol,
                range_start_ms,
                range_end_ms,
            )
            if extra:
                klines = merge_klines(klines, extra)

        if len(klines) == previous_count:
            break

    candles = build_candles(klines)
    missing = missing_signal_opens(candles, review_start, review_end)
    if not missing:
        print(
            f"  {symbol}: 리뷰 구간 캔들 {EXPECTED_SIGNAL_CANDLES}개 확인 완료"
        )
        return klines, candles, True, ""

    reason = build_missing_data_reason(symbol, missing, review_start, review_end)
    return klines, candles, False, reason


def public_get(api_key: str, endpoint: str, params: dict | None = None) -> object:
    query = urllib.parse.urlencode(params or {})
    url = f"{FUTURES_BASE_URL}{endpoint}"
    if query:
        url = f"{url}?{query}"

    request = urllib.request.Request(
        url,
        headers={"X-MBX-APIKEY": api_key},
        method="GET",
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_klines_before(
    api_key: str,
    symbol: str,
    end_ms: int,
    limit: int,
) -> list[list]:
    params = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": "5m",
            "endTime": end_ms,
            "limit": limit,
        }
    )
    url = f"{FUTURES_BASE_URL}{KLINES_ENDPOINT}?{params}"
    request = urllib.request.Request(
        url,
        headers={"X-MBX-APIKEY": api_key},
        method="GET",
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_klines_range(
    api_key: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    all_klines: list[list] = []
    current_start = start_ms

    while current_start < end_ms:
        params = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "interval": "5m",
                "startTime": current_start,
                "endTime": end_ms,
                "limit": MAX_LIMIT,
            }
        )
        url = f"{FUTURES_BASE_URL}{KLINES_ENDPOINT}?{params}"
        request = urllib.request.Request(
            url,
            headers={"X-MBX-APIKEY": api_key},
            method="GET",
        )

        with urllib.request.urlopen(request, timeout=20) as response:
            batch = json.loads(response.read().decode("utf-8"))

        if not batch:
            break

        all_klines.extend(batch)
        last_open = int(batch[-1][0])
        next_start = last_open + INTERVAL_MS

        if next_start <= current_start:
            break

        current_start = next_start

    return all_klines


def get_eligible_symbols(api_key: str) -> set[str]:
    exchange_info = public_get(api_key, EXCHANGE_INFO_ENDPOINT)
    eligible: set[str] = set()

    for symbol_info in exchange_info.get("symbols", []):
        symbol = symbol_info.get("symbol", "")
        order_types = symbol_info.get("orderTypes", [])

        if symbol_info.get("status") != "TRADING":
            continue
        if symbol_info.get("contractType") != "PERPETUAL":
            continue
        if symbol_info.get("quoteAsset") != "USDT":
            continue
        if "MARKET" not in order_types:
            continue
        if symbol in EXCLUDED_SYMBOLS:
            continue

        eligible.add(symbol)

    return eligible


def build_candles(klines: list[list]) -> list[dict]:
    candles: list[dict] = []

    for candle in klines:
        open_dt = datetime.fromtimestamp(int(candle[0]) / 1000, tz=timezone.utc)
        candles.append(
            {
                "open_dt": open_dt,
                "open_time": format_time(open_dt),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
                "quote_volume": float(candle[7]),
            }
        )

    return candles


def calculate_heikin_ashi(candles: list[dict]) -> None:
    prev_ha_open: float | None = None
    prev_ha_close: float | None = None

    for candle in candles:
        open_price = candle["open"]
        high_price = candle["high"]
        low_price = candle["low"]
        close_price = candle["close"]

        ha_close = (open_price + high_price + low_price + close_price) / 4
        ha_open = (
            (open_price + close_price) / 2
            if prev_ha_open is None or prev_ha_close is None
            else (prev_ha_open + prev_ha_close) / 2
        )

        candle["ha_open"] = ha_open
        candle["ha_close"] = ha_close
        candle["direction"] = "GREEN" if ha_close >= ha_open else "RED"

        prev_ha_open = ha_open
        prev_ha_close = ha_close


def calculate_ha_body_metrics(candle: dict) -> tuple[float, float]:
    ha_open = candle["ha_open"]
    ha_close = candle["ha_close"]

    if ha_open == 0:
        ha_body_percent = 0.0
    else:
        ha_body_percent = (ha_close - ha_open) / ha_open

    ha_high = max(candle["high"], ha_open, ha_close)
    ha_low = min(candle["low"], ha_open, ha_close)
    candle_range = ha_high - ha_low

    if candle_range == 0:
        body_ratio = 0.0
    else:
        body_ratio = abs(ha_close - ha_open) / candle_range

    return ha_body_percent, body_ratio


def calculate_volume_ratio(candles: list[dict]) -> float:
    if len(candles) < LOOKBACK_3H_CANDLES:
        return 0.0

    recent = candles[-LOOKBACK_3H_CANDLES:]
    current_volume = recent[-1]["volume"]
    average_volume = sum(candle["volume"] for candle in recent) / LOOKBACK_3H_CANDLES

    if average_volume == 0:
        return 0.0

    return current_volume / average_volume


def calculate_momentum_metrics(
    candles: list[dict],
) -> tuple[float, float, float, float]:
    current = candles[-1]
    ha_body_percent, body_ratio = calculate_ha_body_metrics(current)
    volume_ratio = calculate_volume_ratio(candles)
    momentum = ha_body_percent * body_ratio * volume_ratio
    return ha_body_percent, body_ratio, volume_ratio, momentum


def calculate_slot_score(momentum: float, price_change_6h: float) -> float:
    return momentum / math.sqrt(price_change_6h + 1)


def analyze_symbol_at_search(
    api_key: str,
    symbol: str,
    search_dt: datetime,
) -> dict | None:
    end_ms = int(search_dt.timestamp() * 1000)
    klines = fetch_klines_before(api_key, symbol, end_ms, LOOKBACK_6H_CANDLES)

    if len(klines) < LOOKBACK_6H_CANDLES:
        return None

    candles = build_candles(klines)
    calculate_heikin_ashi(candles)

    last_price = candles[-1]["close"]
    if not (MIN_PRICE <= last_price <= MAX_PRICE):
        return None

    recent_1h = candles[-LOOKBACK_1H_CANDLES:]
    volume_1h = sum(candle["quote_volume"] for candle in recent_1h)

    start_price = candles[0]["close"]
    if start_price == 0:
        return None

    price_change_6h = (last_price - start_price) / start_price * 100
    if not (PRICE_CHANGE_MIN <= price_change_6h <= PRICE_CHANGE_MAX):
        return None

    ha_body_percent, body_ratio, volume_ratio, momentum = calculate_momentum_metrics(
        candles
    )

    return {
        "symbol": symbol,
        "last_price": last_price,
        "volume_1h": volume_1h,
        "price_change_6h": price_change_6h,
        "ha_body_percent": ha_body_percent,
        "body_ratio": body_ratio,
        "volume_ratio": volume_ratio,
        "momentum": momentum,
        "slot_score": calculate_slot_score(momentum, price_change_6h),
    }


def run_slot_search(
    api_key: str,
    eligible_symbols: set[str],
    search_dt: datetime,
) -> list[SlotSelection]:
    print("\n=== Five Slot Search 시작 ===")
    print(f"검색 시각 KST: {format_time_kst(search_dt)}")
    print(f"검색 시각 UTC: {format_time_utc(search_dt)}")

    candidates: list[dict] = []
    symbols = sorted(eligible_symbols)
    total = len(symbols)

    for index, symbol in enumerate(symbols, start=1):
        if index % 25 == 0 or index == total:
            print(f"심볼 스크리닝 중... ({index}/{total})")

        try:
            result = analyze_symbol_at_search(api_key, symbol, search_dt)
            if result is not None:
                candidates.append(result)
        except urllib.error.HTTPError:
            continue

        time.sleep(0.05)

    if not candidates:
        print("오류: 검색 조건에 맞는 심볼을 찾지 못했습니다.")
        return []

    top_volume = sorted(candidates, key=lambda item: item["volume_1h"], reverse=True)[
        :TOP_VOLUME_COUNT
    ]
    ranked = sorted(top_volume, key=lambda item: item["slot_score"], reverse=True)

    print(f"1시간 거래대금 TOP{TOP_VOLUME_COUNT} 필터 완료: {len(top_volume)}개")

    selections: list[SlotSelection] = []
    for rank, item in enumerate(ranked[:TOP_RANK_COUNT], start=1):
        slot_name = f"Slot{rank}" if rank <= SLOT_COUNT else ""
        selections.append(
            SlotSelection(
                rank=rank,
                slot_name=slot_name,
                symbol=item["symbol"],
                previous_6h_price_change_percent=item["price_change_6h"],
                ha_body_percent=item["ha_body_percent"],
                body_ratio=item["body_ratio"],
                volume_ratio=item["volume_ratio"],
                momentum=item["momentum"],
                slot_score=item["slot_score"],
                recent_1h_trading_value=item["volume_1h"],
            )
        )

    return selections


def save_slot_score_review(
    search_dt: datetime,
    selections: list[SlotSelection],
) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for selection in selections:
        rows.append(
            {
                "review_date": REVIEW_DATE,
                "search_time_kst": format_time_kst(search_dt),
                "search_time_utc": format_time_utc(search_dt),
                "rank": str(selection.rank),
                "slot": selection.slot_name,
                "symbol": selection.symbol,
                "previous_6h_price_change_percent": (
                    f"{selection.previous_6h_price_change_percent:.4f}"
                ),
                "ha_body_percent": f"{selection.ha_body_percent:.8f}",
                "body_ratio": f"{selection.body_ratio:.8f}",
                "volume_ratio": f"{selection.volume_ratio:.8f}",
                "momentum": f"{selection.momentum:.8f}",
                "slot_score": f"{selection.slot_score:.8f}",
                "trading_value_1h": f"{selection.recent_1h_trading_value:.2f}",
            }
        )

    fieldnames = [
        "review_date",
        "search_time_kst",
        "search_time_utc",
        "rank",
        "slot",
        "symbol",
        "previous_6h_price_change_percent",
        "ha_body_percent",
        "body_ratio",
        "volume_ratio",
        "momentum",
        "slot_score",
        "trading_value_1h",
    ]

    with SLOT_SCORE_OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_top10_table(selections: list[SlotSelection]) -> None:
    print(f"\n=== TOP{TOP_RANK_COUNT} Slot Score Candidates ===")
    header = (
        f"{'Rank':>4}  {'Symbol':<12}  {'6h_chg%':>8}  {'HA_body%':>10}  "
        f"{'Body_r':>8}  {'Vol_r':>8}  {'Momentum':>12}  {'Score':>12}  "
        f"{'1h_value':>14}  Slot"
    )
    print(header)
    print("-" * len(header))

    for selection in selections:
        slot_label = selection.slot_name if selection.slot_name else "-"
        highlight = f"  << {selection.slot_name} >>" if selection.slot_name else ""
        print(
            f"{selection.rank:>4}  {selection.symbol:<12}  "
            f"{selection.previous_6h_price_change_percent:>8.2f}  "
            f"{selection.ha_body_percent:>10.6f}  "
            f"{selection.body_ratio:>8.4f}  "
            f"{selection.volume_ratio:>8.4f}  "
            f"{selection.momentum:>12.8f}  "
            f"{selection.slot_score:>12.8f}  "
            f"{selection.recent_1h_trading_value:>14,.0f}  {slot_label}{highlight}"
        )

    print("=" * len(header))


def parse_event_time(event_time: str) -> datetime:
    cleaned = event_time.replace(" UTC", "").replace(" KST", "")
    dt = datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")
    if event_time.endswith("KST"):
        return dt.replace(tzinfo=KST_TZ).astimezone(timezone.utc)
    return dt.replace(tzinfo=timezone.utc)


def format_event_times(event_time: str) -> tuple[str, str]:
    event_dt = parse_event_time(event_time)
    return format_time_kst(event_dt), format_time_utc(event_dt)


def is_ha_close_highest(candles: list[dict], index: int) -> bool:
    if index < HA_SIGNAL_LOOKBACK - 1:
        return False

    window = candles[index - (HA_SIGNAL_LOOKBACK - 1) : index + 1]
    current_close = candles[index]["ha_close"]
    return current_close == max(candle["ha_close"] for candle in window)


def is_ha_close_lowest(candles: list[dict], index: int) -> bool:
    if index < HA_SIGNAL_LOOKBACK - 1:
        return False

    window = candles[index - (HA_SIGNAL_LOOKBACK - 1) : index + 1]
    current_close = candles[index]["ha_close"]
    return current_close == min(candle["ha_close"] for candle in window)


def is_valid_entry_signal(
    candles: list[dict],
    index: int,
    exit_cooldown_index: int | None,
) -> bool:
    if exit_cooldown_index is not None and index == exit_cooldown_index:
        return False

    candle = candles[index]
    if candle["direction"] != "GREEN":
        return False

    return is_ha_close_highest(candles, index)


def is_valid_exit_signal(candles: list[dict], index: int) -> bool:
    candle = candles[index]
    if candle["direction"] != "RED":
        return False

    return is_ha_close_lowest(candles, index)


def emergency_stop_price(entry_price: float) -> float:
    return entry_price * EMERGENCY_STOP_MULTIPLIER


def is_ha_close_lower_than_previous_five(candles: list[dict], index: int) -> bool:
    if index < HA_SIGNAL_LOOKBACK - 1:
        return False

    current_close = candles[index]["ha_close"]
    previous_closes = [
        candles[prev_index]["ha_close"]
        for prev_index in range(index - (HA_SIGNAL_LOOKBACK - 1), index)
    ]
    return current_close < min(previous_closes)


def is_valid_short_entry_signal(
    candles: list[dict],
    index: int,
    exit_cooldown_index: int | None,
) -> bool:
    if exit_cooldown_index is not None and index == exit_cooldown_index:
        return False

    candle = candles[index]
    if candle["direction"] != "RED":
        return False

    return is_ha_close_lower_than_previous_five(candles, index)


def is_valid_short_exit_signal(candles: list[dict], index: int) -> bool:
    return candles[index]["direction"] == "GREEN"


def long_profit_percent(entry_price: float, exit_price: float) -> float:
    return (exit_price - entry_price) / entry_price * 100


def short_profit_percent(entry_price: float, exit_price: float) -> float:
    return (entry_price - exit_price) / entry_price * 100


def emergency_short_stop_price(entry_price: float) -> float:
    return entry_price * EMERGENCY_SHORT_STOP_MULTIPLIER


def build_fixed_research_selections() -> list[SlotSelection]:
    selections: list[SlotSelection] = []
    for rank, (slot_name, symbol) in enumerate(RESEARCH_FIXED_SLOTS.items(), start=1):
        selections.append(
            SlotSelection(
                rank=rank,
                slot_name=slot_name,
                symbol=symbol,
                previous_6h_price_change_percent=0.0,
                ha_body_percent=0.0,
                body_ratio=0.0,
                volume_ratio=0.0,
                momentum=0.0,
                slot_score=0.0,
                recent_1h_trading_value=0.0,
            )
        )
    return selections


def try_long_emergency_exit(
    result: SlotResult,
    slot_name: str,
    symbol: str,
    candle: dict,
    position: dict,
    research_mode: bool = False,
) -> bool:
    stop_price = emergency_stop_price(position["entry_price"])
    if candle["low"] > stop_price:
        return False

    profit = long_profit_percent(position["entry_price"], stop_price)
    result.total_profit_percent += profit
    result.total_trades += 1
    result.emergency_exits += 1
    result.long_profit_percent += profit
    result.long_trades += 1
    result.long_emergency_exits += 1
    result.events.append(
        TradeEvent(
            event_type="LONG_EMERGENCY_EXIT" if research_mode else "EMERGENCY_EXIT",
            slot=slot_name,
            symbol=symbol,
            event_time=candle["open_time"],
            price=stop_price,
            profit_percent=profit,
            detail=(
                f"highest-priority intrabar stop | side=LONG | "
                f"entry={position['entry_price']:.8f} | "
                f"emergency_stop_price={stop_price:.8f} "
                f"(entry * {EMERGENCY_STOP_MULTIPLIER}) | "
                f"candle_low={candle['low']:.8f}"
            ),
        )
    )
    return True


def try_short_emergency_exit(
    result: SlotResult,
    slot_name: str,
    symbol: str,
    candle: dict,
    position: dict,
) -> bool:
    stop_price = emergency_short_stop_price(position["entry_price"])
    if candle["high"] < stop_price:
        return False

    profit = short_profit_percent(position["entry_price"], stop_price)
    result.total_profit_percent += profit
    result.total_trades += 1
    result.emergency_exits += 1
    result.short_profit_percent += profit
    result.short_trades += 1
    result.short_emergency_exits += 1
    result.events.append(
        TradeEvent(
            event_type="SHORT_EMERGENCY_EXIT",
            slot=slot_name,
            symbol=symbol,
            event_time=candle["open_time"],
            price=stop_price,
            profit_percent=profit,
            detail=(
                f"highest-priority intrabar stop | side=SHORT | "
                f"entry={position['entry_price']:.8f} | "
                f"emergency_stop_price={stop_price:.8f} "
                f"(entry * {EMERGENCY_SHORT_STOP_MULTIPLIER}) | "
                f"candle_high={candle['high']:.8f}"
            ),
        )
    )
    return True


def try_emergency_exit(
    result: SlotResult,
    slot_name: str,
    symbol: str,
    candle: dict,
    position: dict,
) -> bool:
    return try_long_emergency_exit(
        result, slot_name, symbol, candle, position, research_mode=False
    )


def limit_order_filled(candle: dict, limit_price: float) -> bool:
    return candle["low"] <= limit_price <= candle["high"]


def simulate_slot(
    slot_name: str,
    symbol: str,
    candles: list[dict],
    review_start: datetime,
    review_end: datetime,
) -> SlotResult:
    result = SlotResult(slot_name=slot_name, symbol=symbol)
    position: dict | None = None
    pending_exit = False
    pending_exit_reason = ""
    limit_order: dict | None = None
    has_exited_before = False
    exit_cooldown_index: int | None = None

    simulation_candles = get_simulation_candles(candles, review_start, review_end)
    index_by_open = {candle["open_dt"]: index for index, candle in enumerate(candles)}

    result.processed_candle_count = len(simulation_candles)
    result.signal_candle_count = sum(
        1 for candle in simulation_candles if is_signal_candle(candle, review_end)
    )
    result.actual_duration_minutes = calculate_actual_duration_minutes(
        simulation_candles,
        review_start,
        review_end,
    )

    for candle in simulation_candles:
        candle_index = index_by_open[candle["open_dt"]]

        if position is not None and try_emergency_exit(
            result, slot_name, symbol, candle, position
        ):
            exit_cooldown_index = candle_index
            position = None
            pending_exit = False
            pending_exit_reason = ""
            has_exited_before = True
            continue

        if pending_exit and position is not None:
            exit_price = candle["open"]
            profit = (exit_price - position["entry_price"]) / position["entry_price"] * 100
            result.total_profit_percent += profit
            result.total_trades += 1
            result.events.append(
                TradeEvent(
                    event_type="EXIT",
                    slot=slot_name,
                    symbol=symbol,
                    event_time=candle["open_time"],
                    price=exit_price,
                    profit_percent=profit,
                    detail=pending_exit_reason,
                )
            )
            position = None
            pending_exit = False
            pending_exit_reason = ""
            has_exited_before = True

        if limit_order is not None and position is None:
            limit_price = limit_order["price"]
            if limit_order_filled(candle, limit_price):
                event_type = "REENTRY" if has_exited_before else "ENTRY"
                position = {
                    "entry_price": limit_price,
                    "entry_time": candle["open_time"],
                }
                result.events.append(
                    TradeEvent(
                        event_type=event_type,
                        slot=slot_name,
                        symbol=symbol,
                        event_time=candle["open_time"],
                        price=limit_price,
                        detail=(
                            "5HA GREEN highest close | "
                            f"limit filled at signal={limit_order['signal_time']} "
                            f"ha_close={limit_price:.8f}"
                        ),
                    )
                )
            else:
                result.unfilled_limits += 1
                result.events.append(
                    TradeEvent(
                        event_type="UNFILLED_LIMIT",
                        slot=slot_name,
                        symbol=symbol,
                        event_time=candle["open_time"],
                        price=limit_price,
                        detail=(
                            "5HA GREEN highest close | "
                            f"limit not filled | signal={limit_order['signal_time']}"
                        ),
                    )
                )
            limit_order = None

            if position is not None and try_emergency_exit(
                result, slot_name, symbol, candle, position
            ):
                exit_cooldown_index = candle_index
                position = None
                pending_exit = False
                pending_exit_reason = ""
                has_exited_before = True
                continue

        if not is_signal_candle(candle, review_end):
            continue

        if position is not None and is_valid_exit_signal(candles, candle_index):
            pending_exit = True
            exit_cooldown_index = candle_index
            pending_exit_reason = (
                "5HA RED lowest close among last 6 | exit at next candle open | "
                f"signal={candle['open_time']} ha_close={candle['ha_close']:.8f}"
            )
            continue

        if (
            position is None
            and limit_order is None
            and is_valid_entry_signal(candles, candle_index, exit_cooldown_index)
        ):
            limit_order = {
                "price": candle["ha_close"],
                "signal_time": candle["open_time"],
            }

    if limit_order is not None:
        result.unfilled_limits += 1
        result.events.append(
            TradeEvent(
                event_type="UNFILLED_LIMIT",
                slot=slot_name,
                symbol=symbol,
                event_time=format_time(review_end),
                price=limit_order["price"],
                detail=(
                    "5HA GREEN highest close | "
                    f"review_end_pending_limit | signal={limit_order['signal_time']}"
                ),
            )
        )

    if position is not None:
        exit_price = position["entry_price"]
        if simulation_candles:
            last_signal_candles = [
                signal_candle
                for signal_candle in simulation_candles
                if is_signal_candle(signal_candle, review_end)
            ]
            if last_signal_candles:
                exit_price = last_signal_candles[-1]["close"]

        profit = (exit_price - position["entry_price"]) / position["entry_price"] * 100
        result.total_profit_percent += profit
        result.total_trades += 1
        result.forced_exits += 1
        result.events.append(
            TradeEvent(
                event_type="FORCED_EXIT",
                slot=slot_name,
                symbol=symbol,
                event_time=format_time(review_end),
                price=exit_price,
                profit_percent=profit,
                detail="review_end_force_close",
            )
        )

    annotate_cumulative_profit(result.events)
    return result


def simulate_slot_long_short(
    slot_name: str,
    symbol: str,
    candles: list[dict],
    review_start: datetime,
    review_end: datetime,
) -> SlotResult:
    result = SlotResult(slot_name=slot_name, symbol=symbol)
    position: dict | None = None
    pending_exit = False
    pending_exit_reason = ""
    limit_order: dict | None = None
    has_long_before = False
    has_short_before = False
    exit_cooldown_long_index: int | None = None
    exit_cooldown_short_index: int | None = None

    simulation_candles = get_simulation_candles(candles, review_start, review_end)
    index_by_open = {candle["open_dt"]: index for index, candle in enumerate(candles)}

    result.processed_candle_count = len(simulation_candles)
    result.signal_candle_count = sum(
        1 for candle in simulation_candles if is_signal_candle(candle, review_end)
    )
    result.actual_duration_minutes = calculate_actual_duration_minutes(
        simulation_candles,
        review_start,
        review_end,
    )

    for candle in simulation_candles:
        candle_index = index_by_open[candle["open_dt"]]

        if position is not None:
            if position["side"] == "LONG" and try_long_emergency_exit(
                result, slot_name, symbol, candle, position, research_mode=True
            ):
                exit_cooldown_long_index = candle_index
                position = None
                pending_exit = False
                pending_exit_reason = ""
                has_long_before = True
                continue

            if position["side"] == "SHORT" and try_short_emergency_exit(
                result, slot_name, symbol, candle, position
            ):
                exit_cooldown_short_index = candle_index
                position = None
                pending_exit = False
                pending_exit_reason = ""
                has_short_before = True
                continue

        if pending_exit and position is not None:
            exit_price = candle["open"]
            if position["side"] == "LONG":
                profit = long_profit_percent(position["entry_price"], exit_price)
                event_type = "LONG_EXIT"
                has_long_before = True
                result.long_profit_percent += profit
                result.long_trades += 1
            else:
                profit = short_profit_percent(position["entry_price"], exit_price)
                event_type = "SHORT_EXIT"
                has_short_before = True
                result.short_profit_percent += profit
                result.short_trades += 1

            result.total_profit_percent += profit
            result.total_trades += 1
            result.events.append(
                TradeEvent(
                    event_type=event_type,
                    slot=slot_name,
                    symbol=symbol,
                    event_time=candle["open_time"],
                    price=exit_price,
                    profit_percent=profit,
                    detail=pending_exit_reason,
                )
            )
            position = None
            pending_exit = False
            pending_exit_reason = ""

        if limit_order is not None and position is None:
            limit_price = limit_order["price"]
            side = limit_order["side"]
            if limit_order_filled(candle, limit_price):
                position = {
                    "side": side,
                    "entry_price": limit_price,
                    "entry_time": candle["open_time"],
                }
                if side == "LONG":
                    event_type = "LONG_REENTRY" if has_long_before else "LONG_ENTRY"
                else:
                    event_type = "SHORT_REENTRY" if has_short_before else "SHORT_ENTRY"

                result.events.append(
                    TradeEvent(
                        event_type=event_type,
                        slot=slot_name,
                        symbol=symbol,
                        event_time=candle["open_time"],
                        price=limit_price,
                        detail=limit_order["detail"],
                    )
                )
            else:
                result.unfilled_limits += 1
                result.events.append(
                    TradeEvent(
                        event_type="UNFILLED_LIMIT",
                        slot=slot_name,
                        symbol=symbol,
                        event_time=candle["open_time"],
                        price=limit_price,
                        detail=limit_order["detail_unfilled"],
                    )
                )
            limit_order = None

            if position is not None:
                if position["side"] == "LONG" and try_long_emergency_exit(
                    result, slot_name, symbol, candle, position, research_mode=True
                ):
                    exit_cooldown_long_index = candle_index
                    position = None
                    pending_exit = False
                    pending_exit_reason = ""
                    has_long_before = True
                    continue

                if position["side"] == "SHORT" and try_short_emergency_exit(
                    result, slot_name, symbol, candle, position
                ):
                    exit_cooldown_short_index = candle_index
                    position = None
                    pending_exit = False
                    pending_exit_reason = ""
                    has_short_before = True
                    continue

        if not is_signal_candle(candle, review_end):
            continue

        if position is not None and position["side"] == "LONG":
            if is_valid_exit_signal(candles, candle_index):
                pending_exit = True
                exit_cooldown_long_index = candle_index
                pending_exit_reason = (
                    "5HA RED lowest close among last 6 | LONG exit at next open | "
                    f"signal={candle['open_time']} ha_close={candle['ha_close']:.8f}"
                )
            continue

        if position is not None and position["side"] == "SHORT":
            if is_valid_short_exit_signal(candles, candle_index):
                pending_exit = True
                exit_cooldown_short_index = candle_index
                pending_exit_reason = (
                    "HA GREEN close | SHORT exit at next open | "
                    f"signal={candle['open_time']} ha_close={candle['ha_close']:.8f}"
                )
            continue

        if position is None and limit_order is None:
            if is_valid_entry_signal(
                candles, candle_index, exit_cooldown_long_index
            ):
                limit_order = {
                    "side": "LONG",
                    "price": candle["ha_close"],
                    "signal_time": candle["open_time"],
                    "detail": (
                        "5HA GREEN highest close | "
                        f"LONG limit filled at signal={candle['open_time']} "
                        f"ha_close={candle['ha_close']:.8f}"
                    ),
                    "detail_unfilled": (
                        "5HA GREEN highest close | "
                        f"LONG limit not filled | signal={candle['open_time']}"
                    ),
                }
            elif is_valid_short_entry_signal(
                candles, candle_index, exit_cooldown_short_index
            ):
                limit_order = {
                    "side": "SHORT",
                    "price": candle["ha_close"],
                    "signal_time": candle["open_time"],
                    "detail": (
                        "HA RED close lower than previous 5 HA closes | "
                        f"SHORT limit filled at signal={candle['open_time']} "
                        f"ha_close={candle['ha_close']:.8f}"
                    ),
                    "detail_unfilled": (
                        "HA RED close lower than previous 5 HA closes | "
                        f"SHORT limit not filled | signal={candle['open_time']}"
                    ),
                }

    if limit_order is not None:
        result.unfilled_limits += 1
        result.events.append(
            TradeEvent(
                event_type="UNFILLED_LIMIT",
                slot=slot_name,
                symbol=symbol,
                event_time=format_time(review_end),
                price=limit_order["price"],
                detail=(
                    f"{limit_order['detail_unfilled']} | "
                    f"review_end_pending_limit | signal={limit_order['signal_time']}"
                ),
            )
        )

    if position is not None:
        exit_price = position["entry_price"]
        if simulation_candles:
            last_signal_candles = [
                signal_candle
                for signal_candle in simulation_candles
                if is_signal_candle(signal_candle, review_end)
            ]
            if last_signal_candles:
                exit_price = last_signal_candles[-1]["close"]

        if position["side"] == "LONG":
            profit = long_profit_percent(position["entry_price"], exit_price)
            result.long_profit_percent += profit
            result.long_trades += 1
        else:
            profit = short_profit_percent(position["entry_price"], exit_price)
            result.short_profit_percent += profit
            result.short_trades += 1

        result.total_profit_percent += profit
        result.total_trades += 1
        result.forced_exits += 1
        result.events.append(
            TradeEvent(
                event_type="FORCED_EXIT",
                slot=slot_name,
                symbol=symbol,
                event_time=format_time(review_end),
                price=exit_price,
                profit_percent=profit,
                detail=(
                    f"review_end_force_close | side={position['side']} | "
                    f"entry={position['entry_price']:.8f}"
                ),
            )
        )

    annotate_cumulative_profit(result.events)
    return result


def annotate_cumulative_profit(events: list[TradeEvent]) -> None:
    cumulative = 0.0
    for event in events:
        if event.event_type in PROFIT_EVENT_TYPES:
            cumulative += event.profit_percent
            event.cumulative_profit_percent = cumulative


def compute_research_slot_stats(slot_result: SlotResult) -> dict[str, float | int]:
    profits = [
        event.profit_percent
        for event in slot_result.events
        if event.event_type in PROFIT_EVENT_TYPES
    ]
    wins = [profit for profit in profits if profit > 0]
    losses = [profit for profit in profits if profit < 0]

    return {
        "long_profit": slot_result.long_profit_percent,
        "short_profit": slot_result.short_profit_percent,
        "combined_profit": slot_result.total_profit_percent,
        "long_trades": slot_result.long_trades,
        "short_trades": slot_result.short_trades,
        "trade_count": slot_result.total_trades,
        "average_trade": sum(profits) / len(profits) if profits else 0.0,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "emergency_exits": slot_result.emergency_exits,
        "long_emergency_exits": slot_result.long_emergency_exits,
        "short_emergency_exits": slot_result.short_emergency_exits,
        "unfilled_limits": slot_result.unfilled_limits,
        "forced_exits": slot_result.forced_exits,
        "max_winning": max(wins) if wins else 0.0,
        "max_losing": min(losses) if losses else 0.0,
    }


def format_research_slot_summary_detail(stats: dict[str, float | int]) -> str:
    return (
        f"long_profit={stats['long_profit']:.4f}; "
        f"short_profit={stats['short_profit']:.4f}; "
        f"combined_profit={stats['combined_profit']:.4f}; "
        f"long_trades={stats['long_trades']}; "
        f"short_trades={stats['short_trades']}; "
        f"average_trade_percent={stats['average_trade']:.4f}; "
        f"winning_trades={stats['winning_trades']}; "
        f"losing_trades={stats['losing_trades']}; "
        f"long_emergency_exits={stats['long_emergency_exits']}; "
        f"short_emergency_exits={stats['short_emergency_exits']}; "
        f"unfilled_limits={stats['unfilled_limits']}; "
        f"forced_exits={stats['forced_exits']}; "
        f"max_winning_trade={stats['max_winning']:.4f}; "
        f"max_losing_trade={stats['max_losing']:.4f}"
    )


def compute_slot_stats(slot_result: SlotResult) -> dict[str, float | int]:
    profits = [
        event.profit_percent
        for event in slot_result.events
        if event.event_type in PROFIT_EVENT_TYPES
    ]
    wins = [profit for profit in profits if profit > 0]
    losses = [profit for profit in profits if profit < 0]

    return {
        "total_profit": slot_result.total_profit_percent,
        "trade_count": slot_result.total_trades,
        "average_trade": sum(profits) / len(profits) if profits else 0.0,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "emergency_exits": slot_result.emergency_exits,
        "unfilled_limits": slot_result.unfilled_limits,
        "forced_exits": slot_result.forced_exits,
        "max_winning": max(wins) if wins else 0.0,
        "max_losing": min(losses) if losses else 0.0,
    }


def format_slot_summary_detail(stats: dict[str, float | int]) -> str:
    return (
        f"total_trades={stats['trade_count']}; "
        f"average_trade_percent={stats['average_trade']:.4f}; "
        f"winning_trades={stats['winning_trades']}; "
        f"losing_trades={stats['losing_trades']}; "
        f"emergency_exits={stats['emergency_exits']}; "
        f"unfilled_limits={stats['unfilled_limits']}; "
        f"forced_exits={stats['forced_exits']}; "
        f"max_winning_trade={stats['max_winning']:.4f}; "
        f"max_losing_trade={stats['max_losing']:.4f}"
    )


def save_invalid_report(
    search_dt: datetime,
    review_end: datetime,
    selections: list[SlotSelection],
    invalid_symbols: list[tuple[str, str]],
) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = [
        {
            "review_date": REVIEW_DATE,
            "event_type": "REVIEW_INVALID",
            "slot": "ALL",
            "symbol": "",
            "event_time": format_time(search_dt),
            "price": "",
            "profit_percent": "",
            "detail": (
                f"{format_review_window_detail(search_dt, review_end)}; "
                f"expected_minutes={EXPECTED_REVIEW_MINUTES}; "
                f"invalid_symbols={len(invalid_symbols)}"
            ),
        }
    ]

    for selection in selections:
        rows.append(
            {
                "review_date": REVIEW_DATE,
                "event_type": selection.slot_name.upper(),
                "slot": selection.slot_name,
                "symbol": selection.symbol,
                "event_time": format_time(search_dt),
                "price": "",
                "profit_percent": "",
                "detail": "selected_before_data_validation",
            }
        )

    for symbol, reason in invalid_symbols:
        rows.append(
            {
                "review_date": REVIEW_DATE,
                "event_type": "MISSING_DATA",
                "slot": "",
                "symbol": symbol,
                "event_time": format_time(search_dt),
                "price": "",
                "profit_percent": "",
                "detail": reason,
            }
        )

    fieldnames = [
        "review_date",
        "event_type",
        "slot",
        "symbol",
        "event_time",
        "price",
        "profit_percent",
        "detail",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_report(
    search_dt: datetime,
    review_end: datetime,
    selections: list[SlotSelection],
    slot_results: list[SlotResult],
) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    rows.append(
        {
            "review_date": REVIEW_DATE,
            "event_type": "REVIEW_VALID",
            "slot": "ALL",
            "symbol": "",
            "event_time": format_time(search_dt),
            "price": "",
            "profit_percent": "",
            "detail": (
                f"{format_review_window_detail(search_dt, review_end)}; "
                f"mode=LONG_ONLY"
            ),
        }
    )

    rows.append(
        {
            "review_date": REVIEW_DATE,
            "event_type": "SEARCH",
            "slot": "ALL",
            "symbol": "",
            "event_time": format_time_utc(search_dt),
            "price": "",
            "profit_percent": "",
            "detail": format_review_window_detail(search_dt, review_end),
        }
    )

    for selection in selections:
        rows.append(
            {
                "review_date": REVIEW_DATE,
                "event_type": selection.slot_name.upper(),
                "slot": selection.slot_name,
                "symbol": selection.symbol,
                "event_time": format_time(search_dt),
                "price": "",
                "profit_percent": "",
                "detail": (
                    f"previous_6h_price_change_percent={selection.previous_6h_price_change_percent:.4f}; "
                    f"momentum={selection.momentum:.8f}; "
                    f"slot_score={selection.slot_score:.6f}; "
                    f"recent_1h_trading_value={selection.recent_1h_trading_value:.2f}"
                ),
            }
        )

    for slot_result in slot_results:
        stats = compute_slot_stats(slot_result)
        for event in slot_result.events:
            time_kst, time_utc = format_event_times(event.event_time)
            rows.append(
                {
                    "review_date": REVIEW_DATE,
                    "event_type": event.event_type,
                    "slot": event.slot,
                    "symbol": event.symbol,
                    "event_time": event.event_time,
                    "price": f"{event.price:.8f}",
                    "profit_percent": f"{event.profit_percent:.4f}",
                    "detail": (
                        f"time_kst={time_kst}; time_utc={time_utc}; "
                        f"reason={event.detail}; "
                        f"cumulative_profit={event.cumulative_profit_percent:.4f}"
                    ),
                }
            )

        rows.append(
            {
                "review_date": REVIEW_DATE,
                "event_type": "SLOT_SUMMARY",
                "slot": slot_result.slot_name,
                "symbol": slot_result.symbol,
                "event_time": format_time(review_end),
                "price": "",
                "profit_percent": f"{slot_result.total_profit_percent:.4f}",
                "detail": format_slot_summary_detail(stats),
            }
        )

    total_profit = sum(slot.total_profit_percent for slot in slot_results)
    total_trades = sum(slot.total_trades for slot in slot_results)
    all_profits = [
        event.profit_percent
        for slot in slot_results
        for event in slot.events
        if event.event_type in PROFIT_EVENT_TYPES
    ]
    combined_average = sum(all_profits) / len(all_profits) if all_profits else 0.0
    slot1_symbol = slot_results[0].symbol if len(slot_results) > 0 else ""
    slot2_symbol = slot_results[1].symbol if len(slot_results) > 1 else ""

    rows.append(
        {
            "review_date": REVIEW_DATE,
            "event_type": "SUMMARY",
            "slot": "ALL",
            "symbol": "",
            "event_time": format_time(review_end),
            "price": "",
            "profit_percent": f"{total_profit:.4f}",
            "detail": (
                f"{format_review_window_detail(search_dt, review_end)}; "
                f"slot1={slot1_symbol}; slot2={slot2_symbol}; "
                f"combined_trade_count={total_trades}; "
                f"combined_average_trade_percent={combined_average:.4f}; "
                f"review_duration_minutes={EXPECTED_REVIEW_MINUTES}"
            ),
        }
    )

    fieldnames = [
        "review_date",
        "event_type",
        "slot",
        "symbol",
        "event_time",
        "price",
        "profit_percent",
        "detail",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_simulation_duration(
    review_start: datetime,
    review_end: datetime,
    slot_results: list[SlotResult],
) -> None:
    expected_minutes = (review_end - review_start).total_seconds() / 60
    actual_minutes = (
        min(result.actual_duration_minutes for result in slot_results)
        if slot_results
        else 0.0
    )

    print("\n=== Simulation Duration ===")
    print_review_window_times(review_start, review_end)
    print(f"Actual simulated duration (minutes): {actual_minutes:.0f}")

    for slot_result in slot_results:
        print(
            f"  {slot_result.slot_name}: "
            f"{slot_result.actual_duration_minutes:.0f} min, "
            f"signal_candles={slot_result.signal_candle_count}/"
            f"{EXPECTED_SIGNAL_CANDLES}, "
            f"processed_candles={slot_result.processed_candle_count}"
        )

    if actual_minutes + 0.5 < expected_minutes:
        print(
            f"경고: 요청한 {expected_minutes:.0f}분 전체 구간을 채우지 못했습니다. "
            f"(실제 {actual_minutes:.0f}분)"
        )
    elif any(
        result.signal_candle_count < EXPECTED_SIGNAL_CANDLES for result in slot_results
    ):
        print(
            f"경고: 일부 슬롯의 5분 캔들 수가 부족합니다. "
            f"(기대 {EXPECTED_SIGNAL_CANDLES}개)"
        )
    else:
        print("Review coverage: COMPLETE")


def print_invalid_report(
    search_dt: datetime,
    review_end: datetime,
    invalid_symbols: list[tuple[str, str]],
) -> None:
    print("\n===== Five Solo Review Report =====")
    print("Review status: INVALID")
    print_review_window_times(search_dt, review_end)
    print(f"Required duration (minutes): {EXPECTED_REVIEW_MINUTES}")
    print("\n데이터 부족으로 리뷰를 완료할 수 없습니다.")
    for symbol, reason in invalid_symbols:
        print(f"\n  - {symbol}")
        print(f"    {reason}")
    print("\n시뮬레이션을 실행하지 않았습니다.")
    print("===================================")


def print_slot_events(slot_result: SlotResult) -> None:
    print(f"\n--- {slot_result.slot_name} ({slot_result.symbol}) ---")
    if not slot_result.events:
        print("  No events")
        return

    for event in slot_result.events:
        time_kst, time_utc = format_event_times(event.event_time)
        if event.event_type in PROFIT_EVENT_TYPES:
            print(
                f"  {event.event_type} | {slot_result.symbol} | "
                f"time_kst={time_kst} | time_utc={time_utc} | "
                f"price={event.price:.8f} | trade_profit={event.profit_percent:.4f}% | "
                f"cumulative_profit={event.cumulative_profit_percent:.4f}% | "
                f"reason={event.detail}"
            )
        else:
            print(
                f"  {event.event_type} | {slot_result.symbol} | "
                f"time_kst={time_kst} | time_utc={time_utc} | "
                f"price={event.price:.8f} | reason={event.detail}"
            )


def print_research_header(
    review_start: datetime,
    trading_selections: list[SlotSelection],
) -> None:
    search_kst = format_time_kst(review_start)
    search_utc = format_time_utc(review_start)
    print(
        f"\nSEARCH | time_kst={search_kst} | time_utc={search_utc} | "
        f"reason=fixed research symbols (Slot1=JTOUSDT, Slot2=WIFUSDT)"
    )
    for selection in trading_selections:
        print(
            f"{selection.slot_name.upper()} | symbol={selection.symbol} | "
            f"time_kst={search_kst} | time_utc={search_utc}"
        )


def print_long_short_research_review(
    review_start: datetime,
    review_end: datetime,
    trading_selections: list[SlotSelection],
    slot_results: list[SlotResult],
) -> None:
    print("\n===== FIVE LONG + SHORT RESEARCH =====")
    print_review_window_times(review_start, review_end)

    search_kst = format_time_kst(review_start)
    search_utc = format_time_utc(review_start)
    print(
        f"\nSEARCH | time_kst={search_kst} | time_utc={search_utc} | "
        f"reason=fixed research symbols from prior Five review (no new search)"
    )

    for selection in trading_selections:
        print(
            f"{selection.slot_name.upper()} | symbol={selection.symbol} | "
            f"time_kst={search_kst} | time_utc={search_utc} | "
            f"reason=fixed assignment"
        )

    for slot_result in slot_results:
        print_slot_events(slot_result)
        stats = compute_research_slot_stats(slot_result)
        print(f"\n  {slot_result.slot_name} ({slot_result.symbol}) Summary:")
        print(f"    Selected symbol: {slot_result.symbol}")
        print(f"    LONG profit %: {stats['long_profit']:.4f}")
        print(f"    SHORT profit %: {stats['short_profit']:.4f}")
        print(f"    Combined profit %: {stats['combined_profit']:.4f}")
        print(f"    LONG trades: {stats['long_trades']}")
        print(f"    SHORT trades: {stats['short_trades']}")
        print(f"    Average trade %: {stats['average_trade']:.4f}")
        print(f"    Winning trades: {stats['winning_trades']}")
        print(f"    Losing trades: {stats['losing_trades']}")
        print(f"    LONG emergency exits: {stats['long_emergency_exits']}")
        print(f"    SHORT emergency exits: {stats['short_emergency_exits']}")
        print(f"    Unfilled limits: {stats['unfilled_limits']}")
        print(f"    Forced exits: {stats['forced_exits']}")
        print(f"    Max winning trade: {stats['max_winning']:.4f}%")
        print(f"    Max losing trade: {stats['max_losing']:.4f}%")
        print(f"    Total cumulative profit %: {stats['combined_profit']:.4f}")

    total_long = sum(slot.long_profit_percent for slot in slot_results)
    total_short = sum(slot.short_profit_percent for slot in slot_results)
    total_combined = sum(slot.total_profit_percent for slot in slot_results)
    total_trades = sum(slot.total_trades for slot in slot_results)
    slot1_symbol = trading_selections[0].symbol if trading_selections else "-"
    slot2_symbol = trading_selections[1].symbol if len(trading_selections) > 1 else "-"

    print("\n=== COMBINED SUMMARY ===")
    print(f"Selected symbols: {slot1_symbol}, {slot2_symbol}")
    print(f"Review duration: {EXPECTED_REVIEW_MINUTES} minutes")
    print(f"LONG-only profit %: {total_long:.4f}")
    print(f"SHORT-only profit %: {total_short:.4f}")
    print(f"LONG+SHORT combined profit %: {total_combined:.4f}")
    print(f"Total trades: {total_trades}")
    print("Review status: VALID")
    print("\nResearch experiment only.")
    print("No account changes.")
    print("No Binance orders.")
    print("Historical simulation only.")
    print("==================================================")


def save_research_report(
    search_dt: datetime,
    review_end: datetime,
    selections: list[SlotSelection],
    slot_results: list[SlotResult],
) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    rows.append(
        {
            "review_date": REVIEW_DATE,
            "event_type": "REVIEW_VALID",
            "slot": "ALL",
            "symbol": "",
            "event_time": format_time(search_dt),
            "price": "",
            "profit_percent": "",
            "detail": (
                f"{format_review_window_detail(search_dt, review_end)}; "
                f"mode=LONG_SHORT_RESEARCH"
            ),
        }
    )

    rows.append(
        {
            "review_date": REVIEW_DATE,
            "event_type": "SEARCH",
            "slot": "ALL",
            "symbol": "",
            "event_time": format_time_utc(search_dt),
            "price": "",
            "profit_percent": "",
            "detail": "fixed research symbols; no new symbol search",
        }
    )

    for selection in selections:
        rows.append(
            {
                "review_date": REVIEW_DATE,
                "event_type": selection.slot_name.upper(),
                "slot": selection.slot_name,
                "symbol": selection.symbol,
                "event_time": format_time(search_dt),
                "price": "",
                "profit_percent": "",
                "detail": f"fixed_assignment={selection.symbol}",
            }
        )

    for slot_result in slot_results:
        stats = compute_research_slot_stats(slot_result)
        for event in slot_result.events:
            time_kst, time_utc = format_event_times(event.event_time)
            rows.append(
                {
                    "review_date": REVIEW_DATE,
                    "event_type": event.event_type,
                    "slot": event.slot,
                    "symbol": event.symbol,
                    "event_time": event.event_time,
                    "price": f"{event.price:.8f}",
                    "profit_percent": f"{event.profit_percent:.4f}",
                    "detail": (
                        f"time_kst={time_kst}; time_utc={time_utc}; "
                        f"reason={event.detail}; "
                        f"cumulative_profit={event.cumulative_profit_percent:.4f}"
                    ),
                }
            )

        rows.append(
            {
                "review_date": REVIEW_DATE,
                "event_type": "SLOT_SUMMARY",
                "slot": slot_result.slot_name,
                "symbol": slot_result.symbol,
                "event_time": format_time(review_end),
                "price": "",
                "profit_percent": f"{slot_result.total_profit_percent:.4f}",
                "detail": format_research_slot_summary_detail(stats),
            }
        )

    total_long = sum(slot.long_profit_percent for slot in slot_results)
    total_short = sum(slot.short_profit_percent for slot in slot_results)
    total_combined = sum(slot.total_profit_percent for slot in slot_results)
    total_trades = sum(slot.total_trades for slot in slot_results)
    slot1_symbol = slot_results[0].symbol if slot_results else ""
    slot2_symbol = slot_results[1].symbol if len(slot_results) > 1 else ""

    rows.append(
        {
            "review_date": REVIEW_DATE,
            "event_type": "SUMMARY",
            "slot": "ALL",
            "symbol": "",
            "event_time": format_time(review_end),
            "price": "",
            "profit_percent": f"{total_combined:.4f}",
            "detail": (
                f"{format_review_window_detail(search_dt, review_end)}; "
                f"slot1={slot1_symbol}; slot2={slot2_symbol}; "
                f"long_only_profit={total_long:.4f}; "
                f"short_only_profit={total_short:.4f}; "
                f"combined_profit={total_combined:.4f}; "
                f"total_trades={total_trades}; "
                f"review_duration_minutes={EXPECTED_REVIEW_MINUTES}"
            ),
        }
    )

    fieldnames = [
        "review_date",
        "event_type",
        "slot",
        "symbol",
        "event_time",
        "price",
        "profit_percent",
        "detail",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_trading_review(
    review_start: datetime,
    review_end: datetime,
    trading_selections: list[SlotSelection],
    slot_results: list[SlotResult],
) -> None:
    print("\n===== FIVE 5HA HIGH-CLOSE / LOW-CLOSE REVIEW =====")
    print_review_window_times(review_start, review_end)

    search_kst = format_time_kst(review_start)
    search_utc = format_time_utc(review_start)
    print(
        f"\nSEARCH | time_kst={search_kst} | time_utc={search_utc} | "
        f"reason=Five slot selection at review start"
    )

    for selection in trading_selections:
        print(
            f"{selection.slot_name.upper()} | symbol={selection.symbol} | "
            f"time_kst={search_kst} | time_utc={search_utc} | "
            f"slot_score={selection.slot_score:.8f} | "
            f"momentum={selection.momentum:.8f} | "
            f"6h_change={selection.previous_6h_price_change_percent:.4f}% | "
            f"1h_trading_value={selection.recent_1h_trading_value:,.2f}"
        )

    for slot_result in slot_results:
        print_slot_events(slot_result)
        stats = compute_slot_stats(slot_result)
        print(f"\n  {slot_result.slot_name} Summary:")
        print(f"    Total profit %: {stats['total_profit']:.4f}")
        print(f"    Trade count: {stats['trade_count']}")
        print(f"    Average trade %: {stats['average_trade']:.4f}")
        print(f"    Winning trades: {stats['winning_trades']}")
        print(f"    Losing trades: {stats['losing_trades']}")
        print(f"    Emergency exits: {stats['emergency_exits']}")
        print(f"    Unfilled limits: {stats['unfilled_limits']}")
        print(f"    Forced exits: {stats['forced_exits']}")
        print(f"    Maximum winning trade: {stats['max_winning']:.4f}%")
        print(f"    Maximum losing trade: {stats['max_losing']:.4f}%")

    total_profit = sum(slot.total_profit_percent for slot in slot_results)
    total_trades = sum(slot.total_trades for slot in slot_results)
    all_profits = [
        event.profit_percent
        for slot in slot_results
        for event in slot.events
        if event.event_type in PROFIT_EVENT_TYPES
    ]
    combined_average = sum(all_profits) / len(all_profits) if all_profits else 0.0
    slot1_symbol = trading_selections[0].symbol if trading_selections else "-"
    slot2_symbol = trading_selections[1].symbol if len(trading_selections) > 1 else "-"

    print("\n=== TOTAL ===")
    print(f"Selected Slot1: {slot1_symbol}")
    print(f"Selected Slot2: {slot2_symbol}")
    print(f"Total combined profit %: {total_profit:.4f}")
    print(f"Total trades: {total_trades}")
    print(f"Average trade %: {combined_average:.4f}")
    print(f"Review duration (minutes): {EXPECTED_REVIEW_MINUTES}")
    print("Review status: VALID")
    print("\nHistorical simulation only.")
    print("No account changes.")
    print("No Binance orders.")
    print("==================================================")


def main() -> None:
    api_key, secret_key = get_credentials()

    if not api_key:
        print("오류: BINANCE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    if not secret_key:
        print("오류: BINANCE_SECRET_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        return

    try:
        review_start, review_end = get_review_window_utc()

        if review_end <= review_start:
            print("오류: REVIEW_END_KST는 REVIEW_START_KST 이후여야 합니다.")
            return

        now_utc = datetime.now(timezone.utc)
        print(f"Current UTC time: {format_time_utc(now_utc)}")
        print(f"Current KST time: {format_time_kst(now_utc)}")

        if review_end > now_utc:
            print("REVIEW INVALID")
            print("Reason: future candles not available yet.")
            return

        print("Five LONG ONLY historical review를 시작합니다.")
        print_review_window_times(review_start, review_end)
        print("SHORT trading: DISABLED")

        eligible_symbols = get_eligible_symbols(api_key)
        if not eligible_symbols:
            print("오류: TRADING 상태의 USDT 무기한 선물 심볼을 찾을 수 없습니다.")
            return

        selections = run_slot_search(api_key, eligible_symbols, review_start)
        if len(selections) < SLOT_COUNT:
            print("오류: Slot1/Slot2를 선정하지 못했습니다.")
            return

        print_top10_table(selections)
        save_slot_score_review(review_start, selections)

        trading_selections = [s for s in selections if s.slot_name in {"Slot1", "Slot2"}]
        warmup_start = review_start - timedelta(hours=6)
        invalid_symbols: list[tuple[str, str]] = []
        prepared_candles: dict[str, list[dict]] = {}

        for selection in trading_selections:
            print(f"\n{selection.slot_name} ({selection.symbol}) 캔들 데이터 검증 중...")

            klines, candles, is_complete, reason = fetch_klines_for_review(
                api_key,
                selection.symbol,
                warmup_start,
                review_start,
                review_end,
            )

            print_kline_diagnostics(
                selection.symbol,
                klines,
                candles,
                review_start,
                review_end,
            )

            if not is_complete:
                invalid_symbols.append((selection.symbol, reason))
                continue

            calculate_heikin_ashi(candles)
            print(f"  Heikin Ashi candle count: {len(candles)}")
            prepared_candles[selection.slot_name] = candles

        if invalid_symbols:
            save_invalid_report(review_start, review_end, trading_selections, invalid_symbols)
            print_long_only_review(
                review_start, trading_selections, [], review_valid=False
            )
            print(f"\nDetailed report saved: {OUTPUT_CSV}")
            return

        slot_results: list[SlotResult] = []

        for selection in trading_selections:
            print(f"\n{selection.slot_name} ({selection.symbol}) LONG-only 시뮬레이션 중...")
            slot_results.append(
                simulate_slot(
                    selection.slot_name,
                    selection.symbol,
                    prepared_candles[selection.slot_name],
                    review_start,
                    review_end,
                )
            )
            print(f"{selection.slot_name} 시뮬레이션 완료")

        save_report(review_start, review_end, trading_selections, slot_results)
        save_trade_summary(slot_results, long_only=True)
        print_long_only_review(
            review_start, trading_selections, slot_results, review_valid=True
        )
        print(f"\nDetailed report saved: {OUTPUT_CSV}")
        print(f"Trade summary saved: {TRADE_SUMMARY_CSV}")

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
