"""Per-trade outcome facts from timeline — no scoring."""

from __future__ import annotations

from dataclasses import dataclass

from scout_auto_os.engine.guardian.trade_thesis import GuardianTradeThesis

EXIT_ACTIONS = frozenset({"EXIT", "EMERGENCY_EXIT"})
TRAIL_ACTIONS = frozenset({"TRAIL"})
REDUCE_ACTIONS = frozenset({"REDUCE"})
HOLD_ACTIONS = frozenset({"HOLD", "NO_ACTION"})


@dataclass
class TradeOutcomeFacts:
    trade_id: str
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    hold_minutes: int
    final_roi: float
    peak_roi: float
    max_drawdown: float
    hold_count: int
    trail_start_minutes: int | None
    reduce_count: int
    exit_minutes: int | None
    emergency: bool
    final_recommendation: str
    final_state: str
    avg_guardian_score: float
    expected_roi: float
    expected_peak_roi: float
    expected_drawdown: float
    expected_horizon: int
    entry_reason: str
    predicted_dna: str
    confidence: float


def _parse_entry_time(trade_id: str) -> str:
    if "|" in trade_id:
        return trade_id.split("|")[0].strip()
    return ""


def _parse_side(trade_id: str, thesis: GuardianTradeThesis | None) -> str:
    if thesis:
        return thesis.side
    if "|" in trade_id:
        return trade_id.split("|")[-1].lower()
    return "long"


def extract_trade_facts(
    trade_id: str,
    points: list[dict],
    thesis: GuardianTradeThesis | None,
) -> TradeOutcomeFacts | None:
    if not points:
        return None

    running_peak = float("-inf")
    max_dd = 0.0
    hold_count = 0
    reduce_count = 0
    trail_start: int | None = None
    exit_min: int | None = None
    emergency = False
    scores: list[float] = []

    for p in points:
        roi = float(p.get("current_roi", 0))
        running_peak = max(running_peak, roi)
        max_dd = max(max_dd, running_peak - roi)
        elapsed = int(float(p.get("elapsed_minutes", 0)))
        rec = p.get("recommendation", "")
        if rec in HOLD_ACTIONS:
            hold_count += 1
        if rec in REDUCE_ACTIONS:
            reduce_count += 1
        if rec in TRAIL_ACTIONS and trail_start is None:
            trail_start = elapsed
        if rec in EXIT_ACTIONS and exit_min is None:
            exit_min = elapsed
        if rec == "EMERGENCY_EXIT":
            emergency = True
        try:
            scores.append(float(p.get("guardian_score", 0)))
        except (TypeError, ValueError):
            pass

    last = points[-1]
    symbol = thesis.symbol if thesis else (trade_id.split("|")[1] if "|" in trade_id else "")
    entry_time = _parse_entry_time(trade_id)
    peak_roi = running_peak if running_peak != float("-inf") else float(last.get("current_roi", 0))

    return TradeOutcomeFacts(
        trade_id=trade_id,
        symbol=symbol,
        side=_parse_side(trade_id, thesis),
        entry_time=entry_time,
        exit_time=last.get("timestamp", ""),
        hold_minutes=int(float(last.get("elapsed_minutes", 0))),
        final_roi=float(last.get("current_roi", 0)),
        peak_roi=round(peak_roi, 4),
        max_drawdown=round(max_dd, 4),
        hold_count=hold_count,
        trail_start_minutes=trail_start,
        reduce_count=reduce_count,
        exit_minutes=exit_min,
        emergency=emergency,
        final_recommendation=last.get("recommendation", ""),
        final_state=last.get("guardian_state", ""),
        avg_guardian_score=round(sum(scores) / len(scores), 2) if scores else 0.0,
        expected_roi=float(thesis.expected_roi) if thesis else 0.0,
        expected_peak_roi=float(thesis.expected_peak_roi) if thesis else 0.0,
        expected_drawdown=float(thesis.expected_drawdown) if thesis else 1.0,
        expected_horizon=int(thesis.expected_horizon) if thesis else 120,
        entry_reason=thesis.entry_reason if thesis else "",
        predicted_dna=thesis.predicted_dna if thesis else "",
        confidence=float(thesis.confidence) if thesis else 0.0,
    )
