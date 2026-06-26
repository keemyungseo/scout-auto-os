"""Guardian Decision Engine V1 — contract vs position → single action."""

from __future__ import annotations

from dataclasses import dataclass

from scout_auto_os.engine.guardian.decision_rules import (
    compute_metrics,
    evaluate_rules,
    expected_horizon_minutes,
)
from scout_auto_os.engine.predator.trade_contract import build_trade_contract


@dataclass
class GuardianDecision:
    symbol: str
    action: str
    reason: str
    progress_ratio: float
    time_progress: float
    drawdown_pressure: float
    contract_id: str
    current_roi: float = 0.0
    elapsed_minutes: int = 0
    peak_roi: float = 0.0
    drawdown_from_peak: float = 0.0
    overperformance: bool = False
    expected_horizon: int = 0

    def to_row(self) -> dict:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "reason": self.reason,
            "progress_ratio": round(self.progress_ratio, 4),
            "time_progress": round(self.time_progress, 4),
            "drawdown_pressure": round(self.drawdown_pressure, 4),
            "contract_id": self.contract_id,
            "current_roi": round(self.current_roi, 4),
            "elapsed_minutes": self.elapsed_minutes,
            "peak_roi": round(self.peak_roi, 4),
            "drawdown_from_peak": round(self.drawdown_from_peak, 4),
            "overperformance": int(self.overperformance),
            "expected_horizon": self.expected_horizon,
        }


def contract_from_replay_row(row: dict) -> dict:
    """Build trade contract from Predator replay bundle row."""
    return build_trade_contract({
        "symbol": row.get("symbol", ""),
        "side": row.get("side", row.get("direction", "long")),
        "predicted_roi": row.get("predicted_roi", 0),
        "predicted_peak_roi": row.get("predicted_peak_roi", 0),
        "predicted_drawdown": row.get("predicted_drawdown", 0),
        "predicted_win_prob": row.get("predicted_win_prob", 0),
        "value_score": row.get("value_score", 0),
        "recommended_size": row.get("recommended_size", 0.2),
        "predicted_dna_type": row.get("predicted_dna_type", ""),
        "gate_action": row.get("gate_action", "ENTER"),
        "gate_reason": row.get("gate_reason", ""),
    })


def position_from_replay_outcome(
    row: dict,
    *,
    elapsed_minutes: int = 240,
) -> dict:
    """Snapshot position from labeled replay outcome."""
    peak = float(row.get("actual_peak_roi", row.get("predicted_peak_roi", 0)))
    current = float(row.get("actual_roi", row.get("predicted_roi", 0)))
    dd = float(row.get("actual_drawdown", 0))
    dd_from_peak = max(0.0, peak - current)
    if dd > 0:
        dd_from_peak = max(dd_from_peak, dd)
    return {
        "current_roi": current,
        "elapsed_minutes": elapsed_minutes,
        "peak_roi": peak,
        "drawdown_from_peak": dd_from_peak,
    }


def decide(
    contract: dict,
    position: dict,
    *,
    contract_id: str = "",
    manual: bool = False,
) -> GuardianDecision:
    """Compare contract expectations to live position — one explainable action."""
    pos = dict(position)
    if "expected_horizon" not in pos:
        pos["expected_horizon"] = expected_horizon_minutes(contract)

    metrics = compute_metrics(contract, pos)
    action, reason_lines = evaluate_rules(metrics, contract, manual=manual)
    symbol = contract.get("symbol", "")

    return GuardianDecision(
        symbol=symbol,
        action=action,
        reason=" | ".join(reason_lines),
        progress_ratio=metrics.progress_ratio,
        time_progress=metrics.time_progress,
        drawdown_pressure=metrics.drawdown_pressure,
        contract_id=contract_id or f"{symbol}|{contract.get('side', 'long')}",
        current_roi=metrics.current_roi,
        elapsed_minutes=metrics.elapsed_minutes,
        peak_roi=metrics.peak_roi,
        drawdown_from_peak=metrics.drawdown_from_peak,
        overperformance=metrics.overperformance,
        expected_horizon=metrics.expected_horizon,
    )
