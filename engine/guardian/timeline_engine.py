"""Evaluate Guardian progress at each timeline point — no rule changes."""

from __future__ import annotations

from dataclasses import dataclass, field

from scout_auto_os.engine.guardian.decision_engine import contract_from_replay_row
from scout_auto_os.engine.guardian.progress_engine import evaluate_progress
from scout_auto_os.engine.guardian.trade_thesis import GuardianTradeThesis, build_thesis_from_replay_row

TIMELINE_FIELDS = (
    "trade_id",
    "timestamp",
    "elapsed_minutes",
    "current_roi",
    "progress_ratio",
    "guardian_score",
    "guardian_state",
    "recommendation",
    "reason",
)


@dataclass
class StateTransition:
    trade_id: str
    from_state: str
    to_state: str
    elapsed_minutes: int
    timestamp: str
    current_roi: float
    progress_ratio: float

    def to_row(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "elapsed_minutes": self.elapsed_minutes,
            "timestamp": self.timestamp,
            "current_roi": round(self.current_roi, 4),
            "progress_ratio": round(self.progress_ratio, 4),
        }


@dataclass
class TradeTimeline:
    trade_id: str
    symbol: str
    side: str
    points: list[dict] = field(default_factory=list)
    transitions: list[StateTransition] = field(default_factory=list)
    recommendation_changes: int = 0

    @property
    def transition_count(self) -> int:
        return len(self.transitions)


def evaluate_trade_timeline(
    trade_row: dict,
    snapshots: list[dict],
    *,
    thesis: GuardianTradeThesis | None = None,
    config: dict | None = None,
) -> TradeTimeline:
    trade_id = trade_row.get("trade_key", "")
    thesis = thesis or build_thesis_from_replay_row(trade_row)
    contract = contract_from_replay_row(trade_row)
    contract["expected_horizon"] = thesis.expected_horizon
    contract["contract_id"] = thesis.contract_id

    timeline = TradeTimeline(
        trade_id=trade_id,
        symbol=trade_row.get("symbol", ""),
        side=trade_row.get("direction", trade_row.get("side", "long")),
    )

    prev_state: str | None = None
    prev_rec: str | None = None

    for snap in snapshots:
        progress = evaluate_progress(
            contract,
            snap,
            contract_id=trade_id,
            config=config,
        )
        point = {
            "trade_id": trade_id,
            "timestamp": snap["timestamp"],
            "elapsed_minutes": snap["elapsed_minutes"],
            "current_roi": round(snap["current_roi"], 4),
            "progress_ratio": round(progress.progress_ratio, 4),
            "guardian_score": round(progress.guardian_score, 2),
            "guardian_state": progress.guardian_state,
            "recommendation": progress.recommendation,
            "reason": progress.reason,
        }
        timeline.points.append(point)

        if prev_state is not None and progress.guardian_state != prev_state:
            timeline.transitions.append(StateTransition(
                trade_id=trade_id,
                from_state=prev_state,
                to_state=progress.guardian_state,
                elapsed_minutes=snap["elapsed_minutes"],
                timestamp=snap["timestamp"],
                current_roi=snap["current_roi"],
                progress_ratio=progress.progress_ratio,
            ))
        if prev_rec is not None and progress.recommendation != prev_rec:
            timeline.recommendation_changes += 1

        prev_state = progress.guardian_state
        prev_rec = progress.recommendation

    return timeline
