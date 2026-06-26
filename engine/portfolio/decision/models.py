"""Portfolio Decision Engine V1 — data models."""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_LONG_SLOTS = 3
MAX_SHORT_SLOTS = 3

DECISION_KEEP = "KEEP"
DECISION_REPLACE = "REPLACE"
DECISION_IGNORE = "IGNORE"
DECISION_WAIT = "WAIT"


@dataclass
class PortfolioPosition:
    """Open slot holding — Guardian-enriched."""

    slot_id: str
    trade_id: str
    symbol: str
    side: str
    entry_time: str
    guardian_score: float
    guardian_state: str
    recommendation: str
    current_roi: float
    elapsed_minutes: int
    value_score: float
    expected_roi: float
    confidence: float
    portfolio_value_score: float = 0.0
    expected_horizon: int = 120
    actual_roi: float = 0.0

    def to_slot_dict(self) -> dict:
        return {
            "slot_id": self.slot_id,
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_time": self.entry_time,
            "guardian_score": self.guardian_score,
            "guardian_state": self.guardian_state,
            "recommendation": self.recommendation,
            "current_roi": self.current_roi,
            "portfolio_value_score": self.portfolio_value_score,
        }


@dataclass
class PredatorCandidate:
    """New Predator entry candidate for portfolio comparison."""

    trade_id: str
    symbol: str
    side: str
    timestamp: str
    value_score: float
    expected_roi: float
    expected_win_prob: float
    confidence: float
    gate_action: str
    gate_reason: str
    contract_id: str
    thesis_id: str = ""
    portfolio_value_score: float = 0.0
    actual_roi: float = 0.0

    @property
    def is_enter(self) -> bool:
        return self.gate_action.upper() == "ENTER"


@dataclass
class PortfolioDecisionRecord:
    timestamp: str
    slot: str
    side: str
    current_symbol: str
    candidate_symbol: str
    decision: str
    reason: str
    current_score: float = 0.0
    candidate_score: float = 0.0

    def to_row(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "slot": self.slot,
            "side": self.side,
            "current_symbol": self.current_symbol,
            "candidate_symbol": self.candidate_symbol,
            "decision": self.decision,
            "reason": self.reason,
            "current_score": round(self.current_score, 2) if self.current_score else "",
            "candidate_score": round(self.candidate_score, 2) if self.candidate_score else "",
        }


@dataclass
class PortfolioSlotBook:
    long_slots: list[PortfolioPosition] = field(default_factory=list)
    short_slots: list[PortfolioPosition] = field(default_factory=list)

    def slots_for(self, side: str) -> list[PortfolioPosition]:
        return self.long_slots if side == "long" else self.short_slots

    def set_slots(self, side: str, slots: list[PortfolioPosition]) -> None:
        if side == "long":
            self.long_slots = slots
        else:
            self.short_slots = slots

    def max_slots(self, side: str) -> int:
        return MAX_LONG_SLOTS if side == "long" else MAX_SHORT_SLOTS

    def slot_snapshot(self) -> dict:
        return {
            "long": [p.to_slot_dict() for p in self.long_slots],
            "short": [p.to_slot_dict() for p in self.short_slots],
            "long_used": len(self.long_slots),
            "short_used": len(self.short_slots),
        }
