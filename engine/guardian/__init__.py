"""Guardian decision engine — rule-based hold / trail / exit from trade contract."""

from scout_auto_os.engine.guardian.decision_engine import GuardianDecision, decide
from scout_auto_os.engine.guardian.progress_engine import GuardianProgressResult, evaluate_progress
from scout_auto_os.engine.guardian.trade_thesis import (
    GuardianTradeThesis,
    build_thesis_from_predator_entry,
    build_thesis_from_replay_row,
)

__all__ = [
    "GuardianDecision",
    "GuardianProgressResult",
    "GuardianTradeThesis",
    "build_thesis_from_predator_entry",
    "build_thesis_from_replay_row",
    "decide",
    "evaluate_progress",
]
