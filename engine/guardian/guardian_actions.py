"""Guardian action vocabulary."""

from __future__ import annotations

from enum import Enum


class GuardianAction(str, Enum):
    NO_ACTION = "NO_ACTION"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    TRAIL = "TRAIL"
    EXIT = "EXIT"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"
