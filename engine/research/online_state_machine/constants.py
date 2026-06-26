"""Online state machine constants."""

from __future__ import annotations

BAR_MINUTES = 15  # forward bundle resolution; target cadence was 5m

STATES = (
    "EARLY_BREAKOUT",
    "HEALTHY_TREND",
    "WEAK_TREND",
    "PULLBACK",
    "ACCELERATION",
    "EXHAUSTION",
    "FAKE_BREAKOUT",
    "REVERSAL",
    "DEAD",
    "EXIT",
)

TERMINAL_STATES = frozenset({"DEAD", "FAKE_BREAKOUT", "REVERSAL", "EXIT"})

MOMENTUM_LOOKBACK = 4
SLOPE_LOOKBACK = 4
