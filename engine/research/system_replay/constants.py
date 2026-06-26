"""System replay constants — A/B/C/D stacks (no new engines)."""

from __future__ import annotations

REPLAY_SEED = 42
REPLAY_DAYS = 15
MAX_LONG_SLOTS = 3
MAX_SHORT_SLOTS = 3

SYSTEMS: dict[str, dict] = {
    "A": {
        "label": "Search Only (hold 2h)",
        "exit_mode": "hold_2h",
        "modules": ("search",),
    },
    "B": {
        "label": "Search + Exit",
        "exit_mode": "state_exit",
        "modules": ("search", "exit"),
    },
    "C": {
        "label": "Search + Position Evaluation + Exit",
        "exit_mode": "pe_proxy",
        "modules": ("search", "position_evaluation", "exit"),
    },
    "D": {
        "label": "Full Runtime",
        "exit_mode": "full_exit",
        "modules": ("search", "thesis", "expectation", "position_evaluation", "exit", "portfolio"),
    },
}

BARS_PER_SCAN = 8  # 15m bars in 2h scan interval
