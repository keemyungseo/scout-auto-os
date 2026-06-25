"""Map pattern labels to research engine names."""

from __future__ import annotations

PATTERN_TO_ENGINE: dict[str, tuple[str, str]] = {
    "DOWN_BASE_UP": ("LONG_BASE_REVERSAL", "long"),
    "DOWN_UP": ("LONG_V_REVERSAL", "long"),
    "UP_CONTINUATION": ("LONG_CONTINUATION", "long"),
    "UP_ACCELERATION": ("LONG_ACCELERATION", "long"),
    "UP_BASE_DOWN": ("SHORT_TOP_REVERSAL", "short"),
    "UP_DOWN": ("SHORT_V_REVERSAL", "short"),
    "DOWN_CONTINUATION": ("SHORT_CONTINUATION", "short"),
    "DOWN_ACCELERATION": ("SHORT_ACCELERATION", "short"),
}

RESEARCH_ENGINES = tuple(v[0] for v in PATTERN_TO_ENGINE.values())

ENGINE_TO_PATTERN = {v[0]: k for k, v in PATTERN_TO_ENGINE.items()}
