"""LIVE-facing pattern labels mapped from scan-time direction patterns."""

from __future__ import annotations

from scout_auto_os.engine.research.directional.patterns import label_direction_pattern

# Internal pattern -> LIVE research label
PATTERN_TO_LIVE: dict[str, str] = {
    "UP_CONTINUATION": "LONG_CONTINUATION",
    "DOWN_UP": "LONG_REVERSAL",
    "DOWN_BASE_UP": "LONG_REVERSAL",
    "UP_ACCELERATION": "LONG_BREAKOUT",
    "DOWN_CONTINUATION": "SHORT_CONTINUATION",
    "UP_DOWN": "SHORT_REVERSAL",
    "UP_BASE_DOWN": "SHORT_REVERSAL",
    "DOWN_ACCELERATION": "SHORT_BREAKDOWN",
}

LONG_LIVE_PATTERNS = ("LONG_CONTINUATION", "LONG_REVERSAL", "LONG_BREAKOUT")
SHORT_LIVE_PATTERNS = ("SHORT_CONTINUATION", "SHORT_REVERSAL", "SHORT_BREAKDOWN")


def live_pattern(features: dict) -> str:
    raw = label_direction_pattern(features)
    return PATTERN_TO_LIVE.get(raw, "UNLABELED")


def attach_live_pattern(signals: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in signals:
        lp = live_pattern(s["features"])
        out.append({**s, "live_pattern": lp, "raw_pattern": label_direction_pattern(s["features"])})
    return out
