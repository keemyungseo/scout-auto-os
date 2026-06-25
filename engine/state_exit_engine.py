"""State-based exit engine V1.4 — replaces ROI/EFR primary exit path."""

from __future__ import annotations

import os
from dataclasses import dataclass

from scout_research_r006_pilot_execution_engine import Bar

from scout_auto_os.engine.state_engine import AliveScore


@dataclass
class StateExitDecision:
    should_exit: bool
    reason: str = ""
    review_reason: str = ""


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


class StateExitEngine:
    def __init__(self, config: dict) -> None:
        sc = config.get("state_engine", {})
        self.min_hold = _int_env("STATE_MIN_HOLD_MINUTES", int(sc.get("min_hold_before_exit_min", 30)))
        self.hold_target = _int_env("STATE_HOLD_TARGET_MINUTES", int(sc.get("hold_target_minutes", 120)))
        self.hold_alive = _float_env("STATE_HOLD_ALIVE_SCORE", float(sc.get("hold_alive_score", 70)))
        self.exit_alive = _float_env("STATE_EXIT_ALIVE_SCORE", float(sc.get("exit_alive_score", 45)))
        self.delta_exit = _float_env("STATE_ALIVE_DELTA_EXIT", float(sc.get("alive_delta_exit", -25)))
        self.protective_sl_pct = _float_env("STATE_PROTECTIVE_SL_PCT", float(sc.get("protective_sl_pct", 10)))

    def evaluate(
        self,
        bars: list[Bar],
        entry_px: float,
        entry_alive: AliveScore,
        current_alive: AliveScore,
        hold_minutes: int,
    ) -> StateExitDecision:
        if not bars or entry_px <= 0:
            return StateExitDecision(False)

        # Hard protective SL (safety floor, not primary exit logic)
        sl_px = entry_px * (1 - self.protective_sl_pct / 100)
        if bars[-1].l <= sl_px:
            return StateExitDecision(True, "protective_sl", "price_hit_stop")

        if hold_minutes < self.min_hold:
            return StateExitDecision(
                False,
                review_reason=f"min_hold_{self.min_hold}m",
            )

        delta = current_alive.alive_score - entry_alive.alive_score

        # High alive score → hold beyond 2h target
        if current_alive.alive_score >= self.hold_alive:
            return StateExitDecision(
                False,
                review_reason="alive_score_strong_hold",
            )

        if delta <= self.delta_exit:
            return StateExitDecision(True, "state_momentum_collapse", f"alive_delta={delta:.1f}")

        if current_alive.trend_dead and entry_alive.trend_alive > 0:
            return StateExitDecision(True, "trend_dead", "trend_lost_since_entry")

        if current_alive.momentum_collapse and current_alive.volume_collapse:
            return StateExitDecision(True, "volume_momentum_collapse", "momentum_and_volume_weak")

        if current_alive.exhausted:
            return StateExitDecision(True, "exhaustion", f"exhaustion={current_alive.exhaustion:.0f}")

        if hold_minutes >= self.hold_target and current_alive.alive_score < self.exit_alive:
            return StateExitDecision(
                True,
                "state_score_low_after_target",
                f"hold>{self.hold_target}m score={current_alive.alive_score:.0f}",
            )

        if current_alive.momentum_collapse and hold_minutes >= self.hold_target // 2:
            return StateExitDecision(True, "momentum_collapse", "momentum_weak_mid_hold")

        return StateExitDecision(False, review_reason=current_alive.hold_recommendation.lower())
