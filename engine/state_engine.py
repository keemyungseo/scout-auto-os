"""SCOUT State Engine V1.4 — Alive Score from forward bar state (R008-compatible)."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from scout_research_r006_pilot_execution_engine import Bar
from scout_research_r008_exit_engine import StateSnap, state_snapshot


@dataclass
class StateFormulaWeights:
    """Configurable Alive Formula — LIVE uses LIVE_V14; Research evolves variants."""
    name: str
    trend: float
    momentum: float
    volume: float
    expansion: float
    acceleration: float
    exhaustion_scale: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


# Frozen LIVE weights (V1.4) — Research must not auto-override this.
LIVE_STATE_FORMULA = StateFormulaWeights(
    "LIVE_V14", trend=25, momentum=25, volume=25, expansion=15, acceleration=10,
)


def _exhaustion_penalty(snap: StateSnap, scale: float = 1.0) -> float:
    ex = 0.0
    if not snap.trend_alive and snap.momentum_weak:
        ex += 20.0
    if snap.volume_weak and snap.momentum_weak:
        ex += 15.0
    if not snap.trend_alive and not snap.expansion:
        ex += 10.0
    return min(35.0, ex * scale)


def compute_alive_from_snap(
    snap: StateSnap,
    weights: StateFormulaWeights,
    hold_alive: float = 70.0,
    exit_alive: float = 45.0,
) -> AliveScore:
    trend = weights.trend if snap.trend_alive else 0.0
    momentum = weights.momentum if not snap.momentum_weak else weights.momentum * 0.32
    volume = weights.volume if not snap.volume_weak else weights.volume * 0.32
    expansion = weights.expansion if snap.expansion else weights.expansion * 0.33
    accel_bonus = weights.acceleration if snap.acceleration else 0.0
    exhaustion = _exhaustion_penalty(snap, weights.exhaustion_scale)

    total = max(0.0, min(100.0, trend + momentum + volume + expansion + accel_bonus - exhaustion))
    return AliveScore(
        alive_score=round(total, 2),
        trend_alive=round(trend, 2),
        momentum_alive=round(momentum, 2),
        volume_alive=round(volume, 2),
        expansion_alive=round(expansion, 2),
        exhaustion=round(exhaustion, 2),
        acceleration_bonus=round(accel_bonus, 2),
        hold_recommendation=hold_recommendation(total, hold_alive, exit_alive),
        trend_dead=not snap.trend_alive,
        momentum_collapse=snap.momentum_weak,
        volume_collapse=snap.volume_weak,
        exhausted=exhaustion >= 20.0,
    )


@dataclass
class AliveScore:
    alive_score: float
    trend_alive: float
    momentum_alive: float
    volume_alive: float
    expansion_alive: float
    exhaustion: float
    acceleration_bonus: float
    hold_recommendation: str
    trend_dead: bool
    momentum_collapse: bool
    volume_collapse: bool
    exhausted: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AliveScore":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


def hold_recommendation(score: float, hold_alive: float, exit_alive: float) -> str:
    if score >= hold_alive:
        return "HOLD"
    if score >= exit_alive:
        return "REVIEW"
    return "EXIT"


def compute_alive_score(
    bars: list[Bar],
    entry_i: int = 0,
    hold_alive: float = 70.0,
    exit_alive: float = 45.0,
    weights: StateFormulaWeights | None = None,
) -> AliveScore | None:
    if not bars:
        return None
    i = len(bars) - 1
    snap = state_snapshot(bars, i, entry_i)
    w = weights or LIVE_STATE_FORMULA
    return compute_alive_from_snap(snap, w, hold_alive, exit_alive)


def snap_summary(snap: StateSnap) -> dict:
    return {
        "trend_alive": snap.trend_alive,
        "acceleration": snap.acceleration,
        "expansion": snap.expansion,
        "volume_weak": snap.volume_weak,
        "momentum_weak": snap.momentum_weak,
        "ret_1h_proxy": round(snap.ret_1h_proxy, 4),
        "vol_ratio": round(snap.vol_ratio, 4),
    }
