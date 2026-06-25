"""SCOUT State Engine V1.4 — Alive Score from forward bar state (R008-compatible)."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from scout_research_r006_pilot_execution_engine import Bar
from scout_research_r008_exit_engine import StateSnap, state_snapshot


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
) -> AliveScore | None:
    if not bars:
        return None
    i = len(bars) - 1
    snap = state_snapshot(bars, i, entry_i)

    trend = 25.0 if snap.trend_alive else 0.0
    momentum = 25.0 if not snap.momentum_weak else 8.0
    volume = 25.0 if not snap.volume_weak else 8.0
    expansion = 15.0 if snap.expansion else 5.0
    accel_bonus = 10.0 if snap.acceleration else 0.0

    exhaustion = 0.0
    if not snap.trend_alive and snap.momentum_weak:
        exhaustion += 20.0
    if snap.volume_weak and snap.momentum_weak:
        exhaustion += 15.0
    if not snap.trend_alive and not snap.expansion:
        exhaustion += 10.0
    exhaustion = min(35.0, exhaustion)

    total = max(0.0, min(100.0, trend + momentum + volume + expansion + accel_bonus - exhaustion))
    trend_dead = not snap.trend_alive
    momentum_collapse = snap.momentum_weak
    volume_collapse = snap.volume_weak
    exhausted = exhaustion >= 20.0

    return AliveScore(
        alive_score=round(total, 2),
        trend_alive=round(trend, 2),
        momentum_alive=round(momentum, 2),
        volume_alive=round(volume, 2),
        expansion_alive=round(expansion, 2),
        exhaustion=round(exhaustion, 2),
        acceleration_bonus=round(accel_bonus, 2),
        hold_recommendation=hold_recommendation(total, hold_alive, exit_alive),
        trend_dead=trend_dead,
        momentum_collapse=momentum_collapse,
        volume_collapse=volume_collapse,
        exhausted=exhausted,
    )


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
