"""State League — blind replay ranking of Alive Formula variants."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from scout_research_r006_pilot_execution_engine import Bar, load_forward_bars
from scout_research_r008_exit_engine import state_snapshot

from scout_auto_os.engine.state_engine import StateFormulaWeights, compute_alive_from_snap
from scout_auto_os.engine.research.state_formula_generator import generate_state_formulas
from scout_auto_os.engine.research.state_promotion import promotion_tier

HOLD_ALIVE = 70.0
EXIT_ALIVE = 45.0
MIN_HOLD_BARS = 6  # 30m
BAR_MINUTES = 5
CHECKPOINTS = (6, 12, 18, 24)  # 30m 60m 90m 120m


@dataclass
class ReplayOutcome:
    return_pct: float
    bars_held: int
    mfe_pct: float
    mae_pct: float
    alive_at_2h: float
    alive_at_4h: float
    exited_by_state: bool


def _mfe_mae(bars: list[Bar], entry_i: int, entry_px: float, end_i: int) -> tuple[float, float]:
    max_h = max(b.h for b in bars[entry_i : end_i + 1])
    min_l = min(b.l for b in bars[entry_i : end_i + 1])
    mfe = (max_h - entry_px) / entry_px * 100 if entry_px else 0
    mae = (min_l - entry_px) / entry_px * 100 if entry_px else 0
    return mfe, mae


def replay_formula(
    bars: list[Bar],
    weights: StateFormulaWeights,
    hold_target_bars: int = 24,
) -> ReplayOutcome | None:
    if len(bars) < MIN_HOLD_BARS + 2:
        return None
    entry_i = 0
    entry_px = bars[0].o if bars[0].o > 0 else bars[0].c
    if entry_px <= 0:
        return None

    exit_i = len(bars) - 1
    exited = False
    for j in range(MIN_HOLD_BARS, len(bars)):
        snap = state_snapshot(bars, j, entry_i)
        alive = compute_alive_from_snap(snap, weights, HOLD_ALIVE, EXIT_ALIVE)
        hold_min_bars = hold_target_bars
        if j >= hold_min_bars and alive.alive_score < EXIT_ALIVE:
            exit_i = j
            exited = True
            break
        if alive.trend_dead and alive.momentum_collapse and j >= MIN_HOLD_BARS * 2:
            exit_i = j
            exited = True
            break

    ret = (bars[exit_i].c - entry_px) / entry_px * 100
    mfe, mae = _mfe_mae(bars, entry_i, entry_px, exit_i)

    def _alive_at(bar_i: int) -> float:
        if bar_i >= len(bars):
            bar_i = len(bars) - 1
        snap = state_snapshot(bars, bar_i, entry_i)
        return compute_alive_from_snap(snap, weights).alive_score

    a2 = _alive_at(min(24, len(bars) - 1))
    a4 = _alive_at(min(48, len(bars) - 1))

    return ReplayOutcome(
        return_pct=round(ret, 4),
        bars_held=exit_i - entry_i,
        mfe_pct=round(mfe, 4),
        mae_pct=round(mae, 4),
        alive_at_2h=a2,
        alive_at_4h=a4,
        exited_by_state=exited,
    )


def _profit_factor(returns: list[float]) -> float:
    wins = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    if losses <= 0:
        return round(wins, 2) if wins else 0.0
    return round(wins / losses, 2)


def compute_state_league(
    forward_rows: list[dict],
    cache_dir=None,
    max_formulas: int = 64,
    blind_holdout_ratio: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (full_league, blind_league).
    Blind holdout = last 20% of scans by time — formulas ranked on train only.
    """
    complete = [r for r in forward_rows if r.get("return_2h") not in ("", None)]
    if not complete:
        return [], []

    scans = sorted(set(r["scan_time_kst"] for r in complete))
    split = max(1, int(len(scans) * (1 - blind_holdout_ratio)))
    train_scans = set(scans[:split])
    train_rows = [r for r in complete if r["scan_time_kst"] in train_scans]

    formulas = generate_state_formulas(max_formulas)
    league_rows: list[dict] = []

    for fw in formulas:
        outcomes: list[ReplayOutcome] = []
        for row in train_rows:
            sym = row["symbol"]
            scan = row["scan_time_kst"]
            try:
                bars = load_forward_bars(sym, scan)
            except Exception:
                continue
            if not bars or len(bars) < 10:
                continue
            out = replay_formula(bars, fw)
            if out:
                outcomes.append(out)

        if len(outcomes) < 3:
            league_rows.append(_empty_row(fw.name))
            continue

        rets = [o.return_pct for o in outcomes]
        wins = [r for r in rets if r >= 3.0]
        losses = [r for r in rets if r < 0]
        hold_2h = sum(1 for o in outcomes if o.alive_at_2h >= HOLD_ALIVE) / len(outcomes) * 100
        hold_4h = sum(1 for o in outcomes if o.alive_at_4h >= HOLD_ALIVE) / len(outcomes) * 100
        avg_hold_min = statistics.mean(o.bars_held for o in outcomes) * BAR_MINUTES

        league_rows.append({
            "formula_name": fw.name,
            "trend_w": fw.trend,
            "momentum_w": fw.momentum,
            "volume_w": fw.volume,
            "expansion_w": fw.expansion,
            "acceleration_w": fw.acceleration,
            "sample_count": len(outcomes),
            "win_rate": round(len(wins) / len(rets) * 100, 2),
            "avg_return": round(statistics.mean(rets), 4),
            "avg_loss": round(statistics.mean(losses), 4) if losses else 0.0,
            "profit_factor": _profit_factor(rets),
            "avg_hold_minutes": round(avg_hold_min, 1),
            "avg_mfe": round(statistics.mean(o.mfe_pct for o in outcomes), 4),
            "avg_mae": round(statistics.mean(o.mae_pct for o in outcomes), 4),
            "max_drawdown_avg": round(statistics.mean(o.mae_pct for o in outcomes), 4),
            "trend_hold_2h_pct": round(hold_2h, 2),
            "trend_hold_4h_pct": round(hold_4h, 2),
            "state_exit_rate": round(sum(1 for o in outcomes if o.exited_by_state) / len(outcomes) * 100, 2),
            "league_score": round(
                len(wins) / len(rets) * 40
                + statistics.mean(rets) * 5
                + _profit_factor(rets) * 10
                + hold_2h * 0.2,
                2,
            ),
            "tier": "hypothesis",
        })

    league_rows.sort(key=lambda x: x.get("league_score", 0), reverse=True)
    for rank, row in enumerate(league_rows, 1):
        row["league_rank"] = rank
        row["tier"] = promotion_tier(row, rank, blind_pass=False)

    blind_rows = _blind_validate(formulas, complete, train_scans)
    apply_blind_tiers(league_rows, blind_rows)
    return league_rows, blind_rows


def _empty_row(name: str) -> dict:
    return {"formula_name": name, "sample_count": 0, "league_score": 0, "tier": "insufficient_sample"}


def _blind_pass(blind_rows: list[dict], formula_name: str) -> bool:
    for b in blind_rows:
        if b.get("formula_name") == formula_name:
            n = int(b.get("blind_sample_count") or 0)
            wr = float(b.get("blind_win_rate") or 0)
            return n >= 10 and wr >= 40
    return False


def apply_blind_tiers(league_rows: list[dict], blind_rows: list[dict]) -> None:
    for row in league_rows:
        rank = int(row.get("league_rank") or 99)
        row["tier"] = promotion_tier(row, rank, blind_pass=_blind_pass(blind_rows, row.get("formula_name", "")))


def _blind_validate(formulas, all_rows, train_scans) -> list[dict]:
    """Evaluate top formulas on holdout scans only."""
    holdout = [r for r in all_rows if r["scan_time_kst"] not in train_scans]
    if not holdout:
        return []
    top = formulas[:5]
    out: list[dict] = []
    for fw in top:
        rets: list[float] = []
        for row in holdout:
            try:
                bars = load_forward_bars(row["symbol"], row["scan_time_kst"])
            except Exception:
                continue
            if not bars:
                continue
            ro = replay_formula(bars, fw)
            if ro:
                rets.append(ro.return_pct)
        out.append({
            "formula_name": fw.name,
            "blind_sample_count": len(rets),
            "blind_win_rate": round(sum(1 for r in rets if r >= 3) / len(rets) * 100, 2) if rets else 0,
            "blind_avg_return": round(statistics.mean(rets), 4) if rets else 0,
        })
    return out
