"""Research formula league — parallel ranking formulas on same candidate pool."""

from __future__ import annotations

import statistics
from typing import Callable

import scout_phase20_winner_state_ranking as p20
import scout_phase22_search_formula_evolution as p22
import scout_phase23_search_formula_league as p23

FORMULA_NAMES = (
    "A6_CURRENT",
    "A6_NO_EV",
    "MOMENTUM_ONLY",
    "VOLUME_BREAKOUT",
    "COMPRESSION_BREAKOUT",
    "TREND_ALIVE_ONLY",
    "ACCELERATION_ONLY",
    "RANGE_EXPANSION",
)

TREND_ALIVE = {"TrendAlive", "Acceleration", "Expansion", "ExpansionStart", "VolumeSupport"}
ACCEL_STATES = {"Acceleration", "ExpansionStart"}


def _g(f: dict, key: str, default: float = 0.0) -> float:
    return float(f.get(key, default))


def _score_momentum(row: dict, _peers: list[dict], base: float, _th, _stats) -> float:
    f = row["features"]
    return _g(f, "5m_momentum") + _g(f, "15m_current_return_pct") * 0.5 + base * 0.1


def _score_volume_breakout(row: dict, peers: list[dict], base: float, _th, _stats) -> float:
    f = row["features"]
    vol = p22.within_scan_pct(row, peers, "15m_current_volume_ratio")
    energy = _g(f, "5m_seq_volume_energy_6")
    return base * 0.2 + vol * 3.0 + min(energy, 5.0) * 0.3


def _score_compression_breakout(row: dict, peers: list[dict], base: float, _th, stats) -> float:
    f = row["features"]
    comp = _g(f, "5m_compression")
    release = _g(f, "5m_release")
    b4 = p22.bonus_a4_raw(row, stats)
    return base * 0.1 + b4 * 2.0 + release * 1.5 + (1.0 / (1.0 + comp)) * 2.0


def _score_trend_alive(row: dict, _peers: list[dict], base: float, _th, _stats) -> float:
    st = row.get("states", {})
    alive = sum(1 for tf in ("1h", "2h") if st.get(tf) in TREND_ALIVE)
    return base + alive * 2.0


def _score_acceleration(row: dict, _peers: list[dict], base: float, _th, _stats) -> float:
    st = row.get("states", {})
    accel = sum(1 for tf in ("5m", "15m", "1h") if st.get(tf) in ACCEL_STATES)
    f = row["features"]
    return base * 0.2 + accel * 2.5 + _g(f, "5m_seq_return_sum_6") * 0.2


def _score_range_expansion(row: dict, peers: list[dict], base: float, _th, stats) -> float:
    b2 = p22.within_scan_pct(row, peers, "1h_current_range_pct")
    b5 = p22.bonus_a5_raw(row, peers, stats)
    return base * 0.1 + b2 * 3.0 + b5 * 2.0


def _score_a6(row: dict, peers: list[dict], base: float, th, stats) -> float:
    return p23.formula_scores_a6(row, peers, base, th, stats)["A6"]


FORMULA_SCORERS: dict[str, Callable] = {
    "A6_CURRENT": _score_a6,
    "A6_NO_EV": _score_a6,
    "MOMENTUM_ONLY": _score_momentum,
    "VOLUME_BREAKOUT": _score_volume_breakout,
    "COMPRESSION_BREAKOUT": _score_compression_breakout,
    "TREND_ALIVE_ONLY": _score_trend_alive,
    "ACCELERATION_ONLY": _score_acceleration,
    "RANGE_EXPANSION": _score_range_expansion,
}


def rank_by_formula(rows: list[dict], profile: dict, th, stats: dict, top_n: int = 5) -> dict[str, list[str]]:
    scored_rows: list[dict] = []
    for r in rows:
        peers = rows
        base = p20.state_match_score(r["states"], r["transitions"], profile)
        scored_rows.append({**r, "base_score": base})

    picks: dict[str, list[str]] = {}
    for name in FORMULA_NAMES:
        scorer = FORMULA_SCORERS[name]
        ranked = sorted(
            scored_rows,
            key=lambda x: scorer(x, rows, x["base_score"], th, stats),
            reverse=True,
        )
        picks[name] = [x["symbol"] for x in ranked[:top_n]]
    return picks


def _fval(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def _bool(row: dict, key: str) -> bool:
    v = str(row.get(key, "")).lower()
    return v in ("1", "true", "yes")


def compute_formula_league(
    forward_rows: list[dict],
    picks_history: list[dict],
) -> list[dict]:
    """Aggregate forward outcomes per formula TOP5 picks."""
    fwd_by_key = {(r["scan_time_kst"], r["symbol"]): r for r in forward_rows}
    by_formula: dict[str, list[dict]] = {n: [] for n in FORMULA_NAMES}

    for entry in picks_history:
        scan = entry["scan_time_kst"]
        for formula, symbols in entry.get("picks", {}).items():
            if formula not in by_formula:
                continue
            for sym in symbols:
                row = fwd_by_key.get((scan, sym))
                if row and row.get("return_2h") not in ("", None):
                    by_formula[formula].append(row)

    out: list[dict] = []
    for name in FORMULA_NAMES:
        samples = by_formula[name]
        if not samples:
            out.append({
                "formula_name": name,
                "sample_count": 0,
                "win_rate_30m": 0,
                "win_rate_1h": 0,
                "win_rate_2h": 0,
                "avg_return_2h": 0,
                "median_return_2h": 0,
                "max_drawdown_avg": 0,
                "big_winner_capture_rate": 0,
                "trap_rate": 0,
                "score": 0,
            })
            continue
        r30 = [_fval(s, "return_30m") for s in samples]
        r1h = [_fval(s, "return_1h") for s in samples]
        r2h = [_fval(s, "return_2h") for s in samples]
        mdd = [_fval(s, "max_drawdown_2h") for s in samples]
        big = sum(1 for s in samples if _bool(s, "label_big_winner"))
        trap = sum(1 for s in samples if _bool(s, "label_trap"))
        win2 = sum(1 for x in r2h if x >= 3.0)
        score = (
            (win2 / len(samples)) * 40
            + statistics.mean(r2h) * 5
            + (big / len(samples)) * 30
            - (trap / len(samples)) * 20
        )
        out.append({
            "formula_name": name,
            "sample_count": len(samples),
            "win_rate_30m": round(sum(1 for x in r30 if x >= 2.0) / len(samples) * 100, 2),
            "win_rate_1h": round(sum(1 for x in r1h if x >= 2.5) / len(samples) * 100, 2),
            "win_rate_2h": round(win2 / len(samples) * 100, 2),
            "avg_return_2h": round(statistics.mean(r2h), 4),
            "median_return_2h": round(statistics.median(r2h), 4),
            "max_drawdown_avg": round(statistics.mean(mdd), 4),
            "big_winner_capture_rate": round(big / len(samples) * 100, 2),
            "trap_rate": round(trap / len(samples) * 100, 2),
            "score": round(score, 2),
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def random_baseline_stats(forward_rows: list[dict]) -> dict:
    """Random baseline from all tracked TOP20 forward rows with complete 2h."""
    complete = [r for r in forward_rows if r.get("return_2h") not in ("", None)]
    if not complete:
        return {"sample_count": 0, "win_rate_2h": 0, "avg_return_2h": 0}
    r2h = [_fval(r, "return_2h") for r in complete]
    return {
        "sample_count": len(complete),
        "win_rate_2h": round(sum(1 for x in r2h if x >= 3.0) / len(r2h) * 100, 2),
        "avg_return_2h": round(statistics.mean(r2h), 4),
    }
