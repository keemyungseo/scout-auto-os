"""A6 frozen search + R009 exit helpers."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import scout_phase19_winner_ranking_dna as p19
import scout_phase20_winner_state_ranking as p20
import scout_phase22_search_formula_evolution as p22
import scout_phase23_search_formula_league as p23
from scout_phase19_winner_ranking_dna import extract_dna_features
from scout_research_r005_execution_statistics import parse_kst
from scout_research_r006_pilot_execution_engine import Bar, load_forward_bars
from scout_research_r008_exit_engine import state_snapshot
from scout_research_r009_dynamic_exit_engine import efr_from_bar
from scout_research_r010_dynamic_entry_engine import ENTRY_A, simulate_entry
from season2_universe_blind_test import load_eligible_symbols

import scout_phase16_human_blind_test as p16

if TYPE_CHECKING:
    from scout_auto_os.engine.live_data import LiveDataEngine

from scout_auto_os.engine.expected_ev_engine import compute_live_ev


def brief_reason(states: dict, features: dict) -> str:
    return (
        f"1h={states.get('1h', '?')} 2h={states.get('2h', '?')} | "
        f"rng={features.get('1h_current_range_pct', 0):.1f}%"
    )


def _train_context(cutoff: str) -> tuple[dict, object, object, dict]:
    raw: list[dict] = []
    for line in p19.CANDIDATES_PATH.open(encoding="utf-8"):
        r = json.loads(line)
        if r["scan_kst"] < cutoff:
            raw.append(r)
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by_scan[r["scan_kst"]].append(r)
    for s in by_scan:
        by_scan[s].sort(key=lambda x: x.get("outcome_rank", 999))
    winner_feats = [r["features"] for rows in by_scan.values() for r in rows[:3] if len(rows) >= 4]
    th = p20.build_thresholds(winner_feats)
    annotated = p20.annotate(raw, th)
    train_by: dict[str, list[dict]] = defaultdict(list)
    for r in annotated:
        train_by[r["scan_kst"]].append(r)
    w_train, _ = p20.winner_loser_sets(train_by)
    profile = p20.build_profile(w_train, annotated) if w_train else p20.build_profile([], annotated)
    stats = p22.build_train_stats(annotated, train_by, th)
    return profile, th, stats, by_scan


def run_universe_features(
    scan_kst: str,
    cache_dir: Path,
    max_symbols: int = 0,
    live_engine: LiveDataEngine | None = None,
) -> list[dict]:
    """Full universe rows with DNA features (for Portfolio Engine)."""
    p19.CACHE_DIR = cache_dir
    p16.CACHE_DIR = cache_dir
    end_ms = int(parse_kst(scan_kst).timestamp() * 1000)
    cache_only = os.environ.get("SCOUT_UNIVERSE_CACHE_ONLY", "true").lower() in ("1", "true", "yes")
    symbols = sorted(load_eligible_symbols(refresh=False, cache_only=cache_only))
    if max_symbols > 0:
        symbols = symbols[:max_symbols]
    rows: list[dict] = []
    for sym in symbols:
        feats = extract_dna_features(sym, end_ms)
        if feats:
            rows.append({"scan_kst": scan_kst, "symbol": sym, "features": feats})
    if live_engine and live_engine.enabled:
        live_engine.subscribe([r["symbol"] for r in rows[:40]])
    return rows


def run_a6_scan(
    scan_kst: str,
    cache_dir: Path,
    top_n: int = 5,
    max_symbols: int = 0,
    live_engine: LiveDataEngine | None = None,
    ev_logger=None,
) -> list[dict]:
    p19.CACHE_DIR = cache_dir
    p16.CACHE_DIR = cache_dir
    profile, th, stats, _ = _train_context(scan_kst)
    end_ms = int(parse_kst(scan_kst).timestamp() * 1000)
    cache_only = os.environ.get("SCOUT_UNIVERSE_CACHE_ONLY", "true").lower() in ("1", "true", "yes")
    symbols = sorted(load_eligible_symbols(refresh=False, cache_only=cache_only))
    if max_symbols > 0:
        symbols = symbols[:max_symbols]

    raw_rows: list[dict] = []
    for sym in symbols:
        feats = extract_dna_features(sym, end_ms)
        if feats:
            raw_rows.append({"scan_kst": scan_kst, "symbol": sym, "features": feats})

    if not raw_rows:
        return []

    annotated = p20.annotate(raw_rows, th)
    for r in annotated:
        peers = annotated
        base = p20.state_match_score(r["states"], r["transitions"], profile)
        r["a6_score"] = round(p23.formula_scores_a6(r, peers, base, th, stats)["A6"], 4)

    ranked = sorted(annotated, key=lambda x: x["a6_score"], reverse=True)[:top_n]
    if live_engine and live_engine.enabled:
        live_engine.subscribe([r["symbol"] for r in ranked])

    out: list[dict] = []
    for i, r in enumerate(ranked, 1):
        feats = r["features"]
        st = r["states"]
        sym = r["symbol"]
        bars: list[Bar] = []
        entry_price = feats.get("price", 0)
        if live_engine and live_engine.enabled:
            bars = live_engine.get_forward_bars(sym, scan_kst)
            live_px = live_engine.get_price(sym)
            if live_px > 0:
                entry_price = live_px
        if not bars:
            bars = load_forward_bars(sym, scan_kst)
            if bars:
                entry_price = float(bars[0].o)

        ev = compute_live_ev(sym, bars, r["a6_score"], scan_kst)
        if ev_logger:
            ev_logger.log(sym, ev)
        snap = state_snapshot(bars, len(bars) - 1, 0) if bars else None
        out.append({
            "rank": i,
            "symbol": sym,
            "a6_score": r["a6_score"],
            "reason": brief_reason(st, feats),
            "entry_price": float(entry_price),
            "expected_ev": ev["expected_ev"],
            "remaining_ev": ev["remaining_ev"],
            "trend_alive": ev["trend_alive"] if live_engine else st.get("2h", ""),
            "acceleration": ev["acceleration"] if live_engine else st.get("1h", ""),
            "volume_state": ev["volume_state"] if live_engine else ("weak" if snap and snap.volume_weak else "ok"),
        })
    return out


def check_dynamic_exit(bars: list[Bar], entry_px: float, entry_bar: int = 0) -> tuple[bool, float, str]:
    if not bars or entry_px <= 0:
        return False, 0.0, ""
    end_i = len(bars) - 1
    sl_px = entry_px * 0.90
    for j in range(max(entry_bar + 1, 1), end_i + 1):
        b = bars[j]
        current_ret = (b.c - entry_px) / entry_px * 100
        e30 = efr_from_bar(bars, j, 6)
        e60 = efr_from_bar(bars, j, 12)
        snap = state_snapshot(bars, j, entry_bar)
        if b.l <= sl_px:
            return True, -10.0, "protective_sl"
        if current_ret > 0.5 and e30 < current_ret and e60 < current_ret:
            return True, current_ret, "efr_exit"
        if current_ret > 1.0 and not snap.trend_alive and snap.momentum_weak:
            return True, current_ret, "state_exhaustion"
    return False, 0.0, ""


def current_efr(bars: list[Bar]) -> float:
    if not bars:
        return 0.0
    return efr_from_bar(bars, len(bars) - 1, 12)
