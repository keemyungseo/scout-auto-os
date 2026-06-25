"""Full-universe research scan — scoring only, no orders."""

from __future__ import annotations

import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

import scout_phase16_human_blind_test as p16
import scout_phase19_winner_ranking_dna as p19
import scout_phase20_winner_state_ranking as p20
import scout_phase23_search_formula_league as p23
from scout_phase19_winner_ranking_dna import extract_dna_features
from scout_research_r005_execution_statistics import parse_kst
from scout_research_r006_pilot_execution_engine import Bar, load_forward_bars
from season2_universe_blind_test import load_eligible_symbols

from scout_auto_os.engine.expected_ev_engine import compute_live_ev
from scout_auto_os.engine.strategy_core import _train_context, brief_reason
from scout_auto_os.engine.research.formula_league import rank_by_formula
from scout_auto_os.engine.research.market_data import btc_returns, classify_regime

KST = timezone(timedelta(hours=9))


def _g(f: dict, key: str, default: float = 0.0) -> float:
    return float(f.get(key, default))


def _btc_context_label(btc_1h: float, btc_4h: float) -> str:
    if btc_4h >= 1.0:
        return f"bullish_4h={btc_4h:.2f}"
    if btc_4h <= -1.0:
        return f"bearish_4h={btc_4h:.2f}"
    return f"neutral_1h={btc_1h:.2f}"


def run_research_scan(
    scan_kst: str,
    cache_dir: Path,
    rest_base: str,
    top_n: int = 20,
    workers: int = 8,
    live_engine=None,
    max_symbols: int = 0,
    symbol_list: list[str] | None = None,
) -> dict:
    p19.CACHE_DIR = cache_dir
    p16.CACHE_DIR = cache_dir
    end_ms = int(parse_kst(scan_kst).timestamp() * 1000)
    cache_only = os.environ.get("SCOUT_UNIVERSE_CACHE_ONLY", "false").lower() in ("1", "true", "yes")
    symbols = sorted(symbol_list) if symbol_list else sorted(load_eligible_symbols(refresh=False, cache_only=cache_only))
    if max_symbols > 0 and not symbol_list:
        symbols = symbols[:max_symbols]

    profile, th, stats, _ = _train_context(scan_kst)
    btc_1h, btc_4h = btc_returns(rest_base)
    regime = classify_regime(btc_1h, btc_4h)
    btc_ctx = _btc_context_label(btc_1h, btc_4h)

    raw_rows: list[dict] = []

    def _one(sym: str) -> dict | None:
        try:
            feats = extract_dna_features(sym, end_ms)
            if feats:
                return {"scan_kst": scan_kst, "symbol": sym, "features": feats}
        except Exception:
            return None
        finally:
            time.sleep(0.02)
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, sym): sym for sym in symbols}
        for fut in as_completed(futs):
            row = fut.result()
            if row:
                raw_rows.append(row)

    if not raw_rows:
        return {
            "scan_time_kst": scan_kst,
            "total_symbols": len(symbols),
            "market_regime": regime,
            "btc_1h_return": btc_1h,
            "btc_4h_return": btc_4h,
            "alt_market_strength": 0.0,
            "top20_symbols": "",
            "candidates": [],
            "annotated": [],
            "formula_picks": {},
        }

    annotated = p20.annotate(raw_rows, th)
    for r in annotated:
        peers = annotated
        base = p20.state_match_score(r["states"], r["transitions"], profile)
        r["a6_score"] = round(p23.formula_scores_a6(r, peers, base, th, stats)["A6"], 4)

    ranked = sorted(annotated, key=lambda x: x["a6_score"], reverse=True)[:top_n]
    rets = [_g(r["features"], "1h_current_return_pct") for r in raw_rows]
    alt_strength = round(statistics.median(rets), 4) if rets else 0.0

    candidates: list[dict] = []
    for i, r in enumerate(ranked, 1):
        feats = r["features"]
        st = r["states"]
        sym = r["symbol"]
        price = float(feats.get("price", 0))
        if live_engine and live_engine.enabled:
            live_px = live_engine.get_price(sym)
            if live_px > 0:
                price = live_px
        bars: list[Bar] = []
        if live_engine and live_engine.enabled:
            bars = live_engine.get_forward_bars(sym, scan_kst)
        if not bars:
            bars = load_forward_bars(sym, scan_kst)
        ev = compute_live_ev(sym, bars, r["a6_score"], scan_kst) if bars else {
            "expected_ev": 0.0,
        }
        candidates.append({
            "scan_time_kst": scan_kst,
            "rank": i,
            "symbol": sym,
            "current_price": round(price, 8),
            "a6_score": r["a6_score"],
            "expected_ev": ev.get("expected_ev", 0),
            "reason_1h": st.get("1h", ""),
            "reason_2h": st.get("2h", ""),
            "range_pct": round(_g(feats, "1h_current_range_pct"), 4),
            "volume_ratio": round(_g(feats, "15m_current_volume_ratio"), 4),
            "atr_pct": round(_g(feats, "1h_current_range_pct"), 4),
            "momentum_15m": round(_g(feats, "15m_current_return_pct"), 4),
            "momentum_1h": round(_g(feats, "1h_current_return_pct"), 4),
            "compression_score": round(_g(feats, "5m_compression"), 4),
            "breakout_score": round(_g(feats, "5m_release") + _g(feats, "5m_range_energy") * 0.1, 4),
            "btc_context": btc_ctx,
            "selected_by_live_engine": False,
            "_reason": brief_reason(st, feats),
        })

    formula_picks = rank_by_formula(annotated, profile, th, stats, top_n=5)

    return {
        "scan_time_kst": scan_kst,
        "total_symbols": len(symbols),
        "market_regime": regime,
        "btc_1h_return": btc_1h,
        "btc_4h_return": btc_4h,
        "alt_market_strength": alt_strength,
        "top20_symbols": "|".join(c["symbol"] for c in candidates),
        "candidates": candidates,
        "annotated": annotated,
        "formula_picks": formula_picks,
    }
