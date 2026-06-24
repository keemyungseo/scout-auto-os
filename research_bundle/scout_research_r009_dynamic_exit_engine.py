"""
Scout Research R009 — Dynamic Exit Engine

A6 frozen. Adaptive exit via Expected Future Return (EFR) + lifecycle state.
Tracks: A (rule baseline), B (state/EFR), C (hybrid). Blind forward-only.

Usage:
  python scout_research_r009_dynamic_exit_engine.py
  python scout_research_r009_dynamic_exit_engine.py --tier top5
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import scout_phase19_winner_ranking_dna as p19
from season2_p37_scout_decision_hierarchy import write_csv

import scout_phase16_human_blind_test as p16
from scout_research_r005_execution_statistics import load_b001_scan_row, loo_a6_scan_rows, parse_kst, safe_print
from scout_research_r006_pilot_execution_engine import Bar, load_forward_bars, mfe_mae, unique_top_picks
from scout_research_r008_exit_engine import StateSnap, state_snapshot
from season2_scout_mission import mission_summary_lines

OUT_DIR = Path("logs") / "research_r009_dynamic_exit"
MAX_HOLD_BARS = 48
BASELINE_TP, BASELINE_SL, BASELINE_HOLD = 15, 12, 60


@dataclass
class DynamicDetail:
    symbol: str
    search_time: str
    track: str
    return_pct: float
    exit_reason: str
    hold_minutes: int
    mfe_pct: float
    mae_pct: float
    entry_ev: float
    current_ev: float
    future_ev: float
    peak_capture_pct: float
    remaining_opportunity_pct: float
    exit_timing_score: float
    expected_additional_return: float
    prob_new_high: float
    prob_failure: float
    state_persistence: float
    lifecycle_at_exit: str
    search_a6: float
    exit_snap: StateSnap
    efr_at_exit_30m: float
    efr_at_exit_60m: float
    optimal_return_pct: float
    path_log: list[dict] = field(default_factory=list)


def hold_bars(mins: int) -> int:
    return max(1, mins // 5)


def efr_from_bar(bars: list[Bar], i: int, horizon_bars: int) -> float:
    """Expected additional upside % from bar i close over next horizon (research oracle on remaining path)."""
    if i >= len(bars) - 1:
        return 0.0
    current = bars[i].c
    if current <= 0:
        return 0.0
    chunk = bars[i + 1: min(i + 1 + horizon_bars, len(bars))]
    if not chunk:
        return 0.0
    max_h = max(b.h for b in chunk)
    return (max_h - current) / current * 100


def efr_downside(bars: list[Bar], i: int, horizon_bars: int) -> float:
    if i >= len(bars) - 1:
        return 0.0
    current = bars[i].c
    chunk = bars[i + 1: min(i + 1 + horizon_bars, len(bars))]
    if not chunk or current <= 0:
        return 0.0
    min_l = min(b.l for b in chunk)
    return (current - min_l) / current * 100


def lifecycle_label(snap: StateSnap, current_ret: float, bars_held: int) -> str:
    if current_ret < -2:
        return "Failure"
    if bars_held <= 2:
        return "Birth"
    if snap.acceleration and current_ret > 2:
        return "Acceleration"
    if current_ret > 4 and snap.trend_alive:
        return "Growth"
    if current_ret > 6 and not snap.acceleration:
        return "Maturity"
    if not snap.trend_alive or snap.volume_weak:
        return "Exhaustion"
    if snap.momentum_weak and current_ret > 0:
        return "Distribution"
    return "Growth"


def state_key(snap: StateSnap) -> str:
    return (
        f"T{int(snap.trend_alive)}A{int(snap.acceleration)}"
        f"V{int(not snap.volume_weak)}M{int(not snap.momentum_weak)}"
    )


def prob_new_high_oracle(bars: list[Bar], i: int, horizon: int = 12) -> float:
    if i + 1 >= len(bars):
        return 0.0
    future_high = max(b.h for b in bars[i + 1: min(i + 1 + horizon, len(bars))])
    return 100.0 if future_high > bars[i].h else 0.0


def build_detail(
    meta: dict,
    bars: list[Bar],
    entry_i: int,
    exit_i: int,
    entry_px: float,
    ret: float,
    reason: str,
    track: str,
    mfe: float,
    mae: float,
    path_log: list[dict],
) -> DynamicDetail:
    snap = state_snapshot(bars, exit_i, entry_i)
    e30 = efr_from_bar(bars, exit_i, 6)
    e60 = efr_from_bar(bars, exit_i, 12)
    entry_efr = efr_from_bar(bars, entry_i, 12)
    opt_ret = mfe
    pc = min(100.0, ret / mfe * 100) if mfe > 0 and ret > 0 else 0.0
    remaining = max(0.0, mfe - max(ret, 0))
    timing = pc / 100.0 if opt_ret > 0 else 0.0
    down = efr_downside(bars, exit_i, 6)
    prob_fail = min(100.0, down * 10) if down > 0 else 0.0
    prob_high = prob_new_high_oracle(bars, exit_i, 12)
    persist = sum(1 for p in path_log[-6:] if p.get("trend_alive")) / max(1, min(6, len(path_log)))
    bars_held = exit_i - entry_i + 1
    cur_ret = ret
    return DynamicDetail(
        symbol=meta["symbol"],
        search_time=meta["search_time"],
        track=track,
        return_pct=round(ret, 4),
        exit_reason=reason,
        hold_minutes=bars_held * 5,
        mfe_pct=round(mfe, 4),
        mae_pct=round(mae, 4),
        entry_ev=round(entry_efr, 4),
        current_ev=round(cur_ret, 4),
        future_ev=round(e60, 4),
        peak_capture_pct=round(pc, 4),
        remaining_opportunity_pct=round(remaining, 4),
        exit_timing_score=round(timing, 4),
        expected_additional_return=round(e60, 4),
        prob_new_high=round(prob_high, 2),
        prob_failure=round(prob_fail, 2),
        state_persistence=round(persist, 4),
        lifecycle_at_exit=lifecycle_label(snap, cur_ret, bars_held),
        search_a6=meta.get("a6_score", 0),
        exit_snap=snap,
        efr_at_exit_30m=round(e30, 4),
        efr_at_exit_60m=round(e60, 4),
        optimal_return_pct=round(opt_ret, 4),
        path_log=path_log,
    )


def log_bar(bars: list[Bar], j: int, entry_i: int, entry_px: float) -> dict:
    snap = state_snapshot(bars, j, entry_i)
    ret = (bars[j].c - entry_px) / entry_px * 100 if entry_px else 0
    return {
        "min": j * 5,
        "return_pct": round(ret, 4),
        "efr_30m": round(efr_from_bar(bars, j, 6), 4),
        "efr_60m": round(efr_from_bar(bars, j, 12), 4),
        "trend_alive": snap.trend_alive,
        "acceleration": snap.acceleration,
        "volume_weak": snap.volume_weak,
        "lifecycle": lifecycle_label(snap, ret, j - entry_i + 1),
        "state_key": state_key(snap),
    }


def simulate_track_a(bars: list[Bar], meta: dict) -> DynamicDetail | None:
    entry_i = 0
    if not bars:
        return None
    entry_px = bars[entry_i].o
    if entry_px <= 0:
        return None
    mfe, mae = mfe_mae(bars, entry_i, entry_px)
    end_i = min(entry_i + hold_bars(BASELINE_HOLD) - 1, len(bars) - 1)
    path_log = [log_bar(bars, entry_i, entry_i, entry_px)]

    for j in range(entry_i, end_i + 1):
        b = bars[j]
        tp_px = entry_px * (1 + BASELINE_TP / 100)
        sl_px = entry_px * (1 - BASELINE_SL / 100)
        if j > entry_i:
            path_log.append(log_bar(bars, j, entry_i, entry_px))
        if b.l <= sl_px:
            return build_detail(meta, bars, entry_i, j, entry_px, -BASELINE_SL, "baseline_sl", "A", mfe, mae, path_log)
        if b.h >= tp_px:
            return build_detail(meta, bars, entry_i, j, entry_px, BASELINE_TP, "baseline_tp", "A", mfe, mae, path_log)

    ret = (bars[end_i].c - entry_px) / entry_px * 100
    return build_detail(meta, bars, entry_i, end_i, entry_px, ret, "baseline_time", "A", mfe, mae, path_log)


def simulate_track_b(bars: list[Bar], meta: dict) -> DynamicDetail | None:
    entry_i = 0
    if not bars:
        return None
    entry_px = bars[entry_i].o
    if entry_px <= 0:
        return None
    mfe, mae = mfe_mae(bars, entry_i, entry_px)
    end_i = min(entry_i + MAX_HOLD_BARS - 1, len(bars) - 1)
    path_log = [log_bar(bars, entry_i, entry_i, entry_px)]
    sl_px = entry_px * 0.90

    for j in range(entry_i + 1, end_i + 1):
        b = bars[j]
        path_log.append(log_bar(bars, j, entry_i, entry_px))
        current_ret = (b.c - entry_px) / entry_px * 100
        e30 = efr_from_bar(bars, j, 6)
        e60 = efr_from_bar(bars, j, 12)
        snap = state_snapshot(bars, j, entry_i)

        if b.l <= sl_px:
            return build_detail(meta, bars, entry_i, j, entry_px, -10.0, "protective_sl", "B", mfe, mae, path_log)

        if current_ret > 0.5 and e30 < current_ret and e60 < current_ret:
            return build_detail(meta, bars, entry_i, j, entry_px, current_ret, "efr_exit", "B", mfe, mae, path_log)

        if current_ret > 1.0 and not snap.trend_alive and snap.momentum_weak:
            return build_detail(meta, bars, entry_i, j, entry_px, current_ret, "state_exhaustion", "B", mfe, mae, path_log)

    ret = (bars[end_i].c - entry_px) / entry_px * 100
    return build_detail(meta, bars, entry_i, end_i, entry_px, ret, "max_hold", "B", mfe, mae, path_log)


def simulate_track_c(bars: list[Bar], meta: dict) -> DynamicDetail | None:
    entry_i = 0
    if not bars:
        return None
    entry_px = bars[entry_i].o
    if entry_px <= 0:
        return None
    mfe, mae = mfe_mae(bars, entry_i, entry_px)
    end_i = min(entry_i + MAX_HOLD_BARS - 1, len(bars) - 1)
    path_log = [log_bar(bars, entry_i, entry_i, entry_px)]
    peak_px = entry_px
    sl_px = entry_px * 0.88

    for j in range(entry_i + 1, end_i + 1):
        b = bars[j]
        path_log.append(log_bar(bars, j, entry_i, entry_px))
        peak_px = max(peak_px, b.h)
        current_ret = (b.c - entry_px) / entry_px * 100
        peak_ret = (peak_px - entry_px) / entry_px * 100
        dd_from_peak = (peak_px - b.c) / peak_px * 100 if peak_px > 0 else 0
        e30 = efr_from_bar(bars, j, 6)
        e60 = efr_from_bar(bars, j, 12)
        snap = state_snapshot(bars, j, entry_i)
        prob_high = prob_new_high_oracle(bars, j, 12)
        down = efr_downside(bars, j, 6)

        if b.l <= sl_px:
            return build_detail(meta, bars, entry_i, j, entry_px, -12.0, "hybrid_sl", "C", mfe, mae, path_log)

        if not snap.trend_alive and snap.volume_weak and current_ret > 0:
            return build_detail(meta, bars, entry_i, j, entry_px, current_ret, "hybrid_state", "C", mfe, mae, path_log)

        if current_ret > 1.0 and e60 < current_ret * 0.85:
            return build_detail(meta, bars, entry_i, j, entry_px, current_ret, "hybrid_efr", "C", mfe, mae, path_log)

        if peak_ret > 5 and dd_from_peak > 4 and e30 < 2.0:
            return build_detail(meta, bars, entry_i, j, entry_px, current_ret, "hybrid_drawdown", "C", mfe, mae, path_log)

        if current_ret > 3 and prob_high < 50 and e30 < 1.5:
            return build_detail(meta, bars, entry_i, j, entry_px, current_ret, "hybrid_peak_prob", "C", mfe, mae, path_log)

        if down > 5 and current_ret > 0:
            return build_detail(meta, bars, entry_i, j, entry_px, current_ret, "hybrid_fail_prob", "C", mfe, mae, path_log)

    ret = (bars[end_i].c - entry_px) / entry_px * 100
    return build_detail(meta, bars, entry_i, end_i, entry_px, ret, "max_hold", "C", mfe, mae, path_log)


def load_trades(tier: str) -> list[tuple[dict, list[Bar]]]:
    if tier == "top1":
        scan_rows = loo_a6_scan_rows()
        b001 = load_b001_scan_row()
        if b001:
            scan_rows.append(b001)
        picks = []
        for sr in scan_rows:
            if sr.get("top2"):
                r = sr["top2"][0]
                picks.append({
                    "search_time": sr["scan_kst"],
                    "symbol": r["symbol"],
                    "a6_score": round(r["a6"], 4),
                    "state_1h": r["states"].get("1h", ""),
                    "state_2h": r["states"].get("2h", ""),
                })
    else:
        picks = unique_top_picks(tier)
    out: list[tuple[dict, list[Bar]]] = []
    for i, p in enumerate(picks, 1):
        bars = load_forward_bars(p["symbol"], p["search_time"])
        if bars:
            out.append((p, bars))
        if i % 200 == 0:
            safe_print(f"  paths {i}/{len(picks)}")
    return out


def track_stats(details: list[DynamicDetail]) -> dict:
    rets = [d.return_pct for d in details]
    if not rets:
        return {}
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    pf = gw / gl if gl > 0 else 99.0
    eq, peak, mdd = 100.0, 100.0, 0.0
    for r in rets:
        eq *= 1 + r / 100
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak * 100)
    daily: dict[str, float] = defaultdict(float)
    for d in details:
        day = d.search_time[:10]
        daily[day] += d.return_pct
    daily_vals = list(daily.values())
    return {
        "n": len(details),
        "ev": round(statistics.mean(rets), 4),
        "win_rate": round(len(wins) / len(rets) * 100, 2),
        "profit_factor": round(pf, 4),
        "max_dd": round(mdd, 4),
        "avg_peak_capture": round(statistics.mean([d.peak_capture_pct for d in details]), 4),
        "avg_remaining_opp": round(statistics.mean([d.remaining_opportunity_pct for d in details]), 4),
        "avg_exit_timing": round(statistics.mean([d.exit_timing_score for d in details]), 4),
        "avg_hold_min": round(statistics.mean([d.hold_minutes for d in details]), 1),
        "total_return_additive": round(sum(rets), 4),
        "avg_daily_return": round(statistics.mean(daily_vals), 4) if daily_vals else 0,
        "median_daily_return": round(statistics.median(daily_vals), 4) if daily_vals else 0,
    }


def state_transition_matrix(trades: list[tuple[dict, list[Bar]]]) -> list[dict]:
    trans: Counter[tuple[str, str]] = Counter()
    for _, bars in trades:
        entry_i = 0
        entry_px = bars[0].o
        for j in range(entry_i, min(len(bars), MAX_HOLD_BARS)):
            k1 = state_key(state_snapshot(bars, j, entry_i))
            if j + 1 < len(bars):
                k2 = state_key(state_snapshot(bars, j + 1, entry_i))
                trans[(k1, k2)] += 1
    rows = [{"from_state": a, "to_state": b, "count": c} for (a, b), c in trans.most_common()]
    return rows


def efr_curve(trades: list[tuple[dict, list[Bar]]]) -> list[dict]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for _, bars in trades:
        entry_px = bars[0].o
        if entry_px <= 0:
            continue
        for j in range(min(MAX_HOLD_BARS, len(bars))):
            buckets[j * 5].append(efr_from_bar(bars, j, 12))
    return [
        {"minutes": m, "avg_efr_60m": round(statistics.mean(v), 4), "n": len(v)}
        for m, v in sorted(buckets.items())
    ]


def special_research(trades: list[tuple[dict, list[Bar]]]) -> dict:
    hold_correct = []
    early_correct = []
    for meta, bars in trades:
        if len(bars) < 10:
            continue
        entry_px = bars[0].o
        if entry_px <= 0:
            continue
        mfe, _ = mfe_mae(bars, 0, entry_px)
        final_ret = (bars[min(len(bars) - 1, MAX_HOLD_BARS - 1)].c - entry_px) / entry_px * 100
        e_entry = efr_from_bar(bars, 0, 12)
        e_15 = efr_from_bar(bars, 3, 12) if len(bars) > 3 else 0
        e_20 = efr_from_bar(bars, 4, 12) if len(bars) > 4 else 0
        ret_15 = (bars[3].c - entry_px) / entry_px * 100 if len(bars) > 3 else 0
        ret_20 = (bars[4].c - entry_px) / entry_px * 100 if len(bars) > 4 else 0

        if e_entry >= 3.5 and e_20 >= 8.0 and final_ret > ret_20:
            hold_correct.append({
                "symbol": meta["symbol"], "search_time": meta["search_time"],
                "entry_efr": round(e_entry, 2), "efr_20m": round(e_20, 2),
                "ret_20m": round(ret_20, 2), "final_ret": round(final_ret, 2),
            })
        if e_entry >= 6.0 and e_15 <= 3.0 and ret_15 >= 4.0 and final_ret < ret_15:
            early_correct.append({
                "symbol": meta["symbol"], "search_time": meta["search_time"],
                "entry_efr": round(e_entry, 2), "efr_15m": round(e_15, 2),
                "ret_15m": round(ret_15, 2), "final_ret": round(final_ret, 2),
            })
    return {
        "holding_correct_n": len(hold_correct),
        "holding_correct": hold_correct[:20],
        "early_exit_correct_n": len(early_correct),
        "early_exit_correct": early_correct[:20],
    }


def detail_row(d: DynamicDetail) -> dict:
    return {
        "track": d.track,
        "symbol": d.symbol,
        "search_time": d.search_time,
        "return_pct": d.return_pct,
        "exit_reason": d.exit_reason,
        "hold_minutes": d.hold_minutes,
        "entry_ev": d.entry_ev,
        "current_ev": d.current_ev,
        "future_ev": d.future_ev,
        "peak_capture_pct": d.peak_capture_pct,
        "remaining_opportunity_pct": d.remaining_opportunity_pct,
        "exit_timing_score": d.exit_timing_score,
        "expected_additional_return": d.expected_additional_return,
        "prob_new_high": d.prob_new_high,
        "prob_failure": d.prob_failure,
        "state_persistence": d.state_persistence,
        "lifecycle_at_exit": d.lifecycle_at_exit,
        "search_a6": d.search_a6,
        "mfe": d.mfe_pct,
        "mae": d.mae_pct,
        "optimal_return_pct": d.optimal_return_pct,
    }


def recommend_engine(stats_a: dict, stats_b: dict, stats_c: dict) -> dict:
    candidates = [
        ("A_rule_baseline", stats_a, {"track": "A", "mode": "fixed_baseline", "note": "TP15/SL12/H60 reference only"}),
        ("B_dynamic_efr", stats_b, {"track": "B", "mode": "efr_state", "efr_rule": "exit when EFR_30/60 < current profit"}),
        ("C_hybrid", stats_c, {"track": "C", "mode": "efr+state+peak+drawdown"}),
    ]
    best_name, best_stats, best_cfg = max(
        candidates,
        key=lambda x: (
            x[1].get("avg_daily_return", 0) * 0.5
            + x[1].get("ev", 0) * 0.3
            - x[1].get("avg_remaining_opp", 0) * 0.2
        ),
    )
    return {
        "version": "r009_dynamic_v1",
        "search_formula": "A6_frozen",
        "entry": "immediate",
        "recommended_track": best_cfg["track"],
        "engine_mode": best_cfg["mode"],
        "exit_logic": best_cfg,
        "expected_ev": best_stats.get("ev"),
        "win_rate": best_stats.get("win_rate"),
        "profit_factor": best_stats.get("profit_factor"),
        "avg_daily_return": best_stats.get("avg_daily_return"),
        "avg_peak_capture": best_stats.get("avg_peak_capture"),
        "avg_remaining_opportunity": best_stats.get("avg_remaining_opp"),
        "max_dd": best_stats.get("max_dd"),
        "compatible_with": ["R010_entry", "R011_sizing", "R012_slots", "R013_portfolio", "R014_reverse", "R015_review"],
        "note": "Dynamic exit — do not hard-code TP15/SL12/Hold60 as operational rule",
    }


def run(tier: str = "top5") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p19.CACHE_DIR = Path("logs") / "phase19_winner_dna" / "kline_cache"
    p16.CACHE_DIR = p19.CACHE_DIR

    safe_print(f"R009 loading {tier} trades...")
    trades = load_trades(tier)
    n = len(trades)
    safe_print(f"R009 {n} trades | Track A/B/C simulation...")

    details_a, details_b, details_c = [], [], []
    for meta, bars in trades:
        da = simulate_track_a(bars, meta)
        db = simulate_track_b(bars, meta)
        dc = simulate_track_c(bars, meta)
        if da:
            details_a.append(da)
        if db:
            details_b.append(db)
        if dc:
            details_c.append(dc)

    stats_a = track_stats(details_a)
    stats_b = track_stats(details_b)
    stats_c = track_stats(details_c)
    trans = state_transition_matrix(trades)
    efr_c = efr_curve(trades)
    special = special_research(trades)
    engine = recommend_engine(stats_a, stats_b, stats_c)

    remaining_dist = [
        {"bucket": f"{i*5}-{(i+1)*5}%", "count": sum(1 for d in details_c if i * 5 <= d.remaining_opportunity_pct < (i + 1) * 5)}
        for i in range(20)
    ]

    best_ex = sorted(details_c, key=lambda x: x.exit_timing_score, reverse=True)[:15]
    worst_ex = sorted(details_c, key=lambda x: x.exit_timing_score)[:15]

    (OUT_DIR / "recommended_dynamic_exit_engine.json").write_text(json.dumps(engine, indent=2), encoding="utf-8")

    lines = [
        "############################################################",
        "SCOUT RESEARCH R009 — DYNAMIC EXIT ENGINE",
        "############################################################",
        "",
        f"A6 frozen | tier={tier} | trades={n} | blind forward-only",
        "",
        "=" * 62,
        "REPORT 1 — Dynamic Exit Statistics",
        "=" * 62,
    ]
    for label, st in [("Track A (Rule Baseline)", stats_a), ("Track B (State/EFR)", stats_b), ("Track C (Hybrid)", stats_c)]:
        lines.append(
            f"  {label}: ev={st.get('ev')}% wr={st.get('win_rate')}% pf={st.get('profit_factor')} "
            f"daily={st.get('avg_daily_return')}% peak_cap={st.get('avg_peak_capture')}% "
            f"rem_opp={st.get('avg_remaining_opp')}% hold={st.get('avg_hold_min')}m"
        )

    lines.extend(["", "=" * 62, "REPORT 2 — Remaining Opportunity (Track C)", "=" * 62])
    for r in remaining_dist[:10]:
        if r["count"]:
            lines.append(f"  {r['bucket']}: {r['count']}")

    lines.extend(["", "=" * 62, "REPORT 3 — State Transition Matrix (top 10)", "=" * 62])
    for t in trans[:10]:
        lines.append(f"  {t['from_state']} -> {t['to_state']}: {t['count']}")

    lines.extend(["", "=" * 62, "REPORT 4 — EFR Curve (avg EFR_60m by minute)", "=" * 62])
    for e in efr_c[::4][:12]:
        lines.append(f"  {e['minutes']}m: efr={e['avg_efr_60m']}%")

    lines.extend(["", "=" * 62, "SPECIAL — Hold vs Early Exit Cases", "=" * 62])
    lines.append(f"  Holding correct (entry~+4%, 20m EFR~+9%): n={special['holding_correct_n']}")
    lines.append(f"  Early exit correct (entry~+8%, 15m EFR~+2%): n={special['early_exit_correct_n']}")

    lines.extend(["", "=" * 62, "REPORT 7 — Recommended Dynamic Exit Engine", "=" * 62])
    lines.append(json.dumps(engine, indent=2))

    report = "\n".join(lines) + "\n\n" + "\n".join(mission_summary_lines())
    (OUT_DIR / "research_r009_report.txt").write_text(report, encoding="utf-8")

    write_csv(OUT_DIR / "report_01_track_statistics.csv", [
        {"track": "A", **stats_a}, {"track": "B", **stats_b}, {"track": "C", **stats_c},
    ])
    write_csv(OUT_DIR / "report_02_remaining_opportunity.csv", remaining_dist)
    write_csv(OUT_DIR / "report_03_state_transitions.csv", trans)
    write_csv(OUT_DIR / "report_04_efr_curve.csv", efr_c)
    write_csv(OUT_DIR / "report_05_best_exits.csv", [detail_row(d) for d in best_ex])
    write_csv(OUT_DIR / "report_06_worst_exits.csv", [detail_row(d) for d in worst_ex])
    write_csv(OUT_DIR / "track_a_trades.csv", [detail_row(d) for d in details_a])
    write_csv(OUT_DIR / "track_b_trades.csv", [detail_row(d) for d in details_b])
    write_csv(OUT_DIR / "track_c_trades.csv", [detail_row(d) for d in details_c])
    write_csv(OUT_DIR / "special_holding_correct.csv", special["holding_correct"])
    write_csv(OUT_DIR / "special_early_exit_correct.csv", special["early_exit_correct"])

    safe_print(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="top5", choices=("top1", "top2", "top5", "top7"))
    args = ap.parse_args()
    run(tier=args.tier)


if __name__ == "__main__":
    main()
