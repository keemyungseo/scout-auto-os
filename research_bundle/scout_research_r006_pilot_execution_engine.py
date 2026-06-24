"""
Scout Research R006 — Pilot Execution Engine

A6 frozen search picks. Execution-rule optimization only.
Simulates entry / TP / SL / hold / trailing on 5m forward paths.

Usage:
  python scout_research_r006_pilot_execution_engine.py
  python scout_research_r006_pilot_execution_engine.py --tier top5
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import scout_phase19_winner_ranking_dna as p19
from season2_p37_scout_decision_hierarchy import write_csv
from season2_universe_blind_test import ohlcv

import scout_phase16_human_blind_test as p16
from scout_research_r005_execution_statistics import (
    load_b001_scan_row,
    loo_a6_scan_rows,
    parse_kst,
    safe_print,
)
from season2_scout_mission import mission_summary_lines

OUT_DIR = Path("logs") / "research_r006_pilot_execution"

ENTRY_DELAYS = (
    ("immediate", 0),
    ("30s", 0),
    ("1m", 0),
    ("2m", 0),
    ("5m", 1),
    ("10m", 2),
    ("15m", 3),
)
TP_LEVELS = (3, 5, 7, 10, 12, 15, 18, 20)
SL_LEVELS = (2, 3, 4, 5, 6, 8, 10, 12)
HOLD_MINUTES = (30, 60, 90, 120, 150, 180, 240)
TRAIL_GAPS = (2, 3, 4, 5, 6)
GRID_TP = (3, 5, 7, 10, 12, 15)
GRID_SL = (2, 3, 4, 5, 6, 8, 10)
GRID_HOLD = (60, 90, 105, 120, 150, 180, 240)
GRID_TRAIL = (2, 3, 4, 5, 6)


@dataclass
class Bar:
    t_ms: int
    o: float
    h: float
    l: float
    c: float


@dataclass
class SimResult:
    return_pct: float
    exit_reason: str
    bars_held: int
    mfe_pct: float
    mae_pct: float


def load_forward_bars(symbol: str, scan_kst: str) -> list[Bar]:
    p16.CACHE_DIR = p19.CACHE_DIR
    start_ms = int(parse_kst(scan_kst).timestamp() * 1000)
    try:
        fwd = p16.fetch_forward_5m(symbol, start_ms, 48)
    except Exception:
        return []
    out: list[Bar] = []
    for k in fwd[:48]:
        o, h, l, c, _ = ohlcv(k)
        out.append(Bar(int(k[0]), float(o), float(h), float(l), float(c)))
    return out


def unique_top_picks(tier: str) -> list[dict]:
    scan_rows = loo_a6_scan_rows()
    b001 = load_b001_scan_row()
    if b001:
        scan_rows.append(b001)
    seen: set[tuple[str, str]] = set()
    picks: list[dict] = []
    for sr in scan_rows:
        for r in sr[tier]:
            key = (sr["scan_kst"], r["symbol"])
            if key in seen:
                continue
            seen.add(key)
            picks.append({
                "search_time": sr["scan_kst"],
                "symbol": r["symbol"],
                "a6_score": round(r["a6"], 4),
                "state_1h": r["states"].get("1h", ""),
                "state_2h": r["states"].get("2h", ""),
            })
    return picks


def mfe_mae(bars: list[Bar], entry_i: int, entry_px: float) -> tuple[float, float]:
    if entry_px <= 0 or entry_i >= len(bars):
        return 0.0, 0.0
    max_h = entry_px
    min_l = entry_px
    for b in bars[entry_i:]:
        max_h = max(max_h, b.h)
        min_l = min(min_l, b.l)
    mfe = (max_h - entry_px) / entry_px * 100
    mae = (entry_px - min_l) / entry_px * 100
    return mfe, mae


def simulate(
    bars: list[Bar],
    entry_i: int,
    *,
    tp_pct: float | None = None,
    sl_pct: float | None = None,
    hold_bars: int | None = None,
    trail_gap_pct: float | None = None,
    trail_arm_pct: float = 0.0,
) -> SimResult | None:
    if entry_i >= len(bars):
        return None
    entry_px = bars[entry_i].o
    if entry_px <= 0:
        return None

    mfe, mae = mfe_mae(bars, entry_i, entry_px)
    peak = entry_px
    armed = trail_arm_pct <= 0
    end_i = len(bars) - 1
    if hold_bars is not None:
        end_i = min(entry_i + hold_bars - 1, len(bars) - 1)

    for j in range(entry_i, end_i + 1):
        b = bars[j]
        peak = max(peak, b.h)
        if not armed and (peak - entry_px) / entry_px * 100 >= trail_arm_pct:
            armed = True

        tp_px = entry_px * (1 + tp_pct / 100) if tp_pct is not None else None
        sl_px = entry_px * (1 - sl_pct / 100) if sl_pct is not None else None
        trail_px = peak * (1 - trail_gap_pct / 100) if trail_gap_pct is not None and armed else None

        if sl_px is not None and b.l <= sl_px:
            return SimResult(-sl_pct, "sl", j - entry_i + 1, mfe, mae)
        if tp_px is not None and b.h >= tp_px:
            return SimResult(tp_pct, "tp", j - entry_i + 1, mfe, mae)
        if trail_px is not None and b.l <= trail_px:
            ret = (trail_px - entry_px) / entry_px * 100
            return SimResult(ret, "trail", j - entry_i + 1, mfe, mae)

    close_px = bars[end_i].c
    ret = (close_px - entry_px) / entry_px * 100
    reason = "time" if hold_bars is not None else "eod"
    return SimResult(ret, reason, end_i - entry_i + 1, mfe, mae)


def metrics(returns: list[float]) -> dict:
    if not returns:
        return {}
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    ev = statistics.mean(returns)
    win_rate = len(wins) / len(returns) * 100
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 99.0
    equity = 100.0
    peak_eq = 100.0
    max_dd = 0.0
    for r in returns:
        equity *= 1 + r / 100
        peak_eq = max(peak_eq, equity)
        max_dd = max(max_dd, (peak_eq - equity) / peak_eq * 100)
    return {
        "n": len(returns),
        "ev": round(ev, 4),
        "win_rate": round(win_rate, 2),
        "median": round(statistics.median(returns), 4),
        "avg_return": round(ev, 4),
        "profit_factor": round(pf, 4),
        "expectancy": round(ev, 4),
        "max_dd": round(max_dd, 4),
    }


def load_trade_set(tier: str) -> list[tuple[dict, list[Bar]]]:
    picks = unique_top_picks(tier)
    out: list[tuple[dict, list[Bar]]] = []
    for i, p in enumerate(picks, 1):
        bars = load_forward_bars(p["symbol"], p["search_time"])
        if bars:
            out.append((p, bars))
        if i % 200 == 0:
            safe_print(f"  loaded paths {i}/{len(picks)}")
    return out


def report_entry_delay(trades: list[tuple[dict, list[Bar]]]) -> list[dict]:
    rows: list[dict] = []
    for label, entry_i in ENTRY_DELAYS:
        rets: list[float] = []
        finals: list[float] = []
        mfes: list[float] = []
        opp_loss: list[float] = []
        for _, bars in trades:
            r = simulate(bars, entry_i)
            if not r:
                continue
            rets.append(r.return_pct)
            finals.append(r.return_pct)
            mfes.append(r.mfe_pct)
            if label != "immediate":
                imm_r = simulate(bars, 0)
                if imm_r:
                    opp_loss.append(max(0, imm_r.mfe_pct - r.mfe_pct))
        m = metrics(rets)
        row = {
            "entry": label,
            "entry_bar": entry_i,
            **m,
            "avg_max": round(statistics.mean(mfes), 4) if mfes else 0,
            "avg_final": round(statistics.mean(finals), 4) if finals else 0,
            "opportunity_loss_avg": round(statistics.mean(opp_loss), 4) if opp_loss else 0,
        }
        rows.append(row)
    return rows


def report_fixed_tp(trades: list[tuple[dict, list[Bar]]], entry_i: int = 0) -> list[dict]:
    rows: list[dict] = []
    for tp in TP_LEVELS:
        rets: list[float] = []
        for _, bars in trades:
            r = simulate(bars, entry_i, tp_pct=tp)
            if r:
                rets.append(r.return_pct)
        m = metrics(rets)
        rows.append({"tp_pct": tp, **m})
    return rows


def report_time_exit(trades: list[tuple[dict, list[Bar]]], entry_i: int = 0) -> list[dict]:
    rows: list[dict] = []
    for mins in HOLD_MINUTES:
        hold_bars = max(1, mins // 5)
        rets: list[float] = []
        for _, bars in trades:
            r = simulate(bars, entry_i, hold_bars=hold_bars)
            if r:
                rets.append(r.return_pct)
        m = metrics(rets)
        rows.append({"hold_minutes": mins, "hold_bars": hold_bars, **m})
    return rows


def report_stop_loss(trades: list[tuple[dict, list[Bar]]], entry_i: int = 0) -> list[dict]:
    rows: list[dict] = []
    for sl in SL_LEVELS:
        rets: list[float] = []
        for _, bars in trades:
            r = simulate(bars, entry_i, sl_pct=sl)
            if r:
                rets.append(r.return_pct)
        m = metrics(rets)
        rows.append({"sl_pct": sl, **m})
    return rows


def report_trailing(trades: list[tuple[dict, list[Bar]]], entry_i: int = 0) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for gap in TRAIL_GAPS:
        rets: list[float] = []
        for _, bars in trades:
            r = simulate(bars, entry_i, trail_gap_pct=gap, trail_arm_pct=0)
            if r:
                rets.append(r.return_pct)
        m = metrics(rets)
        rows.append({"trail_gap_pct": gap, **m})

    best_tp_row = max(report_fixed_tp(trades, entry_i), key=lambda x: x["ev"])
    tp_rets: list[float] = []
    for _, bars in trades:
        r = simulate(bars, entry_i, tp_pct=best_tp_row["tp_pct"])
        if r:
            tp_rets.append(r.return_pct)
    tp_m = metrics(tp_rets)
    compare = {
        "best_fixed_tp": best_tp_row["tp_pct"],
        "fixed_tp_ev": tp_m["ev"],
        "fixed_tp_win_rate": tp_m["win_rate"],
        "fixed_tp_pf": tp_m["profit_factor"],
        "best_trail_gap": max(rows, key=lambda x: x["ev"])["trail_gap_pct"],
        "best_trail_ev": max(rows, key=lambda x: x["ev"])["ev"],
    }
    return rows, compare


def grid_search(trades: list[tuple[dict, list[Bar]]], entry_i: int = 0) -> list[dict]:
    strategies: list[dict] = []

    for tp in GRID_TP:
        for sl in GRID_SL:
            rets: list[float] = []
            for _, bars in trades:
                r = simulate(bars, entry_i, tp_pct=tp, sl_pct=sl)
                if r:
                    rets.append(r.return_pct)
            m = metrics(rets)
            score = m["ev"] * min(m["profit_factor"], 5) - m["max_dd"] * 0.05
            strategies.append({
                "kind": "tp_x_sl",
                "tp_pct": tp, "sl_pct": sl,
                "hold_minutes": None, "trail_gap_pct": None,
                "score": round(score, 4),
                **m,
            })

    for tp in GRID_TP:
        for hold in GRID_HOLD:
            hold_bars = max(1, hold // 5)
            rets = []
            for _, bars in trades:
                r = simulate(bars, entry_i, tp_pct=tp, hold_bars=hold_bars)
                if r:
                    rets.append(r.return_pct)
            m = metrics(rets)
            score = m["ev"] * min(m["profit_factor"], 5) - m["max_dd"] * 0.05
            strategies.append({
                "kind": "tp_x_hold",
                "tp_pct": tp, "sl_pct": None,
                "hold_minutes": hold, "trail_gap_pct": None,
                "score": round(score, 4),
                **m,
            })

    for hold in GRID_HOLD:
        hold_bars = max(1, hold // 5)
        for trail in GRID_TRAIL:
            rets = []
            for _, bars in trades:
                r = simulate(bars, entry_i, hold_bars=hold_bars, trail_gap_pct=trail, trail_arm_pct=3)
                if r:
                    rets.append(r.return_pct)
            m = metrics(rets)
            score = m["ev"] * min(m["profit_factor"], 5) - m["max_dd"] * 0.05
            strategies.append({
                "kind": "hold_x_trail",
                "tp_pct": None, "sl_pct": None,
                "hold_minutes": hold, "trail_gap_pct": trail,
                "score": round(score, 4),
                **m,
            })

    for tp in GRID_TP:
        for sl in GRID_SL:
            for hold in GRID_HOLD:
                hold_bars = max(1, hold // 5)
                rets = []
                for _, bars in trades:
                    r = simulate(bars, entry_i, tp_pct=tp, sl_pct=sl, hold_bars=hold_bars)
                    if r:
                        rets.append(r.return_pct)
                m = metrics(rets)
                score = m["ev"] * min(m["profit_factor"], 5) - m["max_dd"] * 0.05
                strategies.append({
                    "kind": "tp_x_sl_x_hold",
                    "tp_pct": tp, "sl_pct": sl,
                    "hold_minutes": hold, "trail_gap_pct": None,
                    "score": round(score, 4),
                    **m,
                })

    strategies.sort(key=lambda x: (x["score"], x["profit_factor"]), reverse=True)
    return strategies


def build_auto_json(
    best: dict,
    entry_rows: list[dict],
    tier: str,
    n_scans: int,
) -> dict:
    best_entry = max(entry_rows, key=lambda x: x["ev"]) if entry_rows else {"entry": "immediate"}
    return {
        "version": "r006_pilot_v1",
        "search_formula": "A6_frozen",
        "search_tier": tier,
        "sample_trades": best.get("n", 0),
        "sample_scans": n_scans,
        "entry": best_entry["entry"],
        "take_profit": best.get("tp_pct"),
        "stop_loss": best.get("sl_pct"),
        "max_hold_minutes": best.get("hold_minutes"),
        "trailing_after_profit": 3.0 if best.get("trail_gap_pct") else None,
        "trailing_gap": best.get("trail_gap_pct"),
        "position_size": "100%",
        "expected_ev": best.get("ev"),
        "win_rate": best.get("win_rate"),
        "profit_factor": best.get("profit_factor"),
        "max_drawdown_pct": best.get("max_dd"),
        "strategy_kind": best.get("kind"),
        "median_return": best.get("median"),
    }


def format_report(title: str, rows: list[dict], keys: list[str]) -> str:
    lines = [title, "=" * 60]
    for r in rows:
        parts = [f"{k}={r.get(k)}" for k in keys if k in r]
        lines.append("  " + " | ".join(parts))
    return "\n".join(lines)


def run(tier: str = "top5") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p19.CACHE_DIR = Path("logs") / "phase19_winner_dna" / "kline_cache"
    p16.CACHE_DIR = p19.CACHE_DIR

    safe_print(f"R006 loading {tier} trade paths...")
    trades = load_trade_set(tier)
    n_trades = len(trades)
    n_scans = len({p["search_time"] for p, _ in trades})
    safe_print(f"R006 {n_trades} paths from {n_scans} scans")

    safe_print("R006 Report 1 — Entry Delay...")
    r1 = report_entry_delay(trades)

    safe_print("R006 Report 2 — Fixed TP...")
    r2 = report_fixed_tp(trades, entry_i=0)

    safe_print("R006 Report 3 — Time Exit...")
    r3 = report_time_exit(trades, entry_i=0)

    safe_print("R006 Report 4 — Stop Loss...")
    r4 = report_stop_loss(trades, entry_i=0)

    safe_print("R006 Report 5 — Trailing...")
    r5, trail_cmp = report_trailing(trades, entry_i=0)

    safe_print("R006 Report 6 — Grid Search...")
    grid = grid_search(trades, entry_i=0)
    top20 = grid[:20]

    best = top20[0] if top20 else {}
    auto_json = build_auto_json(best, r1, tier, n_scans)
    (OUT_DIR / "pilot_execution_rule.json").write_text(
        json.dumps(auto_json, indent=2), encoding="utf-8",
    )

    header = [
        "############################################################",
        "SCOUT RESEARCH R006 — PILOT EXECUTION ENGINE",
        "############################################################",
        "",
        f"A6 frozen | tier={tier} | trades={n_trades} | scans={n_scans}",
        "Simulation: 5m OHLC forward | SL checked before TP (conservative)",
        "Entry delay <5m uses same 5m bar open (sub-5m data unavailable)",
        "",
    ]

    rep1 = format_report(
        "REPORT 1 — ENTRY DELAY",
        r1,
        ["entry", "ev", "win_rate", "avg_max", "avg_final", "opportunity_loss_avg"],
    )
    rep2 = format_report(
        "REPORT 2 — FIXED TP",
        r2,
        ["tp_pct", "win_rate", "avg_return", "median", "profit_factor", "expectancy"],
    )
    rep3 = format_report(
        "REPORT 3 — TIME EXIT",
        r3,
        ["hold_minutes", "avg_return", "median", "win_rate", "ev"],
    )
    rep4 = format_report(
        "REPORT 4 — STOP LOSS",
        r4,
        ["sl_pct", "win_rate", "ev", "max_dd", "profit_factor"],
    )
    rep5_lines = [
        "REPORT 5 — TRAILING STOP",
        "=" * 60,
        format_report("", r5, ["trail_gap_pct", "ev", "win_rate", "profit_factor", "max_dd"]),
        "",
        "Trailing vs Fixed TP:",
        f"  best_fixed_tp={trail_cmp['best_fixed_tp']}% ev={trail_cmp['fixed_tp_ev']}% "
        f"wr={trail_cmp['fixed_tp_win_rate']}% pf={trail_cmp['fixed_tp_pf']}",
        f"  best_trail_gap={trail_cmp['best_trail_gap']}% ev={trail_cmp['best_trail_ev']}%",
    ]
    rep5 = "\n".join(rep5_lines)

    rep6_lines = ["REPORT 6 — COMBINATION SEARCH (Top20)", "=" * 60]
    for i, s in enumerate(top20, 1):
        rep6_lines.append(
            f"  #{i} {s['kind']} tp={s.get('tp_pct')} sl={s.get('sl_pct')} "
            f"hold={s.get('hold_minutes')} trail={s.get('trail_gap_pct')} | "
            f"ev={s['ev']}% wr={s['win_rate']}% pf={s['profit_factor']} dd={s['max_dd']}%"
        )
    rep6 = "\n".join(rep6_lines)

    rep7 = [
        "REPORT 7 — AUTO TRADING JSON",
        "=" * 60,
        json.dumps(auto_json, indent=2),
    ]

    master = "\n".join(header + [rep1, "", rep2, "", rep3, "", rep4, "", rep5, "", rep6, "", "\n".join(rep7)])
    master += "\n\n" + "\n".join(mission_summary_lines())

    (OUT_DIR / "research_r006_report.txt").write_text(master, encoding="utf-8")
    write_csv(OUT_DIR / "report_01_entry_delay.csv", r1)
    write_csv(OUT_DIR / "report_02_fixed_tp.csv", r2)
    write_csv(OUT_DIR / "report_03_time_exit.csv", r3)
    write_csv(OUT_DIR / "report_04_stop_loss.csv", r4)
    write_csv(OUT_DIR / "report_05_trailing.csv", r5)
    write_csv(OUT_DIR / "report_06_grid_top20.csv", top20)
    write_csv(OUT_DIR / "report_06_grid_all.csv", grid)

    safe_print(master)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="top5", choices=("top2", "top5", "top7"))
    args = ap.parse_args()
    run(tier=args.tier)


if __name__ == "__main__":
    main()
