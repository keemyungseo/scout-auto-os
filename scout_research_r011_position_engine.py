"""
Scout Research R011 — Position Engine (Portfolio Simulation)

Fixed: A6 search + Immediate entry + R009-B dynamic exit.
Optimizes 24hr total daily return, not single-trade EV.

Usage:
  python scout_research_r011_position_engine.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scout_phase19_winner_ranking_dna as p19
from season2_p37_scout_decision_hierarchy import write_csv

import scout_phase16_human_blind_test as p16
from scout_research_r005_execution_statistics import load_b001_scan_row, loo_a6_scan_rows, parse_kst, safe_print
from scout_research_r006_pilot_execution_engine import Bar, load_forward_bars
from scout_research_r010_dynamic_entry_engine import ENTRY_A, simulate_entry
from season2_scout_mission import mission_summary_lines

OUT_DIR = Path("logs") / "research_r011_position_engine"
KST = timezone(timedelta(hours=9))
FEE_PCT = 0.05
REENTRY_THRESH = 3.0


@dataclass
class CatalogTrade:
    scan_kst: str
    symbol: str
    a6_score: float
    a6_rank: int
    return_pct: float
    hold_minutes: int
    entry_dt: datetime
    exit_dt: datetime
    bars: list[Bar]
    entry_px: float


@dataclass
class OpenPos:
    trade: CatalogTrade
    weight: float
    entry_dt: datetime


def build_catalog() -> dict[str, list[CatalogTrade]]:
    scan_rows = loo_a6_scan_rows()
    b001 = load_b001_scan_row()
    if b001:
        scan_rows.append(b001)
    catalog: dict[str, list[CatalogTrade]] = {}
    for sr in scan_rows:
        scan = sr["scan_kst"]
        ranked = sorted(sr.get("top7", []), key=lambda x: x.get("a6", 0), reverse=True)
        trades: list[CatalogTrade] = []
        for rank, r in enumerate(ranked, 1):
            meta = {
                "symbol": r["symbol"],
                "search_time": scan,
                "a6_score": round(r["a6"], 4),
            }
            bars = load_forward_bars(r["symbol"], scan)
            if not bars:
                continue
            res = simulate_entry(bars, meta, "A_immediate_100", list(ENTRY_A))
            if not res:
                continue
            entry_dt = parse_kst(scan)
            exit_dt = entry_dt + timedelta(minutes=res.hold_minutes)
            trades.append(CatalogTrade(
                scan_kst=scan,
                symbol=r["symbol"],
                a6_score=res.a6_score,
                a6_rank=rank,
                return_pct=res.return_pct,
                hold_minutes=res.hold_minutes,
                entry_dt=entry_dt,
                exit_dt=exit_dt,
                bars=bars,
                entry_px=bars[0].o,
            ))
        if trades:
            catalog[scan] = trades
    return catalog


def partial_return(trade: CatalogTrade, at_dt: datetime) -> float:
    mins = int((at_dt - trade.entry_dt).total_seconds() / 60)
    if mins < 0:
        return 0.0
    bar_i = min(len(trade.bars) - 1, max(0, mins // 5))
    px = trade.bars[bar_i].c
    if trade.entry_px <= 0:
        return 0.0
    return (px - trade.entry_px) / trade.entry_px * 100


def alloc_weights(scheme: str, n_slots: int, candidates: list[CatalogTrade]) -> list[float]:
    if scheme == "equal":
        return [1.0 / n_slots] * n_slots
    if scheme == "score_50_30_20":
        base = [0.5, 0.3, 0.2, 0.1, 0.1, 0.1][:n_slots]
        s = sum(base)
        return [b / s for b in base]
    if scheme == "score_40_35_25":
        base = [0.4, 0.35, 0.25, 0.15, 0.1, 0.1][:n_slots]
        return [b / sum(base) for b in base]
    if scheme == "score_40_30_20_10":
        base = [0.4, 0.3, 0.2, 0.1, 0.05, 0.05][:n_slots]
        return [b / sum(base) for b in base]
    if scheme == "ev_proportional":
        scores = [max(c.a6_score, 0.1) for c in candidates[:n_slots]]
        s = sum(scores)
        return [x / s for x in scores] if s else [1 / n_slots] * n_slots
    if scheme == "kelly_lite":
        raw = [min(c.a6_score / 10.0, 0.5) for c in candidates[:n_slots]]
        s = sum(raw) or 1.0
        return [min(0.4, x / s) for x in raw]
    if scheme == "risk_adjusted":
        raw = [max(c.a6_score - 3.0, 0.5) for c in candidates[:n_slots]]
        s = sum(raw)
        return [x / s for x in raw] if s else [1 / n_slots] * n_slots
    return [1.0 / n_slots] * n_slots


def simulate_portfolio(
    catalog: dict[str, list[CatalogTrade]],
    *,
    slots: int = 3,
    alloc_scheme: str = "equal",
    replacement: str = "keep",
    replace_threshold: float = 0.0,
    cooldown_min: int = 0,
    exposure: float = 1.0,
) -> dict:
    scans = sorted(catalog.keys())
    open_pos: list[OpenPos] = []
    daily_pnl: dict[str, float] = defaultdict(float)
    trade_count = 0
    replacements = 0
    fees = 0.0
    realized_rets: list[float] = []
    hold_times: list[int] = []
    last_exit: dict[str, datetime] = {}
    equity = 100.0
    peak_eq = 100.0
    mdd = 0.0
    util_samples: list[float] = []

    def realize(ret_pct: float, weight: float, day: str, fee: float = 0.0) -> None:
        nonlocal equity, peak_eq, mdd, fees
        contrib = weight * exposure * ret_pct / 100
        fee_cost = fee * weight * exposure
        fees += fee_cost
        daily_pnl[day] += (contrib - fee_cost) * 100
        equity *= 1 + contrib - fee_cost
        peak_eq = max(peak_eq, equity)
        mdd = max(mdd, (peak_eq - equity) / peak_eq * 100 if peak_eq else 0)
        realized_rets.append(ret_pct)

    def close_position(pos: OpenPos, at_dt: datetime, reason: str, partial: bool) -> None:
        nonlocal trade_count
        if partial:
            ret = partial_return(pos.trade, at_dt)
            hold_times.append(int((at_dt - pos.entry_dt).total_seconds() / 60))
        else:
            ret = pos.trade.return_pct
            hold_times.append(pos.trade.hold_minutes)
        day = at_dt.strftime("%Y-%m-%d")
        fee = FEE_PCT if reason == "replace" else 0.0
        realize(ret, pos.weight, day, fee)
        last_exit[pos.trade.symbol] = at_dt
        trade_count += 1

    for scan_kst in scans:
        now = parse_kst(scan_kst)
        day = now.strftime("%Y-%m-%d")

        closed: list[OpenPos] = []
        still: list[OpenPos] = []
        for p in open_pos:
            if p.trade.exit_dt <= now:
                close_position(p, p.trade.exit_dt, "natural", partial=False)
                closed.append(p)
            else:
                still.append(p)
        open_pos = still
        util_samples.append(sum(p.weight for p in open_pos) * exposure)

        cands = [c for c in catalog[scan_kst] if c.entry_dt == now]
        weights = alloc_weights(alloc_scheme, slots, cands)

        for idx, cand in enumerate(cands):
            if idx >= slots:
                break
            if cand.symbol in {p.trade.symbol for p in open_pos}:
                continue
            if cand.symbol in last_exit:
                cd = (now - last_exit[cand.symbol]).total_seconds() / 60
                if cd < cooldown_min:
                    continue
            if cand.return_pct < -REENTRY_THRESH and cand.symbol in last_exit:
                continue

            w = weights[idx] if idx < len(weights) else weights[-1]

            if len(open_pos) >= slots:
                if replacement == "keep":
                    continue
                victim_i = 0
                if replacement == "replace_oldest":
                    victim_i = min(range(len(open_pos)), key=lambda i: open_pos[i].entry_dt)
                elif replacement == "replace_weakest":
                    victim_i = min(range(len(open_pos)), key=lambda i: open_pos[i].trade.a6_score)
                elif replacement == "replace_low_ev":
                    victim_i = min(range(len(open_pos)), key=lambda i: open_pos[i].trade.a6_score)
                elif replacement == "replace_threshold":
                    weakest = min(open_pos, key=lambda p: p.trade.a6_score)
                    if cand.a6_score - weakest.trade.a6_score < replace_threshold:
                        continue
                    victim_i = open_pos.index(weakest)
                else:
                    continue
                close_position(open_pos[victim_i], now, "replace", partial=True)
                open_pos.pop(victim_i)
                replacements += 1

            if len(open_pos) < slots:
                open_pos.append(OpenPos(trade=cand, weight=w, entry_dt=now))

    for p in open_pos:
        close_position(p, p.trade.exit_dt, "natural", partial=False)

    daily_vals = list(daily_pnl.values())
    wins = [r for r in realized_rets if r > 0]
    losses = [r for r in realized_rets if r <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    pf = gw / gl if gl > 0 else 99.0
    total_days = len({s[:10] for s in catalog.keys()}) or 1

    return {
        "slots": slots,
        "alloc_scheme": alloc_scheme,
        "replacement": replacement,
        "replace_threshold": replace_threshold,
        "cooldown_min": cooldown_min,
        "exposure": exposure,
        "trade_count": trade_count,
        "replacements": replacements,
        "turnover": replacements,
        "fee_cost_pct": round(fees * 100, 4),
        "daily_return_mean": round(statistics.mean(daily_vals), 4) if daily_vals else 0,
        "total_daily_return": round(sum(daily_vals) / total_days, 4) if daily_vals else 0,
        "capital_efficiency": round(sum(daily_vals) / max(1, statistics.mean(util_samples) * 100), 4) if util_samples else 0,
        "capital_utilization": round(statistics.mean(util_samples) * exposure * 100, 4) if util_samples else 0,
        "ev": round(statistics.mean(realized_rets), 4) if realized_rets else 0,
        "win_rate": round(len(wins) / len(realized_rets) * 100, 2) if realized_rets else 0,
        "profit_factor": round(pf, 4),
        "max_dd": round(mdd, 4),
        "avg_hold_min": round(statistics.mean(hold_times), 1) if hold_times else 0,
        "risk_adjusted_return": round(
            (statistics.mean(daily_vals) / statistics.stdev(daily_vals)) if len(daily_vals) > 1 and statistics.stdev(daily_vals) else 0,
            4,
        ),
        "final_equity": round(equity, 4),
    }


def portfolio_score(row: dict) -> float:
    return (
        row.get("total_daily_return", 0) * 0.50
        + row.get("daily_return_mean", 0) * 0.20
        + row.get("capital_efficiency", 0) * 0.15
        + row.get("risk_adjusted_return", 0) * 0.10
        - row.get("max_dd", 0) * 0.05
        - row.get("fee_cost_pct", 0) * 0.05
    )


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p19.CACHE_DIR = Path("logs") / "phase19_winner_dna" / "kline_cache"
    p16.CACHE_DIR = p19.CACHE_DIR

    safe_print("R011 building trade catalog (A6 + immediate + R009-B exit)...")
    catalog = build_catalog()
    n_scans = len(catalog)
    safe_print(f"R011 {n_scans} scans in catalog")

    safe_print("R011 Track A — slot count...")
    track_a = []
    for slots in range(1, 7):
        row = simulate_portfolio(catalog, slots=slots, exposure=1.0)
        row["track"] = "A_slots"
        row["score"] = round(portfolio_score(row), 4)
        track_a.append(row)

    safe_print("R011 Track B — capital allocation...")
    track_b = []
    schemes = (
        ("equal", "equal"),
        ("equal_4", "equal"),
        ("equal_5", "equal"),
        ("score_50_30_20", "score_50_30_20"),
        ("score_40_35_25", "score_40_35_25"),
        ("score_40_30_20_10", "score_40_30_20_10"),
        ("ev_proportional", "ev_proportional"),
        ("kelly_lite", "kelly_lite"),
        ("risk_adjusted", "risk_adjusted"),
    )
    slot_map = {"equal": 3, "equal_4": 4, "equal_5": 5,
                "score_50_30_20": 3, "score_40_35_25": 3, "score_40_30_20_10": 4,
                "ev_proportional": 4, "kelly_lite": 3, "risk_adjusted": 4}
    for label, scheme in schemes:
        sl = slot_map[label]
        row = simulate_portfolio(catalog, slots=sl, alloc_scheme=scheme, exposure=1.0)
        row["track"] = "B_allocation"
        row["label"] = label
        row["score"] = round(portfolio_score(row), 4)
        track_b.append(row)

    safe_print("R011 Track C — replacement engine...")
    track_c = []
    for repl in ("keep", "replace_weakest", "replace_oldest", "replace_low_ev"):
        row = simulate_portfolio(catalog, slots=4, replacement=repl, exposure=1.0)
        row["track"] = "C_replacement"
        row["label"] = repl
        row["score"] = round(portfolio_score(row), 4)
        track_c.append(row)
    for thr in (3, 5, 8, 10, 15):
        row = simulate_portfolio(
            catalog, slots=4, replacement="replace_threshold",
            replace_threshold=thr, exposure=1.0,
        )
        row["track"] = "C_replacement"
        row["label"] = f"threshold_{thr}"
        row["score"] = round(portfolio_score(row), 4)
        track_c.append(row)

    safe_print("R011 Track D — reentry...")
    track_d = []
    for cd in (0, 15, 30, 60, 120):
        row = simulate_portfolio(catalog, slots=3, cooldown_min=cd, exposure=1.0)
        row["track"] = "D_reentry"
        row["label"] = f"cooldown_{cd}m"
        row["score"] = round(portfolio_score(row), 4)
        track_d.append(row)

    safe_print("R011 Track E — capital exposure...")
    track_e = []
    for exp in (1.0, 0.8, 0.6, 0.4):
        row = simulate_portfolio(catalog, slots=3, exposure=exp)
        row["track"] = "E_exposure"
        row["label"] = f"exposure_{int(exp*100)}pct"
        row["score"] = round(portfolio_score(row), 4)
        track_e.append(row)

    best_a = max(track_a, key=lambda x: x["total_daily_return"])
    best_b = max(track_b, key=lambda x: x["total_daily_return"])
    best_c = max(track_c, key=lambda x: x["total_daily_return"])
    best_d = max(track_d, key=lambda x: x["total_daily_return"])
    best_e = max(track_e, key=lambda x: x["total_daily_return"])

    opt_slots = int(best_a["slots"])
    opt_alloc = best_b["alloc_scheme"] if opt_slots > 1 else "equal"
    opt_alloc_label = best_b.get("label", "equal") if opt_slots > 1 else "n/a_single_slot"
    opt_cd = int(best_d["cooldown_min"])
    opt_exp = float(best_e["exposure"])
    combined = simulate_portfolio(
        catalog,
        slots=opt_slots,
        alloc_scheme=opt_alloc,
        replacement=str(best_c.get("replacement", "keep")),
        replace_threshold=float(best_c.get("replace_threshold", 0)),
        cooldown_min=opt_cd,
        exposure=opt_exp,
    )

    pilot = {
        "version": "scout_long_pilot_v1",
        "search": {"formula": "A6_frozen", "tier": "top7_per_scan"},
        "entry": {"method": "immediate_100pct", "source": "R010"},
        "exit": {"engine": "R009_dynamic_efr_state", "source": "R009"},
        "position": {
            "slots": opt_slots,
            "allocation": opt_alloc,
            "allocation_label": opt_alloc_label,
            "replacement": best_c.get("replacement", "keep"),
            "replacement_label": best_c.get("label", "keep"),
            "replace_threshold": best_c.get("replace_threshold", 0),
            "reentry_cooldown_min": opt_cd,
            "capital_exposure": opt_exp,
        },
        "portfolio_metrics": {
            "total_daily_return": combined.get("total_daily_return"),
            "daily_return_mean": combined.get("daily_return_mean"),
            "max_dd": combined.get("max_dd"),
            "profit_factor": combined.get("profit_factor"),
            "win_rate": combined.get("win_rate"),
            "capital_utilization": combined.get("capital_utilization"),
            "trade_count": combined.get("trade_count"),
            "risk_adjusted_return": combined.get("risk_adjusted_return"),
        },
        "track_winners": {
            "A_slots": {k: best_a[k] for k in ("slots", "total_daily_return", "score")},
            "B_allocation": {k: best_b[k] for k in ("label", "alloc_scheme", "total_daily_return", "score")},
            "C_replacement": {k: best_c[k] for k in ("label", "replacement", "total_daily_return", "score")},
            "D_reentry": {k: best_d[k] for k in ("label", "cooldown_min", "total_daily_return", "score")},
            "E_exposure": {k: best_e[k] for k in ("label", "exposure", "total_daily_return", "score")},
            "combined_config": combined,
        },
        "compatible_with": ["R012_slots", "R013_portfolio", "R014_reverse", "R015_review"],
        "mode": "paper_trading_ready",
    }
    (OUT_DIR / "recommended_position_engine.json").write_text(json.dumps(pilot, indent=2), encoding="utf-8")

    lines = [
        "############################################################",
        "SCOUT RESEARCH R011 — POSITION ENGINE",
        "############################################################",
        "",
        f"Fixed: A6 + Immediate Entry + R009-B Exit | scans={n_scans}",
        "Optimize: TOTAL DAILY RETURN (not single-trade EV)",
        "",
        "=" * 62,
        "TRACK A — Slot Count (top by total_daily_return)",
        "=" * 62,
    ]
    for r in sorted(track_a, key=lambda x: x["total_daily_return"], reverse=True):
        lines.append(
            f"  Slot{r['slots']}: daily={r['total_daily_return']}% trades={r['trade_count']} "
            f"util={r['capital_utilization']}% pf={r['profit_factor']} mdd={r['max_dd']}%"
        )

    lines.extend(["", "=" * 62, "TRACK B — Capital Allocation", "=" * 62])
    for r in sorted(track_b, key=lambda x: x["total_daily_return"], reverse=True)[:8]:
        lines.append(
            f"  {r['label']}: daily={r['total_daily_return']}% mdd={r['max_dd']}% "
            f"util={r['capital_utilization']}% pf={r['profit_factor']}"
        )

    lines.extend(["", "=" * 62, "TRACK C — Replacement", "=" * 62])
    for r in sorted(track_c, key=lambda x: x["total_daily_return"], reverse=True)[:6]:
        lines.append(
            f"  {r['label']}: daily={r['total_daily_return']}% turnover={r['turnover']} "
            f"fees={r['fee_cost_pct']}% pf={r['profit_factor']}"
        )

    lines.extend(["", "=" * 62, "TRACK D — Reentry Cooldown", "=" * 62])
    for r in track_d:
        lines.append(f"  {r['label']}: daily={r['total_daily_return']}% trades={r['trade_count']} pf={r['profit_factor']}")

    lines.extend(["", "=" * 62, "TRACK E — Capital Exposure", "=" * 62])
    for r in track_e:
        lines.append(
            f"  {r['label']}: daily={r['total_daily_return']}% mdd={r['max_dd']}% "
            f"risk_adj={r['risk_adjusted_return']}"
        )

    lines.extend(["", "=" * 62, "SCOUT LONG PILOT ENGINE v1", "=" * 62, json.dumps(pilot, indent=2)])
    report = "\n".join(lines) + "\n\n" + "\n".join(mission_summary_lines())
    (OUT_DIR / "position_engine_report.txt").write_text(report, encoding="utf-8")

    write_csv(OUT_DIR / "slot_simulation.csv", track_a)
    write_csv(OUT_DIR / "allocation_simulation.csv", track_b)
    write_csv(OUT_DIR / "replacement_engine.csv", track_c)
    write_csv(OUT_DIR / "reentry_engine.csv", track_d)
    write_csv(OUT_DIR / "capital_exposure.csv", track_e)

    safe_print(report)


if __name__ == "__main__":
    run()
