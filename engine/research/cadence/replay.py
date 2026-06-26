"""Cadence portfolio replay simulation."""

from __future__ import annotations

import statistics
from datetime import datetime

from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.portfolio.slot_manager import SlotBook
from scout_auto_os.engine.research.cadence.returns import return_between_times
from scout_auto_os.engine.research.cadence.schedule import build_cadence_schedule


def _parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def replay_cadence(
    interval_min: int,
    base_scans: list[str],
    by_scan: dict,
    fwd: dict,
    engine: PortfolioEngine,
) -> dict:
    schedule = build_cadence_schedule(base_scans, interval_min)
    if not schedule:
        return {"interval_min": interval_min, "error": "empty_schedule"}

    engine.book = SlotBook()
    open_positions: list[dict] = []
    portfolio_log: list[dict] = []
    slot_history: list[dict] = []
    replacement_log: list[dict] = []
    equity_curve: list[dict] = []
    turnover_events: list[dict] = []

    realized: list[float] = []
    long_rets: list[float] = []
    short_rets: list[float] = []
    hold_mins: list[float] = []
    slot_occ: list[float] = []
    pass_long_total = 0
    pass_short_total = 0

    new_entry_count = 0
    replace_count = 0
    reselect_count = 0
    dup_direction_count = 0
    symbol_entry_history: dict[tuple[str, str], int] = {}

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0

    scan_times = [s[0] for s in schedule]
    days = len({s[:10] for s in scan_times}) or 1

    for idx, (scan_kst, feature_scan) in enumerate(schedule):
        hold_until = scan_times[idx + 1] if idx + 1 < len(scan_times) else scan_kst
        rows = [{"symbol": r["symbol"], "features": r["features"]} for r in by_scan.get(feature_scan, [])]

        closed: list[dict] = []
        still_open: list[dict] = []
        for pos in open_positions:
            if pos["hold_until_scan"] <= scan_kst:
                klines = fwd.get((pos["feature_scan"], pos["symbol"]))
                r = return_between_times(
                    klines or [],
                    pos["feature_scan"],
                    pos["entry_scan"],
                    scan_kst,
                    pos["direction"],
                )
                realized.append(r)
                if pos["direction"] == "long":
                    long_rets.append(r)
                else:
                    short_rets.append(r)
                cumulative += r
                peak = max(peak, cumulative)
                max_dd = min(max_dd, cumulative - peak)
                hm = (_parse(scan_kst) - _parse(pos["entry_scan"])).total_seconds() / 60
                hold_mins.append(hm)
                portfolio_log.append({
                    "interval_min": interval_min,
                    "event": "exit",
                    "scan_time_kst": scan_kst,
                    "symbol": pos["symbol"],
                    "direction": pos["direction"],
                    "return_pct": r,
                    "hold_minutes": round(hm, 2),
                })
                closed.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        result = engine.process_scan(rows, scan_kst, hold_until_scan=hold_until)

        pass_long_total += result["long_pass_count"]
        pass_short_total += result["short_pass_count"]

        for rep in result["replacements"]:
            replacement_log.append({**rep, "interval_min": interval_min})
            replace_count += 1
            turnover_events.append({"type": "replace", "interval_min": interval_min, **rep})

        for entry in result["new_entries"]:
            sym = entry["symbol"]
            direction = entry["direction"]
            key = (direction, sym)
            symbol_entry_history[key] = symbol_entry_history.get(key, 0) + 1
            if symbol_entry_history[key] > 1:
                reselect_count += 1
            if entry.get("action") == "enter":
                new_entry_count += 1
            else:
                replace_count += 1

            dup = sum(1 for p in open_positions if p["symbol"] == sym and p["direction"] == direction)
            if dup:
                dup_direction_count += 1

            open_positions.append({
                "symbol": sym,
                "direction": direction,
                "entry_scan": scan_kst,
                "feature_scan": feature_scan,
                "hold_until_scan": hold_until,
                "entry_score": entry["entry_score"],
                "live_pattern": entry.get("live_pattern"),
            })
            portfolio_log.append({
                "interval_min": interval_min,
                "event": entry.get("action", "enter"),
                "scan_time_kst": scan_kst,
                "feature_scan_kst": feature_scan,
                "symbol": sym,
                "direction": direction,
                "entry_score": entry["entry_score"],
            })
            turnover_events.append({"type": entry.get("action", "enter"), "interval_min": interval_min, "symbol": sym})

        n_long = len(result["long_selected"])
        n_short = len(result["short_selected"])
        slot_occ.append((n_long + n_short) / 6.0)

        for side, selected in (("long", result["long_selected"]), ("short", result["short_selected"])):
            for s in selected:
                slot_history.append({
                    "interval_min": interval_min,
                    "scan_time_kst": scan_kst,
                    "feature_scan_kst": feature_scan,
                    "direction": side,
                    "symbol": s["symbol"],
                    "entry_score": s["entry_score"],
                })

        equity_curve.append({
            "interval_min": interval_min,
            "scan_time_kst": scan_kst,
            "cumulative_return_pct": round(cumulative, 4),
            "open_positions": len(open_positions),
            "long_pass": result["long_pass_count"],
            "short_pass": result["short_pass_count"],
        })

    n_trades = len(realized)
    n_scans = len(schedule)
    total_pass = pass_long_total + pass_short_total
    turnover = new_entry_count + replace_count
    avg_occ = statistics.mean(slot_occ) if slot_occ else 0.0

    summary = {
        "interval_min": interval_min,
        "scan_count": n_scans,
        "long_avg_return": round(statistics.mean(long_rets), 4) if long_rets else 0.0,
        "short_avg_return": round(statistics.mean(short_rets), 4) if short_rets else 0.0,
        "combined_avg_return": round(statistics.mean(realized), 4) if realized else 0.0,
        "win_rate_pct": round(sum(1 for r in realized if r >= 3.0) / n_trades * 100, 2) if n_trades else 0.0,
        "mdd_pct": round(max_dd, 4),
        "trade_count": n_trades,
        "pass_count": total_pass,
        "pass_per_day": round(total_pass / days, 4),
        "replacement_count": replace_count,
        "new_entry_count": new_entry_count,
        "reselect_count": reselect_count,
        "dup_direction_count": dup_direction_count,
        "avg_hold_minutes": round(statistics.mean(hold_mins), 2) if hold_mins else 0.0,
        "avg_slot_occupancy": round(avg_occ, 4),
        "turnover": turnover,
        "turnover_rate": round(turnover / max(n_scans, 1), 4),
        "return_per_trade": round(statistics.mean(realized), 4) if realized else 0.0,
        "return_per_day": round(cumulative / days, 4),
        "return_per_turnover": round(cumulative / turnover, 4) if turnover else 0.0,
        "cumulative_return_pct": round(cumulative, 4),
        "long_trades": len(long_rets),
        "short_trades": len(short_rets),
    }

    return {
        "summary": summary,
        "portfolio_log": portfolio_log,
        "slot_history": slot_history,
        "replacement_log": replacement_log,
        "equity_curve": equity_curve,
        "turnover_events": turnover_events,
    }
