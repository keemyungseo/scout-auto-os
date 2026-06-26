"""Portfolio backtest — 2h interval Long3/Short3 simulation."""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta
from pathlib import Path

from scout_auto_os.engine.portfolio.constants import HOLD_HOURS, SCAN_INTERVAL_HOURS
from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.portfolio.report import build_portfolio_report
from scout_auto_os.engine.research.directional.evaluation import to_long_metrics, to_short_metrics
from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines
from scout_auto_os.engine.research.safe import research_safe


def _parse_scan(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def filter_2h_scans(scans: list[str]) -> list[str]:
    """Keep scans at least SCAN_INTERVAL_HOURS apart."""
    if not scans:
        return []
    scans = sorted(scans)
    kept = [scans[0]]
    last = _parse_scan(scans[0])
    for s in scans[1:]:
        dt = _parse_scan(s)
        if (dt - last).total_seconds() >= SCAN_INTERVAL_HOURS * 3600:
            kept.append(s)
            last = dt
    return kept


def _forward_return(direction: str, klines: list) -> dict:
    raw = compute_forward_metrics(klines)
    if not raw:
        return {}
    if direction == "short":
        m = to_short_metrics(raw)
        return {
            "return_2h": float(m.get("short_return_2h", -float(m.get("return_2h", 0)))),
            "return_4h": float(m.get("return_4h", 0)),
        }
    m = to_long_metrics(raw)
    return {"return_2h": float(m.get("return_2h", 0)), "return_4h": float(m.get("return_4h", 0))}


class PortfolioBacktestRunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "portfolio"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path

    @research_safe("portfolio_backtest")
    def run(self, lookback_days: int = 180) -> dict:
        print("[PORTFOLIO BACKTEST] started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        all_scans = sorted(by_scan.keys())
        if all_scans:
            max_dt = _parse_scan(all_scans[-1])
            min_dt = max_dt - timedelta(days=lookback_days)
            all_scans = [s for s in all_scans if _parse_scan(s) >= min_dt]
        scans = filter_2h_scans(all_scans)

        engine = PortfolioEngine.from_paths(self.data_dir, self.pkg_root)
        open_positions: list[dict] = []
        portfolio_log: list[dict] = []
        slot_history: list[dict] = []
        replacement_log: list[dict] = []
        equity_curve: list[dict] = []
        realized_returns: list[float] = []
        long_returns: list[float] = []
        short_returns: list[float] = []
        replacement_count = 0
        hold_durations: list[float] = []
        slot_occupancy: list[float] = []

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for i, scan_kst in enumerate(scans):
            hold_until = scans[i + 1] if i + 1 < len(scans) else scan_kst
            rows = [
                {"symbol": r["symbol"], "features": r["features"]}
                for r in by_scan[scan_kst]
            ]

            closed: list[dict] = []
            still_open: list[dict] = []
            for pos in open_positions:
                if pos["hold_until_scan"] <= scan_kst:
                    klines = fwd.get((pos["entry_scan"], pos["symbol"]))
                    ret = _forward_return(pos["direction"], klines or [])
                    r2h = ret.get("return_2h", 0.0)
                    realized_returns.append(r2h)
                    if pos["direction"] == "long":
                        long_returns.append(r2h)
                    else:
                        short_returns.append(r2h)
                    cumulative += r2h
                    peak = max(peak, cumulative)
                    max_dd = min(max_dd, cumulative - peak)
                    hold_h = (_parse_scan(scan_kst) - _parse_scan(pos["entry_scan"])).total_seconds() / 3600
                    hold_durations.append(hold_h)
                    closed.append({**pos, "exit_scan": scan_kst, "return_2h": r2h})
                    portfolio_log.append({
                        "event": "exit",
                        "scan_time_kst": scan_kst,
                        "symbol": pos["symbol"],
                        "direction": pos["direction"],
                        "entry_score": pos["entry_score"],
                        "return_2h": r2h,
                        "live_pattern": pos.get("live_pattern"),
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

            result = engine.process_scan(rows, scan_kst, hold_until_scan=hold_until)

            for rep in result["replacements"]:
                replacement_log.append(rep)
                replacement_count += 1

            for entry in result["new_entries"]:
                open_positions.append({
                    "symbol": entry["symbol"],
                    "direction": entry["direction"],
                    "entry_score": entry["entry_score"],
                    "entry_scan": scan_kst,
                    "hold_until_scan": hold_until,
                    "live_pattern": entry.get("live_pattern"),
                    "action": entry.get("action"),
                })
                portfolio_log.append({
                    "event": entry.get("action", "enter"),
                    "scan_time_kst": scan_kst,
                    "symbol": entry["symbol"],
                    "direction": entry["direction"],
                    "entry_score": entry["entry_score"],
                    "live_pattern": entry.get("live_pattern"),
                    "direction_confidence": entry.get("direction_confidence"),
                    "pattern_confidence": entry.get("pattern_confidence"),
                    "rule_margin": entry.get("rule_margin"),
                })

            n_long = len(result["long_selected"])
            n_short = len(result["short_selected"])
            slot_occupancy.append((n_long + n_short) / 6.0)

            for side, selected in (("long", result["long_selected"]), ("short", result["short_selected"])):
                for s in selected:
                    slot_history.append({
                        "scan_time_kst": scan_kst,
                        "direction": side,
                        "symbol": s["symbol"],
                        "entry_score": s["entry_score"],
                        "live_pattern": s.get("live_pattern"),
                        "open_positions": len(open_positions),
                    })

            equity_curve.append({
                "scan_time_kst": scan_kst,
                "cumulative_return_2h": round(cumulative, 4),
                "open_positions": len(open_positions),
                "long_slots": n_long,
                "short_slots": n_short,
                "long_pass": result["long_pass_count"],
                "short_pass": result["short_pass_count"],
            })

        wins = sum(1 for r in realized_returns if r >= 3.0)
        stats = {
            "validation_scans": len(scans),
            "lookback_days": lookback_days,
            "total_trades": len(realized_returns),
            "cumulative_return_2h": round(cumulative, 4),
            "max_drawdown": round(max_dd, 4),
            "win_rate_pct": round(wins / len(realized_returns) * 100, 2) if realized_returns else 0,
            "avg_return_2h": round(statistics.mean(realized_returns), 4) if realized_returns else 0,
            "long_avg_2h": round(statistics.mean(long_returns), 4) if long_returns else 0,
            "short_avg_2h": round(statistics.mean(short_returns), 4) if short_returns else 0,
            "long_trades": len(long_returns),
            "short_trades": len(short_returns),
            "replacement_count": replacement_count,
            "avg_hold_hours": round(statistics.mean(hold_durations), 2) if hold_durations else 0,
            "avg_slot_occupancy": round(statistics.mean(slot_occupancy), 4) if slot_occupancy else 0,
            "pass_per_scan_long": round(
                statistics.mean([e["long_pass"] for e in equity_curve]), 2,
            ) if equity_curve else 0,
            "pass_per_scan_short": round(
                statistics.mean([e["short_pass"] for e in equity_curve]), 2,
            ) if equity_curve else 0,
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "portfolio_log.csv", portfolio_log)
        write_csv(self.out_dir / "slot_history.csv", slot_history)
        write_csv(self.out_dir / "replacement_log.csv", replacement_log)
        write_csv(self.out_dir / "equity_curve.csv", equity_curve)

        report = build_portfolio_report(stats, engine.rules)
        (self.out_dir / "portfolio_report.md").write_text(report, encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in {
            "portfolio_report.md": "portfolio_report_v1.md",
            "portfolio_log.csv": "portfolio_log_v1.csv",
            "slot_history.csv": "slot_history_v1.csv",
            "replacement_log.csv": "replacement_log_v1.csv",
            "equity_curve.csv": "equity_curve_v1.csv",
        }.items():
            s = self.out_dir / src
            if s.exists():
                (reports_dir / dst).write_text(s.read_text(encoding="utf-8"), encoding="utf-8")

        print("[PORTFOLIO BACKTEST] complete")
        return {"stats": stats, "report_path": str(self.out_dir / "portfolio_report.md")}
