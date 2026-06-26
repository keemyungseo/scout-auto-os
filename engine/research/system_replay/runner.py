"""System replay validation runner."""

from __future__ import annotations

from pathlib import Path

from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.system_replay.constants import REPLAY_DAYS, SYSTEMS
from scout_auto_os.engine.research.system_replay.metrics import (
    aggregate_trades,
    compute_expectation_lift,
    compute_pe_lift,
    entry_exit_breakdown,
)
from scout_auto_os.engine.research.system_replay.report import SystemReplayReport
from scout_auto_os.engine.research.system_replay.simulator import filter_replay_scans, simulate_system
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines


class SystemReplayRunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
    ) -> None:
        self.data_dir = data_dir
        self.out_dir = data_dir / "system_replay"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pkg_root = pkg_root
        self.candidates_path = candidates_path
        self.forward_path = forward_path

    @research_safe("system_replay_validation")
    def run(self, replay_days: int = REPLAY_DAYS) -> dict:
        print("[SYSTEM REPLAY] Full System Replay Validation V1")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        scans = filter_replay_scans(sorted(by_scan.keys()), replay_days)
        print(f"[SYSTEM REPLAY] scans={len(scans)} days={replay_days}")

        all_trades: dict[str, list[dict]] = {}
        for sid in SYSTEMS:
            engine = PortfolioEngine.from_paths(self.data_dir, self.pkg_root)
            trades = simulate_system(sid, scans, by_scan, fwd, engine)
            all_trades[sid] = trades
            print(f"[SYSTEM REPLAY] system {sid} trades={len(trades)}")

        pe_lift = compute_pe_lift(all_trades.get("C", []), all_trades.get("A", []))
        exp_lift = compute_expectation_lift(all_trades.get("D", []), all_trades.get("C", []))

        portfolio_rows = []
        long_rows = []
        short_rows = []
        for sid in SYSTEMS:
            trades = all_trades[sid]
            long_t = [t for t in trades if t["direction"] == "long"]
            short_t = [t for t in trades if t["direction"] == "short"]
            prow = aggregate_trades(trades, sid)
            prow["long_avg_roi"] = aggregate_trades(long_t).get("avg_roi", 0)
            prow["short_avg_roi"] = aggregate_trades(short_t).get("avg_roi", 0)
            portfolio_rows.append(prow)
            long_rows.append(aggregate_trades(long_t, sid))
            short_rows.append(aggregate_trades(short_t, sid))

        for row in portfolio_rows:
            if row["system_id"] == "C":
                row["pe_improvement_pct"] = pe_lift
            if row["system_id"] == "D":
                row["expectation_improvement_pct"] = exp_lift

        breakdown: list[dict] = []
        for sid, trades in all_trades.items():
            for r in entry_exit_breakdown(trades):
                r["system_id"] = sid
                breakdown.append(r)

        reporter = SystemReplayReport(self.out_dir)
        meta = reporter.write_all(
            all_trades=all_trades,
            portfolio_rows=portfolio_rows,
            long_rows=long_rows,
            short_rows=short_rows,
            breakdown=breakdown,
            pe_lift=pe_lift,
            exp_lift=exp_lift,
            scan_count=len(scans),
        )
        return meta
