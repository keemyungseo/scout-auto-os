"""Scan Cadence Optimizer V1 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.portfolio.backtest import filter_2h_scans
from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.research.cadence.constants import BASE_SNAPSHOT_MINUTES, CADENCE_INTERVALS_MIN
from scout_auto_os.engine.research.cadence.replay import replay_cadence
from scout_auto_os.engine.research.cadence.report import build_cadence_report, recommend_cadence
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines

KST = timezone(timedelta(hours=9))


class ScanCadenceOptimizerRunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
        lookback_days: int = 180,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "cadence"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path
        self.lookback_days = lookback_days

    @research_safe("scan_cadence_optimizer")
    def run(self) -> dict:
        print("[CADENCE OPTIMIZER] started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)

        from scout_auto_os.engine.portfolio.backtest import _parse_scan

        all_scans = sorted(by_scan.keys())
        if all_scans:
            max_dt = _parse_scan(all_scans[-1])
            min_dt = max_dt - timedelta(days=self.lookback_days)
            all_scans = [s for s in all_scans if _parse_scan(s) >= min_dt]
        base_scans = filter_2h_scans(all_scans)

        summaries: list[dict] = []
        all_portfolio_log: list[dict] = []
        all_slot_history: list[dict] = []
        all_replacement_log: list[dict] = []
        all_equity: list[dict] = []
        all_turnover: list[dict] = []

        for interval_min in CADENCE_INTERVALS_MIN:
            print(f"[CADENCE] replay interval={interval_min}m")
            engine = PortfolioEngine.from_paths(self.data_dir, self.pkg_root)
            result = replay_cadence(interval_min, base_scans, by_scan, fwd, engine)
            if "error" in result:
                continue
            summaries.append(result["summary"])
            all_portfolio_log.extend(result["portfolio_log"])
            all_slot_history.extend(result["slot_history"])
            all_replacement_log.extend(result["replacement_log"])
            all_equity.extend(result["equity_curve"])
            all_turnover.extend(result["turnover_events"])

        rec = recommend_cadence(summaries)
        meta = {
            "base_snapshot_minutes": BASE_SNAPSHOT_MINUTES,
            "base_scan_count": len(base_scans),
            "lookback_days": self.lookback_days,
            "intervals_tested": list(CADENCE_INTERVALS_MIN),
            "recommendation": rec,
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "cadence_summary.csv", summaries)
        write_csv(self.out_dir / "cadence_portfolio_log.csv", all_portfolio_log)
        write_csv(self.out_dir / "cadence_slot_history.csv", all_slot_history)
        write_csv(self.out_dir / "cadence_replacement_log.csv", all_replacement_log)
        write_csv(self.out_dir / "cadence_equity_curve.csv", all_equity)
        write_csv(self.out_dir / "cadence_turnover_report.csv", all_turnover)

        report = build_cadence_report(meta, summaries, rec)
        report_path = self.out_dir / "cadence_report.md"
        report_path.write_text(report, encoding="utf-8")
        (self.out_dir / "cadence_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in {
            "cadence_report.md": "scan_cadence_v1_report.md",
            "cadence_summary.csv": "cadence_summary_v1.csv",
            "cadence_meta.json": "scan_cadence_v1_meta.json",
        }.items():
            p = self.out_dir / src
            if p.exists():
                (reports_dir / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        print("[CADENCE OPTIMIZER] complete")
        return {"meta": meta, "summaries": summaries, "report_path": str(report_path)}
