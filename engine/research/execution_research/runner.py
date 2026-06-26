"""Execution Engine V1 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.portfolio.backtest import filter_2h_scans
from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.research.directional.dna.validator import split_scans
from scout_auto_os.engine.research.execution_research.constants import TOP2_SIZE, TRAIN_RATIO
from scout_auto_os.engine.research.execution_research.report import build_execution_report
from scout_auto_os.engine.research.execution_research.selector import (
    eval_return_2h,
    pick_top2_entry_score,
    pick_top2_execution,
    rank_by_execution,
    top5_pass_candidates,
)
from scout_auto_os.engine.research.execution_research.validation import compare_strategies, tune_weights_on_train
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines

KST = timezone(timedelta(hours=9))


class ExecutionResearchRunner:
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
        self.out_dir = data_dir / "execution_engine"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path
        self.lookback_days = lookback_days

    @research_safe("execution_research")
    def run(self) -> dict:
        print("[EXECUTION ENGINE V1] started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)

        from scout_auto_os.engine.portfolio.backtest import _parse_scan

        all_scans = sorted(by_scan.keys())
        if all_scans:
            max_dt = _parse_scan(all_scans[-1])
            min_dt = max_dt - timedelta(days=self.lookback_days)
            all_scans = [s for s in all_scans if _parse_scan(s) >= min_dt]
        scans = filter_2h_scans(all_scans)
        train_scans, blind_scans = split_scans(scans, TRAIN_RATIO)
        train_set, blind_set = set(train_scans), set(blind_scans)

        engine = PortfolioEngine.from_paths(self.data_dir, self.pkg_root)

        score_rows: list[dict] = []
        pick_rows: list[dict] = []
        train_tune_rows: list[dict] = []

        def _process_scan_list(scan_list: list[str], split: str, weights: dict | None = None) -> dict:
            top5_rets: list[float] = []
            entry2_rets: list[float] = []
            exec2_rets: list[float] = []
            long_top5: list[float] = []
            long_entry2: list[float] = []
            long_exec2: list[float] = []
            short_top5: list[float] = []
            short_entry2: list[float] = []
            short_exec2: list[float] = []

            for scan_kst in scan_list:
                rows = [{"symbol": r["symbol"], "features": r["features"]} for r in by_scan[scan_kst]]
                long5, short5 = top5_pass_candidates(rows, scan_kst, engine)

                for direction, pool in (("long", long5), ("short", short5)):
                    if len(pool) < TOP2_SIZE:
                        continue
                    scored = rank_by_execution(pool, direction, fwd, scan_kst, weights)
                    if len(scored) < TOP2_SIZE:
                        continue

                    entry2 = pick_top2_entry_score(pool)
                    exec2 = pick_top2_execution(scored)

                    for c in scored:
                        klines = fwd.get((scan_kst, c["symbol"]))
                        r2h = eval_return_2h(klines or [], direction)
                        row = {
                            "scan_time_kst": scan_kst,
                            "split": split,
                            "direction": direction,
                            "symbol": c["symbol"],
                            "entry_score": c["entry_score"],
                            "execution_score": c["execution_score"],
                            "return_2h": r2h,
                            **{k: c[k] for k in c if k.startswith("obs_") or k in (
                                "volume_ratio_scan", "vwap_deviation_pct", "atr_increase_ratio",
                                "false_breakout_flag", "new_high_breakout",
                            )},
                        }
                        score_rows.append(row)
                        if split == "train":
                            train_tune_rows.append(row)

                    for c in pool:
                        klines = fwd.get((scan_kst, c["symbol"]))
                        top5_rets.append(eval_return_2h(klines or [], direction))
                        if direction == "long":
                            long_top5.append(eval_return_2h(klines or [], direction))
                        else:
                            short_top5.append(eval_return_2h(klines or [], direction))

                    for c in entry2:
                        klines = fwd.get((scan_kst, c["symbol"]))
                        r = eval_return_2h(klines or [], direction)
                        entry2_rets.append(r)
                        (long_entry2 if direction == "long" else short_entry2).append(r)

                    for c in exec2:
                        klines = fwd.get((scan_kst, c["symbol"]))
                        r = eval_return_2h(klines or [], direction)
                        exec2_rets.append(r)
                        (long_exec2 if direction == "long" else short_exec2).append(r)
                        pick_rows.append({
                            "scan_time_kst": scan_kst,
                            "split": split,
                            "direction": direction,
                            "symbol": c["symbol"],
                            "selection": "execution_top2",
                            "entry_score": c["entry_score"],
                            "execution_score": c["execution_score"],
                            "return_2h": r,
                        })

            return {
                "top5": top5_rets,
                "entry2": entry2_rets,
                "exec2": exec2_rets,
                "long_top5": long_top5,
                "long_entry2": long_entry2,
                "long_exec2": long_exec2,
                "short_top5": short_top5,
                "short_entry2": short_entry2,
                "short_exec2": short_exec2,
            }

        train_stats = _process_scan_list(train_scans, "train")
        weights = tune_weights_on_train(train_tune_rows)
        blind_stats = _process_scan_list(blind_scans, "blind", weights)

        blind_rows: list[dict] = []
        for direction in ("long", "short"):
            blind_rows.extend(compare_strategies(
                blind_stats[f"{direction}_top5"],
                blind_stats[f"{direction}_entry2"],
                blind_stats[f"{direction}_exec2"],
                direction,
                "blind",
            ))

        def _avg(xs: list[float]) -> float:
            return round(sum(xs) / len(xs), 4) if xs else 0.0

        combined = {
            "top5_avg": _avg(blind_stats["top5"]),
            "entry_top2_avg": _avg(blind_stats["entry2"]),
            "exec_top2_avg": _avg(blind_stats["exec2"]),
        }
        combined["lift_vs_top5_pct"] = round(
            (combined["exec_top2_avg"] - combined["top5_avg"]) / abs(combined["top5_avg"] or 0.01) * 100, 2,
        )
        combined["lift_vs_entry_top2_pct"] = round(
            (combined["exec_top2_avg"] - combined["entry_top2_avg"]) / abs(combined["entry_top2_avg"] or 0.01) * 100,
            2,
        )

        meta = {
            "train_scan_count": len(train_scans),
            "blind_scan_count": len(blind_scans),
            "observation_minutes": 15,
            "total_picks": len(pick_rows),
            "weights": weights,
            "combined_blind": combined,
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "execution_scores.csv", score_rows)
        write_csv(self.out_dir / "execution_picks.csv", pick_rows)
        write_csv(self.out_dir / "execution_blind_comparison.csv", blind_rows)

        report = build_execution_report(meta, blind_rows, combined)
        report_path = self.out_dir / "execution_report.md"
        report_path.write_text(report, encoding="utf-8")
        (self.out_dir / "execution_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in {
            "execution_report.md": "execution_engine_v1_report.md",
            "execution_blind_comparison.csv": "execution_blind_comparison_v1.csv",
            "execution_picks.csv": "execution_picks_v1.csv",
            "execution_meta.json": "execution_engine_v1_meta.json",
        }.items():
            p = self.out_dir / src
            if p.exists():
                (reports_dir / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        print(f"[EXECUTION ENGINE V1] blind lift vs entry top2={combined['lift_vs_entry_top2_pct']}%")
        return {"meta": meta, "combined": combined, "report_path": str(report_path)}
