"""Short Execution Research V1 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.short_execution.exit_simulator import (
    blind_exit_comparison,
    simulate_picks_with_rule,
)
from scout_auto_os.engine.research.short_execution.frozen_picks import load_frozen_blind_picks
from scout_auto_os.engine.research.short_execution.hold_compare import (
    hold_strategy_compare,
    pick_best_dynamic_rule,
)
from scout_auto_os.engine.research.short_execution.holding_dna import (
    holding_distribution,
    holding_dna_rows,
)
from scout_auto_os.engine.research.short_execution.lifecycle import (
    aggregate_lifecycle,
    analyze_pick_lifecycle,
    best_checkpoint_by_roi,
)
from scout_auto_os.engine.research.short_execution.live_audit import run_live_audit
from scout_auto_os.engine.research.short_execution.portfolio_analysis import portfolio_mix_analysis
from scout_auto_os.engine.research.short_execution.report import build_decision, build_report

KST = timezone(timedelta(hours=9))


class ShortExecutionRunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
        workspace_root: Path | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.workspace_root = workspace_root or pkg_root.parent
        self.out_dir = data_dir / "short_execution"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path

    @research_safe("short_execution")
    def run(self) -> dict:
        print("[SHORT EXECUTION V1] started - frozen constitutions, execution layer only")

        short_picks, long_picks, pick_meta = load_frozen_blind_picks(
            self.candidates_path, self.forward_path, self.data_dir, self.pkg_root,
        )
        print(
            f"[SHORT EXECUTION V1] short_picks={len(short_picks)} long_picks={len(long_picks)}",
        )

        lifecycle_detail: list[dict] = []
        for pick in short_picks:
            lifecycle_detail.extend(analyze_pick_lifecycle(pick))
        lifecycle_agg = aggregate_lifecycle(lifecycle_detail)
        best_cp = best_checkpoint_by_roi(lifecycle_agg)

        exit_ranking = blind_exit_comparison(short_picks)
        best_dynamic = pick_best_dynamic_rule(exit_ranking)
        exit_trades = simulate_picks_with_rule(short_picks, best_dynamic)
        hold_compare = hold_strategy_compare(short_picks, best_dynamic)

        dna_detail, dna_summary = holding_dna_rows(short_picks)
        dna_dist = holding_distribution(dna_detail)

        portfolio_scans, portfolio_summary = portfolio_mix_analysis(long_picks, short_picks)

        live_audit = run_live_audit(self.pkg_root, self.workspace_root)

        decision = build_decision(
            exit_ranking, dna_summary, hold_compare, live_audit, best_cp,
        )

        meta = {
            **pick_meta,
            "best_exit_rule": exit_ranking[0] if exit_ranking else {},
            "best_checkpoint": best_cp,
            "holding_dna_summary": dna_summary,
            "portfolio_summary": portfolio_summary,
            "live_audit_summary": {
                "live_data_available": live_audit.get("live_data_available"),
                "manual_protection_verified": live_audit.get("manual_protection_verified"),
            },
            "decision": decision,
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "lifecycle_detail.csv", lifecycle_detail)
        write_csv(self.out_dir / "lifecycle_aggregate.csv", lifecycle_agg)
        write_csv(self.out_dir / "exit_ranking.csv", exit_ranking)
        write_csv(self.out_dir / "exit_trades_best_dynamic.csv", exit_trades)
        write_csv(self.out_dir / "holding_dna.csv", dna_detail)
        write_csv(self.out_dir / "holding_distribution.csv", dna_dist)
        write_csv(self.out_dir / "hold_compare.csv", hold_compare)
        write_csv(self.out_dir / "portfolio_scan.csv", portfolio_scans)
        write_csv(self.out_dir / "portfolio_summary.csv", [portfolio_summary])
        write_csv(self.out_dir / "live_issues.csv", live_audit.get("top10_issues", []))
        write_csv(self.out_dir / "code_fix_priority.csv", decision.get("q5_code_fix_priority", []))

        report_md = build_report(
            meta, exit_ranking, dna_summary, hold_compare, portfolio_summary, live_audit, decision,
        )
        (self.out_dir / "short_execution_report.md").write_text(report_md, encoding="utf-8")
        (self.out_dir / "execution_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        mirror = {
            "short_execution_report.md": "short_execution_v1_report.md",
            "exit_ranking.csv": "short_exit_ranking_v1.csv",
            "lifecycle_aggregate.csv": "short_lifecycle_v1.csv",
            "holding_dna.csv": "short_holding_dna_v1.csv",
            "hold_compare.csv": "short_hold_compare_v1.csv",
            "portfolio_summary.csv": "short_portfolio_mix_v1.csv",
            "live_issues.csv": "short_live_issues_v1.csv",
            "execution_meta.json": "short_execution_v1_meta.json",
        }
        for src, dst in mirror.items():
            p = self.out_dir / src
            if p.exists():
                (reports_dir / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        top = exit_ranking[0].get("rule_id") if exit_ranking else "n/a"
        print(f"[SHORT EXECUTION V1] best_exit={top} peak_median={dna_summary.get('median_peak_at_minutes')}m")
        return {"meta": meta, "decision": decision}
