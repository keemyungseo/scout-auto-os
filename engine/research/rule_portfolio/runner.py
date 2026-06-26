"""Rule Portfolio Engine V1 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.portfolio.backtest import filter_2h_scans
from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.research.directional.dna.validator import split_scans
from scout_auto_os.engine.research.execution_generalization.regime import attach_regime_to_groups
from scout_auto_os.engine.research.execution_rule_discovery.constants import TRAIN_RATIO
from scout_auto_os.engine.research.execution_rule_discovery.dataset import collect_execution_groups
from scout_auto_os.engine.research.rule_portfolio.activation import build_activation_matrix
from scout_auto_os.engine.research.rule_portfolio.cluster import assign_cluster, summarize_clusters
from scout_auto_os.engine.research.rule_portfolio.collectors import collect_all_rules
from scout_auto_os.engine.research.rule_portfolio.metadata import build_metadata
from scout_auto_os.engine.research.rule_portfolio.profiler import profile_rule
from scout_auto_os.engine.research.rule_portfolio.report import build_portfolio_report, flatten_library_rows
from scout_auto_os.engine.research.rule_portfolio.router import simulate_regime_router
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines

KST = timezone(timedelta(hours=9))


class RulePortfolioRunner:
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
        self.out_dir = data_dir / "rule_portfolio"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path
        self.lookback_days = lookback_days

    @research_safe("rule_portfolio")
    def run(self) -> dict:
        print("[RULE PORTFOLIO] started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        from scout_auto_os.engine.portfolio.backtest import _parse_scan

        all_scans = sorted(by_scan.keys())
        if all_scans:
            max_dt = _parse_scan(all_scans[-1])
            min_dt = max_dt - timedelta(days=self.lookback_days)
            all_scans = [s for s in all_scans if _parse_scan(s) >= min_dt]
        scans = filter_2h_scans(all_scans)
        train_scans, _ = split_scans(scans, TRAIN_RATIO)
        train_set = set(train_scans)

        engine = PortfolioEngine.from_paths(self.data_dir, self.pkg_root)
        all_groups = collect_execution_groups(by_scan, fwd, scans, engine)
        attach_regime_to_groups(all_groups, by_scan)

        train_groups = [g for g in all_groups if g[0]["scan_time_kst"] in train_set]
        long_train = [g for g in train_groups if g[0]["direction"] == "long"]
        short_train = [g for g in train_groups if g[0]["direction"] == "short"]

        portfolio = collect_all_rules(self.data_dir, self.pkg_root, long_train, short_train)
        print(f"[RULE PORTFOLIO] rules collected={len(portfolio)}")

        profiles = [profile_rule(pr, all_groups) for pr in portfolio]
        cluster_rows = [assign_cluster(p) for p in profiles]
        cluster_summary = summarize_clusters(cluster_rows)
        metadata_rows = [
            build_metadata(p, c)
            for p, c in zip(profiles, cluster_rows, strict=True)
        ]
        activation_rows = build_activation_matrix(portfolio, all_groups)
        router = simulate_regime_router(portfolio, profiles, all_groups)

        library_rows = flatten_library_rows(profiles)
        meta = {
            "rule_count": len(portfolio),
            "long_rule_count": sum(1 for p in portfolio if p.direction == "long" and p.rule is not None),
            "scan_count": len(scans),
            "group_count": len(all_groups),
            "router": router,
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "rule_library.csv", library_rows)
        write_csv(self.out_dir / "rule_metadata.csv", metadata_rows)
        write_csv(self.out_dir / "rule_cluster.csv", cluster_rows)
        write_csv(self.out_dir / "rule_activation_matrix.csv", activation_rows)
        write_csv(self.out_dir / "rule_cluster_summary.csv", cluster_summary)

        report_md = build_portfolio_report(meta, library_rows, cluster_summary, router)
        (self.out_dir / "rule_portfolio.md").write_text(report_md, encoding="utf-8")
        (self.out_dir / "rule_portfolio_meta.json").write_text(
            json.dumps({**meta, "cluster_summary": cluster_summary}, indent=2),
            encoding="utf-8",
        )

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        mirror = {
            "rule_library.csv": "rule_library_v1.csv",
            "rule_metadata.csv": "rule_metadata_v1.csv",
            "rule_cluster.csv": "rule_cluster_v1.csv",
            "rule_activation_matrix.csv": "rule_activation_matrix_v1.csv",
            "rule_portfolio.md": "rule_portfolio_v1.md",
            "rule_portfolio_meta.json": "rule_portfolio_meta_v1.json",
        }
        for src, dst in mirror.items():
            p = self.out_dir / src
            if p.exists():
                (reports_dir / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        print(f"[RULE PORTFOLIO] router lift={router.get('lift_pct')}% beats_baseline={router.get('router_beats_baseline')}")
        return {"meta": meta, "profiles": profiles, "router": router}
