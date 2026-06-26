"""Trade DNA Engine V1 runner."""

from __future__ import annotations

from pathlib import Path

from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.trade_dna.clustering import cluster_trades
from scout_auto_os.engine.research.trade_dna.collector import collect_pass_candidates
from scout_auto_os.engine.research.trade_dna.entry_predictor import evaluate_entry_predictability
from scout_auto_os.engine.research.trade_dna.exit_optimizer import best_exit_per_cluster, estimate_portfolio_lift
from scout_auto_os.engine.research.trade_dna.report import TradeDNAReport, type_statistics, winner_loser_analysis


class TradeDNARunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
    ) -> None:
        self.data_dir = data_dir
        self.out_dir = data_dir / "trade_dna"
        self.pkg_root = pkg_root
        self.candidates_path = candidates_path
        self.forward_path = forward_path

    @research_safe("trade_dna_v1")
    def run(self, replay_days: int = 15) -> dict:
        print("[TRADE DNA] V1 started")
        records = collect_pass_candidates(
            self.data_dir, self.pkg_root,
            self.candidates_path, self.forward_path,
            replay_days=replay_days,
        )
        print(f"[TRADE DNA] trades collected: {len(records)}")

        cluster_rows, meta = cluster_trades(records)
        labels = [int(r["cluster_id"]) for r in cluster_rows]

        type_stats = type_statistics(cluster_rows, records)

        exit_table = best_exit_per_cluster(records, labels)
        for ex in exit_table:
            stat = next((s for s in type_stats if s["trade_type_id"] == ex["trade_type_id"]), {})
            ex["data_derived_label"] = stat.get("data_derived_label", "")

        lift = estimate_portfolio_lift(records, labels, exit_table)
        entry_pred = evaluate_entry_predictability(records, labels)
        wl = winner_loser_analysis(cluster_rows)

        reporter = TradeDNAReport(self.out_dir)
        report_path = reporter.write_all(
            cluster_rows, records, meta, type_stats, exit_table, lift, entry_pred, wl,
        )
        print(f"[TRADE DNA] types={meta.get('n_clusters')} lift={lift.get('expected_lift_pp')}%p")
        print(f"[TRADE DNA] report: {report_path}")
        return {
            "n_trades": len(records),
            "n_types": meta.get("n_clusters"),
            "lift": lift,
            "entry_pred": entry_pred,
            "wl": wl,
            "report_path": str(report_path),
        }
