"""Cluster Prediction Engine V1 orchestrator."""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.research.directional.dna.collector import collect_samples
from scout_auto_os.engine.research.directional.dna.formulas import ClusterFormula
from scout_auto_os.engine.research.directional.dna.validator import split_scans
from scout_auto_os.engine.research.directional.prediction.engine import predict_symbol
from scout_auto_os.engine.research.directional.prediction.loader import (
    best_cluster_formula,
    load_blind_validation,
    load_expected_returns,
    load_formulas,
    resolve_formulas_path,
)
from scout_auto_os.engine.research.directional.prediction.picker import (
    make_cluster_champion_picker,
    make_direction_champion_picker,
    make_prediction_picker,
    make_random_picker,
    make_zero_base_champion_picker,
)
from scout_auto_os.engine.research.directional.prediction.report import (
    build_cluster_prediction_report,
    build_prediction_engine_report,
)
from scout_auto_os.engine.research.directional.prediction.slot_sim import simulate_prediction_slots
from scout_auto_os.engine.research.directional.prediction.validation import run_comparison
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.runner import (
    TRAIN_CUTOFF,
    _is_validation,
    load_candidates_jsonl,
    load_forward_klines,
)

KST = timezone(timedelta(hours=9))


def _estimate_train_returns(
    formulas: list[ClusterFormula],
    all_samples: dict[str, list[dict]],
    train_scans: set[str],
) -> dict[str, dict]:
    """Top-quintile formula assignment on train scans — no blind leakage."""
    out: dict[str, dict] = {}
    for formula in formulas:
        pool = [s for s in all_samples.get(formula.engine, []) if s["scan_time_kst"] in train_scans]
        if len(pool) < 10:
            continue
        ranked = sorted(pool, key=lambda s: formula.score(s["features"]), reverse=True)
        n = max(10, len(ranked) // 5)
        top = ranked[:n]
        if formula.direction == "short":
            rets = [
                float(s["metrics"].get("short_return_2h", -float(s["metrics"].get("return_2h", 0))))
                for s in top
            ]
        else:
            rets = [float(s["metrics"].get("return_2h", 0)) for s in top]
        wins = sum(1 for r in rets if r >= 3.0)
        out[formula.name] = {
            "avg_return_2h": round(statistics.mean(rets), 4),
            "win_rate": round(wins / len(rets) * 100, 2),
            "trap_rate": 0.0,
            "sample_count": len(rets),
        }
    return out


class ClusterPredictionRunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
        train_cutoff: str = TRAIN_CUTOFF,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "zero_base"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path
        self.train_cutoff = train_cutoff

    @research_safe("cluster_prediction")
    def run(self, max_scans: int | None = None, top_k: int = 5) -> dict:
        print("[CLUSTER PREDICTION] started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        val_scans = sorted(s for s in by_scan if _is_validation(s, self.train_cutoff))
        if max_scans:
            val_scans = val_scans[:max_scans]
        train_scans, blind_scans = split_scans(val_scans, 0.7)
        train_set = set(train_scans)

        formulas_path = resolve_formulas_path(self.data_dir, self.pkg_root)
        formulas = load_formulas(formulas_path)
        long_formulas = [f for f in formulas if f.direction == "long"]
        short_formulas = [f for f in formulas if f.direction == "short"]

        clusters_csv = self.out_dir / "directional_dna_clusters.csv"
        if not clusters_csv.exists():
            bundle_clusters = self.pkg_root / "research_bundle" / "reports" / "directional_dna_v1_clusters.csv"
            if bundle_clusters.exists():
                clusters_csv = bundle_clusters

        expected_returns = load_expected_returns(clusters_csv, split="train")
        if len(expected_returns) < len(formulas) // 2:
            all_samples = collect_samples(by_scan, fwd, val_scans)
            estimated = _estimate_train_returns(formulas, all_samples, train_set)
            for name, row in estimated.items():
                expected_returns.setdefault(name, row)

        validation_csv = self.out_dir / "directional_dna_validation.csv"
        if not validation_csv.exists():
            validation_csv = self.pkg_root / "research_bundle" / "reports" / "directional_dna_v1_validation.csv"
        blind_stats = load_blind_validation(validation_csv)
        best_long_name = best_cluster_formula(blind_stats, "long")
        best_short_name = best_cluster_formula(blind_stats, "short")
        best_long_formula = next((f for f in long_formulas if f.name == best_long_name), long_formulas[0] if long_formulas else None)
        best_short_formula = next((f for f in short_formulas if f.name == best_short_name), short_formulas[0] if short_formulas else None)

        prob_rows: list[dict] = []
        sample_predictions: list[dict] = []
        for scan_kst in val_scans:
            for row in by_scan[scan_kst]:
                pred = predict_symbol(row["features"], long_formulas, short_formulas, expected_returns)
                base = {
                    "scan_time_kst": scan_kst,
                    "symbol": row["symbol"],
                    **pred,
                }
                sample_predictions.append(base)
                for c in pred["contributions_long"]:
                    prob_rows.append({
                        "scan_time_kst": scan_kst,
                        "symbol": row["symbol"],
                        "direction": "long",
                        "cluster": c["cluster"],
                        "probability_pct": c["probability_pct"],
                        "expected_return_2h": c["expected_return_2h"],
                        "contribution": c["contribution"],
                    })
                for c in pred["contributions_short"]:
                    prob_rows.append({
                        "scan_time_kst": scan_kst,
                        "symbol": row["symbol"],
                        "direction": "short",
                        "cluster": c["cluster"],
                        "probability_pct": c["probability_pct"],
                        "expected_return_2h": c["expected_return_2h"],
                        "contribution": c["contribution"],
                    })

        sample_predictions.sort(key=lambda x: x["scan_time_kst"], reverse=True)

        methods: list[tuple[str, str, object]] = [
            ("RANDOM", "long", make_random_picker(42)),
            ("RANDOM", "short", make_random_picker(43)),
            ("ZERO_BASE_CHAMPION", "long", make_zero_base_champion_picker("long")),
            ("ZERO_BASE_CHAMPION", "short", make_zero_base_champion_picker("short")),
            ("DIRECTION_CHAMPION", "long", make_direction_champion_picker("long")),
            ("DIRECTION_CHAMPION", "short", make_direction_champion_picker("short")),
            ("PREDICTION_ENGINE", "long", make_prediction_picker(long_formulas, short_formulas, expected_returns, "long")),
            ("PREDICTION_ENGINE", "short", make_prediction_picker(long_formulas, short_formulas, expected_returns, "short")),
        ]
        if best_long_formula:
            methods.append(("CLUSTER_CHAMPION", "long", make_cluster_champion_picker(best_long_formula)))
        if best_short_formula:
            methods.append(("CLUSTER_CHAMPION", "short", make_cluster_champion_picker(best_short_formula)))

        comparison = run_comparison(methods, by_scan, fwd, blind_scans, top_k=top_k)
        slot_summary, slot_detail = simulate_prediction_slots(
            by_scan, fwd, blind_scans, long_formulas, short_formulas, expected_returns,
        )

        expected_validation_rows: list[dict] = []
        for name, stats in expected_returns.items():
            direction = "long" if name.startswith("LONG_") else "short"
            blind = blind_stats.get(name, {})
            expected_validation_rows.append({
                "formula_name": name,
                "direction": direction,
                "train_avg_return_2h": stats.get("avg_return_2h"),
                "train_sample_count": stats.get("sample_count"),
                "blind_avg_return_2h": blind.get("avg_return_2h"),
                "blind_delta_vs_random": blind.get("delta_vs_random"),
                "blind_delta_vs_champion": blind.get("delta_vs_champion"),
            })

        meta = {
            "validation_scans": len(val_scans),
            "train_scans": len(train_scans),
            "blind_scans": len(blind_scans),
            "long_formula_count": len(long_formulas),
            "short_formula_count": len(short_formulas),
            "best_long_cluster": best_long_name,
            "best_short_cluster": best_short_name,
            "top_k": top_k,
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        cluster_report = build_cluster_prediction_report(sample_predictions, meta)
        engine_report = build_prediction_engine_report(comparison, slot_summary, meta)

        (self.out_dir / "cluster_prediction_report.md").write_text(cluster_report, encoding="utf-8")
        (self.out_dir / "prediction_engine_report.md").write_text(engine_report, encoding="utf-8")
        write_csv(self.out_dir / "cluster_probability.csv", prob_rows)
        write_csv(self.out_dir / "expected_return_validation.csv", expected_validation_rows)
        write_csv(self.out_dir / "slot_simulation.csv", slot_detail)
        write_csv(self.out_dir / "prediction_engine_comparison.csv", comparison)
        (self.out_dir / "cluster_prediction_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        bundle_names = {
            "cluster_prediction_report.md": "cluster_prediction_report_v1.md",
            "prediction_engine_report.md": "prediction_engine_report_v1.md",
            "cluster_probability.csv": "cluster_probability_v1.csv",
            "expected_return_validation.csv": "expected_return_validation_v1.csv",
            "slot_simulation.csv": "slot_simulation_v1.csv",
            "prediction_engine_comparison.csv": "prediction_engine_comparison_v1.csv",
            "cluster_prediction_meta.json": "cluster_prediction_meta_v1.json",
        }
        for src_name, dst_name in bundle_names.items():
            src = self.out_dir / src_name
            if src.exists():
                (reports_dir / dst_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        print("[CLUSTER PREDICTION] report generated")
        return {
            "meta": meta,
            "comparison": comparison,
            "slot_summary": slot_summary,
            "report_path": str(self.out_dir / "prediction_engine_report.md"),
        }
