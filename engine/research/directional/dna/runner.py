"""Directional DNA Discovery orchestrator."""

from __future__ import annotations

import json
import string
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.research.directional.dna.analyzer import analyze_feature_importance, pattern_dna_summary
from scout_auto_os.engine.research.directional.dna.clustering import (
    build_feature_matrix,
    choose_k,
    kmeans,
    normalize_matrix,
)
from scout_auto_os.engine.research.directional.dna.collector import collect_samples, numeric_feature_keys
from scout_auto_os.engine.research.directional.dna.constants import RESEARCH_ENGINES
from scout_auto_os.engine.research.directional.dna.formulas import build_cluster_formula
from scout_auto_os.engine.research.directional.dna.report import build_dna_report
from scout_auto_os.engine.research.directional.dna.validator import (
    blind_validate_engine,
    cluster_performance,
    split_scans,
)
from scout_auto_os.engine.research.directional.evaluation import aggregate_directional
from scout_auto_os.engine.research.directional.evaluation import to_long_metrics, to_short_metrics
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics
from scout_auto_os.engine.research.zero_base.random_baseline import generate_random_draws
from scout_auto_os.engine.research.zero_base.runner import (
    TRAIN_CUTOFF,
    _is_validation,
    load_candidates_jsonl,
    load_forward_klines,
)

KST = timezone(timedelta(hours=9))
MIN_ENGINE_SAMPLES = 30


class DirectionalDnaRunner:
    def __init__(
        self,
        data_dir: Path,
        candidates_path: Path,
        forward_path: Path,
        train_cutoff: str = TRAIN_CUTOFF,
    ) -> None:
        self.data_dir = data_dir
        self.out_dir = data_dir / "zero_base"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path
        self.train_cutoff = train_cutoff

    @research_safe("directional_dna")
    def run(self, max_scans: int | None = None) -> dict:
        print("[DNA DISCOVERY] started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        val_scans = sorted(s for s in by_scan if _is_validation(s, self.train_cutoff))
        if max_scans:
            val_scans = val_scans[:max_scans]
        train_scans, blind_scans = split_scans(val_scans, 0.7)

        all_samples = collect_samples(by_scan, fwd, val_scans)
        feature_keys = numeric_feature_keys(
            next(iter(by_scan.values()))[0]["features"],
        )

        dna_summaries: list[dict] = []
        all_importance: list[dict] = []
        cluster_stats: list[dict] = []
        formulas_out: list[dict] = []
        validation_rows: list[dict] = []
        live_candidates: list[dict] = []

        for eng in RESEARCH_ENGINES:
            samples = all_samples.get(eng, [])
            if len(samples) < MIN_ENGINE_SAMPLES:
                continue
            direction = samples[0]["direction"]
            importance = analyze_feature_importance(samples, feature_keys)
            for row in importance:
                row["engine"] = eng
            all_importance.extend(importance)

            summary = pattern_dna_summary(eng, samples, importance)
            top_feats = [r["feature"] for r in importance[:15]]
            train_set = [s for s in samples if s["scan_time_kst"] in train_scans]
            if len(train_set) < MIN_ENGINE_SAMPLES:
                summary["cluster_count"] = 0
                dna_summaries.append(summary)
                continue

            matrix = normalize_matrix(build_feature_matrix(train_set, top_feats))
            k = choose_k(len(train_set))
            labels, _centroids = kmeans(matrix, k, seed=hash(eng) % 10000)
            summary["cluster_count"] = k
            dna_summaries.append(summary)

            engine_formulas = []
            letters = list(string.ascii_uppercase)
            for ci in range(k):
                cluster_samples = [train_set[i] for i, lbl in enumerate(labels) if lbl == ci]
                if len(cluster_samples) < 10:
                    continue
                cluster_imp = analyze_feature_importance(cluster_samples, feature_keys)
                formula = build_cluster_formula(
                    eng, letters[ci], direction, cluster_samples, cluster_imp,
                )
                engine_formulas.append(formula)
                formulas_out.append(formula.to_dict())

                perf = cluster_performance(cluster_samples, direction)
                cluster_stats.append({
                    "engine": eng,
                    "formula_name": formula.name,
                    "cluster_id": letters[ci],
                    "split": "train",
                    **perf,
                })

            # blind baseline random on blind scans
            random_blind: list[dict] = []
            for scan_kst in blind_scans:
                rows = by_scan[scan_kst]
                syms = [r["symbol"] for r in rows]
                for pick in generate_random_draws(syms, 5, 1, 42):
                    for sym in pick:
                        raw = compute_forward_metrics(fwd.get((scan_kst, sym)) or [])
                        if raw:
                            random_blind.append(
                                to_long_metrics(raw) if direction == "long" else to_short_metrics(raw),
                            )
            random_agg = aggregate_directional(random_blind, direction)

            val_rows = blind_validate_engine(
                eng, direction, engine_formulas, by_scan, fwd,
                train_scans, blind_scans, random_agg,
            )
            validation_rows.extend(val_rows)

            for vr in val_rows:
                if vr["formula_name"] in ("RANDOM", "PATTERN_CHAMPION"):
                    continue
                if (
                    float(vr.get("delta_vs_champion", 0)) > 0.5
                    and float(vr.get("delta_vs_random", 0)) > 0.5
                    and int(vr.get("sample_count", 0)) >= 20
                ):
                    live_candidates.append({
                        "formula_name": vr["formula_name"],
                        "engine": eng,
                        "blind_avg2h": vr["avg_return_2h"],
                        "tier": "verification_needed",
                        "reason": "blind beat champion and random",
                    })

        all_importance.sort(key=lambda x: abs(x.get("effect_size", 0)), reverse=True)
        global_top = all_importance[:20]
        for i, row in enumerate(global_top, 1):
            row["global_rank"] = i

        meta = {
            "validation_scans": len(val_scans),
            "train_scans": len(train_scans),
            "blind_scans": len(blind_scans),
            "generated_at": datetime.now(KST).isoformat(),
        }

        report = build_dna_report(
            dna_summaries, global_top, cluster_stats, validation_rows, live_candidates, meta,
        )
        report_path = self.out_dir / "directional_dna_report.md"
        report_path.write_text(report, encoding="utf-8")

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "directional_dna_importance.csv", all_importance)
        write_csv(self.out_dir / "directional_dna_clusters.csv", cluster_stats)
        write_csv(self.out_dir / "directional_dna_validation.csv", validation_rows)
        (self.out_dir / "directional_dna_formulas.json").write_text(
            json.dumps(formulas_out, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        (self.out_dir / "directional_dna_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print("[DNA DISCOVERY] report generated")
        return {
            "meta": meta,
            "dna_summaries": dna_summaries,
            "live_candidates": live_candidates,
            "report_path": str(report_path),
        }
