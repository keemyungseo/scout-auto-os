"""Short Constitution Research V1 orchestrator."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scout_auto_os.engine.portfolio.rule_loader import load_portfolio_rules
from scout_auto_os.engine.research.directional.prediction.loader import load_formulas, resolve_formulas_path
from scout_auto_os.engine.research.ranking_engine.features import feature_matrix
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.short_constitution.analysis import (
    feature_importance,
    feature_shift_vs_long,
    load_long_importance,
    rank_short_labels,
)
from scout_auto_os.engine.research.short_constitution.candidate_generator import generate_short_label_candidates
from scout_auto_os.engine.research.short_constitution.constants import (
    BASELINE_LABEL_ID,
    LONG_CONSTITUTION,
    RANDOM_SEED,
    TRAIN_RATIO,
)
from scout_auto_os.engine.research.short_constitution.dataset import (
    collect_short_dataset,
    prepare_annotated,
    split_by_scans,
)
from scout_auto_os.engine.research.short_constitution.evaluation import (
    evaluate_short_label,
    extended_metrics,
    leak_check_rows,
    train_short_ranker,
)
from scout_auto_os.engine.research.short_constitution.regime import build_short_regime_index
from scout_auto_os.engine.research.short_constitution.report import build_decision, build_report, completeness_score
from scout_auto_os.engine.research.ranking_engine.models import predict_scores
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines

KST = timezone(timedelta(hours=9))


class ShortConstitutionRunner:
    def __init__(self, data_dir: Path, pkg_root: Path, candidates_path: Path, forward_path: Path) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "short_constitution"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path

    @research_safe("short_constitution")
    def run(self) -> dict:
        print("[SHORT CONSTITUTION] started - independent from long")
        np.random.seed(RANDOM_SEED)

        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        rules = load_portfolio_rules(self.data_dir, self.pkg_root)
        formulas = load_formulas(resolve_formulas_path(self.data_dir, self.pkg_root))

        annotated, th, stats = prepare_annotated(by_scan)
        dataset = collect_short_dataset(annotated, fwd, rules, formulas, th, stats)
        train_rows, blind_rows = split_by_scans(dataset, TRAIN_RATIO)
        feat_names, _ = feature_matrix(train_rows)
        leak = leak_check_rows(dataset)
        print(f"[SHORT CONSTITUTION] n={len(dataset)} features={len(feat_names)} labels pending")

        specs = generate_short_label_candidates()
        blind_results: list[dict] = []
        gen_rows: list[dict] = []
        bundles: dict = {}

        for i, spec in enumerate(specs, 1):
            print(f"[SHORT CONSTITUTION] [{i}/{len(specs)}] {spec.label_id}")
            row, gen, bundle = evaluate_short_label(train_rows, blind_rows, feat_names, spec)
            blind_results.append(row)
            gen_rows.extend(gen)
            if bundle and not row.get("error"):
                bundles[spec.label_id] = bundle

        label_ranking = rank_short_labels(blind_results, BASELINE_LABEL_ID)
        best_metrics = label_ranking[0] if label_ranking else {}
        best_id = best_metrics.get("label_id", BASELINE_LABEL_ID)
        best_spec = next(s for s in specs if s.label_id == best_id)
        best_bundle = bundles.get(best_id)

        long_imp = load_long_importance(self.data_dir)
        short_imp = feature_importance(best_bundle, train_rows, best_spec) if best_bundle else []
        shift_rows, shift_summary = feature_shift_vs_long(long_imp, short_imp)

        regime_index = build_short_regime_index(by_scan)
        regime_rows: list[dict] = []
        if best_bundle:
            from scout_auto_os.engine.research.short_constitution.label_builder import apply_short_label

            labeled_blind = apply_short_label(blind_rows, best_spec)

            def score_fn(row, peers, b=best_bundle):
                return float(predict_scores(b, [row])[0])

            by_scan_picks: dict[str, list] = defaultdict(list)
            for r in labeled_blind:
                by_scan_picks[r["scan_kst"]].append(r)
            for scan, rows in sorted(by_scan_picks.items()):
                scores = {r["symbol"]: score_fn(r, rows) for r in rows}
                ranked = sorted(rows, key=lambda r: -scores[r["symbol"]])
                picks = [{**r, "rank": i + 1, "score": scores[r["symbol"]]} for i, r in enumerate(ranked[:5])]
                m = extended_metrics(picks, f"regime_{scan}")
                reg = regime_index.get(scan, {}).get("short_regime", "Mixed")
                regime_rows.append({"regime": reg, "scan": scan, **m})

        long_blind = self._load_long_blind()
        long_cmp = {
            "long_avg_return_2h": long_blind.get("avg_return_2h", LONG_CONSTITUTION["blind_avg_2h"]),
            "long_sharpe": long_blind.get("sharpe"),
            "long_ndcg5": long_blind.get("rank_ndcg5"),
            "short_avg_return_2h": best_metrics.get("avg_return_2h"),
            "short_sharpe": best_metrics.get("sharpe"),
            "short_ndcg5": best_metrics.get("rank_ndcg5"),
            "short_minus_long_avg": round(
                float(best_metrics.get("avg_return_2h", 0)) - float(long_blind.get("avg_return_2h", 0)), 4,
            ),
        }

        score = completeness_score(best_metrics, long_blind, shift_summary)
        decision = build_decision(best_metrics, label_ranking, shift_summary, long_blind, score, leak)

        meta = {
            "direction": "short",
            "sample_count": len(dataset),
            "feature_count": len(feat_names),
            "label_candidates": len(specs),
            "best_label": best_metrics,
            "long_comparison": long_cmp,
            "shift_summary": shift_summary,
            "completeness_score": score,
            "leak_check": leak,
            "leak_passed": leak.get("passed"),
            "decision": decision,
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "blind_report.csv", blind_results)
        write_csv(self.out_dir / "label_ranking.csv", label_ranking)
        write_csv(self.out_dir / "feature_importance.csv", short_imp)
        write_csv(self.out_dir / "feature_shift.csv", shift_rows)
        write_csv(self.out_dir / "regime_report.csv", regime_rows)
        write_csv(self.out_dir / "comparison_vs_long.csv", [long_cmp])
        write_csv(self.out_dir / "generalization_report.csv", gen_rows)

        report_md = build_report(meta, label_ranking, best_metrics, long_cmp, decision, regime_rows)
        (self.out_dir / "short_constitution_report.md").write_text(report_md, encoding="utf-8")
        (self.out_dir / "constitution_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in {
            "short_constitution_report.md": "short_constitution_v1_report.md",
            "blind_report.csv": "short_blind_report_v1.csv",
            "label_ranking.csv": "short_label_ranking_v1.csv",
            "feature_importance.csv": "short_feature_importance_v1.csv",
            "feature_shift.csv": "short_feature_shift_v1.csv",
            "regime_report.csv": "short_regime_report_v1.csv",
            "comparison_vs_long.csv": "short_vs_long_v1.csv",
            "constitution_meta.json": "short_constitution_v1_meta.json",
        }.items():
            p = self.out_dir / src
            if p.exists():
                (reports_dir / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        print(
            f"[SHORT CONSTITUTION] best={best_id} avg={best_metrics.get('avg_return_2h')} "
            f"long={long_cmp.get('long_avg_return_2h')} score={score}/100",
        )
        return {"meta": meta, "decision": decision}

    def _load_long_blind(self) -> dict:
        p = self.data_dir / "constitution_validation" / "constitution_validation_meta.json"
        if p.exists():
            meta = json.loads(p.read_text(encoding="utf-8"))
            return meta.get("blind", {})
        return {"avg_return_2h": LONG_CONSTITUTION["blind_avg_2h"]}
