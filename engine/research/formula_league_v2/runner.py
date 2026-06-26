"""Formula League V2 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scout_phase22_search_formula_evolution as p22

from scout_auto_os.engine.research.directional.dna.validator import split_scans
from scout_auto_os.engine.research.formula_league_v2.annotate import (
    annotate_universe,
    attach_base_scores,
    attach_scan_rank_context,
    label_winner_cohort,
)
from scout_auto_os.engine.research.formula_league_v2.constants import BASELINE_FORMULA_ID, RANK_FEATURES, TRAIN_RATIO
from scout_auto_os.engine.research.formula_league_v2.dna import analyze_formula_dna
from scout_auto_os.engine.research.formula_league_v2.evaluator import evaluate_formula_on_scans
from scout_auto_os.engine.research.formula_league_v2.features import enrich_derived_features
from scout_auto_os.engine.research.formula_league_v2.generator import generate_search_formulas
from scout_auto_os.engine.research.formula_league_v2.report import build_formula_report
from scout_auto_os.engine.research.formula_league_v2.survivor import (
    run_survivor_league,
    scan_regime_map,
    scan_volatility_band,
)
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines

KST = timezone(timedelta(hours=9))
SURVIVOR_CANDIDATE_CAP = 800


class FormulaLeagueV2Runner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "formula_league_v2"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path

    @research_safe("formula_league_v2")
    def run(self) -> dict:
        print("[FORMULA LEAGUE V2] started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        all_scans = sorted(by_scan.keys())
        train_scans, blind_scans = split_scans(all_scans, TRAIN_RATIO)
        train_set = set(train_scans)

        for rows in by_scan.values():
            for r in rows:
                enrich_derived_features(r)

        annotated, th, profile = annotate_universe(by_scan)
        attach_base_scores(annotated, profile)
        attach_scan_rank_context(annotated, list(RANK_FEATURES))
        for rows in annotated.values():
            label_winner_cohort(rows)

        train_flat = [r for scan in train_scans for r in annotated[scan]]
        by_scan_train = {s: annotated[s] for s in train_scans}
        stats = p22.build_train_stats(train_flat, by_scan_train, th)

        formulas = generate_search_formulas(train_flat)
        print(f"[FORMULA LEAGUE V2] generated formulas={len(formulas)}")

        formula_library = [f.to_dict() for f in formulas]
        formula_exprs = {f.formula_id: f.formula_expr for f in formulas}

        blind_scores: list[dict] = []
        for i, formula in enumerate(formulas):
            if i and i % 500 == 0:
                print(f"[FORMULA LEAGUE V2] blind eval {i}/{len(formulas)}")
            _, agg = evaluate_formula_on_scans(
                formula, annotated, blind_scans, fwd, th, stats,
            )
            blind_scores.append(agg)

        blind_scores.sort(key=lambda x: -float(x.get("generalization_score", 0)))
        baseline = next((s for s in blind_scores if s["formula_id"] == BASELINE_FORMULA_ID), {})

        survivor_pool = sorted(
            [f for f in formulas if f.formula_id != BASELINE_FORMULA_ID],
            key=lambda f: -float(next(
                (s["generalization_score"] for s in blind_scores if s["formula_id"] == f.formula_id),
                0,
            )),
        )[:SURVIVOR_CANDIDATE_CAP]
        survivor_pool.append(next(f for f in formulas if f.formula_id == BASELINE_FORMULA_ID))

        regime_map = scan_regime_map(annotated)
        vol_map = scan_volatility_band(annotated)
        round_results, survivor_rows, surviving = run_survivor_league(
            survivor_pool, annotated, all_scans, fwd, th, stats, regime_map, vol_map, TRAIN_RATIO,
        )

        best_survivor_row = next(
            (s for s in survivor_rows if s.get("survived") and s["formula_id"] != BASELINE_FORMULA_ID),
            None,
        )
        best_survivor_score = None
        if best_survivor_row:
            best_survivor_score = next(
                (s for s in blind_scores if s["formula_id"] == best_survivor_row["formula_id"]),
                None,
            )

        dna_rows, importance_rows = analyze_formula_dna(survivor_rows, formula_exprs)
        top_dna = ", ".join(r["token"] for r in dna_rows if r["dna_type"] == "feature")[:120]

        baseline_avg = float(baseline.get("avg_return_2h", 0))
        best_avg = float(best_survivor_score.get("avg_return_2h", 0)) if best_survivor_score else 0
        lift = round((best_avg - baseline_avg) / abs(baseline_avg or 0.01) * 100, 2)
        beats = best_avg > baseline_avg and bool(best_survivor_row)

        meta = {
            "formulas_generated": len(formulas),
            "total_scans": len(all_scans),
            "blind_scan_count": len(blind_scans),
            "survivor_count": sum(1 for s in survivor_rows if s.get("survived")),
            "blind_lift_pct": lift,
            "best_beats_baseline": beats,
            "top_dna_features": top_dna,
            "generated_at": datetime.now(KST).isoformat(),
        }

        generalization_rows = [
            {**r, "eval_split": "blind_temporal"}
            for r in blind_scores
        ]
        for r in round_results:
            generalization_rows.append({**r, "eval_split": r.get("split_type", "round")})

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "formula_league.csv", formula_library)
        write_csv(self.out_dir / "formula_scores.csv", blind_scores)
        write_csv(self.out_dir / "formula_survivors.csv", survivor_rows)
        write_csv(self.out_dir / "formula_dna.csv", dna_rows)
        write_csv(self.out_dir / "formula_feature_importance.csv", importance_rows)
        write_csv(self.out_dir / "formula_generalization.csv", generalization_rows)

        report_md = build_formula_report(
            meta, blind_scores, survivor_rows, baseline,
            best_survivor_score,
        )
        (self.out_dir / "formula_report.md").write_text(report_md, encoding="utf-8")
        (self.out_dir / "formula_league_meta.json").write_text(
            json.dumps({**meta, "baseline": baseline, "best_survivor": best_survivor_score}, indent=2),
            encoding="utf-8",
        )

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        mirror = {
            "formula_league.csv": "formula_league_v2.csv",
            "formula_scores.csv": "formula_scores_v2.csv",
            "formula_survivors.csv": "formula_survivors_v2.csv",
            "formula_dna.csv": "formula_dna_v2.csv",
            "formula_feature_importance.csv": "formula_feature_importance_v2.csv",
            "formula_generalization.csv": "formula_generalization_v2.csv",
            "formula_report.md": "formula_league_v2_report.md",
            "formula_league_meta.json": "formula_league_v2_meta.json",
        }
        for src, dst in mirror.items():
            p = self.out_dir / src
            if p.exists():
                (reports_dir / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        print(f"[FORMULA LEAGUE V2] survivors={meta['survivor_count']} lift={lift}% beats={beats}")
        return {"meta": meta, "blind_scores": blind_scores, "survivors": survivor_rows}
