"""Regime Engine orchestrator — situation → engine router research."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.research.regime.classifier import (
    build_scan_snapshot,
    classify_regime_8,
)
from scout_auto_os.engine.research.regime.report import build_regime_report
from scout_auto_os.engine.research.regime.router import (
    ROUTER_ENGINES,
    build_regime_engine_matrix,
    detect_transitions,
    select_regime_champions,
)
from scout_auto_os.engine.research.regime.rule_search import composite_router_rules, score_rules
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics
from scout_auto_os.engine.research.zero_base.runner import (
    TRAIN_CUTOFF,
    _is_validation,
    load_candidates_jsonl,
    load_forward_klines,
)
from scout_auto_os.engine.research.zero_base.validation import rank_validation_engine

KST = timezone(timedelta(hours=9))

# Map router research engines (includes REVERSAL/COMPRESSION for regime-specific routing)
ENGINE_MAP = {
    "MOMENTUM": "MOMENTUM",
    "BREAKOUT": "BREAKOUT",
    "FORMULA_LEAGUE": "FORMULA_LEAGUE",
    "FEATURE_LEAGUE": "FEATURE_LEAGUE",
    "STATE_LEAGUE": "STATE_LEAGUE",
    "REVERSAL": "REVERSAL_AFTER_DUMP",
    "COMPRESSION": "COMPRESSION",
}


class RegimeEngineRunner:
    def __init__(
        self,
        data_dir: Path,
        candidates_path: Path,
        forward_path: Path,
        train_cutoff: str = TRAIN_CUTOFF,
    ) -> None:
        self.data_dir = data_dir
        self.out_dir = data_dir / "regime"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path
        self.train_cutoff = train_cutoff

    @research_safe("regime_engine")
    def run(self, max_scans: int | None = None) -> dict:
        print("[REGIME ENGINE] scan classification started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        val_scans = sorted(s for s in by_scan if _is_validation(s, self.train_cutoff))
        if max_scans:
            val_scans = val_scans[:max_scans]

        samples: list[dict] = []
        regime_counts: dict[str, int] = defaultdict(int)
        labeled_snaps: list = []
        scan_regime_seq: list[tuple[str, str]] = []

        for scan_kst in val_scans:
            rows = by_scan[scan_kst]
            snap = build_scan_snapshot(scan_kst, rows)
            regime = classify_regime_8(snap)
            regime_counts[regime] += 1
            labeled_snaps.append((snap, regime))
            scan_regime_seq.append((scan_kst, regime))

            def metric_fn(sym: str) -> dict | None:
                klines = fwd.get((scan_kst, sym))
                if not klines:
                    return None
                m = compute_forward_metrics(klines)
                if m:
                    m["scan_time_kst"] = scan_kst
                    m["symbol"] = sym
                return m

            for eng in ROUTER_ENGINES:
                rank_key = ENGINE_MAP.get(eng, eng)
                syms = rank_validation_engine(rows, rank_key, top_k=5)
                for sym in syms:
                    m = metric_fn(sym)
                    if m:
                        samples.append({**m, "regime": regime, "engine": eng})

        print("[REGIME ENGINE] regime matrix computed")
        matrix = build_regime_engine_matrix(samples)
        champions = select_regime_champions(matrix)
        transitions = detect_transitions(scan_regime_seq)
        rule_scores = score_rules(labeled_snaps)
        composites = composite_router_rules(labeled_snaps)

        meta = {
            "validation_scans": len(val_scans),
            "train_cutoff": self.train_cutoff,
            "generated_at": datetime.now(KST).isoformat(),
        }

        report = build_regime_report(
            dict(regime_counts), matrix, champions, transitions, rule_scores, composites, meta,
        )
        report_path = self.out_dir / "regime_engine_report.md"
        report_path.write_text(report, encoding="utf-8")

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "regime_engine_matrix.csv", matrix)
        write_csv(self.out_dir / "regime_champion_router.csv", champions)
        write_csv(self.out_dir / "regime_detection_rules.csv", rule_scores)
        write_csv(self.out_dir / "regime_transitions.csv", transitions)
        (self.out_dir / "regime_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print("[REGIME ENGINE] report generated")
        return {
            "meta": meta,
            "regime_counts": dict(regime_counts),
            "champions": champions,
            "matrix": matrix,
            "report_path": str(report_path),
        }
