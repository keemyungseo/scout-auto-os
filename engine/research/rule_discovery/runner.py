"""Rule Discovery Engine V1 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.portfolio.rule_loader import load_portfolio_rules
from scout_auto_os.engine.research.directional.dna.validator import split_scans
from scout_auto_os.engine.research.directional.entry_filter.constants import LOOKBACK_MONTHS
from scout_auto_os.engine.research.directional.entry_filter.signals_loader import (
    label_signals,
    load_dna_feature_sets,
    resolve_signals,
)
from scout_auto_os.engine.research.rule_discovery.generator import build_scan_rank_context, generate_candidate_rules
from scout_auto_os.engine.research.rule_discovery.report import build_rule_discovery_report
from scout_auto_os.engine.research.rule_discovery.search import decide_adoption, rank_candidates
from scout_auto_os.engine.research.rule_discovery.validator import (
    evaluate_discovered_rule,
    evaluate_hybrid,
    evaluate_v2_tree,
)
from scout_auto_os.engine.research.safe import research_safe

KST = timezone(timedelta(hours=9))


class RuleDiscoveryRunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
        lookback_months: int = LOOKBACK_MONTHS,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "rule_discovery"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path
        self.lookback_months = lookback_months

    @research_safe("rule_discovery")
    def run(self) -> dict:
        print("[RULE DISCOVERY] started")
        long_raw, short_raw, _ = resolve_signals(
            self.data_dir, self.candidates_path, self.forward_path, self.lookback_months,
        )
        long_signals, _ = label_signals(long_raw)
        short_signals, _ = label_signals(short_raw)

        rules_v2 = load_portfolio_rules(self.data_dir, self.pkg_root)
        dna = load_dna_feature_sets(self.data_dir, self.pkg_root)

        all_scans = sorted({s["scan_time_kst"] for s in long_signals + short_signals})
        train_scans, blind_scans = split_scans(all_scans, 0.7)
        train_set, blind_set = set(train_scans), set(blind_scans)

        def _split(signals: list[dict]) -> tuple[list[dict], list[dict]]:
            tr = [s for s in signals if s["scan_time_kst"] in train_set]
            bl = [s for s in signals if s["scan_time_kst"] in blind_set]
            return tr, bl

        long_train, long_blind = _split(long_signals)
        short_train, short_blind = _split(short_signals)

        rank_feats = ["1h_current_body_pct", "1h_current_range_pct", "1h_current_return_pct"]
        build_scan_rank_context(long_blind, rank_feats)
        build_scan_rank_context(short_blind, rank_feats)

        v2_long_blind = evaluate_v2_tree(
            long_blind, rules_v2.long_tree, rules_v2.long_meta.get("rule_id", "V2_LONG"), "long",
        )
        v2_short_blind = evaluate_v2_tree(
            short_blind, rules_v2.short_tree, rules_v2.short_meta.get("rule_id", "V2_SHORT"), "short",
        )

        long_candidates = generate_candidate_rules(long_train, "long", dna.get("long", []))
        short_candidates = generate_candidate_rules(short_train, "short", dna.get("short", []))
        print(f"[RULE DISCOVERY] candidates long={len(long_candidates)} short={len(short_candidates)}")

        long_blind_metrics: list[dict] = []
        for rule in long_candidates:
            m = evaluate_discovered_rule(
                long_blind, rule, float(v2_long_blind["precision"]), int(v2_long_blind["pass_count"]),
            )
            long_blind_metrics.append(m)

        short_blind_metrics: list[dict] = []
        for rule in short_candidates:
            m = evaluate_discovered_rule(
                short_blind, rule, float(v2_short_blind["precision"]), int(v2_short_blind["pass_count"]),
            )
            short_blind_metrics.append(m)

        top_long = rank_candidates(long_blind_metrics, float(v2_long_blind["precision"]))
        top_short = rank_candidates(short_blind_metrics, float(v2_short_blind["precision"]))

        hybrid_rows: list[dict] = []
        for direction, blind, v2_tree, top, candidates in (
            ("long", long_blind, rules_v2.long_tree, top_long, long_candidates),
            ("short", short_blind, rules_v2.short_tree, top_short, short_candidates),
        ):
            if not top:
                continue
            best_rule = next((c for c in candidates if c.rule_id == top[0]["rule_id"]), None)
            if not best_rule:
                continue
            h = evaluate_hybrid(blind, v2_tree, best_rule, direction)
            h["candidate_rule_id"] = top[0]["rule_id"]
            hybrid_rows.append({"scenario": "current_v2", "direction": direction, **evaluate_v2_tree(
                blind, v2_tree, "V2", direction,
            )})
            hybrid_rows.append({"scenario": "candidate", "direction": direction, **top[0]})
            hybrid_rows.append({"scenario": "hybrid_or", "direction": direction, **h})

        recommendation = decide_adoption(top_long, top_short, v2_long_blind, v2_short_blind, hybrid_rows)

        rec_payload = {
            **recommendation,
            "v2_blind_long": v2_long_blind,
            "v2_blind_short": v2_short_blind,
            "top_long_rule": top_long[0] if top_long else None,
            "top_short_rule": top_short[0] if top_short else None,
            "generated_at": datetime.now(KST).isoformat(),
        }
        if top_long:
            rec_rule = next((c for c in long_candidates if c.rule_id == top_long[0]["rule_id"]), None)
            if rec_rule:
                rec_payload["long_ast"] = rec_rule.to_dict()
        if top_short:
            rec_rule = next((c for c in short_candidates if c.rule_id == top_short[0]["rule_id"]), None)
            if rec_rule:
                rec_payload["short_ast"] = rec_rule.to_dict()

        days = len({s[:10] for s in all_scans}) or 1
        v2_pass_total = (
            sum(1 for s in long_signals if rules_v2.long_tree.evaluate(s["features"]))
            + sum(1 for s in short_signals if rules_v2.short_tree.evaluate(s["features"]))
        )
        meta = {
            "total_champion": len(long_signals) + len(short_signals),
            "v2_pass_total": v2_pass_total,
            "v2_coverage_pct": round(v2_pass_total / (len(long_signals) + len(short_signals)) * 100, 2),
            "train_scan_count": len(train_scans),
            "blind_scan_count": len(blind_scans),
            "long_candidates_generated": len(long_candidates),
            "short_candidates_generated": len(short_candidates),
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        top20_rows = sorted(
            top_long + top_short,
            key=lambda x: (float(x.get("pass_per_day", 0)), float(x.get("avg_return_2h", 0))),
            reverse=True,
        )[:20]
        for i, row in enumerate(top20_rows, start=1):
            row["rank"] = i
        write_csv(self.out_dir / "top20_candidate_rules.csv", top20_rows)
        write_csv(self.out_dir / "hybrid_rule_comparison.csv", hybrid_rows)

        report = build_rule_discovery_report(meta, v2_long_blind, v2_short_blind, top_long, top_short, recommendation)
        report_path = self.out_dir / "rule_discovery_report.md"
        report_path.write_text(report, encoding="utf-8")
        (self.out_dir / "recommended_rule.json").write_text(json.dumps(rec_payload, indent=2), encoding="utf-8")
        (self.out_dir / "rule_discovery_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in {
            "rule_discovery_report.md": "rule_discovery_v1_report.md",
            "top20_candidate_rules.csv": "top20_candidate_rules_v1.csv",
            "hybrid_rule_comparison.csv": "hybrid_rule_comparison_v1.csv",
            "recommended_rule.json": "recommended_rule_v1.json",
        }.items():
            p = self.out_dir / src
            if p.exists():
                (reports_dir / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        print(f"[RULE DISCOVERY] decision={recommendation.get('decision')}")
        return {
            "meta": meta,
            "recommendation": recommendation,
            "top_long_count": len(top_long),
            "top_short_count": len(top_short),
            "report_path": str(report_path),
        }
