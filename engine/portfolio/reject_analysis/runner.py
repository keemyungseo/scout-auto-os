"""Reject Analysis Engine V1 — coverage funnel orchestrator."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.portfolio.backtest import filter_2h_scans
from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.portfolio.reject_analysis.portfolio_audit import trace_portfolio_decisions
from scout_auto_os.engine.portfolio.reject_analysis.report import build_coverage_report
from scout_auto_os.engine.portfolio.reject_analysis.rule_audit import audit_rule
from scout_auto_os.engine.portfolio.scoring import score_candidate
from scout_auto_os.engine.research.directional.engines import rank_long, rank_short
from scout_auto_os.engine.research.directional.entry_filter.constants import (
    CHAMPION_TOP_K,
    LONG_DIRECTION_CHAMPION,
    SHORT_DIRECTION_CHAMPION,
)
from scout_auto_os.engine.research.directional.entry_filter.pattern_labels import live_pattern
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl

KST = timezone(timedelta(hours=9))


def _parse_scan(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


class RejectAnalysisRunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        lookback_days: int = 180,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "portfolio"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.lookback_days = lookback_days

    @research_safe("reject_analysis")
    def run(self) -> dict:
        print("[REJECT ANALYSIS] started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        all_scans = sorted(by_scan.keys())
        if all_scans:
            max_dt = _parse_scan(all_scans[-1])
            min_dt = max_dt - timedelta(days=self.lookback_days)
            all_scans = [s for s in all_scans if _parse_scan(s) >= min_dt]
        scans = filter_2h_scans(all_scans)

        engine = PortfolioEngine.from_paths(self.data_dir, self.pkg_root)
        rules = engine.rules
        book = engine.book

        reject_reason_rows: list[dict] = []
        reject_feature_rows: list[dict] = []
        near_pass_rows: list[dict] = []
        portfolio_reject_rows: list[dict] = []

        stage_counts = Counter()
        rule_fail_by_label = Counter()
        portfolio_fail = Counter()
        total_champion = 0
        total_rule_pass = 0
        total_portfolio_pass = 0

        for i, scan_kst in enumerate(scans):
            hold_until = scans[i + 1] if i + 1 < len(scans) else scan_kst
            rows = [{"symbol": r["symbol"], "features": r["features"]} for r in by_scan[scan_kst]]

            long_syms = rank_long(rows, LONG_DIRECTION_CHAMPION, CHAMPION_TOP_K)
            short_syms = rank_short(rows, SHORT_DIRECTION_CHAMPION, CHAMPION_TOP_K)

            long_pass: list[dict] = []
            short_pass: list[dict] = []

            for direction, syms in (("long", long_syms), ("short", short_syms)):
                for sym in syms:
                    total_champion += 1
                    stage_counts["direction_champion"] += 1
                    row = next(r for r in rows if r["symbol"] == sym)
                    features = row["features"]
                    pattern = live_pattern(features)

                    if direction == "long":
                        tree = rules.pattern_trees.get(pattern, rules.long_tree)
                    else:
                        tree = rules.pattern_trees.get(pattern, rules.short_tree)

                    audit = audit_rule(features, tree, direction)
                    stage_counts["rule_evaluated"] += 1

                    for cr in audit["condition_results"]:
                        reject_feature_rows.append({
                            "scan_time_kst": scan_kst,
                            "symbol": sym,
                            "direction": direction,
                            "feature": cr["feature"],
                            "feature_label": cr["feature_label"],
                            "passed": cr["passed"],
                            "gap_pct": cr["gap_pct"],
                            "value": cr["value"],
                            "threshold": cr["threshold"],
                        })
                        if not cr["passed"]:
                            rule_fail_by_label[cr["feature_label"]] += 1

                    final_reason = audit["primary_reason"]
                    portfolio_result = ""

                    if audit["rule_pass"] and audit["primary_reason"] != "Freshness_Reject":
                        total_rule_pass += 1
                        stage_counts["rule_pass"] += 1
                        scored = score_candidate(row, direction, rules, rows, scan_kst, scan_kst)
                        if scored:
                            if direction == "long":
                                long_pass.append(scored)
                            else:
                                short_pass.append(scored)
                    else:
                        if audit["primary_reason"] == "Freshness_Reject":
                            final_reason = "Freshness_Reject"
                        stage_counts["rule_reject"] += 1

                    reject_reason_rows.append({
                        "scan_time_kst": scan_kst,
                        "symbol": sym,
                        "direction": direction,
                        "live_pattern": pattern,
                        "stage": "rule",
                        "final_result": final_reason,
                        "reject_tier": audit["reject_tier"],
                        "failed_conditions": audit["failed_condition_count"],
                        "freshness_score": audit["freshness_score"],
                        "rule_pass": audit["rule_pass"],
                    })

                    if audit["reject_tier"] == "near_pass" and not audit["rule_pass"]:
                        for fc in audit["failed_conditions"]:
                            near_pass_rows.append({
                                "scan_time_kst": scan_kst,
                                "symbol": sym,
                                "direction": direction,
                                "feature": fc["feature"],
                                "feature_label": fc["feature_label"],
                                "value": fc["value"],
                                "threshold": fc["threshold"],
                                "gap": fc["gap"],
                                "gap_pct": fc["gap_pct"],
                                "almost_pass": fc["gap_pct"] <= 5.0,
                            })

            port_decisions, book = trace_portfolio_decisions(
                book, long_pass, short_pass, scan_kst, hold_until,
            )

            for d in port_decisions:
                portfolio_reject_rows.append(d)
                pr = d["portfolio_result"]
                portfolio_fail[pr] += 1
                if pr in ("PASS", "Replacement"):
                    total_portfolio_pass += 1
                    stage_counts["portfolio_pass"] += 1

                reject_reason_rows.append({
                    "scan_time_kst": scan_kst,
                    "symbol": d["symbol"],
                    "direction": d["direction"],
                    "live_pattern": d.get("live_pattern"),
                    "stage": "portfolio",
                    "final_result": pr,
                    "entry_score": d.get("entry_score"),
                    "reject_tier": "",
                    "failed_conditions": "",
                    "freshness_score": "",
                    "rule_pass": True,
                })

        funnel = {
            "direction_champion": total_champion,
            "rule_pass": total_rule_pass,
            "rule_pass_rate_pct": round(total_rule_pass / total_champion * 100, 2) if total_champion else 0,
            "portfolio_pass": total_portfolio_pass,
            "portfolio_pass_rate_pct": round(total_portfolio_pass / total_champion * 100, 2) if total_champion else 0,
            "final_fill_rate_pct": round(total_portfolio_pass / total_champion * 100, 2) if total_champion else 0,
        }

        rule_fail_total = sum(rule_fail_by_label.values()) or 1
        feature_fail_pct = {
            k: round(v / rule_fail_total * 100, 2)
            for k, v in rule_fail_by_label.most_common()
        }

        bottleneck = _find_bottleneck(funnel, rule_fail_by_label, portfolio_fail, total_champion)

        meta = {
            "scans": len(scans),
            "lookback_days": self.lookback_days,
            "champion_candidates": total_champion,
            "generated_at": datetime.now(KST).isoformat(),
            "funnel": funnel,
            "rule_fail_feature_pct": feature_fail_pct,
            "portfolio_reject_counts": dict(portfolio_fail),
            "bottleneck": bottleneck,
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "reject_reason.csv", reject_reason_rows)
        write_csv(self.out_dir / "reject_feature.csv", reject_feature_rows)
        write_csv(self.out_dir / "near_pass.csv", near_pass_rows)
        write_csv(self.out_dir / "portfolio_reject.csv", portfolio_reject_rows)

        report = build_coverage_report(meta, feature_fail_pct, portfolio_fail, near_pass_rows)
        (self.out_dir / "coverage_report.md").write_text(report, encoding="utf-8")
        (self.out_dir / "reject_analysis_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in {
            "coverage_report.md": "coverage_report_v1.md",
            "reject_reason.csv": "reject_reason_v1.csv",
            "reject_feature.csv": "reject_feature_v1.csv",
            "near_pass.csv": "near_pass_v1.csv",
            "portfolio_reject.csv": "portfolio_reject_v1.csv",
        }.items():
            s = self.out_dir / src
            if s.exists():
                (reports_dir / dst).write_text(s.read_text(encoding="utf-8"), encoding="utf-8")

        print("[REJECT ANALYSIS] complete")
        return {"meta": meta, "report_path": str(self.out_dir / "coverage_report.md")}


def _find_bottleneck(
    funnel: dict,
    rule_fail: Counter,
    portfolio_fail: Counter,
    total: int,
) -> dict:
    rule_pass = int(funnel.get("rule_pass", 0))
    port_pass = int(funnel.get("portfolio_pass", 0))
    rule_drop = total - rule_pass
    port_drop = rule_pass - port_pass

    if rule_drop >= port_drop:
        top_feat = rule_fail.most_common(1)
        fail_total = sum(rule_fail.values()) or 1
        return {
            "stage": "entry_rule_v2",
            "severity_pct": round(rule_drop / total * 100, 2) if total else 0,
            "candidates_lost": rule_drop,
            "top_feature_blocker": top_feat[0][0] if top_feat else "unknown",
            "top_feature_block_pct": round(top_feat[0][1] / fail_total * 100, 2) if top_feat else 0,
            "recommendation": "Primary coverage gap at Entry Rule V2 — Body/Range/Momentum thresholds block ~95% of champion picks",
            "priority": "high",
        }

    top_port = portfolio_fail.most_common(1)
    return {
        "stage": "portfolio_engine",
        "severity_pct": round(port_drop / total * 100, 2) if total else 0,
        "candidates_lost": port_drop,
        "top_blocker": top_port[0][0] if top_port else "unknown",
        "recommendation": "Portfolio slot/replacement/diversification limits PASS after rule filter",
        "priority": "medium",
    }
