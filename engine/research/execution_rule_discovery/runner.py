"""Execution Rule Discovery V1 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.portfolio.backtest import filter_2h_scans
from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.research.directional.dna.validator import split_scans
from scout_auto_os.engine.research.execution_rule_discovery.baselines import (
    pick_top2_entry_score,
    pick_top2_execution_score,
)
from scout_auto_os.engine.research.execution_rule_discovery.constants import TOP_OUTPUT, TRAIN_RATIO
from scout_auto_os.engine.research.execution_rule_discovery.dataset import collect_execution_groups
from scout_auto_os.engine.research.execution_rule_discovery.generator import (
    avg_top2_return,
    generate_execution_rules,
    rank_rules_on_train,
)
from scout_auto_os.engine.research.execution_rule_discovery.report import build_execution_rule_report
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines

KST = timezone(timedelta(hours=9))


def _split_groups(groups: list[list[dict]], train_set: set[str]) -> tuple[list[list[dict]], list[list[dict]]]:
    train_g = [g for g in groups if g[0]["scan_time_kst"] in train_set]
    blind_g = [g for g in groups if g[0]["scan_time_kst"] not in train_set]
    return train_g, blind_g


def _lift(new: float, base: float) -> float:
    return round((new - base) / abs(base or 0.01) * 100, 2)


class ExecutionRuleDiscoveryRunner:
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
        self.out_dir = data_dir / "execution_rule_discovery"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path
        self.lookback_days = lookback_days

    @research_safe("execution_rule_discovery")
    def run(self) -> dict:
        print("[EXECUTION RULE DISCOVERY] started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        from scout_auto_os.engine.portfolio.backtest import _parse_scan

        all_scans = sorted(by_scan.keys())
        if all_scans:
            max_dt = _parse_scan(all_scans[-1])
            min_dt = max_dt - timedelta(days=self.lookback_days)
            all_scans = [s for s in all_scans if _parse_scan(s) >= min_dt]
        scans = filter_2h_scans(all_scans)
        train_scans, blind_scans = split_scans(scans, TRAIN_RATIO)
        train_set = set(train_scans)

        engine = PortfolioEngine.from_paths(self.data_dir, self.pkg_root)
        all_groups = collect_execution_groups(by_scan, fwd, scans, engine)
        train_groups, blind_groups = _split_groups(all_groups, train_set)

        long_train = [g for g in train_groups if g[0]["direction"] == "long"]
        short_train = [g for g in train_groups if g[0]["direction"] == "short"]
        long_blind = [g for g in blind_groups if g[0]["direction"] == "long"]
        short_blind = [g for g in blind_groups if g[0]["direction"] == "short"]

        all_rules: list = []
        rule_map: dict = {}
        for direction, tg in (("long", long_train), ("short", short_train)):
            rules = generate_execution_rules(tg, direction)
            all_rules.extend(rules)
            for r in rules:
                rule_map[r.rule_id] = r
            print(f"[EXECUTION RULE DISCOVERY] {direction} rules={len(rules)}")

        train_ranked = rank_rules_on_train(long_train + short_train, all_rules)

        blind_entry = avg_top2_return(blind_groups, None, pick_top2_entry_score)
        blind_exec = avg_top2_return(blind_groups, None, pick_top2_execution_score)

        blind_rule_rows: list[dict] = []
        for row in train_ranked[:TOP_OUTPUT * 2]:
            rule = rule_map.get(row["rule_id"])
            if not rule:
                continue
            bg = long_blind if rule.direction == "long" else short_blind
            bm = avg_top2_return(bg, rule)
            bm["rule_id"] = row["rule_id"]
            bm["rule_expr"] = row["rule_expr"]
            bm["direction"] = rule.direction
            bm["lift_vs_exec_score_pct"] = _lift(bm["avg_return_2h"], blind_exec["avg_return_2h"])
            bm["lift_vs_entry_top2_pct"] = _lift(bm["avg_return_2h"], blind_entry["avg_return_2h"])
            blind_rule_rows.append(bm)

        blind_rule_rows.sort(
            key=lambda x: (float(x["avg_return_2h"]), float(x.get("lift_vs_exec_score_pct", 0))),
            reverse=True,
        )
        top_rules = blind_rule_rows[:TOP_OUTPUT]

        best = top_rules[0] if top_rules else None
        beats_exec = best and float(best["avg_return_2h"]) > float(blind_exec["avg_return_2h"])

        if beats_exec and best:
            if int(best.get("trade_count", 0)) >= 20:
                decision = "Adopt"
                reason = (
                    f"Rule `{best['rule_expr'][:80]}` beats Execution Score on blind "
                    f"({best['avg_return_2h']} vs {blind_exec['avg_return_2h']})"
                )
            else:
                decision = "Needs further validation"
                reason = (
                    f"Rule beats Execution Score on blind ({best['avg_return_2h']} vs {blind_exec['avg_return_2h']}) "
                    f"but sample is small (n={best.get('trade_count')}) - extend blind window before LIVE"
                )
            rec_rule = rule_map[best["rule_id"]].to_dict()
            rec_rule["rule_expr"] = best["rule_expr"]
        else:
            decision = "Keep current Execution Engine"
            reason = "No discovered rule beats Execution Score Top2 on blind validation"
            rec_rule = {"engine": "execution_score_v1", "weights": "manual"}

        recommendation = {
            "decision": decision,
            "reason": reason,
            "baseline_exec_avg": blind_exec["avg_return_2h"],
            "baseline_entry_top2_avg": blind_entry["avg_return_2h"],
            "best_rule_avg": best["avg_return_2h"] if best else None,
            "best_rule_id": best["rule_id"] if best else None,
            "best_rule_expr": best["rule_expr"] if best else None,
            "recommended_rule": rec_rule,
            "generated_at": datetime.now(KST).isoformat(),
        }

        blind_comparison = [
            {"strategy": "top2_entry_score", "direction": "combined", **blind_entry},
            {"strategy": "top2_execution_score", "direction": "combined", **blind_exec},
        ]
        if best:
            blind_comparison.append({
                "strategy": "top2_discovered_rule",
                "direction": best["direction"],
                **best,
            })
            blind_comparison[2]["lift_vs_exec_score_pct"] = best["lift_vs_exec_score_pct"]
            blind_comparison[2]["lift_vs_entry_top2_pct"] = best["lift_vs_entry_top2_pct"]

        meta = {
            "train_groups": len(train_groups),
            "blind_groups": len(blind_groups),
            "blind_trades": blind_exec["trade_count"],
            "rules_mined": len(all_rules),
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "top20_execution_rules.csv", top_rules)
        write_csv(self.out_dir / "execution_blind_comparison.csv", blind_comparison)
        write_csv(self.out_dir / "execution_rules_train_rank.csv", train_ranked[:50])

        report = build_execution_rule_report(meta, blind_comparison, top_rules, recommendation)
        report_path = self.out_dir / "execution_replacement_report.md"
        report_path.write_text(report, encoding="utf-8")
        (self.out_dir / "recommended_execution_rule.json").write_text(
            json.dumps(recommendation, indent=2), encoding="utf-8",
        )
        (self.out_dir / "execution_rule_discovery_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in {
            "execution_replacement_report.md": "execution_rule_discovery_v1_report.md",
            "top20_execution_rules.csv": "top20_execution_rules_v1.csv",
            "recommended_execution_rule.json": "recommended_execution_rule_v1.json",
            "execution_blind_comparison.csv": "execution_rule_blind_comparison_v1.csv",
        }.items():
            p = self.out_dir / src
            if p.exists():
                (reports_dir / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        print(f"[EXECUTION RULE DISCOVERY] decision={decision}")
        return {"meta": meta, "recommendation": recommendation, "report_path": str(report_path)}
