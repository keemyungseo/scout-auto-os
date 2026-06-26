"""Execution Rule Generalization Test V1 orchestrator."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.portfolio.backtest import filter_2h_scans
from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.research.execution_generalization.constants import WALK_FORWARD_DAYS
from scout_auto_os.engine.research.execution_generalization.decision import decide_keep_or_reject
from scout_auto_os.engine.research.execution_generalization.metrics import (
    collect_trade_returns,
    equity_max_drawdown,
    evaluate_fold,
    return_stability,
    sharpe_approx,
)
from scout_auto_os.engine.research.execution_generalization.regime import attach_regime_to_groups
from scout_auto_os.engine.research.execution_generalization.report import (
    build_generalization_report,
    build_regime_report,
)
from scout_auto_os.engine.research.execution_generalization.rule_loader import (
    load_frozen_execution_rule,
    resolve_rule_path,
)
from scout_auto_os.engine.research.execution_generalization.splits import (
    expanding_window_splits,
    leave_one_period_out,
    monthly_splits,
    rolling_walk_forward,
    temporal_blind_split,
    weekly_splits,
)
from scout_auto_os.engine.research.execution_rule_discovery.baselines import pick_top2_execution_score
from scout_auto_os.engine.research.execution_rule_discovery.dataset import collect_execution_groups
from scout_auto_os.engine.research.execution_rule_discovery.generator import avg_top2_return
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines

KST = timezone(timedelta(hours=9))


def _groups_for_scans(all_groups: list[list[dict]], scans: set[str]) -> list[list[dict]]:
    return [g for g in all_groups if g[0]["scan_time_kst"] in scans]


def _evaluate_regime_bucket(
    groups: list[list[dict]],
    rule,
    key_fn,
) -> list[dict]:
    buckets: dict[str, list[list[dict]]] = defaultdict(list)
    for g in groups:
        if g[0].get("direction") != rule.direction:
            continue
        buckets[key_fn(g[0])].append(g)

    rows: list[dict] = []
    for key, bg in sorted(buckets.items()):
        fold = evaluate_fold(bg, rule, str(key), "regime")
        if "regime" in key or key in ("bull", "bear", "sideway"):
            fold["regime"] = key
        else:
            fold["volatility_band"] = key
        rows.append(fold)
    return rows


class ExecutionGeneralizationRunner:
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
        self.out_dir = data_dir / "execution_generalization"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path
        self.lookback_days = lookback_days

    @research_safe("execution_generalization")
    def run(self) -> dict:
        print("[EXECUTION GENERALIZATION] started")
        rule_path = resolve_rule_path(self.data_dir, self.pkg_root)
        rule = load_frozen_execution_rule(rule_path)
        print(f"[EXECUTION GENERALIZATION] frozen rule: {rule.rule_expr}")

        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        from scout_auto_os.engine.portfolio.backtest import _parse_scan

        all_scans = sorted(by_scan.keys())
        if all_scans:
            max_dt = _parse_scan(all_scans[-1])
            min_dt = max_dt - timedelta(days=self.lookback_days)
            all_scans = [s for s in all_scans if _parse_scan(s) >= min_dt]
        scans = filter_2h_scans(all_scans)

        engine = PortfolioEngine.from_paths(self.data_dir, self.pkg_root)
        all_groups = collect_execution_groups(by_scan, fwd, scans, engine)
        attach_regime_to_groups(all_groups, by_scan)

        long_groups = [g for g in all_groups if g[0]["direction"] == rule.direction]

        rule_rets = collect_trade_returns(long_groups, rule)
        base_rets = collect_trade_returns(long_groups, None, pick_top2_execution_score, rule.direction)
        days = len({g[0]["scan_time_kst"][:10] for g in long_groups}) or 1

        fold_rows: list[dict] = []
        scan_set = set(scans)

        split_plan = [
            ("monthly", monthly_splits(scans)),
            ("weekly", weekly_splits(scans)),
            ("walk_forward", rolling_walk_forward(scans, WALK_FORWARD_DAYS)),
            ("expanding", expanding_window_splits(scans)),
            ("leave_one_out", leave_one_period_out(scans, "week")),
            ("temporal_blind", temporal_blind_split(scans)),
        ]

        for split_type, splits in split_plan:
            for fold_id, fold_scans in splits.items():
                if split_type == "temporal_blind" and fold_id != "blind_holdout":
                    continue
                fg = _groups_for_scans(all_groups, set(fold_scans) & scan_set)
                if not fg:
                    continue
                row = evaluate_fold(fg, rule, fold_id, split_type)
                fold_rows.append(row)

        monthly_rows = [f for f in fold_rows if f["split_type"] == "monthly"]
        monthly_avgs = [float(f["rule_avg_return_2h"]) for f in monthly_rows if f["rule_trade_count"]]

        regime_rows = _evaluate_regime_bucket(long_groups, rule, lambda r: r.get("regime", "unknown"))
        vol_rows = _evaluate_regime_bucket(long_groups, rule, lambda r: r.get("volatility_band", "unknown"))

        overall_rule = avg_top2_return(long_groups, rule)
        overall_base = avg_top2_return(long_groups, None, pick_top2_execution_score)

        decision = decide_keep_or_reject(
            fold_rows,
            regime_rows,
            float(overall_rule["avg_return_2h"]),
            float(overall_base["avg_return_2h"]),
            len(rule_rets),
        )

        meta = {
            "rule_id": rule.rule_id,
            "rule_expr": rule.rule_expr,
            "direction": rule.direction,
            "scan_count": len(scans),
            "group_count": len(long_groups),
            "overall_rule_avg": overall_rule["avg_return_2h"],
            "overall_base_avg": overall_base["avg_return_2h"],
            "overall_rule_win": overall_rule["win_rate_pct"],
            "overall_base_win": overall_base["win_rate_pct"],
            "overall_rule_trades": len(rule_rets),
            "overall_base_trades": len(base_rets),
            "overall_rule_mdd": equity_max_drawdown(rule_rets),
            "overall_base_mdd": equity_max_drawdown(base_rets),
            "overall_rule_sharpe": sharpe_approx(rule_rets),
            "overall_base_sharpe": sharpe_approx(base_rets),
            "overall_rule_rpd": round(sum(rule_rets) / days, 4),
            "overall_base_rpd": round(sum(base_rets) / days, 4),
            "monthly_stability": return_stability(monthly_avgs),
            "decision": decision,
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "monthly_report.csv", monthly_rows)
        write_csv(self.out_dir / "walkforward_report.csv", [f for f in fold_rows if f["split_type"] == "walk_forward"])
        write_csv(self.out_dir / "generalization_folds.csv", fold_rows)
        write_csv(self.out_dir / "regime_report.csv", regime_rows)
        write_csv(self.out_dir / "volatility_report.csv", vol_rows)

        (self.out_dir / "generalization_report.md").write_text(
            build_generalization_report(meta, fold_rows, decision), encoding="utf-8",
        )
        (self.out_dir / "regime_report.md").write_text(
            build_regime_report(regime_rows, vol_rows), encoding="utf-8",
        )
        (self.out_dir / "generalization_decision.json").write_text(
            json.dumps({**decision, "rule": rule.to_dict(), "meta": meta}, indent=2), encoding="utf-8",
        )

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in {
            "generalization_report.md": "execution_generalization_v1_report.md",
            "regime_report.md": "execution_regime_v1_report.md",
            "monthly_report.csv": "execution_monthly_v1.csv",
            "walkforward_report.csv": "execution_walkforward_v1.csv",
            "generalization_decision.json": "execution_generalization_decision_v1.json",
        }.items():
            p = self.out_dir / src
            if p.exists():
                (reports_dir / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        print(f"[EXECUTION GENERALIZATION] decision={decision['decision']}")
        return {"meta": meta, "decision": decision}
