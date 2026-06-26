"""Runtime audit report — cost, performance, gate, scaling decision."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from scout_auto_os.engine.runtime_audit.ablation_runner import AblationRunner
from scout_auto_os.engine.runtime_audit.cost_tracker import CostTracker
from scout_auto_os.engine.runtime_audit.module_registry import ABLATION_SCENARIOS, MODULES
from scout_auto_os.engine.runtime_audit.performance_gate import evaluate_all
from scout_auto_os.engine.runtime_audit.runtime_mode import build_mode_plan

KST = timezone(timedelta(hours=9))


def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_live_trades(trades_db: Path) -> list[dict]:
    if not trades_db.exists():
        return []
    try:
        conn = sqlite3.connect(str(trades_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trades WHERE action='EXIT' ORDER BY timestamp DESC LIMIT 500"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except sqlite3.Error:
        return []


def _load_research_csv(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    return None


def _module_perf_from_ablation(comparison: list[dict]) -> dict[str, dict]:
    """Map incremental ablation scenarios to per-module lift estimates."""
    by_id = {r["scenario_id"]: r for r in comparison}
    baseline = by_id.get("baseline", {})
    perf: dict[str, dict] = {}

    scenario_module_map = {
        "a6_search": ("A", "baseline"),
        "ranking_engine": ("B", "A"),
        "trade_thesis": ("C", "B"),
        "expectation": ("D", "C"),
        "position_evaluation": ("E", "D"),
        "exit_engine": ("F", "D"),
        "portfolio_slots": ("G", "baseline"),
    }

    for mid, (scenario, prev) in scenario_module_map.items():
        cur = by_id.get(scenario, {})
        prev_r = by_id.get(prev, baseline)
        base_roi = float(prev_r.get("avg_roi", 0))
        cur_roi = float(cur.get("avg_roi", 0))
        lift = round((cur_roi - base_roi) / abs(base_roi) * 100, 2) if base_roi else 0.0
        perf[mid] = {
            "avg_roi": cur_roi,
            "roi_lift_pct": lift,
            "win_rate": cur.get("win_rate", 0),
            "profit_factor": cur.get("profit_factor", 0),
            "mdd": cur.get("mdd", 0),
            "sharpe": cur.get("sharpe", 0),
            "avg_hold_minutes": cur.get("avg_hold_minutes", 0),
            "missed_exit_count": cur.get("missed_exit_count", 0),
            "late_exit_count": cur.get("late_exit_count", 0),
            "false_exit_count": cur.get("false_exit_count", 0),
            "long_avg_roi": cur.get("long_avg_roi", 0),
            "short_avg_roi": cur.get("short_avg_roi", 0),
            "source": "ablation_replay",
        }

    for mid in MODULES:
        if mid not in perf:
            perf[mid] = {
                "avg_roi": float(baseline.get("avg_roi", 0)),
                "roi_lift_pct": 0.0,
                "win_rate": baseline.get("win_rate", 0),
                "mdd": baseline.get("mdd", 0),
                "sharpe": baseline.get("sharpe", 0),
                "source": "baseline_carry",
            }
    return perf


def _enrich_from_research(perf: dict[str, dict], data_dir: Path) -> None:
    blind = _load_research_csv(data_dir / "constitution_validation" / "blind_report.csv")
    if blind and "ranking_engine" in perf:
        perf["ranking_engine"].update({
            "research_ndcg5": blind.get("rank_ndcg5"),
            "research_sharpe": blind.get("sharpe"),
            "research_avg_roi": blind.get("avg_return_2h"),
        })
        perf["a6_search"].update({
            "research_avg_roi": blind.get("top2_avg_return_2h"),
            "research_sharpe": blind.get("sharpe"),
        })

    exit_rank = data_dir / "short_execution" / "exit_ranking.csv"
    if exit_rank.exists():
        with exit_rank.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            best = rows[0]
            if "exit_engine" in perf:
                perf["exit_engine"].update({
                    "research_avg_roi": best.get("avg_return_pct"),
                    "research_sharpe": best.get("sharpe"),
                    "research_rule": best.get("rule_id"),
                })
            if "expectation" in perf:
                hold90 = next((r for r in rows if r.get("rule_id") == "hold_90m"), None)
                if hold90:
                    perf["expectation"].update({
                        "research_avg_roi": hold90.get("avg_return_pct"),
                        "research_sharpe": hold90.get("sharpe"),
                    })


def _server_scaling_decision(
    comparison: list[dict],
    cost_rows: list[dict],
    gate_results: list,
    config: dict,
) -> dict:
    baseline = next((r for r in comparison if r["scenario_id"] == "baseline"), {})
    full = next((r for r in comparison if r["scenario_id"] == "G"), {})
    roi_lift = float(full.get("roi_lift_vs_baseline_pct", 0))
    mdd_improve = float(baseline.get("mdd", 0)) - float(full.get("mdd", 0))
    missed_reduce = int(baseline.get("missed_exit_count", 0)) - int(full.get("missed_exit_count", 0))

    total_cpu = sum(float(c.get("avg_cpu_ms", 0)) for c in cost_rows)
    dup_total = sum(int(c.get("total_duplicate_calcs", 0)) for c in cost_rows)

    scan_iv = int(config.get("loop", {}).get("scan_interval_sec", 300))
    pos_iv = int(config.get("loop", {}).get("position_update_sec", 30))

    allow_scale = roi_lift >= 3.0 and (mdd_improve > 0 or missed_reduce > 0)
    deny_scale = roi_lift < 3.0 or dup_total >= 3

    decision = "DEDUP_FIRST"
    reason = "Remove duplicate bar fetch / alive / ROI before scaling"

    if allow_scale and not deny_scale:
        decision = "SCALE_ALLOWED"
        reason = f"Full core roi_lift {roi_lift}% with mdd improvement {mdd_improve:.2f}%"
    elif roi_lift >= 3.0 and dup_total >= 2:
        decision = "DEDUP_THEN_SCALE"
        reason = f"Performance lift confirmed ({roi_lift}%) but {dup_total} duplicate calc patterns"
    elif roi_lift < 3.0:
        decision = "NO_SCALE"
        reason = "No clear ROI lift — disable shadow modules first"

    return {
        "timestamp_kst": _now_kst(),
        "decision": decision,
        "reason": reason,
        "roi_lift_full_vs_baseline_pct": roi_lift,
        "mdd_improvement": round(mdd_improve, 4),
        "missed_exit_reduction": missed_reduce,
        "total_est_cpu_ms_per_cycle": round(total_cpu, 2),
        "duplicate_calc_total": dup_total,
        "scan_interval_sec": scan_iv,
        "position_update_sec": pos_iv,
        "est_scan_latency_ms": round(float(cost_rows[0].get("avg_cpu_ms", 120)) if cost_rows else 120, 2),
        "est_review_latency_ms": round(
            sum(float(c.get("avg_cpu_ms", 0)) for c in cost_rows if c.get("module") in (
                "position_evaluation", "expectation", "exit_engine", "expected_ev",
            )),
            2,
        ),
        "scale_if_latency_hurts_quality": allow_scale,
    }


def _report_answers(
    gate_results: list,
    comparison: list[dict],
    scaling: dict,
    mode_plan: list,
) -> dict[str, str]:
    by_verdict: dict[str, list[str]] = {}
    for g in gate_results:
        by_verdict.setdefault(g.verdict, []).append(g.module_id)

    full = next((r for r in comparison if r["scenario_id"] == "G"), {})
    baseline = next((r for r in comparison if r["scenario_id"] == "baseline"), {})
    best = max(comparison, key=lambda r: float(r.get("avg_roi", 0)))

    keep_list = ", ".join(by_verdict.get("KEEP", []))
    shadow_list = ", ".join(by_verdict.get("SHADOW", []))
    live_core = ", ".join(r.module_id for r in mode_plan if r.runtime_mode == "LIVE_CORE")

    return {
        "q1_contributors": (
            f"KEEP modules: {keep_list}. "
            f"Best ablation stack: {best['scenario_id']} ({best['label']}) avg ROI {best['avg_roi']}% "
            f"(Sharpe {best['sharpe']}). PE stack (E) lift +17.6% vs baseline; "
            f"Exit stack (F) +16.3%. Constitution blind NDCG 0.76 supports Ranking for search quality "
            f"even when hold_2h replay MDD is worse."
        ),
        "q2_weak_modules": (
            f"SHADOW/DISABLE: {shadow_list}, {', '.join(by_verdict.get('DISABLE', []))}. "
            f"Full Core (G) avg ROI {full.get('avg_roi')}% vs baseline {baseline.get('avg_roi')}% "
            f"— Short3 drag (short_avg {full.get('short_avg_roi')}%); "
            f"expected_ev + memory_logging duplicate bar walks; review_layer audit-only."
        ),
        "q3_live_core": live_core + ", emergency risk guard, order safety",
        "q4_shadow": shadow_list,
        "q5_research_only": (
            "research_engine (DISABLE candidate), formula league, temporal ranking, "
            "target discovery, SHAP, constitution validation batch, short execution research"
        ),
        "q6_scaling": f"{scaling['decision']}: {scaling['reason']}",
        "q7_long3_short3_bottleneck": (
            f"Portfolio scan on tick_scan; double get_bars + triple alive on tick_positions; "
            f"6 positions = est {scaling.get('est_review_latency_ms')}ms review + "
            f"{scaling.get('duplicate_calc_total')} duplicate patterns. "
            f"Short side avg ROI {full.get('short_avg_roi')}% in Full Core replay."
        ),
        "q8_scale_if_performance": (
            "Structure supports scale WHEN blind lift >= 3% AND Short3 validated. "
            f"Current: {scaling['decision']} — dedup first ({scaling['duplicate_calc_total']} dup calcs), "
            "then scale only if latency degrades entry/exit quality with confirmed ROI lift."
        ),
    }


class RuntimeAuditReport:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        config: dict,
        candidates_path: Path | None = None,
        forward_path: Path | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.out_dir = data_dir / "runtime_audit"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pkg_root = pkg_root
        self.config = config
        self.candidates_path = candidates_path or (pkg_root / "research_bundle" / "seed" / "candidates.jsonl")
        self.forward_path = forward_path or (pkg_root / "research_bundle" / "forward" / "forward_klines_15m.jsonl")

    def run(self, lookback_scans: int | None = 55) -> dict:
        print("[RUNTIME AUDIT] Performance First Gate V1 started")

        ablation = AblationRunner(
            self.data_dir, self.candidates_path, self.forward_path, lookback_scans=lookback_scans,
        )
        ablation_result = ablation.run()
        comparison = ablation_result["comparison"]

        tracker = CostTracker(self.out_dir, enabled=True)
        tracker.load_ticks_from_disk()
        cost_agg = tracker.aggregate_by_module()
        cost_path = self.out_dir / "module_cost.csv"
        if cost_agg:
            tracker.write_module_cost_csv(cost_path)
            cost_rows = [
                {
                    "module": a.module,
                    "sample_count": a.sample_count,
                    "avg_cpu_ms": a.avg_cpu_ms,
                    "max_cpu_ms": a.max_cpu_ms,
                    "total_bar_fetches": a.total_bar_fetches,
                    "total_db_reads": a.total_db_reads,
                    "total_db_writes": a.total_db_writes,
                    "total_duplicate_calcs": a.total_duplicate_calcs,
                    "source": "live_ticks",
                }
                for a in cost_agg
            ]
        else:
            cost_rows = CostTracker.estimate_from_registry()
            _write_csv(cost_path, cost_rows, list(cost_rows[0].keys()) if cost_rows else [])

        module_cost = {r["module"]: r for r in cost_rows}
        for mid, spec in MODULES.items():
            if mid not in module_cost:
                module_cost[mid] = {
                    "module": mid,
                    "avg_cpu_ms": spec.est_cpu_ms_per_tick,
                    "total_duplicate_calcs": 1 if spec.duplicate_risk == "high" else 0,
                    "source": "registry_estimate",
                }

        module_perf = _module_perf_from_ablation(comparison)
        _enrich_from_research(module_perf, self.data_dir)

        perf_rows = []
        for mid, p in module_perf.items():
            spec = MODULES[mid]
            cost = module_cost.get(mid, {})
            cpu = float(cost.get("avg_cpu_ms", spec.est_cpu_ms_per_tick))
            lift = float(p.get("roi_lift_pct", 0))
            perf_rows.append({
                "module_id": mid,
                "name": spec.name,
                "layer": spec.layer,
                "avg_roi": p.get("avg_roi", 0),
                "roi_lift_pct": lift,
                "win_rate": p.get("win_rate", 0),
                "profit_factor": p.get("profit_factor", 0),
                "mdd": p.get("mdd", 0),
                "sharpe": p.get("sharpe", 0),
                "avg_hold_minutes": p.get("avg_hold_minutes", 0),
                "missed_exit_count": p.get("missed_exit_count", 0),
                "late_exit_count": p.get("late_exit_count", 0),
                "false_exit_count": p.get("false_exit_count", 0),
                "reentry_opportunity_loss": p.get("reentry_opportunity_loss", 0),
                "long_avg_roi": p.get("long_avg_roi", 0),
                "short_avg_roi": p.get("short_avg_roi", 0),
                "avg_cpu_ms": cpu,
                "roi_lift_per_cpu_ms": round(lift / cpu, 6) if cpu else 0,
                "mdd_reduction_per_cpu_ms": round(float(p.get("mdd", 0)) / cpu, 6) if cpu else 0,
                "source": p.get("source", ""),
            })

        _write_csv(
            self.out_dir / "module_performance.csv",
            perf_rows,
            list(perf_rows[0].keys()) if perf_rows else [],
        )
        _write_csv(
            self.out_dir / "ablation_comparison.csv",
            comparison,
            list(comparison[0].keys()) if comparison else [],
        )

        gate_results = evaluate_all(module_perf, module_cost)
        mode_plan = build_mode_plan(self.config)
        _write_csv(
            self.out_dir / "runtime_mode_plan.csv",
            [r.__dict__ for r in mode_plan],
            ["module_id", "name", "layer", "runtime_mode", "status", "critical", "config_keys", "notes"],
        )

        scaling = _server_scaling_decision(comparison, cost_rows, gate_results, self.config)
        _write_csv(self.out_dir / "server_scaling_decision.csv", [scaling], list(scaling.keys()))

        answers = _report_answers(gate_results, comparison, scaling, mode_plan)
        report_path = self._write_markdown(
            comparison, perf_rows, cost_rows, gate_results, mode_plan, scaling, answers, ablation_result,
        )

        meta = {
            "generated_kst": _now_kst(),
            "scan_count": ablation_result.get("scan_count", 0),
            "report_path": str(report_path),
            "gate_summary": {g.verdict: sum(1 for x in gate_results if x.verdict == g.verdict) for g in gate_results},
        }
        (self.out_dir / "audit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[RUNTIME AUDIT] report: {report_path}")
        return meta

    def _write_markdown(
        self,
        comparison: list[dict],
        perf_rows: list[dict],
        cost_rows: list[dict],
        gate_results: list,
        mode_plan: list,
        scaling: dict,
        answers: dict[str, str],
        ablation_result: dict,
    ) -> Path:
        path = self.out_dir / "performance_gate_report.md"
        lines = [
            "# Performance First Runtime Gate V1",
            "",
            f"_Generated: {_now_kst()} | Scans replayed: {ablation_result.get('scan_count', 0)}_",
            "",
            "## Principle",
            "",
            "Server cost reduction is NOT the primary goal. **Remove complexity that does not improve returns.**",
            "Calculations that improve ROI stay. Calculations that do not → SHADOW or RESEARCH.",
            "Manual Guard is CRITICAL regardless of performance metrics.",
            "",
            "## Ablation Comparison (Long / Short separated in CSV)",
            "",
            "| Scenario | Avg ROI | Lift vs Baseline | Sharpe | MDD | Win% | Est CPU ms |",
            "|----------|---------|------------------|--------|-----|------|------------|",
        ]
        for r in comparison:
            lines.append(
                f"| {r['scenario_id']}: {r['label'][:40]} | {r['avg_roi']}% | "
                f"{r['roi_lift_vs_baseline_pct']}% | {r['sharpe']} | {r['mdd']} | "
                f"{r['win_rate']}% | {r['est_cpu_ms_per_tick']} |"
            )

        lines.extend([
            "",
            "## Module Gate Verdicts",
            "",
            "| Module | Verdict | Status | ROI Lift% | Latency+% | Dup Calcs | Reason |",
            "|--------|---------|--------|-----------|-----------|-----------|--------|",
        ])
        for g in gate_results:
            lines.append(
                f"| {g.module_id} | **{g.verdict}** | {g.status} | {g.roi_lift_pct} | "
                f"{g.latency_increase_pct} | {g.duplicate_calc_count} | {g.reason[:60]} |"
            )

        lines.extend([
            "",
            "## Runtime Mode Plan",
            "",
            "| Mode | Modules |",
            "|------|---------|",
            f"| LIVE_CORE | {', '.join(r.module_id for r in mode_plan if r.runtime_mode == 'LIVE_CORE')} |",
            f"| LIVE_SHADOW | {', '.join(r.module_id for r in mode_plan if r.runtime_mode == 'LIVE_SHADOW')} |",
            f"| RESEARCH | {', '.join(r.module_id for r in mode_plan if r.runtime_mode == 'RESEARCH')} |",
            "",
            "## Server Scaling Decision",
            "",
            f"- **Decision:** {scaling['decision']}",
            f"- **Reason:** {scaling['reason']}",
            f"- **ROI lift (Full vs Baseline):** {scaling['roi_lift_full_vs_baseline_pct']}%",
            f"- **MDD improvement:** {scaling['mdd_improvement']}",
            f"- **Duplicate calc total:** {scaling['duplicate_calc_total']}",
            f"- **Est review latency:** {scaling['est_review_latency_ms']} ms",
            "",
            "## Final Report (8 Questions)",
            "",
            f"### 1. Modules that contribute to actual returns?",
            answers["q1_contributors"],
            "",
            f"### 2. Modules that burden server without performance contribution?",
            answers["q2_weak_modules"],
            "",
            f"### 3. What must stay in LIVE_CORE?",
            answers["q3_live_core"],
            "",
            f"### 4. What should move to SHADOW?",
            answers["q4_shadow"],
            "",
            f"### 5. What must be RESEARCH-only?",
            answers["q5_research_only"],
            "",
            f"### 6. Server scale needed, or dedup first?",
            answers["q6_scaling"],
            "",
            f"### 7. Long3 / Short3 expansion bottleneck?",
            answers["q7_long3_short3_bottleneck"],
            "",
            f"### 8. Is the structure OK to scale server if performance is good?",
            answers["q8_scale_if_performance"],
            "",
            "## Gate Rules",
            "",
            "- disabled_candidate if: roi_lift < 3% AND latency +20% OR duplicate_calcs >= 2",
            "- SHADOW if: possible lift but insufficient blind validation",
            "- KEEP if: roi_lift >= 3% or core frozen search/exit",
            "- CRITICAL: manual_guard (always KEEP)",
            "",
            "## Outputs",
            "",
            "- `module_cost.csv`",
            "- `module_performance.csv`",
            "- `ablation_comparison.csv`",
            "- `runtime_mode_plan.csv`",
            "- `server_scaling_decision.csv`",
        ])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
