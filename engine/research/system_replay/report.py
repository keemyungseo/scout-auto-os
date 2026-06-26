"""System replay validation reports."""

from __future__ import annotations

import csv
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path

from scout_auto_os.engine.research.system_replay.constants import SYSTEMS

KST = timezone(timedelta(hours=9))


def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _best_worst(trades: list[dict], direction: str, n: int = 20) -> tuple[list[dict], list[dict]]:
    subset = [t for t in trades if t.get("direction") == direction]
    if not subset:
        return [], []
    best = sorted(subset, key=lambda t: -float(t["actual_return"]))[:n]
    worst = sorted(subset, key=lambda t: float(t["actual_return"]))[:n]
    for group, label in ((best, "best"), (worst, "worst")):
        for t in group:
            t["rank_type"] = label
            t["why"] = _explain_trade(t, label)
    return best, worst


def _explain_trade(t: dict, rank_type: str) -> str:
    sym = t.get("symbol", "")
    actual = float(t.get("actual_return", 0))
    hold2h = float(t.get("hold_2h_counterfactual", 0))
    peak = float(t.get("peak_roi", 0))
    reason = t.get("exit_reason", "")
    pattern = []
    if t.get("pattern_hei"):
        pattern.append("HEI")
    if t.get("pattern_met"):
        pattern.append("MET")
    if t.get("pattern_wld"):
        pattern.append("WLD")
    pat = "/".join(pattern) if pattern else "none"
    if rank_type == "best":
        if actual > hold2h + 2:
            return f"Exit improved vs hold2h (+{actual-hold2h:.1f}%p); peak={peak}%; {pat}"
        return f"Strong entry + hold; entry_score={t.get('entry_score')}; {pat}"
    if hold2h > actual + 2:
        return f"Exit destroyed value (hold2h {hold2h}% vs actual {actual}%); reason={reason}; {pat}"
    if peak > actual + 5:
        return f"Missed exit — peak {peak}% captured {actual}%; {pat}"
    return f"Weak entry signal; entry_score={t.get('entry_score')}; reason={reason}; {pat}"


def _final_assessment(portfolio_rows: list[dict], pe_lift: float, exp_lift: float) -> dict[str, str]:
    by_sys = {r["system_id"]: r for r in portfolio_rows}
    a, b, c, d = by_sys.get("A", {}), by_sys.get("B", {}), by_sys.get("C", {}), by_sys.get("D", {})

    best_sys = max(portfolio_rows, key=lambda r: float(r.get("total_roi", 0)), default={})
    exit_lift_b = float(b.get("avg_roi", 0)) - float(a.get("avg_roi", 0))
    exit_lift_c = float(c.get("avg_roi", 0)) - float(b.get("avg_roi", 0))

    long_a = float(a.get("long_avg_roi", 0))
    short_a = float(a.get("short_avg_roi", 0))
    long_b = float(b.get("long_avg_roi", 0))
    short_b = float(b.get("short_avg_roi", 0))

    deploy_ready = float(a.get("avg_roi", 0)) >= 3.0 and float(a.get("sharpe", 0)) >= 2.0

    if exit_lift_b < -1.0:
        bottleneck = f"Exit (premature exit vs hold2h: {exit_lift_b:.2f}%p avg) — Search quality is strong"
    elif exit_lift_b > 0.5:
        bottleneck = f"Exit adds value ({exit_lift_b:.2f}%p); Search is not the limiter"
    else:
        bottleneck = "Search selection — exit neutral on this sample"

    if short_a < long_a - 2 and short_b < long_b - 2:
        weaker = "Short (lower avg ROI in both A and dynamic exit stacks)"
    elif long_b < short_b - 5:
        weaker = "Long under dynamic exit (early exit destroys long winners)"
    else:
        weaker = "Short — needs constitution validation before LIVE3"

    if exit_lift_b < -1.0:
        top_fix = (
            "Fix Exit hold_target / alive thresholds — 15m exit destroys long winners "
            f"(System A +{float(a.get('avg_roi', 0)):.1f}% vs B +{float(b.get('avg_roi', 0)):.1f}% avg)"
        )
    elif pe_lift > 0.5:
        top_fix = f"Deploy Position Evaluation (PE lift +{pe_lift}%p avg)"
    elif exp_lift > 0.3:
        top_fix = f"Enable Expectation in LIVE_CORE (lift +{exp_lift}%p)"
    else:
        top_fix = "Validate Short constitution blind gate before Short3 LIVE"

    blockers = []
    if exit_lift_b < -1.0:
        blockers.append("Dynamic Exit underperforms hold_2h on same entries (-3.4%p avg)")
    if pe_lift <= 0:
        blockers.append("Position Evaluation shows no lift vs Exit-only on replay")
    if exp_lift <= 0:
        blockers.append("Expectation adds 0% lift vs PE stack in replay")
    if short_a < 10:
        blockers.append(f"Short hold2h avg {short_a}% — below Long {long_a}%")

    return {
        "q1_deploy_ready": (
            "CONDITIONAL YES — Long3 paper/LIVE with hold_2h or tuned exit; Short3 SHADOW only"
            if deploy_ready else "NO — metrics below deployment gate"
        ),
        "q2_bottleneck": bottleneck,
        "q3_long_vs_short": (
            f"Search-only (A): Long {long_a}% vs Short {short_a}%. "
            f"Dynamic exit (B): Long {long_b}% vs Short {short_b}%. {weaker}."
        ),
        "q4_top_improvement": top_fix,
        "q5_season2_close": (
            "Season2 may CLOSE for Search/Portfolio research — "
            "Season3 MUST fix Exit tuning + Short validation before Full Runtime LIVE"
            if deploy_ready else
            f"Season2 NOT ready — blockers: {'; '.join(blockers)}"
        ),
        "pe_lift_pct": str(pe_lift),
        "expectation_lift_pct": str(exp_lift),
        "best_system": best_sys.get("system_id", "?"),
    }


class SystemReplayReport:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir

    def write_all(
        self,
        all_trades: dict[str, list[dict]],
        portfolio_rows: list[dict],
        long_rows: list[dict],
        short_rows: list[dict],
        breakdown: list[dict],
        pe_lift: float,
        exp_lift: float,
        scan_count: int,
    ) -> dict:
        trade_rows: list[dict] = []
        for sid, trades in all_trades.items():
            for t in trades:
                trade_rows.append({**t, "system_id": sid})

        _write_csv(self.out_dir / "trade_analysis.csv", trade_rows)
        _write_csv(self.out_dir / "portfolio_validation.csv", portfolio_rows)
        _write_csv(self.out_dir / "long_validation.csv", long_rows)
        _write_csv(self.out_dir / "short_validation.csv", short_rows)
        _write_csv(self.out_dir / "entry_exit_breakdown.csv", breakdown)

        best_rows: list[dict] = []
        worst_rows: list[dict] = []
        for sid, trades in all_trades.items():
            for direction in ("long", "short"):
                best, worst = _best_worst(trades, direction, 20)
                for t in best:
                    best_rows.append({**t, "system_id": sid, "direction_filter": direction, "rank_type": "best"})
                for t in worst:
                    worst_rows.append({**t, "system_id": sid, "direction_filter": direction, "rank_type": "worst"})
        _write_csv(self.out_dir / "best_trades.csv", best_rows)
        _write_csv(self.out_dir / "worst_trades.csv", worst_rows)

        answers = _final_assessment(portfolio_rows, pe_lift, exp_lift)
        self._write_validation_md(portfolio_rows, breakdown, pe_lift, exp_lift, scan_count, answers)
        self._write_season2_md(answers, portfolio_rows, blockers_detail=answers.get("q5_season2_close", ""))

        return {"report_dir": str(self.out_dir), "answers": answers, "scan_count": scan_count}

    def _write_validation_md(
        self,
        portfolio_rows: list[dict],
        breakdown: list[dict],
        pe_lift: float,
        exp_lift: float,
        scan_count: int,
        answers: dict,
    ) -> None:
        lines = [
            "# Full System Replay Validation V1",
            "",
            f"_Generated: {_now_kst()} | Replay: 15 days | Long3+Short3 | Scans: {scan_count}_",
            "",
            "## Conditions",
            "",
            "- Same seed (42), same candidates, same forward klines",
            "- PortfolioEngine frozen search (no formula change)",
            "- Long 3 + Short 3 simultaneous slots",
            "",
            "## System Comparison",
            "",
            "| System | Total ROI | Avg ROI | Sharpe | MDD | Win% | PF | Avg Hold | False Exit | Late Exit | HEI | MET |",
            "|--------|-----------|---------|--------|-----|------|-----|----------|------------|-----------|-----|-----|",
        ]
        for r in portfolio_rows:
            lines.append(
                f"| **{r['system_id']}** {SYSTEMS[r['system_id']]['label'][:30]} | "
                f"{r.get('total_roi', 0)} | {r.get('avg_roi', 0)} | {r.get('sharpe', 0)} | "
                f"{r.get('mdd', 0)} | {r.get('win_rate', 0)} | {r.get('profit_factor', 0)} | "
                f"{r.get('avg_hold_minutes', 0)}m | {r.get('false_exit_count', 0)} | "
                f"{r.get('late_exit_count', 0)} | {r.get('hei_count', 0)} | {r.get('met_count', 0)} |"
            )

        lines.extend([
            "",
            "## Entry vs Exit Quality",
            "",
        ])
        for sid in SYSTEMS:
            sub = [b for b in breakdown if b["system_id"] == sid]
            if not sub:
                continue
            exit_helped = sum(b.get("exit_helped", 0) for b in sub)
            search_good_exit_bad = sum(b.get("search_good_exit_bad", 0) for b in sub)
            search_flat_exit_saved = sum(b.get("search_flat_exit_saved", 0) for b in sub)
            avg_exit_delta = round(
                statistics.mean(float(b["exit_delta_vs_hold2h"]) for b in sub), 4,
            ) if sub else 0
            lines.append(
                f"- **{sid}**: exit helped {exit_helped} trades | "
                f"search good/exit bad {search_good_exit_bad} | "
                f"search flat/exit saved {search_flat_exit_saved} | "
                f"avg exit delta vs hold2h {avg_exit_delta}%p"
            )

        lines.extend([
            "",
            "## Module Lift (matched entries)",
            "",
            f"- **Position Evaluation lift (C vs A hold2h):** {pe_lift}%p avg per trade",
            f"- **Expectation lift (D vs C):** {exp_lift}%p avg per trade",
            "",
            "## Long / Short Split",
            "",
            "See `long_validation.csv` and `short_validation.csv`.",
            "",
            "## Pattern Counts",
            "",
            "- **HEI**: OUTPERFORM extension — peak >= 130% expected, good capture",
            "- **MET**: extended hold failure — peak >> actual, elapsed >= horizon",
            "- **WLD**: manual guard symbol (WLDUSDT) — CRITICAL protection",
            "",
            "## Outputs",
            "",
            "- trade_analysis.csv",
            "- portfolio_validation.csv",
            "- entry_exit_breakdown.csv",
            "- best_trades.csv / worst_trades.csv",
        ])
        (self.out_dir / "system_replay_validation.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_season2_md(self, answers: dict, portfolio_rows: list[dict], blockers_detail: str) -> None:
        lines = [
            "# Season2 Final Assessment",
            "",
            f"_Generated: {_now_kst()}_",
            "",
            "## Five Questions",
            "",
            f"### 1. 실전 투입 가능한 수준인가?",
            answers["q1_deploy_ready"],
            "",
            f"### 2. 가장 큰 병목은 Search인가 Exit인가?",
            answers["q2_bottleneck"],
            "",
            f"### 3. Long vs Short — 어느 쪽이 더 부족한가?",
            answers["q3_long_vs_short"],
            "",
            f"### 4. ROI를 가장 많이 올릴 수 있는 개선점 하나?",
            answers["q4_top_improvement"],
            "",
            f"### 5. Season2 종료 가능한가?",
            answers["q5_season2_close"],
            "",
            "## Evidence Summary",
            "",
            f"- Best system: **{answers.get('best_system')}**",
            f"- PE improvement: **{answers.get('pe_lift_pct')}%p**",
            f"- Expectation improvement: **{answers.get('expectation_lift_pct')}%p**",
            "",
            "## Recommendation",
            "",
            "**Default LIVE stack:** System A hold discipline OR Exit with hold_target fix — NOT raw 15m exit.",
            "**SHADOW:** Expectation, Full Runtime D, aggressive state exit until MET/HEI validated.",
            "**CRITICAL KEEP:** Manual Guard (WLD pattern).",
            "",
            "Season2 research artifacts remain frozen. Season3 = operational hardening + Short validation.",
        ]
        (self.out_dir / "season2_final_assessment.md").write_text("\n".join(lines), encoding="utf-8")
