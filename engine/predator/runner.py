"""Predator Value Gate V1 runner."""

from __future__ import annotations

from pathlib import Path

from scout_auto_os.engine.predator.inference import load_replay_bundle
from scout_auto_os.engine.predator.predator_output import validate_predator_output
from scout_auto_os.engine.predator.report import ValueGateReport
from scout_auto_os.engine.predator.shadow_runner import (
    best_missed_trade,
    false_accept_cases,
    false_skip_cases,
    portfolio_metrics,
    process_trade_shadow,
    worst_accepted_trade,
)
from scout_auto_os.engine.research.safe import research_safe


def _compute_verdict(
    baseline: dict,
    gated: dict,
    false_skips: list[dict],
    false_accepts: list[dict],
    contract_ok: bool,
) -> tuple[str, dict]:
    sharpe_improved = gated["sharpe"] > baseline["sharpe"]
    mdd_improved = gated["mdd"] > baseline["mdd"]
    skip_band_bad = gated.get("skipped_avg_roi", 0) < -5
    false_skip_rate = len(false_skips) / max(baseline["trade_count"], 1)

    answers = {
        "Value Gate를 Predator에 연결할 수 있는가?": (
            "Yes — enrich_predator_candidate + trade_contract 연결 완료 (Shadow 모드)"
            if contract_ok else "Partial — contract validation failed"
        ),
        "value_score < 50 skip은 유지할 만한가?": (
            f"Yes — skipped 구간 avg ROI {gated['skipped_avg_roi']:.2f}%, false skip {len(false_skips)}건"
            if skip_band_bad and false_skip_rate < 0.15
            else f"Mixed — false skip {len(false_skips)}건, threshold 재검토 필요"
        ),
        "60~69 / 70~79 size multiplier는 적절한가?": (
            "Reasonable — accepted avg ROI 양수이며 밴드별 분산 존재"
            if gated["accepted_avg_roi"] > 0 else "Needs tuning — accepted trades still weak"
        ),
        "Full Size 대비 Dynamic Size 성과 개선이 재현되는가?": (
            f"Yes — Sharpe {baseline['sharpe']}→{gated['sharpe']}, MDD {baseline['mdd']}→{gated['mdd']}"
            if sharpe_improved and mdd_improved else "Partial — risk-adjusted gain not fully reproduced"
        ),
        "False Skip 위험은 감당 가능한가?": (
            f"Acceptable — {len(false_skips)}건 / {baseline['trade_count']} trades ({false_skip_rate*100:.1f}%)"
            if false_skip_rate <= 0.10 else f"Elevated — {len(false_skips)} good trades missed"
        ),
        "Guardian Contract에 Value Estimator 값을 정상 연결했는가?": (
            "Yes — expected_roi/peak/drawdown/win_prob/value_score/size/dna/exit_profile 포함, hold time 제외"
            if contract_ok else "No — validation errors"
        ),
        "LIVE 적용은 바로 가능한가, 아니면 Shadow가 필요한가?": (
            "Shadow 필요 — 50-59 SHADOW_ONLY 밴드 검증 + false skip/accept 모니터링 후 LIVE"
        ),
    }

    if not contract_ok:
        verdict = "NEEDS_MORE_DATA"
    elif sharpe_improved and mdd_improved and len(false_accepts) <= 5:
        verdict = "KEEP_SHADOW_ONLY"
    elif sharpe_improved and gated["accepted_avg_roi"] > 5 and false_skip_rate < 0.08:
        verdict = "KEEP_LIVE_READY"
    elif gated["total_roi"] < baseline["total_roi"] and not sharpe_improved:
        verdict = "REJECT"
    else:
        verdict = "KEEP_SHADOW_ONLY"

    return verdict, answers


class PredatorValueGateRunner:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.trade_dna_dir = data_dir / "trade_dna"
        self.out_dir = data_dir / "value_gate"

    @research_safe("predator_value_gate_v1")
    def run(self) -> dict:
        print("[VALUE GATE] V1 started")
        rows = load_replay_bundle(self.trade_dna_dir)
        print(f"[VALUE GATE] replay trades={len(rows)}")

        shadow_trades = [process_trade_shadow(r) for r in rows]
        contract_errors = []
        for t in shadow_trades:
            errs = validate_predator_output(t["enriched"])
            if errs:
                contract_errors.extend(errs)
        contract_ok = len(contract_errors) == 0

        baseline_m = portfolio_metrics(shadow_trades, "baseline")
        gated_m = portfolio_metrics(shadow_trades, "gated")
        fskips = false_skip_cases(shadow_trades)
        faccepts = false_accept_cases(shadow_trades)
        best_miss = best_missed_trade(shadow_trades)
        worst_acc = worst_accepted_trade(shadow_trades)

        verdict, answers = _compute_verdict(baseline_m, gated_m, fskips, faccepts, contract_ok)

        reporter = ValueGateReport(self.out_dir)
        report_path = reporter.write_all(
            shadow_trades, baseline_m, gated_m, fskips, faccepts,
            best_miss, worst_acc, verdict, answers,
        )

        print(f"[VALUE GATE] baseline Sharpe={baseline_m['sharpe']} gated={gated_m['sharpe']}")
        print(f"[VALUE GATE] false_skip={len(fskips)} false_accept={len(faccepts)} verdict={verdict}")
        print(f"[VALUE GATE] report: {report_path}")
        return {
            "n_trades": len(rows),
            "verdict": verdict,
            "baseline_sharpe": baseline_m["sharpe"],
            "gated_sharpe": gated_m["sharpe"],
            "false_skip_count": len(fskips),
            "false_accept_count": len(faccepts),
            "report_path": str(report_path),
        }
