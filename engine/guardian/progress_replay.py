"""Guardian progress replay on 157-trade bundle."""

from __future__ import annotations

from pathlib import Path

from scout_auto_os.engine.guardian.decision_engine import (
    contract_from_replay_row,
    position_from_replay_outcome,
)
from scout_auto_os.engine.guardian.progress_engine import evaluate_progress
from scout_auto_os.engine.guardian.progress_output import (
    PROGRESS_CSV,
    REPORT_MD,
    SUMMARY_JSON,
    build_summary,
    write_progress_csv,
    write_progress_report,
    write_summary_json,
)
from scout_auto_os.engine.predator.inference import load_replay_bundle

DEFAULT_ELAPSED_MIN = 240


def _met_scenario_probe(config: dict | None = None) -> dict:
    contract = {
        "contract_id": "met_scenario",
        "symbol": "METUSDT",
        "side": "long",
        "expected_roi": 3.0,
        "expected_peak_roi": 5.0,
        "expected_drawdown": 8.0,
        "expected_win_prob": 0.6,
        "value_score": 55.0,
        "dna_type": "TYPE_1",
        "exit_profile": "early_exit",
        "expected_horizon": 90,
        "gate_action": "ENTER",
    }
    position = {
        "current_roi": 1.0,
        "elapsed_minutes": 2880,
        "peak_roi": 7.0,
        "drawdown_from_peak": 6.0,
    }
    r = evaluate_progress(contract, position, config=config)
    return {
        "guardian_state": r.guardian_state,
        "recommendation": r.recommendation,
        "is_thesis_failed": r.guardian_state == "THESIS_FAILED",
        "reason": r.reason,
    }


def run_progress_replay(
    data_dir: Path,
    *,
    elapsed_minutes: int = DEFAULT_ELAPSED_MIN,
    config: dict | None = None,
) -> dict:
    bundle = load_replay_bundle(data_dir / "trade_dna")
    results = []
    for row in bundle:
        contract = contract_from_replay_row(row)
        contract["expected_horizon"] = contract.get("expected_horizon") or _horizon(contract)
        contract["contract_id"] = row.get("trade_key", "")
        position = position_from_replay_outcome(row, elapsed_minutes=elapsed_minutes)
        results.append(
            evaluate_progress(
                contract,
                position,
                contract_id=row.get("trade_key", ""),
                config=config,
            )
        )

    scenarios = {"met": _met_scenario_probe(config)}
    summary = build_summary(results, scenarios=scenarios)

    out_dir = data_dir / "guardian"
    csv_path = out_dir / PROGRESS_CSV
    json_path = out_dir / SUMMARY_JSON
    report_path = out_dir / REPORT_MD

    write_progress_csv(csv_path, [r.to_row() for r in results])
    write_summary_json(json_path, summary)
    write_progress_report(report_path, results, summary)

    return {
        "ok": True,
        "trade_count": len(results),
        "progress_csv": str(csv_path),
        "summary_json": str(json_path),
        "report_md": str(report_path),
        "summary": summary,
        "met_thesis_failed": scenarios["met"]["is_thesis_failed"],
    }


def _horizon(contract: dict) -> int:
    from scout_auto_os.engine.guardian.decision_rules import expected_horizon_minutes
    return expected_horizon_minutes(contract)
