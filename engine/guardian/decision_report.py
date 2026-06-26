"""Guardian replay batch + human-readable report."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from scout_auto_os.engine.guardian.decision_engine import (
    GuardianDecision,
    contract_from_replay_row,
    decide,
    position_from_replay_outcome,
)
from scout_auto_os.engine.guardian.guardian_decision_log import GuardianDecisionLog
from scout_auto_os.engine.predator.inference import load_replay_bundle
from scout_auto_os.storage.db import now_kst

REPORT_NAME = "guardian_decision_report.md"
DEFAULT_ELAPSED_MIN = 240


def run_replay_decisions(
    data_dir: Path,
    *,
    elapsed_minutes: int = DEFAULT_ELAPSED_MIN,
    reset_log: bool = True,
) -> dict:
    """Apply Guardian to 157-trade replay bundle at outcome snapshot."""
    trade_dna = data_dir / "trade_dna"
    out_dir = data_dir / "guardian"
    rows_bundle = load_replay_bundle(trade_dna)

    decisions: list[GuardianDecision] = []
    for row in rows_bundle:
        contract = contract_from_replay_row(row)
        position = position_from_replay_outcome(row, elapsed_minutes=elapsed_minutes)
        d = decide(
            contract,
            position,
            contract_id=row.get("trade_key", ""),
        )
        decisions.append(d)

    logger = GuardianDecisionLog(out_dir)
    if reset_log:
        logger.reset_log()
    decision_rows = [d.to_row() for d in decisions]
    csv_path = logger.write_decisions(decision_rows)

    action_counts = Counter(d.action for d in decisions)
    scenarios = _scenario_checks()

    report_path = write_decision_report(
        out_dir,
        decisions=decisions,
        action_counts=dict(action_counts),
        scenarios=scenarios,
        elapsed_minutes=elapsed_minutes,
    )

    return {
        "ok": True,
        "trade_count": len(decisions),
        "decision_csv": str(csv_path),
        "log_csv": str(logger.log_path),
        "report_md": str(report_path),
        "action_counts": dict(action_counts),
        "scenarios": scenarios,
    }


def _scenario_checks() -> dict:
    """MET extended hold + HEI outperformance — explicit rule probes."""
    from scout_auto_os.engine.guardian.decision_engine import decide as _decide

    met_contract = {
        "symbol": "METUSDT",
        "side": "long",
        "expected_roi": 3.0,
        "expected_peak_roi": 5.0,
        "expected_drawdown": 8.0,
        "exit_profile": "early_exit",
        "early_exit_allowed": True,
        "dna_type": "TYPE_1",
        "gate_action": "ENTER",
    }
    met_position = {
        "current_roi": 1.0,
        "elapsed_minutes": 2880,
        "peak_roi": 7.0,
        "drawdown_from_peak": 6.0,
        "expected_horizon": 90,
    }
    met_dec = _decide(met_contract, met_position, contract_id="met_scenario")

    hei_contract = {
        "symbol": "HEIUSDT",
        "side": "long",
        "expected_roi": 30.0,
        "expected_peak_roi": 53.0,
        "expected_drawdown": 32.0,
        "exit_profile": "runner",
        "early_exit_allowed": False,
        "trail_priority": True,
        "dna_type": "TYPE_0",
        "gate_action": "ENTER",
    }
    hei_position = {
        "current_roi": 35.0,
        "elapsed_minutes": 60,
        "peak_roi": 53.0,
        "drawdown_from_peak": 0.0,
        "expected_horizon": 90,
    }
    hei_dec = _decide(hei_contract, hei_position, contract_id="hei_scenario")

    return {
        "met": {
            "action": met_dec.action,
            "reason": met_dec.reason,
            "blocks_extended_hold": met_dec.action in ("EXIT", "EMERGENCY_EXIT", "REDUCE"),
        },
        "hei": {
            "action": hei_dec.action,
            "reason": hei_dec.reason,
            "switches_to_trail": hei_dec.action == "TRAIL",
        },
    }


def write_decision_report(
    out_dir: Path,
    *,
    decisions: list[GuardianDecision],
    action_counts: dict,
    scenarios: dict,
    elapsed_minutes: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPORT_NAME
    met = scenarios.get("met", {})
    hei = scenarios.get("hei", {})

    lines = [
        "# Guardian Decision Engine V1 — Report",
        "",
        f"**Generated:** {now_kst()}",
        f"**Replay trades:** {len(decisions)}",
        f"**Snapshot elapsed:** {elapsed_minutes}m",
        "",
        "## Action distribution",
        "",
    ]
    for action, count in sorted(action_counts.items()):
        lines.append(f"- {action}: {count}")

    lines.extend([
        "",
        "## MET scenario (2-day extended hold)",
        "",
        f"- Action: **{met.get('action', '—')}**",
        f"- Blocks extended hold: **{met.get('blocks_extended_hold', False)}**",
        f"- Reason: {met.get('reason', '')}",
        "",
        "## HEI scenario (target exceeded)",
        "",
        f"- Action: **{hei.get('action', '—')}**",
        f"- Switches to TRAIL: **{hei.get('switches_to_trail', False)}**",
        f"- Reason: {hei.get('reason', '')}",
        "",
        "## Design",
        "",
        "- Rule-based only — no ML, no training",
        "- Inputs: trade contract + current position snapshot",
        "- Every action includes human-readable reason (no black box)",
        "",
        "## Sample decisions (first 5)",
        "",
    ])
    for d in decisions[:5]:
        lines.append(f"- **{d.symbol}** → {d.action}: {d.reason[:120]}...")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
