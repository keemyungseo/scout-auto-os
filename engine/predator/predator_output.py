"""Predator-Long output enrichment with Value Estimator + DNA."""

from __future__ import annotations

from scout_auto_os.engine.predator.trade_contract import build_trade_contract
from scout_auto_os.engine.predator.value_gate import evaluate_gate, is_manual_protected


REQUIRED_OUTPUT_KEYS = (
    "symbol",
    "side",
    "entry_score",
    "direction_confidence",
    "dna_type_probability",
    "predicted_roi",
    "predicted_peak_roi",
    "predicted_drawdown",
    "predicted_win_prob",
    "predicted_sharpe",
    "value_score",
    "recommended_size",
    "trade_contract",
)


def enrich_predator_candidate(
    candidate: dict,
    predictions: dict,
    *,
    position: dict | None = None,
) -> dict:
    """
    Extend a Predator-Long candidate with value gate fields.
    predictions: value_score, predicted_*, dna_type, runner_probability, etc.
    """
    sym = candidate.get("symbol", predictions.get("symbol", ""))
    side = candidate.get("side", predictions.get("direction", "long"))
    dna_type = predictions.get("predicted_dna_type", predictions.get("dna_type", "UNKNOWN"))
    runner_prob = float(predictions.get("runner_probability", predictions.get("dna_type_probability", 0)))
    value_score = float(predictions.get("value_score", 0))

    gate = evaluate_gate(
        value_score,
        dna_type=dna_type,
        runner_probability=runner_prob,
        is_manual_protected=is_manual_protected(position),
    )

    out = {
        "symbol": sym,
        "side": side,
        "entry_score": round(float(candidate.get("entry_score", predictions.get("entry_score", 0))), 4),
        "direction_confidence": round(float(
            candidate.get("direction_confidence", predictions.get("direction_confidence", runner_prob))
        ), 4),
        "dna_type_probability": round(runner_prob, 4),
        "predicted_dna_type": dna_type,
        "predicted_roi": round(float(predictions.get("predicted_roi", 0)), 4),
        "predicted_peak_roi": round(float(predictions.get("predicted_peak_roi", 0)), 4),
        "predicted_drawdown": round(float(predictions.get("predicted_drawdown", 0)), 4),
        "predicted_win_prob": round(float(predictions.get("predicted_win_prob", 0)), 4),
        "predicted_sharpe": round(float(predictions.get("predicted_sharpe", 0)), 4),
        "value_score": round(value_score, 2),
        "recommended_size": gate["recommended_size"],
        "gate_action": gate["action"],
        "gate_reason": gate["reason"],
        "shadow_only": gate.get("shadow_only", False),
    }
    out["trade_contract"] = build_trade_contract(out)
    return out


def validate_predator_output(row: dict) -> list[str]:
    missing = [k for k in REQUIRED_OUTPUT_KEYS if k not in row]
    contract = row.get("trade_contract") or {}
    for ck in (
        "expected_roi", "expected_peak_roi", "expected_drawdown",
        "expected_win_prob", "value_score", "size_multiplier", "dna_type", "exit_profile",
    ):
        if ck not in contract:
            missing.append(f"trade_contract.{ck}")
    if "expected_hold_time" in contract or "predicted_hold_time" in row:
        missing.append("holding_time_must_not_be_present")
    return missing
