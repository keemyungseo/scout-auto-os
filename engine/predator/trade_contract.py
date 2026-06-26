"""Guardian Trade Contract from Value Estimator + DNA."""

from __future__ import annotations

from scout_auto_os.engine.predator.value_gate import exit_profile_for_dna


def build_trade_contract(enriched: dict) -> dict:
    """Contract fields for Guardian — no holding time prediction."""
    dna_type = enriched.get("predicted_dna_type") or enriched.get("dna_type", "UNKNOWN")
    exit_meta = exit_profile_for_dna(dna_type)
    return {
        "symbol": enriched.get("symbol", ""),
        "side": enriched.get("side", "long"),
        "expected_roi": round(float(enriched.get("predicted_roi", 0)), 4),
        "expected_peak_roi": round(float(enriched.get("predicted_peak_roi", 0)), 4),
        "expected_drawdown": round(float(enriched.get("predicted_drawdown", 0)), 4),
        "expected_win_prob": round(float(enriched.get("predicted_win_prob", 0)), 4),
        "value_score": round(float(enriched.get("value_score", 0)), 2),
        "size_multiplier": round(float(enriched.get("recommended_size", 0)), 2),
        "dna_type": dna_type,
        "exit_profile": exit_meta["exit_profile"],
        "early_exit_allowed": exit_meta["early_exit_allowed"],
        "trail_priority": exit_meta["trail_priority"],
        "max_hold_managed_by": exit_meta["max_hold_managed_by"],
        "gate_action": enriched.get("gate_action", ""),
        "gate_reason": enriched.get("gate_reason", ""),
    }
