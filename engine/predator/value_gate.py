"""Predator Value Gate — size tiers and decisions."""

from __future__ import annotations

from enum import Enum


class GateAction(str, Enum):
    SKIP = "SKIP"
    SHADOW_ONLY = "SHADOW_ONLY"
    ENTER = "ENTER"
    NO_ACTION = "NO_ACTION"


# Predator Value Gate V1 tiers (no 90+ band — unused in current sample)
def recommended_size(value_score: float) -> float:
    if value_score < 50:
        return 0.0
    if value_score < 60:
        return 0.1
    if value_score < 70:
        return 0.3
    if value_score < 80:
        return 0.6
    return 1.0


def exit_profile_for_dna(dna_type: str) -> dict:
    if dna_type == "TYPE_0":
        return {
            "exit_profile": "runner",
            "early_exit_allowed": False,
            "trail_priority": True,
            "max_hold_managed_by": "guardian",
        }
    return {
        "exit_profile": "early_exit",
        "early_exit_allowed": True,
        "trail_priority": False,
        "max_hold_managed_by": "guardian",
    }


def evaluate_gate(
    value_score: float,
    *,
    dna_type: str,
    runner_probability: float,
    is_manual_protected: bool = False,
) -> dict:
    """Apply Value Gate rules + DNA type contract."""
    if is_manual_protected:
        return {
            "action": GateAction.NO_ACTION.value,
            "recommended_size": 0.0,
            "reason": "manual_protected",
            "shadow_only": False,
        }

    if dna_type == "TYPE_1" or runner_probability < 0.5:
        return {
            "action": GateAction.SKIP.value,
            "recommended_size": 0.0,
            "reason": "dna_type_failed_momentum",
            "shadow_only": True,
        }

    size = recommended_size(value_score)
    if value_score < 50:
        return {
            "action": GateAction.SKIP.value,
            "recommended_size": 0.0,
            "reason": "value_score_below_50",
            "shadow_only": False,
        }
    if value_score < 60:
        return {
            "action": GateAction.SHADOW_ONLY.value,
            "recommended_size": size,
            "reason": "value_score_50_59",
            "shadow_only": True,
        }
    return {
        "action": GateAction.ENTER.value,
        "recommended_size": size,
        "reason": "value_gate_pass",
        "shadow_only": False,
    }


def is_manual_protected(position: dict | None) -> bool:
    if not position:
        return False
    if str(position.get("source", "")).upper() == "MANUAL":
        return True
    if int(position.get("manual_lock") or 0):
        return True
    if not int(position.get("auto_manage", 1)):
        return True
    return False
