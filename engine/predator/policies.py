"""Value Gate policy definitions A–E (no model retrain)."""

from __future__ import annotations

from scout_auto_os.engine.predator.value_gate import GateAction

# Policy C fixed thresholds (set once from TYPE_0 winner descriptive stats, not grid search)
POLICY_C_DD_MAX = 28.0
POLICY_C_WIN_MIN = 0.55
POLICY_C_RUNNER_MIN = 0.70


def _dna_skip(row: dict) -> bool:
    return row["predicted_dna_type"] == "TYPE_1" or float(row["runner_probability"]) < 0.5


def _result(action: str, size: float, reason: str, *, shadow_only: bool = False) -> dict:
    return {
        "action": action,
        "recommended_size": size,
        "reason": reason,
        "shadow_only": shadow_only,
    }


def _tier_size(score: float, bands: list[tuple[float, float, float]]) -> float:
    for lo, hi, mult in bands:
        if lo <= score < hi:
            return mult
    return bands[-1][2] if bands else 0.0


def policy_a_v1(row: dict) -> dict:
    """Current V1."""
    if _dna_skip(row):
        return _result(GateAction.SKIP.value, 0.0, "dna_type_failed_momentum", shadow_only=True)
    score = float(row["value_score"])
    if score < 50:
        return _result(GateAction.SKIP.value, 0.0, "value_score_below_50")
    if score < 60:
        return _result(GateAction.SHADOW_ONLY.value, 0.1, "value_score_50_59", shadow_only=True)
    if score < 70:
        return _result(GateAction.ENTER.value, 0.3, "value_score_60_69")
    if score < 80:
        return _result(GateAction.ENTER.value, 0.6, "value_score_70_79")
    return _result(GateAction.ENTER.value, 1.0, "value_score_80_plus")


def policy_b_soft_50s(row: dict) -> dict:
    """50–59 → ENTER 0.2x."""
    if _dna_skip(row):
        return _result(GateAction.SKIP.value, 0.0, "dna_type_failed_momentum", shadow_only=True)
    score = float(row["value_score"])
    if score < 50:
        return _result(GateAction.SKIP.value, 0.0, "value_score_below_50")
    if score < 60:
        return _result(GateAction.ENTER.value, 0.2, "value_score_50_59_soft")
    if score < 70:
        return _result(GateAction.ENTER.value, 0.3, "value_score_60_69")
    if score < 80:
        return _result(GateAction.ENTER.value, 0.6, "value_score_70_79")
    return _result(GateAction.ENTER.value, 1.0, "value_score_80_plus")


def policy_c_type0_exception(row: dict) -> dict:
    """score<50 TYPE_0 exception → ENTER 0.2x if filters pass; else V1."""
    score = float(row["value_score"])
    if score < 50 and not _dna_skip(row):
        if (
            row["predicted_dna_type"] == "TYPE_0"
            and float(row["runner_probability"]) >= POLICY_C_RUNNER_MIN
            and float(row["predicted_drawdown"]) <= POLICY_C_DD_MAX
            and float(row["predicted_win_prob"]) >= POLICY_C_WIN_MIN
        ):
            return _result(GateAction.ENTER.value, 0.2, "type0_exception_below_50")
    if _dna_skip(row):
        return _result(GateAction.SKIP.value, 0.0, "dna_type_failed_momentum", shadow_only=True)
    return policy_a_v1(row)


def policy_d_conservative(row: dict) -> dict:
    """score < 60 → SKIP."""
    if _dna_skip(row):
        return _result(GateAction.SKIP.value, 0.0, "dna_type_failed_momentum", shadow_only=True)
    score = float(row["value_score"])
    if score < 60:
        return _result(GateAction.SKIP.value, 0.0, "value_score_below_60")
    if score < 70:
        return _result(GateAction.ENTER.value, 0.3, "value_score_60_69")
    if score < 80:
        return _result(GateAction.ENTER.value, 0.6, "value_score_70_79")
    return _result(GateAction.ENTER.value, 1.0, "value_score_80_plus")


def policy_e_balanced(row: dict) -> dict:
    """50–59 split by runner_prob 0.60."""
    if _dna_skip(row):
        return _result(GateAction.SKIP.value, 0.0, "dna_type_failed_momentum", shadow_only=True)
    score = float(row["value_score"])
    rp = float(row["runner_probability"])
    if score < 50:
        return _result(GateAction.SKIP.value, 0.0, "value_score_below_50")
    if score < 60:
        if rp >= 0.60:
            return _result(GateAction.ENTER.value, 0.2, "value_score_50_59_runner_ok")
        return _result(GateAction.SHADOW_ONLY.value, 0.1, "value_score_50_59_low_runner", shadow_only=True)
    if score < 70:
        return _result(GateAction.ENTER.value, 0.3, "value_score_60_69")
    if score < 80:
        return _result(GateAction.ENTER.value, 0.6, "value_score_70_79")
    return _result(GateAction.ENTER.value, 1.0, "value_score_80_plus")


POLICIES: dict[str, dict] = {
    "A": {"name": "Current V1", "fn": policy_a_v1},
    "B": {"name": "Soft 50s", "fn": policy_b_soft_50s},
    "C": {"name": "TYPE0 Exception", "fn": policy_c_type0_exception},
    "D": {"name": "Conservative Core", "fn": policy_d_conservative},
    "E": {"name": "Balanced", "fn": policy_e_balanced},
}


def evaluate_policy(policy_key: str, row: dict) -> dict:
    spec = POLICIES[policy_key]
    return spec["fn"](row)


def score_band(score: float) -> str:
    if score < 50:
        return "0-49"
    if score < 60:
        return "50-59"
    if score < 70:
        return "60-69"
    if score < 80:
        return "70-79"
    return "80+"
