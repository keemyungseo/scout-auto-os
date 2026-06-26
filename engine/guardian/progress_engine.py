"""Guardian Progress Engine V1 — metrics, score, state, recommendation."""

from __future__ import annotations

from dataclasses import dataclass

from scout_auto_os.engine.guardian.progress_config import GuardianProgressWeights, load_progress_weights
from scout_auto_os.engine.guardian.progress_metrics import (
    ProgressMetrics,
    compute_progress_metrics,
    normalize_contract,
)
from scout_auto_os.engine.guardian.progress_recommendation import recommend_action
from scout_auto_os.engine.guardian.progress_score import compute_guardian_score
from scout_auto_os.engine.guardian.progress_state import classify_guardian_state

PROGRESS_FIELDS = (
    "contract_id",
    "symbol",
    "progress_ratio",
    "time_progress",
    "peak_progress",
    "drawdown_pressure",
    "guardian_score",
    "guardian_state",
    "recommendation",
    "reason",
)


@dataclass
class GuardianProgressResult:
    contract_id: str
    symbol: str
    progress_ratio: float
    time_progress: float
    peak_progress: float
    drawdown_pressure: float
    guardian_score: float
    guardian_state: str
    recommendation: str
    reason: str

    def to_row(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "symbol": self.symbol,
            "progress_ratio": round(self.progress_ratio, 4),
            "time_progress": round(self.time_progress, 4),
            "peak_progress": round(self.peak_progress, 4),
            "drawdown_pressure": round(self.drawdown_pressure, 4),
            "guardian_score": round(self.guardian_score, 2),
            "guardian_state": self.guardian_state,
            "recommendation": self.recommendation,
            "reason": self.reason,
        }


def evaluate_progress(
    contract: dict,
    position: dict,
    *,
    contract_id: str = "",
    manual: bool = False,
    weights: GuardianProgressWeights | None = None,
    config: dict | None = None,
) -> GuardianProgressResult:
    """Evaluate whether entry thesis remains valid — explainable progress snapshot."""
    c = normalize_contract(contract)
    if contract_id:
        c["contract_id"] = contract_id
    w = weights or load_progress_weights(config)

    metrics = compute_progress_metrics(c, position)
    score = compute_guardian_score(metrics, c, w)
    state, state_lines = classify_guardian_state(metrics, c)
    rec, rec_lines = recommend_action(state, metrics, c, manual=manual)

    reason_parts = [
        f"metrics progress={metrics.progress_ratio:.3f} time={metrics.time_progress:.3f} "
        f"peak={metrics.peak_progress:.3f} dd_pressure={metrics.drawdown_pressure:.3f}",
        f"score={score:.1f}",
        *state_lines,
        *rec_lines,
    ]

    return GuardianProgressResult(
        contract_id=c.get("contract_id", contract_id),
        symbol=c.get("symbol", ""),
        progress_ratio=metrics.progress_ratio,
        time_progress=metrics.time_progress,
        peak_progress=metrics.peak_progress,
        drawdown_pressure=metrics.drawdown_pressure,
        guardian_score=score,
        guardian_state=state,
        recommendation=rec,
        reason=" | ".join(reason_parts),
    )
