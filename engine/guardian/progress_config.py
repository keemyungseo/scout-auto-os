"""Guardian progress score weights — loaded from config."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardianProgressWeights:
    roi_progress: float = 0.30
    time_alignment: float = 0.15
    drawdown_health: float = 0.20
    value_score: float = 0.20
    win_probability: float = 0.15

    def normalized(self) -> "GuardianProgressWeights":
        total = (
            self.roi_progress + self.time_alignment + self.drawdown_health
            + self.value_score + self.win_probability
        )
        if total <= 0:
            return GuardianProgressWeights()
        return GuardianProgressWeights(
            roi_progress=self.roi_progress / total,
            time_alignment=self.time_alignment / total,
            drawdown_health=self.drawdown_health / total,
            value_score=self.value_score / total,
            win_probability=self.win_probability / total,
        )


def load_progress_weights(config: dict | None = None) -> GuardianProgressWeights:
    cfg = (config or {}).get("guardian", {}).get("progress", {}).get("weights", {})
    w = GuardianProgressWeights(
        roi_progress=float(cfg.get("roi_progress", 0.30)),
        time_alignment=float(cfg.get("time_alignment", 0.15)),
        drawdown_health=float(cfg.get("drawdown_health", 0.20)),
        value_score=float(cfg.get("value_score", 0.20)),
        win_probability=float(cfg.get("win_probability", 0.15)),
    )
    return w.normalized()
