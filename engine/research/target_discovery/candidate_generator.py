"""Auto-generate candidate training labels from forward metrics."""

from __future__ import annotations

from scout_auto_os.engine.research.target_discovery.constants import BASELINE_LABEL_ID
from scout_auto_os.engine.research.target_discovery.label_builder import LabelSpec


def generate_label_candidates() -> list[LabelSpec]:
    specs: list[LabelSpec] = [
        LabelSpec(
            BASELINE_LABEL_ID,
            "Baseline max_up_4h (Ranking V1)",
            "max_up_4h",
            "baseline",
            description="Current Ranking Engine label from seed max_up_4h",
        ),
    ]

    def add(
        label_id: str,
        name: str,
        rank_key: str,
        category: str,
        invert: bool = False,
        description: str = "",
    ) -> None:
        specs.append(LabelSpec(label_id, name, rank_key, category, invert, description))

    for key, name in (
        ("max_up_30m", "30m max favorable excursion"),
        ("max_up_1h", "1h max favorable excursion"),
        ("max_up_2h", "2h max favorable excursion"),
        ("max_up_6h", "6h max favorable excursion"),
        ("max_up_12h", "12h max favorable excursion"),
    ):
        add(f"max_up_{key.split('_')[-1]}", name, key, "mfe")

    for key, name in (
        ("return_30m", "30m close return"),
        ("return_1h", "1h close return"),
        ("return_2h", "2h close return"),
        ("return_4h", "4h close return"),
        ("return_6h", "6h close return"),
        ("return_12h", "12h close return"),
    ):
        add(f"return_{key.split('_')[-1]}", name, key, "return")

    add("avg_return_multi", "Multi-horizon average return", "avg_return_multi", "return")
    add("max_return_multi", "Multi-horizon max return", "max_return_multi", "return")
    add("return_minus_dd", "Return + drawdown (2h)", "return_minus_dd", "risk_adjusted")
    add("return_per_risk", "Return / |drawdown|", "return_per_risk", "risk_adjusted")
    add("drawdown_resilience", "Return minus abs drawdown", "drawdown_resilience", "risk_adjusted")
    add("peak_efficiency", "Close return / MFE", "peak_efficiency", "efficiency")
    add("recovery_speed", "Return minus MAE", "recovery_speed", "efficiency")
    add("intrabar_sharpe", "Intrabar Sharpe (2h window)", "intrabar_sharpe", "risk_adjusted")
    add("intrabar_sortino", "Intrabar Sortino (2h window)", "intrabar_sortino", "risk_adjusted")
    add("mfe_2h", "MFE alias (2h)", "mfe_2h", "mfe")
    add("mae_2h", "MAE (2h, lower better)", "mae_2h", "mae", invert=True)
    add("max_drawdown_2h", "Max drawdown (less negative better)", "max_drawdown_2h", "mae", invert=True)
    add("hit_3pct", "Hit 3% at 2h close", "hit_3pct", "binary")
    add("hit_max_3pct", "Hit 3% MFE within 2h", "hit_max_3pct", "binary")
    add("breakout_success", "Breakout success (MFE>=4, close>=2)", "breakout_success", "binary")
    add("momentum_persist", "Momentum persistence", "momentum_persist", "persistence")
    add("uptrend_duration", "Uptrend bar duration", "uptrend_duration", "persistence")
    add("time_to_peak_score", "Faster peak (inverted time)", "time_to_peak_score", "timing")
    add("time_to_3pct_score", "Faster 3% hit", "time_to_3pct_score", "timing")
    add("label_success_2h", "Success flag (>=3% 2h)", "hit_3pct", "binary")

    seen: set[str] = set()
    out: list[LabelSpec] = []
    for s in specs:
        if s.label_id in seen:
            continue
        seen.add(s.label_id)
        out.append(s)
    return out
