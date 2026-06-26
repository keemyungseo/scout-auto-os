"""Auto-generate short training label candidates."""

from __future__ import annotations

from scout_auto_os.engine.research.short_constitution.constants import BASELINE_LABEL_ID
from scout_auto_os.engine.research.short_constitution.label_builder import ShortLabelSpec


def generate_short_label_candidates() -> list[ShortLabelSpec]:
    specs = [
        ShortLabelSpec(
            BASELINE_LABEL_ID,
            "Baseline max_down_2h (MFE down)",
            "max_down_2h",
            "baseline",
        ),
    ]

    def add(label_id, name, key, cat, invert=False):
        specs.append(ShortLabelSpec(label_id, name, key, cat, invert))

    add("return_short_2h", "2h short close return", "short_return_2h", "return")
    add("return_short_30m", "30m short return", "return_30m", "return")
    add("return_short_1h", "1h short return", "return_1h", "return")
    add("return_short_4h", "4h short return", "return_4h", "return")
    add("return_short_6h", "6h short return", "return_6h", "return")
    add("max_down_2h", "2h max favorable down", "max_down_2h", "mfe")
    add("max_down_4h", "4h max favorable down", "max_down_4h", "mfe")
    add("max_down_6h", "6h max favorable down", "max_down_6h", "mfe")
    add("return_plus_dd", "Short return minus drawup", "return_plus_dd", "risk_adjusted")
    add("drawup_resilience", "Drawup resilience", "drawup_resilience", "risk_adjusted")
    add("risk_adjusted_short", "Risk adjusted short", "risk_adjusted_short", "risk_adjusted")
    add("return_per_risk_short", "Short return / adverse", "return_per_risk_short", "risk_adjusted")
    add("hit_short_3pct", "Hit 3% short at 2h", "hit_short_3pct", "binary")
    add("distribution_success", "Distribution success", "distribution_success", "binary")
    add("capitulation_fade", "Capitulation fade", "capitulation_fade", "binary")
    add("intrabar_sharpe_short", "Intrabar short Sharpe", "intrabar_sharpe_short", "risk_adjusted")
    add("mae_short_2h", "Max adverse (drawup)", "mae_short_2h", "mae", invert=True)
    add("max_up_adverse_2h", "Max drawup 2h (lower better)", "max_up_adverse_2h", "mae", invert=True)

    seen: set[str] = set()
    out: list[ShortLabelSpec] = []
    for s in specs:
        if s.label_id in seen:
            continue
        seen.add(s.label_id)
        out.append(s)
    return out
