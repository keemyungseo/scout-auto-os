"""Generate State Formula variants for State League (Research only)."""

from __future__ import annotations

from scout_auto_os.engine.state_engine import LIVE_STATE_FORMULA, StateFormulaWeights

# Hand-crafted + grid variants (expandable to hundreds)
_BASE_GRID = (
    (25, 25, 25, 15, 10),
    (35, 20, 30, 10, 5),
    (15, 35, 35, 15, 0),
    (30, 30, 20, 15, 5),
    (20, 30, 30, 15, 5),
    (40, 15, 25, 15, 5),
    (15, 25, 25, 20, 15),
    (25, 15, 35, 15, 10),
    (20, 20, 20, 25, 15),
    (30, 25, 15, 20, 10),
    (10, 40, 30, 15, 5),
    (25, 30, 15, 20, 10),
    (20, 25, 30, 15, 10),
    (35, 25, 15, 15, 10),
    (15, 20, 35, 20, 10),
)


def generate_state_formulas(max_count: int = 64) -> list[StateFormulaWeights]:
    formulas: list[StateFormulaWeights] = [LIVE_STATE_FORMULA]
    live_tuple = (LIVE_STATE_FORMULA.trend, LIVE_STATE_FORMULA.momentum,
                  LIVE_STATE_FORMULA.volume, LIVE_STATE_FORMULA.expansion,
                  LIVE_STATE_FORMULA.acceleration)
    for i, (t, m, v, e, a) in enumerate(_BASE_GRID):
        if (t, m, v, e, a) == live_tuple:
            continue
        name = f"STATE_{chr(65 + i)}" if i < 26 else f"STATE_G{i}"
        formulas.append(StateFormulaWeights(name, t, m, v, e, a))
    # Exhaustion scale variants on top performers grid
    for scale in (0.8, 1.2, 1.5):
        formulas.append(StateFormulaWeights(
            f"STATE_EXS_{scale}", 25, 25, 25, 15, 10, exhaustion_scale=scale,
        ))
    return formulas[:max_count]
