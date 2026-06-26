"""Pick functions for blind validation comparison."""

from __future__ import annotations

from typing import Callable

from scout_auto_os.engine.research.directional.dna.formulas import ClusterFormula, rank_by_formula
from scout_auto_os.engine.research.directional.engines import rank_long, rank_short
from scout_auto_os.engine.research.directional.prediction.engine import predict_symbol
from scout_auto_os.engine.research.zero_base.random_baseline import generate_random_draws
from scout_auto_os.engine.research.zero_base.validation import rank_validation_engine

LONG_DIRECTION_CHAMPION = "LONG_CONTINUATION"
SHORT_DIRECTION_CHAMPION = "SHORT_CONTINUATION"


def make_random_picker(seed: int = 42) -> Callable[[list[dict]], list[str]]:
    def pick(rows: list[dict], top_k: int = 5) -> list[str]:
        syms = [r["symbol"] for r in rows]
        return generate_random_draws(syms, top_k, 1, seed)[0]

    return pick


def make_zero_base_champion_picker(direction: str) -> Callable[[list[dict], int], list[str]]:
    def pick(rows: list[dict], top_k: int = 5) -> list[str]:
        if direction == "long":
            return rank_validation_engine(rows, "A6", top_k)
        return rank_validation_engine(rows, "MOMENTUM", top_k)

    return pick


def make_direction_champion_picker(direction: str) -> Callable[[list[dict], int], list[str]]:
    engine = LONG_DIRECTION_CHAMPION if direction == "long" else SHORT_DIRECTION_CHAMPION

    def pick(rows: list[dict], top_k: int = 5) -> list[str]:
        if direction == "long":
            return rank_long(rows, engine, top_k)
        return rank_short(rows, engine, top_k)

    return pick


def make_cluster_champion_picker(
    formula: ClusterFormula,
) -> Callable[[list[dict], int], list[str]]:
    def pick(rows: list[dict], top_k: int = 5) -> list[str]:
        return rank_by_formula(rows, formula, top_k)

    return pick


def make_prediction_picker(
    long_formulas: list[ClusterFormula],
    short_formulas: list[ClusterFormula],
    expected_returns: dict[str, dict],
    direction: str,
) -> Callable[[list[dict], int], list[str]]:
    def pick(rows: list[dict], top_k: int = 5) -> list[str]:
        scored = []
        for row in rows:
            pred = predict_symbol(row["features"], long_formulas, short_formulas, expected_returns)
            score = pred["long_score"] if direction == "long" else pred["short_score"]
            scored.append((row["symbol"], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:top_k]]

    return pick
