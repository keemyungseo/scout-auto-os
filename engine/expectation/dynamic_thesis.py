"""Dynamic thesis extension and exit thesis."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from scout_auto_os.engine.expectation.curve_builder import ExpectedPath, build_expected_path
from scout_auto_os.engine.position_evaluation.thesis import TradeThesis


@dataclass
class ThesisTransition:
    prior_thesis_id: str
    new_thesis_id: str
    prior_path_id: str
    new_path_id: str
    transition_type: str
    reason: str
    thesis_version: int


def maybe_extension_thesis(
    thesis: TradeThesis,
    path: ExpectedPath,
    *,
    current_roi: float,
    elapsed_min: int,
    thesis_state: str,
    data_dir,
) -> tuple[TradeThesis | None, ExpectedPath | None, ThesisTransition | None]:
    if thesis_state not in ("THESIS_COMPLETE", "OUTPERFORM"):
        return None, None, None
    if current_roi < thesis.expected_return_pct * 1.5:
        return None, None, None
    if elapsed_min > thesis.expected_horizon_min * 1.5:
        return None, None, None

    new_return = round(max(current_roi * 1.3, thesis.expected_return_pct * 2), 4)
    new_horizon = min(thesis.max_hold_minutes, max(thesis.expected_horizon_min + 90, elapsed_min + 60))

    new_thesis = TradeThesis(
        thesis_id=f"th_{uuid.uuid4().hex[:12]}",
        position_id=thesis.position_id,
        symbol=thesis.symbol,
        side=thesis.side,
        entry_time=thesis.entry_time,
        entry_price=thesis.entry_price,
        entry_score=thesis.entry_score,
        search_engine_version=thesis.search_engine_version,
        model_version=thesis.model_version,
        label_version=thesis.label_version,
        rank=thesis.rank,
        expected_horizon_min=new_horizon,
        expected_return_pct=new_return,
        success_probability=min(0.95, thesis.success_probability + 0.1),
        primary_reason="extension_thesis_trailing",
        secondary_reasons=[f"extended_from_{thesis.thesis_id}", "mode_trailing"],
        initial_stop_pct=max(3.0, thesis.initial_stop_pct * 0.6),
        initial_take_profit_pct=new_return,
        invalid_condition="extension_trail_breach",
        max_hold_minutes=thesis.max_hold_minutes,
        review_interval_minutes=thesis.review_interval_minutes,
        source=thesis.source,
        auto_manage=thesis.auto_manage,
        engine=thesis.engine,
    )
    new_path = build_expected_path(
        new_thesis.thesis_id,
        thesis.position_id,
        thesis.symbol,
        thesis.side,
        new_return,
        new_horizon,
        new_thesis.success_probability,
        data_dir,
        thesis_version=path.thesis_version + 1,
    )
    trans = ThesisTransition(
        prior_thesis_id=thesis.thesis_id,
        new_thesis_id=new_thesis.thesis_id,
        prior_path_id=path.path_id,
        new_path_id=new_path.path_id,
        transition_type="EXTENSION",
        reason=f"roi={current_roi}% exceeded target — trailing extension",
        thesis_version=path.thesis_version + 1,
    )
    return new_thesis, new_path, trans


def exit_thesis_record(thesis: TradeThesis, reason: str) -> dict:
    return {
        "thesis_id": thesis.thesis_id,
        "position_id": thesis.position_id,
        "symbol": thesis.symbol,
        "transition_type": "EXIT_THESIS",
        "reason": reason,
    }
