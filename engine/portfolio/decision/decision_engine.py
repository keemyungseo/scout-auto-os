"""Portfolio Decision Engine V1 — slot compare and decision rules."""

from __future__ import annotations

from scout_auto_os.engine.portfolio.decision.models import (
    DECISION_IGNORE,
    DECISION_KEEP,
    DECISION_REPLACE,
    DECISION_WAIT,
    PortfolioDecisionRecord,
    PortfolioPosition,
    PortfolioSlotBook,
    PredatorCandidate,
)
from scout_auto_os.engine.portfolio.decision.position_score import (
    rescore_book_positions,
    score_candidate,
)


def _replacement_margin(config: dict | None) -> float:
    if not config:
        return 5.0
    return float(config.get("portfolio_decision", {}).get("replacement_margin", 5.0))


def _min_candidate_score(config: dict | None) -> float:
    if not config:
        return 40.0
    return float(config.get("portfolio_decision", {}).get("min_candidate_score", 40.0))


def evaluate_candidate(
    book: PortfolioSlotBook,
    candidate: PredatorCandidate,
    *,
    config: dict | None = None,
) -> tuple[PortfolioSlotBook, list[PortfolioDecisionRecord], PortfolioPosition | None, PortfolioPosition | None]:
    """
    Compare new Predator candidate against same-side slots.
    Long and Short operate independently.
    Returns updated book, decision records, admitted position, replaced-out position.
    """
    side = candidate.side.lower()
    slots = list(book.slots_for(side))
    max_s = book.max_slots(side)
    records: list[PortfolioDecisionRecord] = []
    admitted: PortfolioPosition | None = None
    replaced_out: PortfolioPosition | None = None

    cand_score = score_candidate(candidate, config)
    candidate.portfolio_value_score = cand_score

    if not candidate.is_enter:
        records.append(PortfolioDecisionRecord(
            timestamp=candidate.timestamp,
            slot=f"{side}_book",
            side=side,
            current_symbol="",
            candidate_symbol=candidate.symbol,
            decision=DECISION_IGNORE,
            reason=f"Predator gate={candidate.gate_action} — {candidate.gate_reason}",
            candidate_score=cand_score,
        ))
        return book, records, None, None

    if cand_score < _min_candidate_score(config):
        records.append(PortfolioDecisionRecord(
            timestamp=candidate.timestamp,
            slot=f"{side}_book",
            side=side,
            current_symbol="",
            candidate_symbol=candidate.symbol,
            decision=DECISION_IGNORE,
            reason=f"Candidate portfolio_value {cand_score:.1f} < min {_min_candidate_score(config):.0f}",
            candidate_score=cand_score,
        ))
        return book, records, None, None

    rescore_book_positions(slots, config)

    if len(slots) < max_s:
        slot_id = f"{side}_{len(slots) + 1}"
        admitted = _position_from_candidate(candidate, slot_id, cand_score)
        slots.append(admitted)
        book.set_slots(side, slots)
        records.append(PortfolioDecisionRecord(
            timestamp=candidate.timestamp,
            slot=slot_id,
            side=side,
            current_symbol="",
            candidate_symbol=candidate.symbol,
            decision=DECISION_REPLACE,
            reason=f"{slot_id} empty — admit candidate portfolio_value {cand_score:.1f}",
            candidate_score=cand_score,
        ))
        return book, records, admitted, None

    weakest = min(slots, key=lambda p: p.portfolio_value_score)
    margin = _replacement_margin(config)
    w_score = weakest.portfolio_value_score

    if cand_score >= w_score + margin:
        slot_id = weakest.slot_id
        records.append(PortfolioDecisionRecord(
            timestamp=candidate.timestamp,
            slot=slot_id,
            side=side,
            current_symbol=weakest.symbol,
            candidate_symbol=candidate.symbol,
            decision=DECISION_REPLACE,
            reason=(
                f"{slot_id} portfolio_value {w_score:.1f} < New Candidate {cand_score:.1f} "
                f"(guardian_score {weakest.guardian_score:.1f}) — replace {weakest.symbol}"
            ),
            current_score=w_score,
            candidate_score=cand_score,
        ))
        replaced_out = weakest
        slots = [p for p in slots if p.slot_id != slot_id]
        admitted = _position_from_candidate(candidate, slot_id, cand_score)
        slots.append(admitted)
        book.set_slots(side, slots)
    else:
        records.append(PortfolioDecisionRecord(
            timestamp=candidate.timestamp,
            slot=weakest.slot_id,
            side=side,
            current_symbol=weakest.symbol,
            candidate_symbol=candidate.symbol,
            decision=DECISION_WAIT,
            reason=(
                f"{weakest.slot_id} portfolio_value {w_score:.1f} >= New Candidate {cand_score:.1f} "
                f"— defer candidate"
            ),
            current_score=w_score,
            candidate_score=cand_score,
        ))
        for p in slots:
            if p.slot_id != weakest.slot_id:
                records.append(PortfolioDecisionRecord(
                    timestamp=candidate.timestamp,
                    slot=p.slot_id,
                    side=side,
                    current_symbol=p.symbol,
                    candidate_symbol=candidate.symbol,
                    decision=DECISION_KEEP,
                    reason=(
                        f"{p.slot_id} portfolio_value {p.portfolio_value_score:.1f} "
                        f">= New Candidate {cand_score:.1f} — keep {p.symbol}"
                    ),
                    current_score=p.portfolio_value_score,
                    candidate_score=cand_score,
                ))

    return book, records, admitted, replaced_out


def _position_from_candidate(
    candidate: PredatorCandidate,
    slot_id: str,
    portfolio_value_score: float,
) -> PortfolioPosition:
    return PortfolioPosition(
        slot_id=slot_id,
        trade_id=candidate.trade_id,
        symbol=candidate.symbol,
        side=candidate.side,
        entry_time=candidate.timestamp,
        guardian_score=candidate.value_score * 0.5 + candidate.confidence * 0.5,
        guardian_state="BUILDING",
        recommendation="HOLD",
        current_roi=0.0,
        elapsed_minutes=0,
        value_score=candidate.value_score,
        expected_roi=candidate.expected_roi,
        confidence=candidate.confidence,
        portfolio_value_score=portfolio_value_score,
        actual_roi=candidate.actual_roi,
    )
