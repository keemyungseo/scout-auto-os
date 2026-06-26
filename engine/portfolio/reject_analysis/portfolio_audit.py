"""Portfolio-stage reject tracing."""

from __future__ import annotations

from scout_auto_os.engine.portfolio.constants import MAX_LONG_SLOTS, MAX_SHORT_SLOTS, REPLACEMENT_MARGIN
from scout_auto_os.engine.portfolio.diversification import diversify_select, movement_group
from scout_auto_os.engine.portfolio.slot_manager import SlotBook, SlotHolding, update_slots


def trace_portfolio_decisions(
    book: SlotBook,
    long_candidates: list[dict],
    short_candidates: list[dict],
    scan_time_kst: str,
    hold_until_scan: str,
    replacement_margin: float = REPLACEMENT_MARGIN,
) -> tuple[list[dict], SlotBook]:
    """
    Annotate each rule-pass candidate with portfolio outcome before mutating book.
    Returns per-candidate portfolio decisions + updated book.
    """
    decisions: list[dict] = []
    snapshots: dict[str, dict] = {}

    for direction, candidates in (("long", long_candidates), ("short", short_candidates)):
        max_s = MAX_LONG_SLOTS if direction == "long" else MAX_SHORT_SLOTS
        slots = list(book.slots_for(direction))
        held = {s.symbol for s in slots}

        diversified = diversify_select(candidates, max_s * 2)
        div_syms = {c["symbol"] for c in diversified}
        div_rank = {c["symbol"]: i for i, c in enumerate(diversified)}

        for c in candidates:
            sym = c["symbol"]
            base = {
                "scan_time_kst": scan_time_kst,
                "symbol": sym,
                "direction": direction,
                "entry_score": c.get("entry_score"),
                "live_pattern": c.get("live_pattern"),
                "movement_group": movement_group(c),
            }
            if sym in held:
                decisions.append({**base, "portfolio_result": "Already_Occupied", "portfolio_stage": "portfolio"})
                continue
            if sym not in div_syms:
                decisions.append({
                    **base,
                    "portfolio_result": "Diversification_Reject",
                    "portfolio_stage": "portfolio",
                })
                continue
            snapshots[sym + direction] = base

        weakest_score = min((s.entry_score for s in slots), default=0.0)
        replace_threshold = weakest_score * (1.0 + replacement_margin) if slots else 0.0

        for c in diversified:
            sym = c["symbol"]
            base = {
                "scan_time_kst": scan_time_kst,
                "symbol": sym,
                "direction": direction,
                "entry_score": c.get("entry_score"),
                "live_pattern": c.get("live_pattern"),
                "movement_group": movement_group(c),
            }
            if len(slots) < max_s:
                slots.append(SlotHolding(
                    sym, direction, c["entry_score"], scan_time_kst,
                    c.get("live_pattern", ""), hold_until_scan,
                ))
                decisions.append({
                    **base,
                    "portfolio_result": "PASS",
                    "portfolio_stage": "portfolio",
                    "action": "enter",
                })
                held.add(sym)
                continue

            if c["entry_score"] >= replace_threshold:
                out = min(slots, key=lambda s: s.entry_score)
                slots.remove(out)
                slots.append(SlotHolding(
                    sym, direction, c["entry_score"], scan_time_kst,
                    c.get("live_pattern", ""), hold_until_scan,
                ))
                decisions.append({
                    **base,
                    "portfolio_result": "Replacement",
                    "portfolio_stage": "portfolio",
                    "action": "replace",
                    "replaced_symbol": out.symbol,
                })
                held.add(sym)
            else:
                decisions.append({
                    **base,
                    "portfolio_result": "Low_Score",
                    "portfolio_stage": "portfolio",
                    "required_score": round(replace_threshold, 2),
                    "score_gap": round(replace_threshold - c["entry_score"], 2),
                })

        for c in candidates:
            sym = c["symbol"]
            if sym in div_syms or sym in held:
                continue
            if any(d.get("symbol") == sym and d.get("direction") == direction for d in decisions):
                continue
            decisions.append({
                "scan_time_kst": scan_time_kst,
                "symbol": sym,
                "direction": direction,
                "entry_score": c.get("entry_score"),
                "live_pattern": c.get("live_pattern"),
                "portfolio_result": "Low_Score",
                "portfolio_stage": "portfolio",
                "note": "below_diversify_cutoff",
            })

        if direction == "long":
            book.long_slots = slots[:max_s]
        else:
            book.short_slots = slots[:max_s]

    return decisions, book
