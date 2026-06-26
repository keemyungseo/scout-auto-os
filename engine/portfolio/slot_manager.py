"""Slot manager with replacement logic."""

from __future__ import annotations

from dataclasses import dataclass, field

from scout_auto_os.engine.portfolio.constants import (
    MAX_LONG_SLOTS,
    MAX_SHORT_SLOTS,
    REPLACEMENT_MARGIN,
)
from scout_auto_os.engine.portfolio.diversification import diversify_select


@dataclass
class SlotHolding:
    symbol: str
    direction: str
    entry_score: float
    entry_scan: str
    live_pattern: str
    hold_until_scan: str = ""


@dataclass
class SlotBook:
    long_slots: list[SlotHolding] = field(default_factory=list)
    short_slots: list[SlotHolding] = field(default_factory=list)

    def slots_for(self, direction: str) -> list[SlotHolding]:
        return self.long_slots if direction == "long" else self.short_slots

    def max_slots(self, direction: str) -> int:
        return MAX_LONG_SLOTS if direction == "long" else MAX_SHORT_SLOTS


def update_slots(
    book: SlotBook,
    long_candidates: list[dict],
    short_candidates: list[dict],
    scan_time_kst: str,
    hold_until_scan: str,
    replacement_margin: float = REPLACEMENT_MARGIN,
    diversify: bool = True,
) -> tuple[SlotBook, list[dict], list[dict]]:
    """
    Returns updated book, new_entries, replacements.
    """
    new_entries: list[dict] = []
    replacements: list[dict] = []

    for direction, candidates in (("long", long_candidates), ("short", short_candidates)):
        slots = book.slots_for(direction)
        max_s = book.max_slots(direction)
        held_syms = {s.symbol for s in slots}

        pool = [c for c in candidates if c["symbol"] not in held_syms]
        if diversify:
            pool = diversify_select(pool, max_s * 2)

        for c in pool:
            if len(slots) < max_s:
                h = SlotHolding(
                    symbol=c["symbol"],
                    direction=direction,
                    entry_score=c["entry_score"],
                    entry_scan=scan_time_kst,
                    live_pattern=c.get("live_pattern", ""),
                    hold_until_scan=hold_until_scan,
                )
                slots.append(h)
                new_entries.append({**c, "action": "enter", "scan_time_kst": scan_time_kst})
                held_syms.add(c["symbol"])
                continue

            weakest = min(slots, key=lambda s: s.entry_score)
            threshold = weakest.entry_score * (1.0 + replacement_margin)
            if c["entry_score"] >= threshold:
                replacements.append({
                    "direction": direction,
                    "scan_time_kst": scan_time_kst,
                    "out_symbol": weakest.symbol,
                    "out_score": weakest.entry_score,
                    "in_symbol": c["symbol"],
                    "in_score": c["entry_score"],
                    "score_gap": round(c["entry_score"] - weakest.entry_score, 2),
                })
                slots.remove(weakest)
                slots.append(SlotHolding(
                    symbol=c["symbol"],
                    direction=direction,
                    entry_score=c["entry_score"],
                    entry_scan=scan_time_kst,
                    live_pattern=c.get("live_pattern", ""),
                    hold_until_scan=hold_until_scan,
                ))
                new_entries.append({**c, "action": "replace", "replaced": weakest.symbol, "scan_time_kst": scan_time_kst})
                held_syms.add(c["symbol"])

        if direction == "long":
            book.long_slots = slots[:max_s]
        else:
            book.short_slots = slots[:max_s]

    return book, new_entries, replacements
