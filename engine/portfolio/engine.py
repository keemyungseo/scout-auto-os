"""Portfolio Engine — Direction Champion → Entry V2 → Long3/Short3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from scout_auto_os.engine.portfolio.constants import (
    HOLD_HOURS,
    MAX_LONG_SLOTS,
    MAX_SHORT_SLOTS,
    REPLACEMENT_MARGIN,
)
from scout_auto_os.engine.portfolio.rule_loader import PortfolioRules, load_portfolio_rules
from scout_auto_os.engine.portfolio.scoring import build_pass_candidates
from scout_auto_os.engine.portfolio.slot_manager import SlotBook, SlotHolding, update_slots


@dataclass
class PortfolioEngine:
    rules: PortfolioRules
    replacement_margin: float = REPLACEMENT_MARGIN
    max_long_slots: int = MAX_LONG_SLOTS
    max_short_slots: int = MAX_SHORT_SLOTS
    hold_hours: int = HOLD_HOURS
    book: SlotBook = field(default_factory=SlotBook)

    @classmethod
    def from_paths(cls, data_dir: Path, pkg_root: Path, **kwargs) -> PortfolioEngine:
        return cls(load_portfolio_rules(data_dir, pkg_root), **kwargs)

    def process_scan(
        self,
        rows: list[dict],
        scan_time_kst: str,
        hold_until_scan: str | None = None,
    ) -> dict:
        """Score PASS candidates and update Long3/Short3 slots."""
        hold_until = hold_until_scan or scan_time_kst
        long_c, short_c = build_pass_candidates(rows, scan_time_kst, self.rules, scan_time_kst)
        self.book, new_entries, replacements = update_slots(
            self.book,
            long_c,
            short_c,
            scan_time_kst,
            hold_until,
            replacement_margin=self.replacement_margin,
        )
        return {
            "scan_time_kst": scan_time_kst,
            "long_pass_count": len(long_c),
            "short_pass_count": len(short_c),
            "long_selected": [self._slot_dict(s) for s in self.book.long_slots],
            "short_selected": [self._slot_dict(s) for s in self.book.short_slots],
            "new_entries": new_entries,
            "replacements": replacements,
            "long_candidates_top": long_c[:10],
            "short_candidates_top": short_c[:10],
        }

    def select_for_entry(self, rows: list[dict], scan_time_kst: str) -> dict:
        """One-shot selection without mutating slot book (for LIVE tick)."""
        long_c, short_c = build_pass_candidates(rows, scan_time_kst, self.rules, scan_time_kst)
        from scout_auto_os.engine.portfolio.diversification import diversify_select

        return {
            "long": diversify_select(long_c, self.max_long_slots),
            "short": diversify_select(short_c, self.max_short_slots),
            "long_pass": len(long_c),
            "short_pass": len(short_c),
        }

    @staticmethod
    def _slot_dict(s: SlotHolding) -> dict:
        return {
            "symbol": s.symbol,
            "direction": s.direction,
            "entry_score": s.entry_score,
            "entry_scan": s.entry_scan,
            "live_pattern": s.live_pattern,
            "hold_until_scan": s.hold_until_scan,
        }
