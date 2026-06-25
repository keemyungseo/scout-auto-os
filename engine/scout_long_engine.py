"""A6 Long engine — scan → select → entry (with Entry Quality Guard V1.3)."""

from __future__ import annotations

from scout_auto_os.storage.db import Database


class ScoutLongEngine:
    def __init__(self, config: dict, db: Database, entry_guard=None, block_summary=None) -> None:
        self.config = config
        self.db = db
        self.enabled = bool(config["long_engine"].get("enabled", True))
        self.entry_guard = entry_guard
        self.block_summary = block_summary

    def select_candidate(
        self,
        top5: list[dict],
        occupied: set[str],
        locked: set[str],
        skip: set[str] | None = None,
    ) -> dict | None:
        if not self.enabled:
            return None
        skip = skip or set()
        for row in top5:
            sym = row["symbol"]
            if sym in occupied or sym in locked or sym in skip:
                continue
            return row
        return None

    def try_entry(
        self,
        candidate: dict,
        position_mgr,
        execution,
        alert_mgr,
        risk_mgr,
        trade_recorder=None,
    ) -> tuple[str | None, bool]:
        """
        Returns (position_id, try_next_candidate).
        try_next_candidate=True when blocked by quality guard (skip symbol, continue).
        """
        if not candidate:
            return None, False
        ok, _reason = risk_mgr.can_enter_long(position_mgr.long_slots_used())
        if not ok:
            return None, False
        if not position_mgr.has_slot():
            return None, False

        sym = candidate["symbol"]
        quality_ctx: dict = {}

        if self.entry_guard:
            result = self.entry_guard.evaluate(sym, candidate)
            quality_ctx = result.metrics
            if not result.passed:
                if self.block_summary:
                    self.block_summary.record_block(result.block_reason)
                if trade_recorder:
                    ctx = self._merge_context(candidate, quality_ctx)
                    trade_recorder.set_entry_context(sym, ctx)
                return None, True

        if self.block_summary:
            self.block_summary.record_pass()

        price = float(candidate["entry_price"])
        if trade_recorder:
            ctx = self._merge_context(candidate, quality_ctx)
            trade_recorder.set_entry_context(sym, ctx)

        pid = position_mgr.create_position(
            sym, "LONG", price, "AUTO", "A6_LONG",
            a6_score=candidate["a6_score"],
            expected_ev=candidate["expected_ev"],
        )
        execution.paper_entry(
            pid, sym, "LONG", price,
            candidate.get("reason", "A6_top_candidate"),
            "A6_LONG",
        )
        alert_mgr.entry_alert(
            sym, "LONG", price,
            candidate.get("reason", ""),
            float(candidate.get("expected_ev", 0)),
        )
        self.db.log_event("scout_long_engine", "entry", {"symbol": sym, "position_id": pid})
        return pid, False

    @staticmethod
    def _merge_context(candidate: dict, quality: dict) -> dict:
        from scout_auto_os.engine.review.parse_reason import parse_reason_fields

        parsed = parse_reason_fields(candidate.get("reason", ""))
        ctx = {
            "scan_rank": candidate.get("rank"),
            "score": candidate.get("a6_score"),
            "expected_ev": candidate.get("expected_ev"),
            "reason_1h": parsed.get("reason_1h"),
            "reason_2h": parsed.get("reason_2h"),
            "range_pct": parsed.get("range_pct"),
            "quote_volume_24h": quality.get("quote_volume_24h"),
            "quote_volume_5m": quality.get("quote_volume_5m"),
            "quote_volume_5m_ratio": quality.get("quote_volume_5m_ratio"),
            "pullback_from_15m_high_pct": quality.get("pullback_from_15m_high_pct"),
            "last_5m_candle_direction": quality.get("last_5m_candle_direction"),
            "estimated_slippage_pct": quality.get("estimated_slippage_pct"),
            "entry_quality_score": quality.get("entry_quality_score"),
            "entry_quality_pass": quality.get("entry_quality_pass", True),
            "entry_block_reason": quality.get("entry_block_reason", ""),
        }
        return ctx

    def try_fill_slots(
        self,
        top5: list[dict],
        occupied: set[str],
        locked: set[str],
        position_mgr,
        execution,
        alert_mgr,
        risk_mgr,
        trade_recorder=None,
    ) -> list[str]:
        """Fill up to max_long_slots independently (no meta rotation)."""
        entered: list[str] = []
        skipped: set[str] = set()
        while position_mgr.has_slot():
            cand = self.select_candidate(top5, occupied, locked, skipped)
            if not cand:
                break
            pid, try_next = self.try_entry(
                cand, position_mgr, execution, alert_mgr, risk_mgr, trade_recorder,
            )
            if try_next:
                skipped.add(cand["symbol"])
                continue
            if not pid:
                break
            entered.append(pid)
            occupied.add(cand["symbol"])
        return entered
