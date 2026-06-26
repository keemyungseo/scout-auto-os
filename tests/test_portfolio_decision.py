"""Portfolio Decision Engine V1 tests."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.portfolio.decision.decision_engine import evaluate_candidate
from scout_auto_os.engine.portfolio.decision.decision_replay import run_portfolio_decision_replay
from scout_auto_os.engine.portfolio.decision.models import (
    DECISION_IGNORE,
    DECISION_REPLACE,
    DECISION_WAIT,
    PortfolioPosition,
    PortfolioSlotBook,
    PredatorCandidate,
)
from scout_auto_os.engine.portfolio.decision.position_score import score_candidate, score_position
from scout_auto_os.engine.control.portfolio_decision_status import build_portfolio_decision_status


class PortfolioScoreTests(unittest.TestCase):
    def test_position_score_range(self) -> None:
        pos = PortfolioPosition(
            slot_id="long_1",
            trade_id="t1",
            symbol="BTCUSDT",
            side="long",
            entry_time="2026-06-01 10:00:00",
            guardian_score=70,
            guardian_state="ON_TRACK",
            recommendation="HOLD",
            current_roi=5.0,
            elapsed_minutes=60,
            value_score=65,
            expected_roi=15.0,
            confidence=75,
        )
        s = score_position(pos)
        self.assertGreaterEqual(s, 0)
        self.assertLessEqual(s, 100)

    def test_candidate_score(self) -> None:
        c = PredatorCandidate(
            trade_id="t2",
            symbol="ETHUSDT",
            side="long",
            timestamp="2026-06-01 12:00:00",
            value_score=80,
            expected_roi=20,
            expected_win_prob=0.85,
            confidence=78,
            gate_action="ENTER",
            gate_reason="test",
            contract_id="t2",
        )
        self.assertGreater(score_candidate(c), 50)


class PortfolioDecisionTests(unittest.TestCase):
    def _candidate(self, symbol: str, score: float = 75.0) -> PredatorCandidate:
        return PredatorCandidate(
            trade_id=f"2026-06-01 10:00:00|{symbol}|long",
            symbol=symbol,
            side="long",
            timestamp="2026-06-01 10:00:00",
            value_score=score,
            expected_roi=15,
            expected_win_prob=0.8,
            confidence=score,
            gate_action="ENTER",
            gate_reason="enter",
            contract_id=f"2026-06-01 10:00:00|{symbol}|long",
        )

    def _position(self, slot: str, symbol: str, gscore: float) -> PortfolioPosition:
        return PortfolioPosition(
            slot_id=slot,
            trade_id=f"key|{symbol}|long",
            symbol=symbol,
            side="long",
            entry_time="2026-06-01 08:00:00",
            guardian_score=gscore,
            guardian_state="ON_TRACK",
            recommendation="HOLD",
            current_roi=2.0,
            elapsed_minutes=120,
            value_score=50,
            expected_roi=10,
            confidence=50,
            portfolio_value_score=gscore,
        )

    def test_empty_slot_admit(self) -> None:
        book = PortfolioSlotBook()
        c = self._candidate("BTCUSDT", 70)
        book, records, admitted, replaced = evaluate_candidate(book, c)
        self.assertIsNotNone(admitted)
        self.assertIsNone(replaced)
        self.assertEqual(records[0].decision, DECISION_REPLACE)
        self.assertEqual(len(book.long_slots), 1)

    def test_replace_weaker(self) -> None:
        book = PortfolioSlotBook(long_slots=[
            self._position("long_1", "A", 80),
            self._position("long_2", "B", 25),
            self._position("long_3", "C", 70),
        ])
        c = self._candidate("NEWUSDT", 85)
        book, records, admitted, replaced = evaluate_candidate(book, c)
        self.assertIsNotNone(admitted)
        self.assertEqual(replaced.symbol if replaced else "", "B")
        self.assertTrue(any(r.decision == DECISION_REPLACE for r in records))

    def test_wait_when_not_better(self) -> None:
        book = PortfolioSlotBook(long_slots=[
            self._position("long_1", "A", 80),
            self._position("long_2", "B", 75),
            self._position("long_3", "C", 70),
        ])
        c = self._candidate("WEAKUSDT", 50)
        book, records, admitted, _ = evaluate_candidate(book, c)
        self.assertIsNone(admitted)
        self.assertEqual(records[0].decision, DECISION_WAIT)

    def test_ignore_skip_gate(self) -> None:
        book = PortfolioSlotBook()
        c = self._candidate("X", 80)
        c.gate_action = "SKIP"
        _, records, admitted, _ = evaluate_candidate(book, c)
        self.assertIsNone(admitted)
        self.assertEqual(records[0].decision, DECISION_IGNORE)

    def test_long_short_independent(self) -> None:
        book = PortfolioSlotBook(long_slots=[self._position("long_1", "A", 90)])
        short_c = PredatorCandidate(
            trade_id="t|ETH|short",
            symbol="ETHUSDT",
            side="short",
            timestamp="2026-06-01 10:00:00",
            value_score=75,
            expected_roi=12,
            expected_win_prob=0.7,
            confidence=70,
            gate_action="ENTER",
            gate_reason="enter",
            contract_id="t|ETH|short",
        )
        book, _, admitted, _ = evaluate_candidate(book, short_c)
        self.assertIsNotNone(admitted)
        self.assertEqual(len(book.long_slots), 1)
        self.assertEqual(len(book.short_slots), 1)


class PortfolioReplayTests(unittest.TestCase):
    def test_replay_157(self) -> None:
        data_dir = ROOT / "scout_auto_os" / "data"
        if not (data_dir / "trade_dna" / "value_prediction.csv").exists():
            self.skipTest("bundle missing")
        result = run_portfolio_decision_replay(data_dir)
        self.assertEqual(result["trade_count"], 157)
        self.assertGreater(result["decision_rows"], 100)
        summary_path = data_dir / "portfolio" / "portfolio_decision_summary.json"
        self.assertTrue(summary_path.exists())

    def test_api_payload(self) -> None:
        data_dir = ROOT / "scout_auto_os" / "data"
        if not (data_dir / "portfolio" / "portfolio_decision_summary.json").exists():
            if not (data_dir / "trade_dna" / "value_prediction.csv").exists():
                self.skipTest("data missing")
            run_portfolio_decision_replay(data_dir)
        payload = build_portfolio_decision_status(data_dir)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["mode"], "PORTFOLIO_DECISION")
        self.assertIn("avg_slot_utilization", payload["summary"])


if __name__ == "__main__":
    unittest.main()
