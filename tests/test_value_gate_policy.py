"""Value Gate policy V2 unit tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.predator.policies import evaluate_policy, policy_b_soft_50s, policy_e_balanced


class PolicyTests(unittest.TestCase):
    def _row(self, score: float, runner: float = 0.9, dna: str = "TYPE_0") -> dict:
        return {
            "value_score": score,
            "runner_probability": runner,
            "predicted_dna_type": dna,
            "predicted_drawdown": 10,
            "predicted_win_prob": 0.8,
        }

    def test_policy_b_opens_50s(self) -> None:
        g = policy_b_soft_50s(self._row(55))
        self.assertEqual(g["action"], "ENTER")
        self.assertEqual(g["recommended_size"], 0.2)

    def test_policy_e_splits_runner(self) -> None:
        high = policy_e_balanced(self._row(55, runner=0.7))
        low = policy_e_balanced(self._row(55, runner=0.5))
        self.assertEqual(high["action"], "ENTER")
        self.assertEqual(low["action"], "SHADOW_ONLY")

    def test_type1_always_skip(self) -> None:
        g = evaluate_policy("A", self._row(90, dna="TYPE_1", runner=0.1))
        self.assertEqual(g["recommended_size"], 0.0)


if __name__ == "__main__":
    unittest.main()
