"""Cluster Prediction Engine V1 — unit tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.directional.dna.formulas import ClusterFormula
from scout_auto_os.engine.research.directional.prediction.engine import predict_symbol
from scout_auto_os.engine.research.directional.prediction.loader import load_formulas, resolve_formulas_path
from scout_auto_os.engine.research.directional.prediction.runner import ClusterPredictionRunner


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_softmax_probs_sum():
    long_f = [
        ClusterFormula("LONG_TEST_A", "LONG_TEST", "A", "long", {"5m_momentum": 1.0}),
        ClusterFormula("LONG_TEST_B", "LONG_TEST", "B", "long", {"5m_momentum": -0.5}),
    ]
    short_f = [
        ClusterFormula("SHORT_TEST_A", "SHORT_TEST", "A", "short", {"5m_momentum": -1.0}),
    ]
    expected = {
        "LONG_TEST_A": {"avg_return_2h": 5.0},
        "LONG_TEST_B": {"avg_return_2h": 1.0},
        "SHORT_TEST_A": {"avg_return_2h": 3.0},
    }
    features = {"5m_momentum": 2.0}
    pred = predict_symbol(features, long_f, short_f, expected)
    long_sum = sum(pred["long_cluster_probs"].values())
    if abs(long_sum - 100.0) > 0.1:
        _fail(f"long probs sum {long_sum}")
    if pred["recommended_direction"] not in ("LONG", "SHORT"):
        _fail("bad direction")
    print(f"OK: long_score={pred['long_score']} recommend={pred['recommended_direction']}")


def test_load_formulas():
    path = resolve_formulas_path(PKG / "data", PKG)
    formulas = load_formulas(path)
    if len(formulas) < 8:
        _fail(f"expected >=8 formulas, got {len(formulas)}")
    print(f"OK: loaded {len(formulas)} cluster formulas")


def test_small_run():
    with tempfile.TemporaryDirectory() as tmp:
        runner = ClusterPredictionRunner(
            Path(tmp),
            pkg_root=PKG,
            candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
            forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        )
        result = runner.run(max_scans=12, top_k=3)
        out = Path(tmp) / "zero_base"
        for fname in (
            "cluster_prediction_report.md",
            "prediction_engine_report.md",
            "cluster_probability.csv",
        ):
            if not (out / fname).exists():
                _fail(f"missing {fname}")
        if not result["comparison"]:
            _fail("comparison empty")
        print(f"OK: methods compared={len(result['comparison'])}")


def main():
    test_softmax_probs_sum()
    test_load_formulas()
    test_small_run()
    print("\nALL CLUSTER PREDICTION V1 TESTS PASSED")


if __name__ == "__main__":
    main()
