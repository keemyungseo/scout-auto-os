"""Directional DNA Discovery V1 — small sample tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.directional.dna.analyzer import analyze_feature_importance
from scout_auto_os.engine.research.directional.dna.clustering import kmeans, normalize_matrix
from scout_auto_os.engine.research.directional.dna.collector import numeric_feature_keys
from scout_auto_os.engine.research.directional.dna.formulas import build_cluster_formula
from scout_auto_os.engine.research.directional.dna.runner import DirectionalDnaRunner


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_importance():
    samples = []
    for i in range(60):
        samples.append({
            "success": i < 30,
            "features": {
                "5m_momentum": (5.0 if i < 30 else -1.0) + i * 0.01,
                "15m_current_return_pct": (3.0 if i < 30 else -0.5),
            },
        })
    imp = analyze_feature_importance(samples, ["5m_momentum", "15m_current_return_pct"])
    if not imp:
        _fail("importance empty")
    print(f"OK: top feature={imp[0]['feature']} effect={imp[0]['effect_size']}")


def test_kmeans():
    data = normalize_matrix([[1, 0], [1.1, 0.1], [5, 5], [5.2, 4.8]])
    labels, _ = kmeans(data, 2, seed=1)
    if len(set(labels)) < 2:
        _fail("kmeans single cluster")
    print(f"OK: kmeans labels={labels}")


def test_small_run():
    with tempfile.TemporaryDirectory() as tmp:
        runner = DirectionalDnaRunner(
            Path(tmp),
            candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
            forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        )
        result = runner.run(max_scans=15)
        if not (Path(tmp) / "zero_base" / "directional_dna_report.md").exists():
            _fail("report missing")
        print(f"OK: engines analyzed={len(result['dna_summaries'])}")


def main():
    test_importance()
    test_kmeans()
    test_small_run()
    print("\nALL DNA DISCOVERY V1 TESTS PASSED")


if __name__ == "__main__":
    main()
