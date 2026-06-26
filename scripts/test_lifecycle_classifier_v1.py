"""Lifecycle Classifier V1 tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.lifecycle_classifier.metrics import per_class_metrics
from scout_auto_os.engine.research.lifecycle_classifier.model import MultinomialLifecycleClassifier


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_multinomial_classifier():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 8))
    y = rng.integers(0, 3, size=120)
    names = ["A", "B", "C"]
    model = MultinomialLifecycleClassifier(names, max_iter=400)
    info = model.fit(X, y)
    probs = model.predict_proba(X)
    if probs.shape != (120, 3):
        _fail("bad proba shape")
    if abs(probs.sum(axis=1).mean() - 1.0) > 1e-5:
        _fail("proba not normalized")
    print(f"OK: train_acc={info['train_accuracy']}")


def test_metrics():
    y_t = ["Fake Breakout", "Slow Trend", "Fake Breakout"]
    y_p = ["Fake Breakout", "Fake Breakout", "Slow Trend"]
    rows = per_class_metrics(y_t, y_p, ["Fake Breakout", "Slow Trend"])
    if not rows:
        _fail("no metrics")
    print("OK: metrics")


def test_full_run():
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        (data / "zero_base").mkdir(parents=True)
        for name in ("entry_filter_rules_v2.json", "directional_dna_formulas.json"):
            for src in (
                PKG / "data" / "zero_base" / name,
                PKG / "research_bundle" / "reports" / name.replace(".json", "_v2.json" if "entry" in name else "_v1_formulas.json"),
            ):
                if src.exists():
                    (data / "zero_base" / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                    break
        from scout_auto_os.engine.research.lifecycle_classifier.runner import LifecycleClassifierRunner

        runner = LifecycleClassifierRunner(
            data,
            PKG,
            PKG / "research_bundle" / "seed" / "candidates.jsonl",
            PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        )
        result = runner.run()
        out = data / "lifecycle_classifier"
        if not (out / "lifecycle_classifier_report.md").exists():
            _fail("report missing")
        print(
            f"OK: long_f1={result['long_metrics']['aggregate']['macro_f1']} "
            f"short_f1={result['short_metrics']['aggregate']['macro_f1']}",
        )


def main():
    test_multinomial_classifier()
    test_metrics()
    test_full_run()
    print("\nALL LIFECYCLE CLASSIFIER V1 TESTS PASSED")


if __name__ == "__main__":
    main()
