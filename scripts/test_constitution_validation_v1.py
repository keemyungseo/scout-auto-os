"""Constitution Validation V1 tests."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))


def test_regime_tags():
    from scout_auto_os.engine.research.constitution_validation.regime_validator import tag_scan_regimes

    rows = [
        {
            "scan_kst": "2026-06-01 00:00:00",
            "symbol": "BTCUSDT",
            "features": {
                "1h_current_return_pct": 1.5,
                "2h_current_return_pct": 1.0,
                "15m_current_volume_ratio": 1.2,
                "5m_compression": 1.0,
                "5m_release": 0.2,
                "1h_current_range_pct": 3.0,
                "h4_score": 80,
            },
        },
    ]
    tags = tag_scan_regimes(rows)
    assert "market_simple" in tags
    assert "volatility" in tags
    print("OK: regime tags")


def test_full_run():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("SKIP: no sklearn")
        return

    from scout_auto_os.engine.research.constitution_validation.runner import ConstitutionValidationRunner

    runner = ConstitutionValidationRunner(
        PKG / "data",
        PKG,
        PKG / "research_bundle" / "seed" / "candidates.jsonl",
        PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run()
    if not result:
        raise SystemExit("run failed")
    out = PKG / "data" / "constitution_validation"
    for name in ("final_constitution_report.md", "blind_report.csv", "rolling_validation.csv"):
        if not (out / name).exists():
            raise SystemExit(f"missing {name}")
    print(f"OK: confidence={result['decision'].get('confidence_tier')}")


def main():
    test_regime_tags()
    test_full_run()
    print("\nALL CONSTITUTION VALIDATION V1 TESTS PASSED")


if __name__ == "__main__":
    main()
