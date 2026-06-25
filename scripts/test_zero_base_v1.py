"""SCOUT Zero-Base Discovery V1 — small sample test (3 engines, 10 scans, 10 random draws)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.zero_base.candidates import rank_all_engines, CANDIDATE_ENGINES
from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics
from scout_auto_os.engine.research.zero_base.runner import ZeroBaseRunner, load_candidates_jsonl, load_forward_klines
from scout_auto_os.engine.research.zero_base.ranking import aggregate_candidate_metrics, candidate_score


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_forward_metrics():
    klines = [[0, 100, 105, 99, 103, 1], [1, 103, 108, 102, 107, 1],
              [2, 107, 110, 106, 109, 1], [3, 109, 112, 108, 111, 1]]
    m = compute_forward_metrics(klines, 100)
    if m.get("return_30m") is None:
        _fail("return_30m missing")
    print(f"OK: forward metrics return_2h={m.get('return_2h')}")


def test_rank_engines():
    rows = load_candidates_jsonl(PKG / "research_bundle" / "seed" / "candidates.jsonl")
    scan = sorted(rows.keys())[0]
    picks = rank_all_engines(rows[scan], top_k=5)
    if "A6_CURRENT" not in picks or len(picks["A6_CURRENT"]) != 5:
        _fail("A6 picks failed")
    print(f"OK: rank engines scan={scan} engines={len(picks)}")


def test_small_run():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        engines = ("RANDOM_BASELINE", "A6_CURRENT", "PURE_MOMENTUM_15M")
        runner = ZeroBaseRunner(
            data_dir,
            candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
            forward_klines_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
            random_draws=10,
            top_k=5,
            engines=engines,
            eval_intervals=("1h",),
        )
        result = runner.run(max_scans=10)
        if not result.get("champion_board"):
            _fail("no champion board")
        if not (data_dir / "zero_base" / "zero_base_report.md").exists():
            _fail("report not written")
        if not (data_dir / "zero_base" / "candidate_results.csv").exists():
            _fail("candidate_results missing")
        a6 = result.get("a6", {})
        print(
            f"OK: small run a6 avg2h={a6.get('avg_return_2h')} "
            f"board={len(result['champion_board'])} scans={result['meta'].get('validation_scans')}"
        )


def test_candidate_score():
    row = {"avg_return_2h": 2, "big_winner_capture_rate": 10, "win_rate": 40, "trap_rate": 5, "max_drawdown_avg": -3}
    s = candidate_score(row)
    if s <= 0:
        _fail(f"score should be positive got {s}")
    print(f"OK: candidate_score={s}")


def main():
    test_forward_metrics()
    test_rank_engines()
    test_candidate_score()
    test_small_run()
    print("\nALL ZERO-BASE V1 TESTS PASSED")


if __name__ == "__main__":
    main()
