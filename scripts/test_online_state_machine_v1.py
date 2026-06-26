"""Online State Machine V1 tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.online_state_machine.state_estimator import annotate_states
from scout_auto_os.engine.research.online_state_machine.transitions import extract_transitions
from scout_auto_os.engine.research.signal_lifecycle.timeline import build_signal_timeline


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _sample_klines(n: int = 30, drift: float = 0.003) -> list:
    bars = []
    px = 100.0
    t0 = 1780239600000
    for i in range(n):
        o = px
        c = px * (1 + drift)
        h = max(o, c) * 1.004
        l = min(o, c) * 0.996
        bars.append([t0 + i * 900_000, o, h, l, c, 1_000_000.0 * (1 + i * 0.02)])
        px = c
    return bars


def test_state_annotation():
    klines = _sample_klines(40)
    tl, _ = build_signal_timeline(klines, "long", "2026-06-01 00:00:00", "T", "long_T")
    from scout_auto_os.engine.research.online_state_machine.online_features import enrich_timeline_online

    stated = annotate_states(enrich_timeline_online(tl))
    if not all("state" in r for r in stated):
        _fail("missing state")
    trans, seq = extract_transitions("long_T", "long", "2026-06-01 00:00:00", "T", stated)
    if "->" not in seq:
        _fail("bad sequence")
    if not trans:
        _fail("need transitions")
    print(f"OK: seq={seq[:60]}... transitions={len(trans)}")


def test_full_run():
    with tempfile.TemporaryDirectory() as tmp:
        from scout_auto_os.engine.research.online_state_machine.runner import OnlineStateMachineRunner

        runner = OnlineStateMachineRunner(
            Path(tmp),
            PKG,
            PKG / "research_bundle" / "seed" / "candidates.jsonl",
            PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        )
        result = runner.run()
        out = Path(tmp) / "online_state_machine"
        for name in (
            "state_timeline.csv",
            "state_sequence.csv",
            "transition_matrix.csv",
            "state_statistics.csv",
            "state_report.md",
        ):
            if not (out / name).exists():
                _fail(f"missing {name}")
        print(f"OK: transitions={result['transition_count']}")


def main():
    test_state_annotation()
    test_full_run()
    print("\nALL ONLINE STATE MACHINE V1 TESTS PASSED")


if __name__ == "__main__":
    main()
