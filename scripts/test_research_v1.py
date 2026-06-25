"""SCOUT Research Engine V1 tests — no real orders."""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_bundle"))

import scout_phase16_human_blind_test as p16
import scout_phase19_winner_ranking_dna as p19

os.environ.setdefault("SCOUT_UNIVERSE_CACHE_ONLY", "true")

from scout_auto_os.engine.research_engine import ResearchEngine, FORBIDDEN_IMPORTS
from scout_auto_os.engine.telegram_commands import TelegramCommandBot
from scout_auto_os.storage.db import now_kst


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_no_order_references() -> None:
    research_dir = ROOT / "scout_auto_os" / "engine" / "research"
    for py in [ROOT / "scout_auto_os" / "engine" / "research_engine.py", *research_dir.glob("*.py")]:
        for line in py.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#") or "FORBIDDEN_IMPORTS" in line:
                continue
            for token in FORBIDDEN_IMPORTS:
                if token in line and ("import" in line or "(" in line):
                    _fail(f"{py.name} references order function: {token}")
            if "execution_engine" in line and "import" in line and "pilot_execution" not in line and "execution_statistics" not in line:
                _fail(f"{py.name} imports execution_engine")
    print("OK: no order function references")


def test_scan_and_storage() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="scout_research_test_"))
    try:
        cache = ROOT / "logs" / "phase19_winner_dna" / "kline_cache"
        candidates = ROOT / "logs" / "phase19_winner_dna" / "candidates.jsonl"
        seed = ROOT / "scout_auto_os" / "research_bundle" / "seed" / "candidates.jsonl"
        if not candidates.exists() and seed.exists():
            candidates.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(seed, candidates)
        if not cache.exists():
            cache = ROOT / "scout_auto_os" / "research_bundle"
        p19.OUT_DIR = candidates.parent
        p19.CACHE_DIR = cache
        p19.CANDIDATES_PATH = candidates
        p16.CACHE_DIR = cache
        os.environ["SCOUT_UNIVERSE_CACHE_ONLY"] = "false"

        scan_symbols: list[str] = []
        if candidates.exists():
            for line in candidates.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("scan_kst") == "2026-06-15 22:00:00":
                    scan_symbols.append(row["symbol"])
        scan_symbols = sorted(set(scan_symbols))[:40]
        cfg = {
            "research": {
                "enabled": True,
                "scan_interval_min": 5,
                "top_n": 20,
                "workers": 4,
                "max_symbols": 40,
                "scan_symbols": scan_symbols,
            },
            "live_data": {"rest_base": "https://fapi.binance.com", "enabled": False},
            "loop": {"daily_report_hour_kst": 8},
            "alerts": {"telegram": False},
            "paper_mode": True,
        }

        def price_fn(sym: str) -> float:
            return 100.0

        engine = ResearchEngine(cfg, tmp, cache, price_fn, live_top5_fn=lambda: ["BTCUSDT"])
        scan_kst = "2026-06-15 22:00:00"
        result = engine.run_scan_once(scan_kst)
        if not result or not result.get("candidates"):
            _fail(f"scan returned no candidates for {scan_kst}")

        n = len(result["candidates"])
        if n > 20:
            _fail(f"expected at most 20 candidates, got {n}")
        print(f"OK: scan returned {n} candidates")

        cand_path = tmp / "research" / "research_candidates.csv"
        if not cand_path.exists():
            _fail("research_candidates.csv missing")
        with cand_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 1:
            _fail("research_candidates.csv empty")
        print(f"OK: {len(rows)} candidate rows saved")

        fwd_path = tmp / "research" / "research_forward_results.csv"
        if not fwd_path.exists():
            _fail("research_forward_results.csv missing")
        with fwd_path.open(encoding="utf-8") as f:
            fwd_rows = list(csv.DictReader(f))
        placeholders = [r for r in fwd_rows if r.get("return_2h") == ""]
        if not placeholders:
            _fail("expected forward placeholder rows")
        print(f"OK: {len(placeholders)} forward placeholders")

        scan_path = tmp / "research" / "research_scans.csv"
        if not scan_path.exists():
            _fail("research_scans.csv missing")
        print("OK: research_scans.csv exists")

        if engine.forward.pending_count < 1:
            _fail("forward pending queue empty")
        print(f"OK: forward pending={engine.forward.pending_count}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_telegram_commands() -> None:
    snap = {
        "enabled": True,
        "last_scan_time": "2026-06-23 12:00:00",
        "scan_count": 3,
        "forward_pending": 12,
        "formula_league": [{"formula_name": "A6_CURRENT", "score": "10", "win_rate_2h": "50", "sample_count": "4"}],
        "feature_league": [{"feature_name": "volume_ratio", "condition": ">=1.5", "win_rate_2h": "55", "comment": "hypothesis"}],
        "report": {"missed_big_winners": [{"scan_time_kst": "t", "symbol": "X", "return_2h": "5", "rank": "8"}]},
    }
    bot = TelegramCommandBot(
        {"paper_mode": True, "alerts": {"telegram": False}},
        Path("."),
        None,  # type: ignore
        None,  # type: ignore
        lambda: {},
        None,  # type: ignore
        lambda: {},
        research_snapshot_fn=lambda: snap,
    )
    for fn, label in (
        (bot._cmd_research, "/research"),
        (bot._cmd_league, "/league"),
        (bot._cmd_features, "/features"),
        (bot._cmd_missed_research, "/missed_research"),
    ):
        text = fn()
        if not text or label not in text:
            _fail(f"{label} command returned empty/invalid")
        print(f"OK: {label} -> {text.splitlines()[0]}")


def main() -> None:
    print("=== SCOUT Research Engine V1 Test ===")
    test_no_order_references()
    test_scan_and_storage()
    test_telegram_commands()
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
