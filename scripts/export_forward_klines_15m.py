"""
Export forward 15m klines for candidates.jsonl backtest.

Each row: {scan_kst, symbol, start_ms, bar_count, forward_klines_15m}
  forward_klines_15m = [[open_time_ms, open, high, low, close, volume], ...]

Forward window: scan_kst timestamp -> +24h (96 x 15m bars).
Caches to logs/phase19_winner_dna/kline_cache/fwd15m_{SYMBOL}_{start_ms}.json

Usage:
  python scripts/export_forward_klines_15m.py
  python scripts/export_forward_klines_15m.py --last-scans 54
  python scripts/export_forward_klines_15m.py --out scout_auto_os/research_bundle/forward/forward_klines_15m.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import top10_gainer_learning_20260613 as t10

KST = timezone(timedelta(hours=9))
FORWARD_BARS = 96
API_SLEEP = 0.04
WORKERS = 10

DEFAULT_CANDIDATES = ROOT / "logs" / "phase19_winner_dna" / "candidates.jsonl"
FALLBACK_CANDIDATES = ROOT / "scout_auto_os" / "research_bundle" / "seed" / "candidates.jsonl"
CACHE_DIR = ROOT / "logs" / "phase19_winner_dna" / "kline_cache"
DEFAULT_OUT = ROOT / "scout_auto_os" / "research_bundle" / "forward" / "forward_klines_15m.jsonl"


def safe_print(msg: str, **kwargs) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"), **kwargs)


def parse_kst(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


def compact_kline(raw: list) -> list:
    return [int(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]), float(raw[4]), float(raw[5])]


def fetch_forward_15m(symbol: str, start_ms: int, count: int = FORWARD_BARS) -> list[list]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"fwd15m_{symbol}_{start_ms}.json"
    path = CACHE_DIR / tag
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    params = urllib.parse.urlencode({
        "symbol": symbol,
        "interval": "15m",
        "startTime": start_ms,
        "limit": min(count, t10.MAX_LIMIT),
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
                data = json.loads(resp.read().decode())
            path.write_text(json.dumps(data), encoding="utf-8")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (418, 429) and attempt < 4:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    return []


def load_candidates(path: Path, last_scans: int | None) -> list[dict]:
    src = path if path.exists() else FALLBACK_CANDIDATES
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    if last_scans is None:
        return rows
    scans = sorted(set(r["scan_kst"] for r in rows))
    keep = set(scans[-last_scans:])
    return [r for r in rows if r["scan_kst"] in keep]


def export_row(row: dict) -> dict:
    scan_kst = row["scan_kst"]
    symbol = row["symbol"]
    start_ms = int(parse_kst(scan_kst).timestamp() * 1000)
    base = {"scan_kst": scan_kst, "symbol": symbol, "start_ms": start_ms}
    try:
        raw = fetch_forward_15m(symbol, start_ms, FORWARD_BARS)
    except Exception as exc:
        time.sleep(API_SLEEP)
        return {**base, "bar_count": 0, "error": str(exc), "forward_klines_15m": []}
    time.sleep(API_SLEEP)
    if not raw:
        return {**base, "bar_count": 0, "forward_klines_15m": []}
    bars = [compact_kline(k) for k in raw[:FORWARD_BARS]]
    return {**base, "bar_count": len(bars), "forward_klines_15m": bars}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--last-scans", type=int, default=None, help="Only last N scan timestamps")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    rows = load_candidates(args.candidates, args.last_scans)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    safe_print(f"Export forward 15m: {len(rows)} pairs -> {args.out}")
    written = 0
    errors = 0
    short = 0

    with args.out.open("w", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(export_row, r): r for r in rows}
            for i, fut in enumerate(as_completed(futs), 1):
                out = fut.result()
                if out is None:
                    continue
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1
                if out.get("error"):
                    errors += 1
                if out.get("bar_count", 0) < FORWARD_BARS:
                    short += 1
                if i % 100 == 0 or i == len(rows):
                    safe_print(f"  progress {i}/{len(rows)} written={written} short={short} err={errors}")

    size_mb = args.out.stat().st_size / 1e6
    safe_print(f"Done: {written} rows, {size_mb:.1f} MB, short_bars={short}, errors={errors}")


if __name__ == "__main__":
    main()
