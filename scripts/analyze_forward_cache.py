"""One-off analysis: forward cache coverage and size estimates."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]
candidates_path = ROOT / "logs" / "phase19_winner_dna" / "candidates.jsonl"
if not candidates_path.exists():
    candidates_path = ROOT / "scout_auto_os" / "research_bundle" / "seed" / "candidates.jsonl"

cache = ROOT / "logs" / "phase19_winner_dna" / "kline_cache"


def parse_kst(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


def start_ms_p19(scan_kst: str) -> int:
    end_ms = int(parse_kst(scan_kst).timestamp() * 1000)
    return end_ms + 5 * 60 * 1000


def start_ms_scan(scan_kst: str) -> int:
    return int(parse_kst(scan_kst).timestamp() * 1000)


rows = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines() if line.strip()]
scans = sorted(set(r["scan_kst"] for r in rows))
last54 = set(scans[-54:])
rows54 = [r for r in rows if r["scan_kst"] in last54]

fwd_files = list(cache.glob("fwd5m_*.json"))
total = sum(f.stat().st_size for f in fwd_files)
avg = total / len(fwd_files) if fwd_files else 0


def coverage(start_fn, label: str) -> None:
    miss = [r for r in rows if not (cache / f"fwd5m_{r['symbol']}_{start_fn(r['scan_kst'])}.json").exists()]
    miss54 = [r for r in rows54 if not (cache / f"fwd5m_{r['symbol']}_{start_fn(r['scan_kst'])}.json").exists()]
    print(f"{label} fwd5m all: {len(rows) - len(miss)}/{len(rows)}")
    print(f"{label} fwd5m last54: {len(rows54) - len(miss54)}/{len(rows54)}")
    if miss:
        print(f"  sample missing: {miss[:3]}")


print("total rows", len(rows))
print("total scans", len(scans), scans[0], "->", scans[-1])
print("last54 rows", len(rows54))
coverage(start_ms_p19, "p19 (+5m)")
coverage(start_ms_scan, "scan_kst")
print("fwd5m files", len(fwd_files), "total MB", round(total / 1e6, 1), "avg bytes", int(avg))
# 96 x 15m bars ~2x 48 x 5m json payload
for label, n in [("all", len(rows)), ("last54", len(rows54))]:
    print(f"est fwd15m jsonl {label} MB", round(n * avg * 2 / 1e6, 1))
