"""Collect replay entries for Trade DNA analysis."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from scout_auto_os.engine.portfolio.backtest import filter_2h_scans, _parse_scan
from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.portfolio.scoring import build_pass_candidates
from scout_auto_os.engine.research.trade_dna.curve_builder import TradeDNARecord, build_trade_dna
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines


def collect_pass_candidates(
    data_dir: Path,
    pkg_root: Path,
    candidates_path: Path,
    forward_path: Path,
    replay_days: int = 15,
) -> list[TradeDNARecord]:
    by_scan = load_candidates_jsonl(candidates_path)
    fwd = load_forward_klines(forward_path)
    scans = filter_2h_scans(sorted(by_scan.keys()))
    if scans:
        cut = _parse_scan(scans[-1]) - timedelta(days=replay_days)
        scans = [s for s in scans if _parse_scan(s) >= cut]

    engine = PortfolioEngine.from_paths(data_dir, pkg_root)
    records: list[TradeDNARecord] = []
    seen: set[str] = set()

    for scan_kst in scans:
        rows = [{"symbol": r["symbol"], "features": r["features"]} for r in by_scan[scan_kst]]
        feat_map = {r["symbol"]: r["features"] for r in rows}
        long_c, short_c = build_pass_candidates(rows, scan_kst, engine.rules, scan_kst)
        for direction, bucket in (("long", long_c), ("short", short_c)):
            for c in bucket:
                sym = c["symbol"]
                key = f"{scan_kst}|{sym}|{direction}"
                if key in seen:
                    continue
                klines = fwd.get((scan_kst, sym))
                if not klines:
                    continue
                rec = build_trade_dna(
                    scan_kst, sym, direction,
                    float(c.get("entry_score", 0)),
                    str(c.get("live_pattern", "")),
                    feat_map.get(sym, {}),
                    klines,
                )
                if rec:
                    seen.add(key)
                    records.append(rec)

    slot_entries = _collect_filled_slots(data_dir, pkg_root, by_scan, fwd, scans)
    for rec in slot_entries:
        if rec.trade_key not in seen:
            seen.add(rec.trade_key)
            records.append(rec)

    return records


def _collect_filled_slots(
    data_dir: Path,
    pkg_root: Path,
    by_scan: dict,
    fwd: dict,
    scans: list[str],
) -> list[TradeDNARecord]:
    engine = PortfolioEngine.from_paths(data_dir, pkg_root)
    out: list[TradeDNARecord] = []
    for scan_kst in scans:
        hold_until = scans[scans.index(scan_kst) + 1] if scans.index(scan_kst) + 1 < len(scans) else scan_kst
        rows = [{"symbol": r["symbol"], "features": r["features"]} for r in by_scan[scan_kst]]
        result = engine.process_scan(rows, scan_kst, hold_until_scan=hold_until)
        for entry in result["new_entries"]:
            klines = fwd.get((scan_kst, entry["symbol"]))
            if not klines:
                continue
            rec = build_trade_dna(
                scan_kst, entry["symbol"], entry["direction"],
                float(entry.get("entry_score", 0)),
                str(entry.get("live_pattern", "")),
                entry.get("features") or {},
                klines,
            )
            if rec:
                rec.features["slot_filled"] = 1
                out.append(rec)
    return out
