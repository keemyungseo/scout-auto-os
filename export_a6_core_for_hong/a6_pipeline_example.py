"""
A6 pipeline example: train -> search -> rank -> eval
Original reference: scout_blind_test_b001.py

External usage:
  1. Prepare multi-TF OHLCV dataframes per symbol (5m, 15m, 30m, 1h, 2h)
  2. Build train set from historical scans (features + outcome_rank)
  3. Call rank_scan_at_timestamp() at live scan time
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

from export_a6_core_for_hong.a6_common import WINNER_TOP_N, ohlcv, parse_kst
from export_a6_core_for_hong.a6_feature_core import (
    extract_dna_features_from_klines,
)
from export_a6_core_for_hong.a6_score_core import build_train_stats, formula_scores_a6, score_candidate_a6
from export_a6_core_for_hong.a6_state_core import (
    annotate,
    build_profile,
    build_thresholds,
    state_match_score,
    winner_loser_sets,
)

KST = timezone(timedelta(hours=9))


def _safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def load_train_from_jsonl(path: Path, cutoff_kst: str) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Load Phase19-style training rows before cutoff.
    Each line: {scan_kst, symbol, features, max_up_4h, outcome_rank?}
    """
    raw: list[dict] = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if r["scan_kst"] < cutoff_kst:
            raw.append(r)
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by_scan[r["scan_kst"]].append(r)
    for s in by_scan:
        by_scan[s].sort(key=lambda x: x.get("outcome_rank", 999))
    return raw, by_scan


def build_train_context(
    train_rows: list[dict],
    train_by_scan: dict[str, list[dict]],
) -> tuple[object, object, dict, dict]:
    """Returns thresholds, profile, stats, train_by_scan."""
    winner_feats = [
        r["features"] for rows in train_by_scan.values()
        for r in rows[:WINNER_TOP_N] if len(rows) >= 4
    ]
    th = build_thresholds(winner_feats)
    annotated = annotate(train_rows, th)
    ann_by: dict[str, list[dict]] = defaultdict(list)
    for r in annotated:
        ann_by[r["scan_kst"]].append(r)
    for s in ann_by:
        ann_by[s].sort(key=lambda x: x.get("outcome_rank", 999))
    w_train, _ = winner_loser_sets(ann_by)
    profile = build_profile(w_train, annotated) if w_train else build_profile([], annotated)
    stats = build_train_stats(annotated, ann_by, th)
    return th, profile, stats, ann_by


def rank_scan_candidates(
    candidates: list[dict],
    profile: dict,
    th,
    stats: dict,
    top_n: int = 5,
) -> list[dict]:
    """
    Score and rank scan candidates with frozen A6 formula.
    Each candidate: {symbol, features, scan_kst, ...}
    """
    rows: list[dict] = []
    for c in candidates:
        f = c["features"]
        row = {"symbol": c["symbol"], "scan_kst": c.get("scan_kst", ""), "features": f}
        ann = annotate([row], th)[0]
        rows.append({**c, **ann})

    for r in rows:
        base = state_match_score(r["states"], r["transitions"], profile)
        r["base_score"] = base
        r["a6_score"] = round(formula_scores_a6(r, rows, base, th, stats)["A6"], 4)

    ranked = sorted(rows, key=lambda x: x["a6_score"], reverse=True)
    for i, r in enumerate(ranked[:top_n], 1):
        r["rank"] = i
    return ranked[:top_n]


def eval_forward_max_return(entry_price: float, forward_k5: list[list]) -> dict:
    """Simple 4h forward eval from 5m klines (48 bars)."""
    if not forward_k5 or entry_price <= 0:
        return {}
    entry_p = float(forward_k5[0][1])
    max_h = max(ohlcv(k)[1] for k in forward_k5[:48])
    close_4h = ohlcv(forward_k5[min(47, len(forward_k5) - 1)])[3]
    return {
        "entry_price": entry_p,
        "max_return_pct": round((max_h - entry_p) / entry_p * 100, 4),
        "final_4h_return_pct": round((close_4h - entry_p) / entry_p * 100, 4),
    }


def rank_scan_at_timestamp(
    symbol_klines: dict[str, dict[str, list[list]]],
    scan_kst: str,
    train_rows: list[dict],
    train_by_scan: dict[str, list[dict]],
    top_n: int = 5,
) -> list[dict]:
    """
    Full search at one timestamp.
    symbol_klines: {symbol: {"5m": [...], "15m": [...], ...}}
    """
    th, profile, stats, _ = build_train_context(train_rows, train_by_scan)
    end_ms = int(parse_kst(scan_kst).timestamp() * 1000)
    candidates: list[dict] = []
    for sym, kl_map in symbol_klines.items():
        feats = extract_dna_features_from_klines(
            kl_map["5m"], kl_map["15m"], end_ms,
            k30=kl_map.get("30m"), k1h=kl_map.get("1h"), k2h=kl_map.get("2h"),
        )
        if feats:
            candidates.append({"symbol": sym, "scan_kst": scan_kst, "features": feats})
    return rank_scan_candidates(candidates, profile, th, stats, top_n=top_n)


def _demo_synthetic_kline(base: float, n: int, start_ms: int, interval_ms: int) -> list[list]:
    """Generate minimal synthetic klines for import test only."""
    out = []
    px = base
    for i in range(n):
        o = px
        h = px * 1.008
        l = px * 0.992
        c = px * 1.003
        v = 1000 + i * 10
        out.append([start_ms + i * interval_ms, o, h, l, c, v])
        px = c
    return out


def run_demo() -> None:
    """Minimal runnable demo without network or project data."""
    print("A6 Core Demo (synthetic data)")
    scan_kst = "2026-06-16 17:00:00"
    end_ms = int(parse_kst(scan_kst).timestamp() * 1000)
    start_ms = end_ms - 120 * 5 * 60 * 1000

    sym = "DEMOUSDT"
    k5 = _demo_synthetic_kline(1.0, 120, start_ms, 5 * 60 * 1000)
    k15 = _demo_synthetic_kline(1.0, 96, start_ms, 15 * 60 * 1000)
    kl_map = {"5m": k5, "15m": k15, "30m": k15, "1h": k15, "2h": k15}

    feats = extract_dna_features_from_klines(k5, k15, end_ms, k30=k15, k1h=k15, k2h=k15)
    if not feats:
        print("Pattern B rejected synthetic demo symbol (expected for flat synthetic data)")
        print("Use real OHLCV dataframes for production ranking.")
        return

    fake_train = [{
        "scan_kst": "2026-06-15 17:00:00",
        "symbol": sym,
        "features": feats,
        "outcome_rank": 1,
        "max_up_4h": 5.0,
    }]
    train_by = {"2026-06-15 17:00:00": fake_train}
    th, profile, stats, _ = build_train_context(fake_train, train_by)
    ranked = rank_scan_at_timestamp({sym: kl_map}, scan_kst, fake_train, train_by, top_n=1)
    if ranked:
        r = ranked[0]
        _safe_print(f"  {r['symbol']} A6={r['a6_score']:.4f} states={r['states']}")


def run_with_project_data() -> None:
    """Optional: run against local Phase19 candidates.jsonl if present."""
    p19 = Path("logs/phase19_winner_dna/candidates.jsonl")
    if not p19.exists():
        print(f"Project data not found: {p19}")
        return
    cutoff = "2026-06-16 17:00:00"
    train, train_by = load_train_from_jsonl(p19, cutoff)
    th, profile, stats, _ = build_train_context(train, train_by)
    scan_rows = [r for r in train if r["scan_kst"] == "2026-06-15 17:00:00"][:20]
    if not scan_rows:
        scan_rows = list(train_by.values())[0][:20]
    candidates = [{"symbol": r["symbol"], "scan_kst": r["scan_kst"], "features": r["features"]} for r in scan_rows]
    top5 = rank_scan_candidates(candidates, profile, th, stats, top_n=5)
    print(f"Ranked {len(top5)} from project cache:")
    for r in top5:
        _safe_print(f"  #{r['rank']} {r['symbol']} A6={r['a6_score']:.4f}")


if __name__ == "__main__":
    run_demo()
    print()
    run_with_project_data()
