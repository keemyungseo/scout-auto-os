"""
Scout Phase 17 — Human Formula Blind Test

Applies fixed human formula at 2026-06-20 11:00 KST. No post-scan data in search.

Usage:
  python scout_phase17_human_formula_blind_test.py
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10
from season2_p37_scout_decision_hierarchy import pf, write_csv
from season2_universe_blind_test import load_eligible_symbols

OUT_DIR = Path("logs") / "phase17_formula"
CACHE_DIR = OUT_DIR / "kline_cache"
P16_CACHE = Path("logs") / "phase16_blind" / "kline_cache"
KST = timezone(timedelta(hours=9))
SCAN_KST = "2026-06-20 11:00:00"
EVAL_END_KST = "2026-06-20 15:00:00"
API_SLEEP = 0.04
WORKERS = 10

EXCLUDED = {"BTCUSDT", "ETHUSDT", "USDCUSDT", "XRPUSDT"}
VOL_RATIO_MIN = 2.3
RET_MIN = 1.2
RET_MAX = 8.0
MA_DIST_MAX = 6.0

PHASE16_TOP2 = (
    ("BTWUSDT", 19.7061, 17.6027),
    ("EDGEUSDT", 7.4501, 6.042),
)


def parse_kst(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


def fetch_klines(symbol: str, interval: str, end_ms: int, limit: int) -> list[list]:
    tag = f"{interval}_{symbol}_{end_ms}.json"
    for base in (CACHE_DIR, P16_CACHE):
        p = base / tag
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    params = urllib.parse.urlencode({
        "symbol": symbol, "interval": interval, "endTime": end_ms,
        "limit": min(limit, t10.MAX_LIMIT),
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
                data = json.loads(resp.read().decode())
            (CACHE_DIR / tag).write_text(json.dumps(data), encoding="utf-8")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (418, 429) and attempt < 4:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    return []


def fetch_forward_5m(symbol: str, start_ms: int, count: int = 48) -> list[list]:
    tag = f"fwd5m_{symbol}_{start_ms}.json"
    for base in (CACHE_DIR, P16_CACHE):
        p = base / tag
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    params = urllib.parse.urlencode({
        "symbol": symbol, "interval": "5m", "startTime": start_ms, "limit": count,
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
                data = json.loads(resp.read().decode())
            (CACHE_DIR / tag).write_text(json.dumps(data), encoding="utf-8")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (418, 429) and attempt < 4:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    return []


def snapshot_30m(symbol: str, end_ms: int) -> dict | None:
    try:
        kl = fetch_klines(symbol, "30m", end_ms, 25)
        if len(kl) < 22:
            return None
        k = kl[-1]
        o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
        vol = float(k[5])
        qv = float(k[7]) if len(k) > 7 else vol * c
        if not (t10.MIN_PRICE <= c <= t10.MAX_PRICE):
            return None

        prev_vols = [float(x[5]) for x in kl[-21:-1]]
        prev_closes = [float(x[4]) for x in kl[-21:-1]]
        vol_ma20 = statistics.mean(prev_vols) if prev_vols else vol
        ma20 = statistics.mean(prev_closes) if prev_closes else c
        ret_pct = (c - o) / o * 100 if o else 0
        dist_ma = (c - ma20) / ma20 * 100 if ma20 else 0
        vol_ratio = vol / vol_ma20 if vol_ma20 else 0

        return {
            "symbol": symbol,
            "price_at_11": c,
            "open_30m": round(o, 8),
            "high_30m": round(h, 8),
            "low_30m": round(l, 8),
            "close_30m": round(c, 8),
            "volume_30m": round(vol, 4),
            "quote_volume_30m": round(qv, 4),
            "volume_ma20_30m": round(vol_ma20, 4),
            "volume_ratio_30m": round(vol_ratio, 4),
            "candle_return_pct": round(ret_pct, 4),
            "ma20_30m": round(ma20, 8),
            "distance_from_ma20_pct": round(dist_ma, 4),
        }
    except Exception:
        return None
    finally:
        time.sleep(API_SLEEP)


def passes_formula(row: dict, use_ma_dist: bool) -> bool:
    if row["volume_ratio_30m"] < VOL_RATIO_MIN:
        return False
    if row["candle_return_pct"] < RET_MIN or row["candle_return_pct"] > RET_MAX:
        return False
    if use_ma_dist and row["distance_from_ma20_pct"] >= MA_DIST_MAX:
        return False
    return True


def eval_forward(symbol: str, entry_ms: int) -> dict:
    fwd = fetch_forward_5m(symbol, entry_ms, 48)
    if not fwd:
        return {}
    entry = float(fwd[0][1])
    if entry <= 0:
        return {}
    highs = [float(k[2]) for k in fwd]
    lows = [float(k[3]) for k in fwd]
    closes = [float(k[4]) for k in fwd]
    times = [int(k[0]) for k in fwd]
    max_h = max(highs)
    min_l = min(lows)
    max_i = highs.index(max_h)
    max_up = (max_h - entry) / entry * 100
    ret = (closes[-1] - entry) / entry * 100
    mdd = (min_l - entry) / entry * 100
    max_up_kst = datetime.fromtimestamp(times[max_i] / 1000, tz=KST).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "entry_price": round(entry, 8),
        "return_4h_pct": round(ret, 4),
        "max_up_4h_pct": round(max_up, 4),
        "mdd_4h_pct": round(mdd, 4),
        "max_up_time_kst": max_up_kst,
    }


def miss_reason(row: dict | None, in_top20: bool) -> str:
    if not row:
        return "excluded_or_no_data"
    parts = []
    if not in_top20:
        parts.append("not_in_top20_quote_volume")
    if row["volume_ratio_30m"] < VOL_RATIO_MIN:
        parts.append(f"volume_ratio={row['volume_ratio_30m']:.2f}<2.3")
    if row["candle_return_pct"] < RET_MIN:
        parts.append(f"return={row['candle_return_pct']:.2f}%<1.2%")
    if row["candle_return_pct"] > RET_MAX:
        parts.append(f"return={row['candle_return_pct']:.2f}%>8.0%")
    if row["distance_from_ma20_pct"] >= MA_DIST_MAX:
        parts.append(f"ma_dist={row['distance_from_ma20_pct']:.2f}%>=6.0%")
    return "; ".join(parts) if parts else "passed_all_but_not_selected"


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scan_dt = parse_kst(SCAN_KST)
    end_ms = int(scan_dt.timestamp() * 1000)
    entry_ms = end_ms

    symbols = sorted(
        s for s in load_eligible_symbols(refresh=False, cache_only=False)
        if s not in EXCLUDED
    )
    print(f"Phase 17 formula blind: {SCAN_KST} | universe={len(symbols)}")

    snapshots: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(snapshot_30m, s, end_ms): s for s in symbols}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                snapshots.append(r)

    snapshots.sort(key=lambda x: x["quote_volume_30m"], reverse=True)
    top20 = snapshots[:20]
    top20_syms = {r["symbol"] for r in top20}
    for i, r in enumerate(top20, 1):
        r["qv_rank_top20"] = i

    snap_by_sym = {r["symbol"]: r for r in snapshots}

    def search(use_ma: bool) -> list[dict]:
        passed = [r for r in top20 if passes_formula(r, use_ma)]
        passed.sort(key=lambda x: x["quote_volume_30m"], reverse=True)
        for i, r in enumerate(passed, 1):
            r["formula_rank"] = i
        return passed

    ma_removed = False
    passed = search(use_ma=True)
    if not passed:
        ma_removed = True
        passed = search(use_ma=False)

    selected: list[dict] = []
    if len(passed) >= 3:
        selected = [passed[1], passed[2]]
    elif len(passed) == 2:
        selected = passed[:2]
    elif len(passed) == 1:
        selected = passed[:1]

    sel_syms = {r["symbol"] for r in selected}
    for r in passed:
        r["selected"] = r["symbol"] in sel_syms

    rows_out = []
    for r in passed:
        rows_out.append({
            **r,
            "qv_rank": r.get("formula_rank", 0),
            "selected": "Y" if r["selected"] else "N",
        })

    print("  evaluating forward performance...")
    all_fwd: list[dict] = []
    eval_syms = set(symbols)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(eval_forward, s, entry_ms): s for s in eval_syms}
        for fut in as_completed(futs):
            sym = futs[fut]
            fwd = fut.result()
            if fwd:
                all_fwd.append({"symbol": sym, **fwd})
            time.sleep(0.02)

    all_fwd.sort(key=lambda x: x["max_up_4h_pct"], reverse=True)
    for i, r in enumerate(all_fwd, 1):
        r["rank_max_up"] = i
    all_fwd.sort(key=lambda x: x["return_4h_pct"], reverse=True)
    ret_rank = {r["symbol"]: i for i, r in enumerate(all_fwd, 1)}

    for r in rows_out:
        f = next((x for x in all_fwd if x["symbol"] == r["symbol"]), {})
        r.update({k: f.get(k) for k in (
            "entry_price", "return_4h_pct", "max_up_4h_pct", "mdd_4h_pct",
            "max_up_time_kst", "rank_max_up",
        )})
        r["rank_return_4h"] = ret_rank.get(r["symbol"])

    sel_results = [r for r in rows_out if r["selected"] == "Y"]

    universe_top10 = sorted(all_fwd, key=lambda x: x["max_up_4h_pct"], reverse=True)[:10]
    missed = [r for r in universe_top10 if r["symbol"] not in sel_syms]

    # Verdict
    sel_max = [r.get("max_up_4h_pct", 0) for r in sel_results]
    scout_max = [x[1] for x in PHASE16_TOP2]
    avg_sel_rank = statistics.mean([r.get("rank_max_up", 99) for r in sel_results]) if sel_results else 99
    hit_btw = any(r["symbol"] == "BTWUSDT" for r in passed)
    hit_edge = any(r["symbol"] == "EDGEUSDT" for r in passed)
    sel_hit_top2_actual = (
        len(sel_results) >= 2
        and sel_results[0].get("rank_max_up", 99) <= 2
        and sel_results[1].get("rank_max_up", 99) <= 2
    ) if len(sel_results) >= 2 else False

    if sel_hit_top2_actual or (sel_max and max(sel_max) >= max(scout_max) * 0.9):
        verdict = "KEEP"
    elif sel_max and max(sel_max) >= 5:
        verdict = "MODIFY"
    else:
        verdict = "DISCARD"

    lines = [
        "##############################################################",
        "SCOUT PHASE 17 — HUMAN FORMULA BLIND TEST",
        "##############################################################",
        "",
        f"Scan: {SCAN_KST} KST | Eval: {SCAN_KST} -> {EVAL_END_KST}",
        f"MA20 distance condition removed: {'YES' if ma_removed else 'NO'}",
        "",
        "=" * 58,
        "1. UNIVERSE",
        "=" * 58,
        f"  Eligible symbols: {len(symbols)}",
        f"  With 30m data: {len(snapshots)}",
        "",
        "=" * 58,
        "2. TOP20 BY 30m QUOTE VOLUME (10:30-11:00)",
        "=" * 58,
    ]
    for r in top20:
        lines.append(
            f"  #{r['qv_rank_top20']} {r['symbol']} qv={r['quote_volume_30m']:.0f} "
            f"ret={r['candle_return_pct']:.2f}% volR={r['volume_ratio_30m']:.2f}"
        )

    lines.extend(["", "=" * 58, "3. FORMULA PASS (all conditions)", "=" * 58])
    if not passed:
        lines.append("  (none)")
    for r in passed:
        lines.append(
            f"  #{r['formula_rank']} {r['symbol']} selected={r['selected']} "
            f"qv={r['quote_volume_30m']:.0f} volR={r['volume_ratio_30m']:.2f} "
            f"ret={r['candle_return_pct']:.2f}% maDist={r['distance_from_ma20_pct']:.2f}%"
        )

    lines.extend(["", "=" * 58, "4. FINAL SELECTION", "=" * 58])
    for r in sel_results:
        lines.append(
            f"  ** {r['symbol']} ** formula_rank=#{r['formula_rank']} "
            f"max_up={r.get('max_up_4h_pct')}% ret={r.get('return_4h_pct')}% "
            f"actual_rank=#{r.get('rank_max_up')}"
        )

    lines.extend(["", "=" * 58, "5. vs PHASE16 SCOUT", "=" * 58])
    lines.append(f"  BTWUSDT in formula pass: {hit_btw}")
    lines.append(f"  EDGEUSDT in formula pass: {hit_edge}")
    lines.append(f"  SCOUT TOP1 BTW max_up=19.71% | TOP2 EDGE max_up=7.45%")
    if sel_max:
        lines.append(f"  Formula TOP2 max_up: {sel_max}")
        better = max(sel_max) > max(scout_max)
        lines.append(f"  vs SCOUT: {'better peak' if better else 'weaker peak'}")
    lines.append(
        f"  Formula picked actual top2 by max_up: {sel_hit_top2_actual}"
    )

    lines.extend(["", "=" * 58, "6. UNIVERSE TOP10 max_up (11:00-15:00)", "=" * 58])
    for r in universe_top10:
        mark = " SELECTED" if r["symbol"] in sel_syms else ""
        lines.append(
            f"  #{r['rank_max_up']} {r['symbol']} max_up={r['max_up_4h_pct']}% "
            f"ret={r['return_4h_pct']}%"
            f"{mark}"
        )

    lines.extend(["", "=" * 58, "7. MISSED WINNERS (top10 not selected)", "=" * 58])
    for m in missed[:5]:
        snap = snap_by_sym.get(m["symbol"])
        lines.append(f"  {m['symbol']} max_up={m['max_up_4h_pct']}% rank=#{m['rank_max_up']}")
        lines.append(f"    reason: {miss_reason(snap, m['symbol'] in top20_syms)}")

    lines.extend(["", "=" * 58, "8. VERDICT", "=" * 58,
        f"  Decision: {verdict}",
        f"  avg selected max_up rank: {avg_sel_rank:.1f}",
        "",
        f"ONE LINE: Human formula verdict: {verdict} — "
        f"selected {[r['symbol'] for r in sel_results]} "
        f"max_up={[r.get('max_up_4h_pct') for r in sel_results]} "
        f"vs SCOUT BTW/EDGE 19.71%/7.45%.",
    ])

    report = OUT_DIR / "phase17_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    write_csv(OUT_DIR / "top20_quote_volume.csv", top20)
    write_csv(OUT_DIR / "formula_pass.csv", rows_out)
    write_csv(OUT_DIR / "universe_forward_top30.csv", universe_top10)

    print("\n".join(lines[-25:]).encode("ascii", "replace").decode("ascii"))
    print(f"\nSaved: {report}")


if __name__ == "__main__":
    run()
