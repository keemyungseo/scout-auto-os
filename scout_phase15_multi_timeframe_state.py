"""
Scout Phase 15 — Multi-Timeframe State Dataset

Builds raw MTF state at scan time. NO filter/ranking/threshold changes.

Usage:
  python scout_phase15_multi_timeframe_state.py
  python scout_phase15_multi_timeframe_state.py --max-scans 3
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import top10_gainer_learning_20260613 as t10
from season2_p40_scout_transition_triggers import ema
from season2_universe_blind_test import ohlcv

OUT_DIR = Path("logs") / "phase15_state"
CACHE_DIR = OUT_DIR / "kline_cache"
PHASE12_CACHE = Path("logs") / "phase12_dna" / "scan_cache"
PHASE14_CACHE = Path("logs") / "phase14_holdout" / "scan_cache"
KST = timezone(timedelta(hours=9))
START_KST = datetime(2026, 5, 1, 0, 0, tzinfo=KST)
END_KST = datetime(2026, 5, 14, 23, 0, tzinfo=KST)
SCAN_INTERVAL_H = 2
API_SLEEP = 0.06
SEQ_N = 6
WIN_THRESHOLD = 7.0
NORMAL_MAX = 5.0

TIMEFRAMES: dict[str, dict] = {
    "5m": {"interval": "5m", "lookback": 120},
    "15m": {"interval": "15m", "lookback": 96},
    "30m": {"interval": "30m", "lookback": 96},
    "1h": {"interval": "1h", "lookback": 96},
    "2h": {"interval": "2h", "lookback": 96},
    "4h": {"interval": "4h", "lookback": 96},
    "6h": {"interval": "6h", "lookback": 96},
    "12h": {"interval": "12h", "lookback": 96},
    "1d": {"interval": "1d", "lookback": 90},
}

STATE_FIELDS = (
    "open", "high", "low", "close", "volume", "volume_ma20", "body", "upper_wick",
    "lower_wick", "range_pct", "atr", "bb_width", "bb_expansion", "vwap_distance",
    "ma5", "ma10", "ma20", "ma60", "ma_slope", "ma_acceleration", "close_position",
    "compression_length", "compression_std", "rs_vs_btc", "dollar_volume",
    "high_distance", "low_distance", "high_close_flag", "body_ratio",
    "positive_count", "return_sum", "volume_energy", "body_energy", "range_energy",
)

COMPARE_FIELDS = (
    "volume_ma20", "range_pct", "ma_slope", "compression_length", "volume_energy",
    "return_sum", "positive_count", "bb_expansion", "rs_vs_btc", "close_position",
)


def gen_scan_times() -> list[tuple[str, datetime]]:
    out: list[tuple[str, datetime]] = []
    t = START_KST
    while t <= END_KST:
        out.append((t.strftime("%Y-%m-%d %H:%M:%S"), t))
        t += timedelta(hours=SCAN_INTERVAL_H)
    return out


def fetch_klines(symbol: str, interval: str, end_ms: int, limit: int) -> list[list]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{interval}_{symbol}_{end_ms}_{limit}.json"
    p = CACHE_DIR / tag
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    params = urllib.parse.urlencode({
        "symbol": symbol, "interval": interval, "endTime": end_ms,
        "limit": min(limit, t10.MAX_LIMIT),
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
                data = json.loads(resp.read().decode())
            p.write_text(json.dumps(data), encoding="utf-8")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (418, 429, 503) and attempt < 3:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    return []


def compression_at(klines: list[list], end_i: int) -> tuple[int, float]:
    ranges: list[float] = []
    n = 0
    for i in range(end_i - 1, max(0, end_i - 48), -1):
        o, h, l, _, _ = ohlcv(klines[i])
        if o <= 0:
            break
        r = (h - l) / o * 100
        if r <= 2.0:
            ranges.append(r)
            n += 1
        else:
            break
    std = statistics.pstdev(ranges) if len(ranges) > 1 else 0.0
    return n, round(std, 4)


def ma_slope_pct(closes: list[float]) -> float:
    if len(closes) < 12:
        return 0.0
    recent = statistics.mean(closes[-6:-1])
    prior = statistics.mean(closes[-12:-7]) if len(closes) >= 12 else recent
    return (recent - prior) / prior * 100 if prior else 0.0


def seq_window(klines: list[list], end_i: int, vol_ma: float) -> dict:
    n = min(SEQ_N, end_i + 1)
    rows = []
    for i in range(end_i - n + 1, end_i + 1):
        o, h, l, c, vol = ohlcv(klines[i])
        ret = (c - o) / o * 100 if o else 0.0
        rng = (h - l) / o * 100 if o else 0.0
        body = abs(c - o) / o * 100 if o else 0.0
        br = body / rng if rng > 0 else 0.0
        vr = vol / vol_ma if vol_ma > 0 else 0.0
        rows.append({"ret": ret, "rng": rng, "br": br, "vr": vr, "pos": ret > 0})
    pos_rets = [r["ret"] for r in rows if r["pos"]]
    return {
        "positive_count": sum(1 for r in rows if r["pos"]),
        "return_sum": round(sum(r["ret"] for r in rows), 4),
        "volume_energy": round(sum(r["vr"] * max(r["ret"], 0) for r in rows), 4),
        "body_energy": round(sum(r["br"] * max(r["ret"], 0) for r in rows), 4),
        "range_energy": round(sum(r["rng"] * max(r["ret"], 0) for r in rows), 4),
    }


def candle_state(klines: list[list], anchor_i: int, btc_slope: float) -> dict | None:
    if anchor_i < 20 or anchor_i >= len(klines):
        return None
    o, h, l, c, vol = ohlcv(klines[anchor_i])
    if o <= 0 or c <= 0:
        return None

    vols = [ohlcv(klines[j])[4] for j in range(max(0, anchor_i - 24), anchor_i)]
    vol_ma = statistics.mean(vols[-20:]) if vols else vol
    closes = [float(klines[i][4]) for i in range(max(0, anchor_i - 65), anchor_i + 1)]
    c_slice = closes[-60:] if len(closes) >= 60 else closes

    rng = (h - l) / o * 100
    body = abs(c - o) / o * 100
    upper = (h - max(o, c)) / o * 100
    lower = (min(o, c) - l) / o * 100
    br = body / rng if rng > 0 else 0.0

    ma5 = statistics.mean(c_slice[-5:]) if len(c_slice) >= 5 else c
    ma10 = statistics.mean(c_slice[-10:]) if len(c_slice) >= 10 else c
    ma20 = statistics.mean(c_slice[-20:]) if len(c_slice) >= 20 else c
    ma60 = statistics.mean(c_slice[-60:]) if len(c_slice) >= 60 else ma20

    slope = ma_slope_pct(closes)
    slope_p = ma_slope_pct(closes[:-1]) if len(closes) > 13 else slope

    bb_mid = statistics.mean(c_slice[-20:]) if len(c_slice) >= 20 else c
    bb_std = statistics.pstdev(c_slice[-20:]) if len(c_slice) >= 20 else 0.0
    bb_width = (2 * bb_std / bb_mid * 100) if bb_mid else 0.0
    bb_prev = 0.0
    if len(c_slice) >= 24:
        m2 = statistics.mean(c_slice[-24:-4])
        s2 = statistics.pstdev(c_slice[-24:-4])
        bb_prev = (2 * s2 / m2 * 100) if m2 else 0.0

    atr_b = t10.average_true_range_percent(klines[max(0, anchor_i - 14):anchor_i], c)
    atr_n = t10.true_range(klines[anchor_i], float(klines[anchor_i - 1][4])) / c * 100 if anchor_i > 0 else 0

    vwap_num = vwap_den = 0.0
    for i in range(max(0, anchor_i - 19), anchor_i + 1):
        ko, kh, kl, kc, kv = ohlcv(klines[i])
        tp = (kh + kl + kc) / 3
        vwap_num += tp * kv
        vwap_den += kv
    vwap = vwap_num / vwap_den if vwap_den else c
    vwap_dist = (c - vwap) / vwap * 100 if vwap else 0.0

    period_hi = max(float(klines[i][2]) for i in range(max(0, anchor_i - 20), anchor_i)) if anchor_i > 0 else h
    period_lo = min(float(klines[i][3]) for i in range(max(0, anchor_i - 20), anchor_i)) if anchor_i > 0 else l
    hi_n = max(c_slice[-20:]) if len(c_slice) >= 20 else c

    comp_len, comp_std = compression_at(klines, anchor_i + 1)
    seq = seq_window(klines, anchor_i, vol_ma)

    return {
        "open": round(o, 8),
        "high": round(h, 8),
        "low": round(l, 8),
        "close": round(c, 8),
        "volume": round(vol, 4),
        "volume_ma20": round(vol_ma, 4),
        "body": round(body, 4),
        "upper_wick": round(upper, 4),
        "lower_wick": round(lower, 4),
        "range_pct": round(rng, 4),
        "atr": round(atr_n / atr_b if atr_b > 0 else 1.0, 4),
        "bb_width": round(bb_width, 4),
        "bb_expansion": round(bb_width / bb_prev if bb_prev > 0 else 1.0, 4),
        "vwap_distance": round(vwap_dist, 4),
        "ma5": round(ma5, 8),
        "ma10": round(ma10, 8),
        "ma20": round(ma20, 8),
        "ma60": round(ma60, 8),
        "ma_slope": round(slope, 4),
        "ma_acceleration": round(slope - slope_p, 4),
        "close_position": round((c - l) / (h - l) if h > l else 0.5, 4),
        "compression_length": float(comp_len),
        "compression_std": comp_std,
        "rs_vs_btc": round(slope - btc_slope, 4),
        "dollar_volume": round(c * vol, 4),
        "high_distance": round((period_hi - c) / c * 100 if c else 0, 4),
        "low_distance": round((c - period_lo) / c * 100 if c else 0, 4),
        "high_close_flag": 1.0 if c >= hi_n * 0.999 else 0.0,
        "body_ratio": round(br, 4),
        **seq,
    }


def build_tf_state(klines: list[list], btc_slope: float) -> dict | None:
    if len(klines) < 25:
        return None
    anchor = len(klines) - 1
    cur = candle_state(klines, anchor, btc_slope)
    prev = candle_state(klines, anchor - 1, btc_slope) if anchor >= 21 else None
    if not cur:
        return None
    return {"current": cur, "previous": prev or {}}


def build_state_vector(symbol: str, end_ms: int, btc_slopes: dict[str, float]) -> dict:
    sv: dict = {}
    for tf, cfg in TIMEFRAMES.items():
        try:
            kl = fetch_klines(symbol, cfg["interval"], end_ms, cfg["lookback"])
            st = build_tf_state(kl, btc_slopes.get(tf, 0.0))
            if st:
                sv[tf] = st
            time.sleep(API_SLEEP)
        except Exception:
            continue
    return sv


def btc_slopes_all(end_ms: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for tf, cfg in TIMEFRAMES.items():
        try:
            kl = fetch_klines("BTCUSDT", cfg["interval"], end_ms, 30)
            if len(kl) >= 12:
                closes = [float(k[4]) for k in kl]
                out[tf] = ma_slope_pct(closes)
        except Exception:
            out[tf] = 0.0
    return out


def load_scan_rows(scan_kst: str) -> list[dict]:
    tag = scan_kst.replace(" ", "_").replace(":", "")
    p14 = PHASE14_CACHE / f"scan_{tag}.json"
    if p14.exists():
        data = json.loads(p14.read_text(encoding="utf-8"))
        rows = data["models"]["A"]
        return [{"symbol": r["symbol"], "rank": r["rank"], "max_up": r["max_up_2h"]} for r in rows]
    p12 = PHASE12_CACHE / f"{tag}.json"
    if p12.exists():
        data = json.loads(p12.read_text(encoding="utf-8"))
        rows = data.get("top30", [])
        return [{"symbol": r["symbol"], "rank": r["rank"], "max_up": r["max_up_12h"]} for r in rows]
    return []


def assign_cluster(f15: dict, max_up: float) -> str:
    if f15.get("volume_ma_ratio", 1) < 1.2 and f15.get("positive_count", 0) >= 3:
        if f15.get("volume_energy", 0) >= 1.5 and f15.get("return_sum", 0) >= 1.0:
            return "ZEST_like"
    if f15.get("volume_ma20", 0) > 0:
        vr = f15.get("volume", 0) / f15["volume_ma20"] if f15["volume_ma20"] else 1
        if vr < 1.0 and max_up >= WIN_THRESHOLD:
            return "Low_Volume_Explosion"
    if f15.get("compression_length", 0) >= 10 and f15.get("range_energy", 0) > 1:
        return "Compression_Release"
    if f15.get("ma_slope", 0) > 1 and f15.get("ma_acceleration", 0) > 0:
        return "Momentum"
    if f15.get("range_pct", 0) >= 3:
        return "Expansion"
    return "Other"


def label_samples(rows: list[dict]) -> list[dict]:
    """Return list of (symbol, rank, max_up, sample_types)."""
    if not rows:
        return []
    by_sym: dict[str, dict] = {}
    for r in rows:
        sym = r["symbol"]
        rank, max_up = r["rank"], r["max_up"]
        types: set[str] = set()
        if rank <= 30:
            types.add("top30")
        if rank <= 5:
            types.add("top5")
        if rank <= 5 and max_up >= WIN_THRESHOLD:
            types.add("winner")
        if rank > 5 and max_up >= WIN_THRESHOLD:
            types.add("missed_winner")
        by_sym[sym] = {"symbol": sym, "rank": rank, "max_up": max_up, "types": types}

    normals = [
        r for r in rows
        if r["rank"] > 5 and r["max_up"] < NORMAL_MAX
    ]
    for r in normals[:3]:
        by_sym.setdefault(r["symbol"], {
            "symbol": r["symbol"], "rank": r["rank"], "max_up": r["max_up"], "types": set(),
        })["types"].add("normal")

    out = []
    for info in by_sym.values():
        if not info["types"]:
            continue
        out.append(info)
    return out


def flatten_for_parquet(rec: dict) -> dict:
    row = {
        "scan_kst": rec["scan_kst"],
        "symbol": rec["symbol"],
        "sample_types": ",".join(sorted(rec["sample_types"])),
        "primary_type": rec["primary_type"],
        "rank": rec["rank"],
        "max_up": rec["max_up"],
        "cluster": rec.get("cluster", ""),
        "state_vector_json": json.dumps(rec["state_vector"], ensure_ascii=False),
    }
    sv = rec["state_vector"]
    for tf in TIMEFRAMES:
        cur = sv.get(tf, {}).get("current", {})
        for f in COMPARE_FIELDS:
            row[f"{tf}_current_{f}"] = cur.get(f)
    return row


def tf_separation(records: list[dict], type_a: str, type_b: str) -> dict[str, float]:
    """Mean abs diff across compare fields between two sample groups."""
    def group_vals(primary: str) -> dict[str, list[float]]:
        g: dict[str, list[float]] = defaultdict(list)
        for rec in records:
            if rec["primary_type"] != primary:
                continue
            for tf in TIMEFRAMES:
                cur = rec["state_vector"].get(tf, {}).get("current", {})
                for f in COMPARE_FIELDS:
                    v = cur.get(f)
                    if isinstance(v, (int, float)) and not math.isnan(v):
                        g[f"{tf}.{f}"].append(float(v))
        return g

    ga, gb = group_vals(type_a), group_vals(type_b)
    per_tf: dict[str, list[float]] = defaultdict(list)
    for key in ga:
        if key not in gb or not ga[key] or not gb[key]:
            continue
        diff = abs(statistics.mean(ga[key]) - statistics.mean(gb[key]))
        tf = key.split(".")[0]
        per_tf[tf].append(diff)
    return {tf: round(statistics.mean(vs), 4) if vs else 0.0 for tf, vs in per_tf.items()}


def correlation_snapshot(records: list[dict], tf: str) -> list[dict]:
    """Raw state correlation among compare fields (winner+missed only)."""
    rows = [r for r in records if r["primary_type"] in ("winner", "missed_winner")]
    if len(rows) < 5:
        return []
    feats = COMPARE_FIELDS
    mat: list[dict] = []
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            xs, ys = [], []
            for r in rows:
                cur = r["state_vector"].get(tf, {}).get("current", {})
                xa, yb = cur.get(a), cur.get(b)
                if isinstance(xa, (int, float)) and isinstance(yb, (int, float)):
                    xs.append(float(xa))
                    ys.append(float(yb))
            if len(xs) < 5:
                continue
            mx, my = statistics.mean(xs), statistics.mean(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
            mat.append({"tf": tf, "a": a, "b": b, "corr": round(num / den if den else 0, 4)})
    mat.sort(key=lambda x: abs(x["corr"]), reverse=True)
    return mat[:15]


def pick_representative(records: list[dict], cluster: str) -> dict | None:
    cands = [r for r in records if r.get("cluster") == cluster and r["primary_type"] == "missed_winner"]
    if not cands:
        cands = [r for r in records if r.get("cluster") == cluster]
    if not cands:
        return None
    return max(cands, key=lambda r: r["max_up"])


def primary_type(types: set[str]) -> str:
    for t in ("winner", "missed_winner", "top5", "normal", "top30"):
        if t in types:
            return t
    return "top30"


def run(max_scans: int | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scans = gen_scan_times()
    if max_scans:
        scans = scans[:max_scans]

    print(f"Phase 15 MTF state: {len(scans)} scans")
    all_records: list[dict] = []
    btc_cache: dict[str, dict[str, float]] = {}
    done_scans: set[str] = set()
    jsonl_path = OUT_DIR / "state_dataset.jsonl"
    if jsonl_path.exists():
        for line in jsonl_path.open(encoding="utf-8"):
            rec = json.loads(line)
            all_records.append(rec)
            done_scans.add(rec["scan_kst"])
        if done_scans:
            print(f"  resume: {len(done_scans)} scans already in dataset")

    for i, (scan_kst, scan_dt) in enumerate(scans, 1):
        if scan_kst in done_scans:
            continue
        print(f"  scan {i}/{len(scans)}: {scan_kst}")
        rows = load_scan_rows(scan_kst)
        if not rows:
            print(f"    skip - no cache for {scan_kst}")
            continue
        samples = label_samples(rows)
        end_ms = int(scan_dt.timestamp() * 1000)
        if scan_kst not in btc_cache:
            btc_cache[scan_kst] = btc_slopes_all(end_ms)
        btc_sl = btc_cache[scan_kst]

        for s in samples:
            sym = s["symbol"]
            sv = build_state_vector(sym, end_ms, btc_sl)
            if not sv:
                continue
            f15 = sv.get("15m", {}).get("current", {})
            cluster = assign_cluster(f15, s["max_up"])
            rec = {
                "scan_kst": scan_kst,
                "symbol": sym,
                "rank": s["rank"],
                "max_up": s["max_up"],
                "sample_types": sorted(s["types"]),
                "primary_type": primary_type(s["types"]),
                "cluster": cluster,
                "state_vector": sv,
            }
            all_records.append(rec)

        # checkpoint per scan
        with jsonl_path.open("w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    jsonl_path = OUT_DIR / "state_dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    flat_rows = [flatten_for_parquet(r) for r in all_records]
    parquet_path = OUT_DIR / "state_dataset.parquet"
    pd.DataFrame(flat_rows).to_parquet(parquet_path, index=False)

    sep_wm = tf_separation(all_records, "winner", "missed_winner")
    sep_wn = tf_separation(all_records, "winner", "normal")
    sep_mm = tf_separation(all_records, "missed_winner", "normal")

    best_wm = max(sep_wm, key=sep_wm.get) if sep_wm else "?"
    best_wn = max(sep_wn, key=sep_wn.get) if sep_wn else "?"
    corr_15 = correlation_snapshot(all_records, "15m")
    corr_1h = correlation_snapshot(all_records, "1h")

    clusters = ("ZEST_like", "Compression_Release", "Momentum", "Low_Volume_Explosion", "Expansion")
    examples: dict[str, dict] = {}
    for cl in clusters:
        ex = pick_representative(all_records, cl)
        if ex:
            examples[cl] = {
                "scan_kst": ex["scan_kst"],
                "symbol": ex["symbol"],
                "max_up": ex["max_up"],
                "state_vector": ex["state_vector"],
            }

    n_winner = sum(1 for r in all_records if r["primary_type"] == "winner")
    n_missed = sum(1 for r in all_records if r["primary_type"] == "missed_winner")
    n_normal = sum(1 for r in all_records if r["primary_type"] == "normal")

    lines = [
        "##############################################################",
        "SCOUT PHASE 15 — MULTI-TIMEFRAME STATE DATASET",
        "##############################################################",
        "",
        f"Period: {START_KST.date()} ~ {END_KST.date()} | Scans: {len(scans)}",
        f"Records: {len(all_records)} | winner={n_winner} missed={n_missed} normal={n_normal}",
        "",
        "=" * 58,
        "1. TIMEFRAME SEPARATION (mean abs diff)",
        "=" * 58,
        "  Winner vs Missed Winner:",
    ]
    for tf in TIMEFRAMES:
        lines.append(f"    {tf:4s}: {sep_wm.get(tf, 0):.4f}")
    lines.append(f"  -> Largest gap: {best_wm} ({sep_wm.get(best_wm, 0):.4f})")
    lines.append("")
    lines.append("  Winner vs Normal:")
    for tf in TIMEFRAMES:
        lines.append(f"    {tf:4s}: {sep_wn.get(tf, 0):.4f}")
    lines.append(f"  -> Largest gap: {best_wn} ({sep_wn.get(best_wn, 0):.4f})")
    lines.append("")
    lines.append("  Missed Winner vs Normal:")
    for tf in TIMEFRAMES:
        lines.append(f"    {tf:4s}: {sep_mm.get(tf, 0):.4f}")

    lines.extend(["", "=" * 58, "2. KEY FINDINGS", "=" * 58])
    lines.append(f"  Winner-Missed best TF: {best_wm}")
    lines.append(f"  Winner-Normal best TF: {best_wn}")
    if sep_wm.get("5m", 0) < sep_wm.get(best_wm, 1) * 0.5:
        lines.append("  5m: small Winner-Missed gap (noise at micro scale)")
    if sep_wm.get("1h", 0) > sep_wm.get("15m", 0):
        lines.append("  1h: larger structural gap than 15m")

    lines.extend(["", "=" * 58, "3. RAW STATE CORRELATION (top pairs)", "=" * 58])
    lines.append("  15m:")
    for c in corr_15[:5]:
        lines.append(f"    {c['a']} x {c['b']}: {c['corr']:+.3f}")
    lines.append("  1h:")
    for c in corr_1h[:5]:
        lines.append(f"    {c['a']} x {c['b']}: {c['corr']:+.3f}")

    lines.extend(["", "=" * 58, "4. REPRESENTATIVE CASES (MTF state)", "=" * 58])
    for cl, ex in examples.items():
        lines.append(f"  [{cl}] {ex['scan_kst']} {ex['symbol']} max_up={ex['max_up']}%")
        for tf in ("5m", "15m", "1h", "4h", "1d"):
            cur = ex["state_vector"].get(tf, {}).get("current", {})
            if cur:
                lines.append(
                    f"    {tf} cur: volE={cur.get('volume_energy')} retSum={cur.get('return_sum')} "
                    f"comp={cur.get('compression_length')} ma_slope={cur.get('ma_slope')} "
                    f"range={cur.get('range_pct')}"
                )

    lines.extend(["", "=" * 58, "5. OUTPUT FILES", "=" * 58,
        f"  {jsonl_path}",
        f"  {parquet_path}",
        f"  {OUT_DIR / 'phase15_report.txt'}",
    ])

    report = OUT_DIR / "phase15_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "tf_separation.csv").write_text(
        "timeframe,winner_vs_missed,winner_vs_normal,missed_vs_normal\n"
        + "\n".join(
            f"{tf},{sep_wm.get(tf,0)},{sep_wn.get(tf,0)},{sep_mm.get(tf,0)}"
            for tf in TIMEFRAMES
        ),
        encoding="utf-8",
    )

    print("\n".join(lines[-25:]).encode("ascii", "replace").decode("ascii"))
    print(f"\nSaved {len(all_records)} records -> {jsonl_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scans", type=int, default=None)
    args = parser.parse_args()
    run(args.max_scans)


if __name__ == "__main__":
    main()
