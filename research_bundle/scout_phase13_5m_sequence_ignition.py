"""
Scout Phase 13 — 5M Sequence Ignition Feature Design

Validates sequence features on 5m candles. NO filter/ranking changes. NO trading.

Usage:
  python scout_phase13_5m_sequence_ignition.py
"""

from __future__ import annotations

import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10
from season2_p37_scout_decision_hierarchy import pf, write_csv
from season2_p40_scout_transition_triggers import ema
from season2_universe_blind_test import ohlcv

OUT_DIR = Path("logs") / "phase13_sequence"
CACHE_DIR = OUT_DIR / "kline_cache"
PHASE12 = Path("logs") / "phase12_dna"
KST = timezone(timedelta(hours=9))
INTERVAL = "5m"
INTERVAL_MS = 5 * 60 * 1000
LOOKBACK = 120
API_SLEEP = 0.12
MAX_RETRIES = 4

WINDOWS = (3, 6, 9, 12)
TIME_OFFSETS = ((0, "0m"), (3, "m15"), (6, "m30"), (9, "m45"), (12, "m60"))

SEQ_BASE = (
    "seq_return_sum", "seq_positive_count", "seq_volume_persistence",
    "seq_volume_energy", "seq_body_energy", "seq_range_energy",
    "seq_slope_delta", "seq_slope_accel", "seq_close_strength",
    "seq_compression_release", "first_abnormal_candle",
)

CANDIDATE_RULES = (
    {
        "id": "C1_ignition_6",
        "desc": "seq_return_sum_6>=1.5, pos>=4, vol_persist>=0.5, slope_delta_6>0",
        "check": lambda f: (
            f.get("seq_return_sum_6", 0) >= 1.5
            and f.get("seq_positive_count_6", 0) >= 4
            and f.get("seq_volume_persistence_6", 0) >= 0.5
            and f.get("seq_slope_delta_6", 0) > 0
        ),
    },
    {
        "id": "C2_compression_release",
        "desc": "compression>=12, seq_compression_release_6=1, close_pos>=0.6",
        "check": lambda f: (
            f.get("compression_length", 0) >= 12
            and f.get("seq_compression_release_6", 0) >= 1
            and f.get("close_position", 0) >= 0.6
        ),
    },
    {
        "id": "C3_volume_energy",
        "desc": "seq_volume_energy_6>=2.0, seq_return_sum_6>=1.0",
        "check": lambda f: (
            f.get("seq_volume_energy_6", 0) >= 2.0
            and f.get("seq_return_sum_6", 0) >= 1.0
        ),
    },
    {
        "id": "C4_low_vol_seq",
        "desc": "volume_ma_ratio<1.0 AND seq_volume_energy_6>=1.5",
        "check": lambda f: (
            f.get("volume_ma_ratio", 0) < 1.0
            and f.get("seq_volume_energy_6", 0) >= 1.5
        ),
    },
    {
        "id": "C5_zest_like",
        "desc": "seq_return_sum_6>=1.2, seq_positive_count_6>=3, first_abnormal_6=1",
        "check": lambda f: (
            f.get("seq_return_sum_6", 0) >= 1.2
            and f.get("seq_positive_count_6", 0) >= 3
            and f.get("first_abnormal_candle_6", 0) >= 1
        ),
    },
    {
        "id": "C6_slope_accel",
        "desc": "seq_slope_accel_6>0.5, seq_close_strength_6>=0.5",
        "check": lambda f: (
            f.get("seq_slope_accel_6", 0) > 0.5
            and f.get("seq_close_strength_6", 0) >= 0.5
        ),
    },
    {
        "id": "C7_body_energy",
        "desc": "seq_body_energy_6>=0.8, seq_range_energy_6>=2.0",
        "check": lambda f: (
            f.get("seq_body_energy_6", 0) >= 0.8
            and f.get("seq_range_energy_6", 0) >= 2.0
        ),
    },
)


def parse_kst(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


def fetch_klines_5m(symbol: str, end_ms: int, limit: int) -> list[list]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{symbol}_{end_ms}_{limit}.json"
    cache_path = CACHE_DIR / tag
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    params = urllib.parse.urlencode({
        "symbol": symbol, "interval": INTERVAL, "endTime": end_ms,
        "limit": min(limit, t10.MAX_LIMIT),
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
                data = json.loads(resp.read().decode())
            cache_path.write_text(json.dumps(data), encoding="utf-8")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (418, 429, 503) and attempt < MAX_RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    return []


def fetch_forward_5m(symbol: str, start_ms: int, count: int) -> list[list]:
    tag = f"{symbol}_fwd_{start_ms}_{count}.json"
    cache_path = CACHE_DIR / tag
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    params = urllib.parse.urlencode({
        "symbol": symbol, "interval": INTERVAL, "startTime": start_ms,
        "limit": min(count, t10.MAX_LIMIT),
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
                data = json.loads(resp.read().decode())
            cache_path.write_text(json.dumps(data), encoding="utf-8")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (418, 429, 503) and attempt < MAX_RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    return []


def ma_slope(closes: list[float]) -> float:
    if len(closes) < 12:
        return 0.0
    recent = statistics.mean(closes[-6:-1])
    prior = statistics.mean(closes[-12:-7]) if len(closes) >= 12 else recent
    return (recent - prior) / prior * 100 if prior else 0.0


def compression_length(klines: list[list], end_i: int, max_rng: float = 1.5) -> int:
    n = 0
    for i in range(end_i - 1, max(0, end_i - 48), -1):
        o, h, l, _, _ = ohlcv(klines[i])
        if o <= 0 or (h - l) / o * 100 > max_rng:
            break
        n += 1
    return n


def candle_row(k: list, vol_ma: float) -> dict:
    o, h, l, c, vol = ohlcv(k)
    ret = (c - o) / o * 100 if o else 0.0
    rng = (h - l) / o * 100 if o else 0.0
    body = abs(c - o) / o * 100 if o else 0.0
    br = body / rng if rng > 0 else 0.0
    vr = vol / vol_ma if vol_ma > 0 else 0.0
    cp = (c - l) / (h - l) if h > l else 0.5
    return {
        "return_pct": ret,
        "range_pct": rng,
        "body_ratio": br,
        "volume_ma_ratio": vr,
        "close_position": cp,
        "positive": ret > 0,
        "pos_ret": max(ret, 0),
    }


def window_seq(klines: list[list], end_i: int, n: int) -> dict:
    if end_i < n or end_i >= len(klines):
        return {}
    vols = [ohlcv(klines[j])[4] for j in range(max(0, end_i - 24), end_i)]
    vol_ma = statistics.mean(vols[-20:]) if vols else 1.0

    rows = []
    for i in range(end_i - n + 1, end_i + 1):
        rows.append(candle_row(klines[i], vol_ma))

    pos_rets = [r["pos_ret"] for r in rows]
    return_sum = sum(r["return_pct"] for r in rows)
    pos_count = sum(1 for r in rows if r["positive"])
    vol_persist = sum(1 for r in rows if r["volume_ma_ratio"] >= 1.0) / n
    vol_energy = sum(r["volume_ma_ratio"] * r["pos_ret"] for r in rows)
    body_energy = sum(r["body_ratio"] * r["pos_ret"] for r in rows)
    range_energy = sum(r["range_pct"] * r["pos_ret"] for r in rows)
    close_strength = sum(1 for r in rows if r["close_position"] >= 0.6) / n

    closes = [float(klines[i][4]) for i in range(max(0, end_i - 24), end_i + 1)]
    slope_now = ma_slope(closes)
    slope_prior = ma_slope(closes[:-n]) if len(closes) > n + 6 else slope_now
    slope_delta = slope_now - slope_prior
    slope_accel = slope_delta / max(abs(slope_prior), 0.01)

    comp_len = compression_length(klines, end_i - n + 1)
    pre_start = max(0, end_i - n - 12)
    pre = [candle_row(klines[i], vol_ma) for i in range(pre_start, end_i - n + 1)]
    comp_release = 0
    if pre and rows:
        pre_rng = statistics.mean([x["range_pct"] for x in pre[-12:]])
        pre_vol = statistics.mean([x["volume_ma_ratio"] for x in pre[-12:]])
        rec_rng = statistics.mean([x["range_pct"] for x in rows])
        rec_vol = statistics.mean([x["volume_ma_ratio"] for x in rows])
        if comp_len >= 4 and rec_rng > pre_rng * 1.2 and rec_vol > pre_vol * 1.1:
            comp_release = 1

    first_abnormal = 0
    for i, r in enumerate(rows):
        if i == 0:
            continue
        prev = rows[i - 1]
        if (
            r["range_pct"] > prev["range_pct"]
            and r["body_ratio"] > prev["body_ratio"]
            and r["volume_ma_ratio"] > prev["volume_ma_ratio"]
        ):
            first_abnormal = 1
            break

    return {
        f"seq_return_sum_{n}": round(return_sum, 4),
        f"seq_positive_count_{n}": pos_count,
        f"seq_volume_persistence_{n}": round(vol_persist, 4),
        f"seq_volume_energy_{n}": round(vol_energy, 4),
        f"seq_body_energy_{n}": round(body_energy, 4),
        f"seq_range_energy_{n}": round(range_energy, 4),
        f"seq_slope_delta_{n}": round(slope_delta, 4),
        f"seq_slope_accel_{n}": round(slope_accel, 4),
        f"seq_close_strength_{n}": round(close_strength, 4),
        f"seq_compression_release_{n}": float(comp_release),
        f"first_abnormal_candle_{n}": float(first_abnormal),
    }


def compute_at_anchor(klines: list[list], anchor_i: int) -> dict:
    feats: dict = {}
    o, h, l, c, vol = ohlcv(klines[anchor_i])
    vols = [ohlcv(klines[j])[4] for j in range(max(0, anchor_i - 24), anchor_i)]
    vol_ma = statistics.mean(vols[-20:]) if vols else 1.0
    feats["volume_ma_ratio"] = round(vol / vol_ma if vol_ma else 0, 4)
    feats["close_position"] = round((c - l) / (h - l) if h > l else 0.5, 4)
    feats["compression_length"] = float(compression_length(klines, anchor_i))
    for n in WINDOWS:
        feats.update(window_seq(klines, anchor_i, n))
    return feats


def forward_from_entry(entry: float, fwd: list[list]) -> dict:
    horizons = {"15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48, "12h": 144}
    out: dict = {}
    if entry <= 0 or not fwd:
        return out
    for label, n in horizons.items():
        chunk = fwd[:n]
        if not chunk:
            continue
        max_h = max(ohlcv(k)[1] for k in chunk)
        min_l = min(ohlcv(k)[2] for k in chunk)
        close = float(chunk[-1][4])
        out[f"ret_{label}"] = round((close - entry) / entry * 100, 4)
        out[f"max_up_{label}"] = round((max_h - entry) / entry * 100, 4)
        out[f"mdd_{label}"] = round((entry - min_l) / entry * 100, 4)
    return out


def analyze_record(symbol: str, scan_kst: str, label: str, cluster: str = "") -> dict | None:
    try:
        scan_dt = parse_kst(scan_kst)
        end_ms = int(scan_dt.timestamp() * 1000)
        klines = fetch_klines_5m(symbol, end_ms, LOOKBACK)
        if len(klines) < 40:
            return None
        anchor = len(klines) - 1

        timeline: dict = {}
        for off_bars, off_label in TIME_OFFSETS:
            ai = anchor - off_bars
            if ai < 12:
                continue
            timeline[off_label] = compute_at_anchor(klines, ai)

        scan_feats = timeline.get("0m", {})
        fwd = fetch_forward_5m(symbol, end_ms + INTERVAL_MS, 150)
        entry_close = float(klines[anchor][4])
        entry_next = float(fwd[0][1]) if fwd else entry_close
        forward_close = forward_from_entry(entry_close, fwd)
        forward_open = forward_from_entry(entry_next, fwd[1:] if len(fwd) > 1 else [])

        single_weak = scan_feats.get("volume_ma_ratio", 0) < 1.0
        seq_strong = (
            scan_feats.get("seq_volume_energy_6", 0) >= 1.5
            or scan_feats.get("seq_return_sum_6", 0) >= 1.5
        )

        zest_like = (
            scan_feats.get("volume_ma_ratio", 0) < 1.2
            and scan_feats.get("seq_return_sum_6", 0) >= 1.0
            and scan_feats.get("seq_positive_count_6", 0) >= 3
            and scan_feats.get("first_abnormal_candle_6", 0) >= 1
        )

        return {
            "symbol": symbol,
            "scan_kst": scan_kst,
            "label": label,
            "cluster": cluster,
            "timeline": timeline,
            "scan_features": scan_feats,
            "single_vol_weak": single_weak,
            "seq_strong": seq_strong,
            "zest_like": zest_like,
            "forward_close_entry": forward_close,
            "forward_next_open": forward_open,
        }
    except Exception:
        return None
    finally:
        time.sleep(API_SLEEP)


def load_missed() -> list[dict]:
    path = PHASE12 / "missed_winners_dna.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_top5_hits() -> list[dict]:
    rows = []
    for p in sorted((PHASE12 / "scan_cache").glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        for r in data.get("top5", []):
            if r.get("max_up_12h", 0) >= 5.0:
                rows.append({
                    "symbol": r["symbol"],
                    "scan_kst": data["scan_kst"],
                    "max_up_12h": r["max_up_12h"],
                    "volume_ma_ratio": r.get("features", {}).get("volume_ma_ratio", 0),
                })
    return rows


def avg_feats(records: list[dict], key: str = "scan_features") -> dict:
    if not records:
        return {}
    keys = set()
    for r in records:
        keys.update(r.get(key, {}).keys())
    out = {}
    for k in sorted(keys):
        vals = [r.get(key, {}).get(k, 0) for r in records if isinstance(r.get(key, {}).get(k), (int, float))]
        if vals:
            out[k] = round(statistics.mean(vals), 4)
    return out


def eval_rule(rule: dict, records: list[dict], label: str) -> dict:
    flagged = [r for r in records if rule["check"](r.get("scan_features", {}))]
    n = len(flagged)
    max_ups = []
    mdds = []
    for r in flagged:
        s = r.get("outcome", {})
        if s.get("max_up_12h"):
            max_ups.append(s["max_up_12h"])
    hits = sum(1 for x in max_ups if x >= 5.0)
    return {
        "rule_id": rule["id"],
        "desc": rule["desc"],
        "cohort": label,
        "sample_n": n,
        "hit_rate_pct": round(hits / max(n, 1) * 100, 1),
        "avg_max_up": round(statistics.mean(max_ups), 2) if max_ups else 0,
        "median_max_up": round(statistics.median(max_ups), 2) if max_ups else 0,
    }


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missed_raw = load_missed()
    top5_raw = load_top5_hits()

    print(f"Phase 13: loading {len(missed_raw)} missed, {len(top5_raw)} top5 hits")

    missed_analyzed: list[dict] = []
    for i, m in enumerate(missed_raw, 1):
        print(f"  missed {i}/{len(missed_raw)}: {m['symbol']} @ {m['scan_kst']}")
        r = analyze_record(m["symbol"], m["scan_kst"], "missed", m.get("cluster", ""))
        if r:
            r["outcome"] = {"max_up_12h": m.get("max_up_12h", 0), "return_12h": m.get("return_12h", 0)}
            missed_analyzed.append(r)

    top5_analyzed: list[dict] = []
    for i, t in enumerate(top5_raw, 1):
        print(f"  top5 {i}/{len(top5_raw)}: {t['symbol']} @ {t['scan_kst']}")
        r = analyze_record(t["symbol"], t["scan_kst"], "top5_hit")
        if r:
            r["outcome"] = {"max_up_12h": t.get("max_up_12h", 0)}
            top5_analyzed.append(r)

    print(f"Phase 13: analyzed missed={len(missed_analyzed)}/{len(missed_raw)}, top5={len(top5_analyzed)}/{len(top5_raw)}")

    low_vol = [r for r in missed_analyzed if r.get("cluster") == "Low_Volume_Explosion"]
    zest_like = [r for r in missed_analyzed if r.get("zest_like")]
    weak_but_seq = [r for r in missed_analyzed if r.get("single_vol_weak") and r.get("seq_strong")]

    missed_avg = avg_feats(missed_analyzed)
    low_vol_avg = avg_feats(low_vol)
    top5_avg = avg_feats(top5_analyzed)

    rule_results = []
    for rule in CANDIDATE_RULES:
        rule_results.append(eval_rule(rule, missed_analyzed, "missed"))
        rule_results.append(eval_rule(rule, top5_analyzed, "top5_hit"))
    rule_results.sort(key=lambda x: (x["cohort"] == "missed", x["hit_rate_pct"]), reverse=True)

    # Entry timing stats for missed (close vs next open at scan)
    entry_stats = []
    for r in missed_analyzed:
        fc = r.get("forward_close_entry", {})
        fo = r.get("forward_next_open", {})
        if fc.get("max_up_12h"):
            entry_stats.append({
                "symbol": r["symbol"],
                "close_12h_max": fc.get("max_up_12h"),
                "open_12h_max": fo.get("max_up_12h"),
            })

    lines = [
        "############################################################",
        "SCOUT PHASE 13 — 5M SEQUENCE IGNITION FEATURE DESIGN",
        "############################################################",
        "",
        "NO filter/ranking/trading changes applied.",
        "",
        "=" * 58,
        "1. SEQUENCE FEATURE LIST",
        "=" * 58,
    ]
    for base in SEQ_BASE:
        lines.append(f"  {base}_{{3,6,9,12}}  (windows: 15/30/45/60 min)")
    lines.append("  + volume_ma_ratio, close_position, compression_length at anchor")
    lines.append("  Timeline offsets: 0m, -15m, -30m, -45m, -60m")

    lines.extend(["", "=" * 58, "2. MISSED WINNER SEQUENCE AVERAGES (scan/0m)", "=" * 58])
    for k in sorted(missed_avg.keys())[:20]:
        lines.append(f"  {k}: {missed_avg[k]}")

    lines.extend(["", "=" * 58, "3. LOW VOLUME EXPLOSION CLUSTER AVERAGES", "=" * 58])
    lines.append(f"  n={len(low_vol)}")
    for k in ("volume_ma_ratio", "seq_return_sum_6", "seq_volume_energy_6", "seq_positive_count_6",
              "seq_volume_persistence_6", "seq_slope_delta_6", "compression_length"):
        lines.append(f"  {k}: {low_vol_avg.get(k, 'n/a')}")

    lines.extend(["", "=" * 58, "4. TOP5 HIT vs MISSED (scan/0m)", "=" * 58])
    for k in ("volume_ma_ratio", "seq_return_sum_6", "seq_volume_energy_6", "seq_body_energy_6",
              "seq_compression_release_6", "first_abnormal_candle_6"):
        lines.append(
            f"  {k}: missed={missed_avg.get(k)} top5={top5_avg.get(k)}"
        )

    lines.extend(["", "=" * 58, "5. SINGLE vol_ma_ratio vs seq_volume_energy_6", "=" * 58])
    m_weak = sum(1 for r in missed_analyzed if r.get("single_vol_weak"))
    m_seq = sum(1 for r in missed_analyzed if r.get("seq_strong"))
    m_both = len(weak_but_seq)
    lines.append(f"  Missed with single vol<1.0: {m_weak}/{len(missed_analyzed)} ({round(m_weak/max(len(missed_analyzed),1)*100,1)}%)")
    lines.append(f"  Missed with seq strong at scan: {m_seq}/{len(missed_analyzed)}")
    lines.append(f"  Weak single BUT seq strong: {m_both}/{len(missed_analyzed)} ({round(m_both/max(len(missed_analyzed),1)*100,1)}%)")

    lines.extend(["", "=" * 58, "6. ZEST-LIKE PATTERN COUNT", "=" * 58])
    lines.append(f"  ZEST-like samples: {len(zest_like)} / {len(missed_analyzed)}")
    for z in zest_like[:8]:
        f = z["scan_features"]
        lines.append(
            f"    {z['symbol']} @{z['scan_kst']} vol={f.get('volume_ma_ratio')} "
            f"seq_ret6={f.get('seq_return_sum_6')} seq_volE6={f.get('seq_volume_energy_6')}"
        )

    lines.extend(["", "=" * 58, "7. TOP5 CANDIDATE SEQUENCE CONDITIONS", "=" * 58])
    seen = set()
    for rr in rule_results:
        if rr["cohort"] != "missed" or rr["rule_id"] in seen:
            continue
        seen.add(rr["rule_id"])
        top5_rr = next((x for x in rule_results if x["rule_id"] == rr["rule_id"] and x["cohort"] == "top5_hit"), {})
        lines.append(
            f"  {rr['rule_id']}: missed n={rr['sample_n']} hit={rr['hit_rate_pct']}% "
            f"avg_max={rr['avg_max_up']}% | top5 n={top5_rr.get('sample_n',0)} "
            f"| {rr['desc']}"
        )

    lines.extend(["", "=" * 58, "8. FIRST MEANINGFUL CANDLE / ENTRY TIMING", "=" * 58])
    if entry_stats:
        close_avg = statistics.mean([e["close_12h_max"] for e in entry_stats if e.get("close_12h_max")])
        open_avg = statistics.mean([e["open_12h_max"] for e in entry_stats if e.get("open_12h_max")])
        lines.append(f"  Avg 12h max_up if enter at scan close: {close_avg:.2f}%")
        lines.append(f"  Avg 12h max_up if enter at next 5m open: {open_avg:.2f}%")

    lines.extend([
        "",
        "=" * 58,
        "9. FINAL CONCLUSION",
        "=" * 58,
        "  Search formula modification: NO (not yet).",
        f"  Sequence validity: {m_both}/{len(missed_analyzed)} missed show weak single-vol but strong 30m sequence.",
        "  Recommendation: Next Phase test seq_volume_energy_6 + seq_return_sum_6 as",
        "  RANKING auxiliary score only (not filter). Do not KEEP any single condition yet.",
        "  Best experimental candidate: C4_low_vol_seq or C3_volume_energy (pending blind replay).",
    ])

    report = OUT_DIR / "phase13_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")

    with (OUT_DIR / "sequence_analysis.jsonl").open("w", encoding="utf-8") as f:
        for r in missed_analyzed + top5_analyzed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_csv(OUT_DIR / "candidate_rules.csv", rule_results)
    write_csv(OUT_DIR / "missed_sequence_avg.csv", [{"feature": k, "mean": v} for k, v in missed_avg.items()])

    print("\n".join(lines[-15:]).encode("ascii", "replace").decode("ascii"))
    print(f"\nSaved: {report}")


if __name__ == "__main__":
    run()
