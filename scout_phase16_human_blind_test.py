"""
Scout Phase 16 — Human vs Scout Blind Test

Blind pick at 2026-06-20 11:00 KST. Forward eval to 15:00 only in AFTER report.

Usage:
  python scout_phase16_human_blind_test.py
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10
from season2_p37_scout_decision_hierarchy import pf, write_csv
from season2_p40_scout_transition_triggers import ema
from season2_universe_blind_test import load_eligible_symbols, ohlcv

from scout_phase13_5m_sequence_ignition import window_seq

OUT_DIR = Path("logs") / "phase16_blind"
CACHE_DIR = OUT_DIR / "kline_cache"
KST = timezone(timedelta(hours=9))
SCAN_KST = "2026-06-20 11:00:00"
FORWARD_END_KST = "2026-06-20 15:00:00"
API_SLEEP = 0.05
WORKERS = 12
MISS_THRESHOLD = 7.0

PATTERN_B = (("macd_signal", "gte", -0.0016), ("range_pct", "gte", 1.4768))
RANK_WEIGHTS = {
    "young_birth": -1.4343174277392308,
    "birth_age_min": 1.5586524217896396,
    "ignition_age_min": -0.22688287801448231,
    "ma_slope_accel": -0.12327724295144488,
    "volume_ma_ratio": -0.06664243036600889,
}

TF_INTERVALS = (
    ("15m", "15m", 96),
    ("30m", "30m", 96),
    ("1h", "1h", 96),
    ("2h", "2h", 96),
    ("4h", "4h", 96),
)


@dataclass
class Candidate:
    symbol: str
    price: float
    ranking_score: float
    explosion_score: float
    rank_h4: int = 0
    rank_scout: int = 0
    seq_5m: dict = field(default_factory=dict)
    tf_states: dict = field(default_factory=dict)
    compression: str = ""
    volume_state: str = ""
    momentum_state: str = ""
    reasons: list[str] = field(default_factory=list)
    forward: dict = field(default_factory=dict)


def parse_kst(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


def fetch_klines(symbol: str, interval: str, end_ms: int, limit: int) -> list[list]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{interval}_{symbol}_{end_ms}.json"
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
            if e.code in (418, 429) and attempt < 3:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    return []


def fetch_forward_5m(symbol: str, start_ms: int, count: int) -> list[list]:
    tag = f"fwd5m_{symbol}_{start_ms}.json"
    p = CACHE_DIR / tag
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    params = urllib.parse.urlencode({
        "symbol": symbol, "interval": "5m", "startTime": start_ms, "limit": count,
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
                data = json.loads(resp.read().decode())
            p.write_text(json.dumps(data), encoding="utf-8")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (418, 429) and attempt < 4:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    return []


def macd_sig(closes: list[float]) -> float:
    if len(closes) < 26:
        return 0.0
    hist = [ema(closes[:i], 12) - ema(closes[:i], 26) for i in range(26, len(closes) + 1)]
    return ema(hist, 9) if hist else 0.0


def ma_slope_pct(klines: list[list]) -> float:
    if len(klines) < 14:
        return 0.0
    closes = [float(k[4]) for k in klines]
    recent = statistics.mean(closes[-6:-1])
    prior = statistics.mean(closes[-12:-7]) if len(closes) >= 12 else recent
    return (recent - prior) / prior * 100 if prior else 0.0


def lifecycle_15m(klines: list[list]) -> dict:
    anchor = len(klines) - 1
    window = klines[max(0, anchor - 48): anchor + 1]
    base = min(ohlcv(k)[2] for k in window) if window else ohlcv(klines[-1])[2]
    birth_i = ign_i = anchor
    for i in range(anchor, max(0, anchor - 32), -1):
        if ohlcv(klines[i])[4] >= base * 1.03:
            birth_i = i
            break
    for i in range(anchor, max(0, anchor - 24), -1):
        vols = [ohlcv(klines[j])[4] for j in range(max(0, i - 8), i)]
        if vols and statistics.mean(vols) > 0 and ohlcv(klines[i])[4] / statistics.mean(vols) >= 1.2:
            ign_i = i
            break
    slope = ma_slope_pct(klines)
    slope_p = ma_slope_pct(klines[:-4]) if len(klines) > 20 else slope
    return {
        "birth_age_min": float((anchor - birth_i) * 15),
        "ignition_age_min": float((anchor - ign_i) * 15),
        "young_birth": 1.0 if (anchor - birth_i) * 15 <= 45 else 0.0,
        "ma_slope_accel": slope - slope_p,
        "ma_slope": slope,
    }


def h4_score(lc: dict, vol_ratio: float) -> float:
    ff = {
        "young_birth": lc.get("young_birth", 0),
        "birth_age_min": lc.get("birth_age_min", 0),
        "ignition_age_min": lc.get("ignition_age_min", 0),
        "ma_slope_accel": lc.get("ma_slope_accel", 0),
        "volume_ma_ratio": vol_ratio,
    }
    return sum(RANK_WEIGHTS[k] * ff.get(k, 0) for k in RANK_WEIGHTS)


def tf_candle_pair(klines: list[list]) -> dict | None:
    if len(klines) < 25:
        return None
    anchor = len(klines) - 1

    def one(i: int) -> dict:
        o, h, l, c, vol = ohlcv(klines[i])
        vols = [ohlcv(klines[j])[4] for j in range(max(0, i - 24), i)]
        vol_ma = statistics.mean(vols[-20:]) if vols else vol
        rng = (h - l) / o * 100 if o else 0
        body = abs(c - o) / o * 100 if o else 0
        return {
            "body_pct": round(body, 4),
            "range_pct": round(rng, 4),
            "volume_ma_ratio": round(vol / vol_ma if vol_ma else 0, 4),
            "close_position": round((c - l) / (h - l) if h > l else 0.5, 4),
        }

    return {"current": one(anchor), "previous": one(anchor - 1)}


def classify_states(seq: dict, tf: dict, lc: dict) -> tuple[str, str, str]:
    vol_15 = tf.get("15m", {}).get("current", {}).get("volume_ma_ratio", 1)
    vol_e = seq.get("seq_volume_energy_6", 0)
    comp_flags = []
    for label in ("2h", "4h", "1h"):
        cur = tf.get(label, {}).get("current", {})
        prev = tf.get(label, {}).get("previous", {})
        if cur.get("range_pct", 99) < 2 and prev.get("range_pct", 99) < 2:
            comp_flags.append(label)
    compression = "Compression_Release" if comp_flags and vol_e >= 1.0 else (
        f"Compression_{'+'.join(comp_flags)}" if comp_flags else "None"
    )
    if vol_15 < 1.0 and vol_e >= 1.5:
        volume = "Low_Volume_Energy_Build"
    elif vol_15 >= 1.5:
        volume = "Volume_Surge"
    elif vol_15 < 1.0:
        volume = "Low_Volume"
    else:
        volume = "Normal"
    slope = lc.get("ma_slope", 0)
    accel = lc.get("ma_slope_accel", 0)
    pos = seq.get("seq_positive_count_6", 0)
    if slope > 0 and accel > 0 and pos >= 3:
        momentum = "Accelerating_Up"
    elif pos >= 4:
        momentum = "Sequence_Positive"
    elif slope > 0:
        momentum = "Mild_Up"
    else:
        momentum = "Weak"
    return compression, volume, momentum


def explosion_score(seq: dict, tf: dict, lc: dict) -> float:
    """Read-only scout pick score (Phase 16 blind only)."""
    s = 0.0
    s += seq.get("seq_volume_energy_6", 0) * 0.45
    s += seq.get("seq_return_sum_6", 0) * 0.20
    s += seq.get("first_abnormal_candle_6", 0) * 2.0
    s += seq.get("seq_positive_count_6", 0) * 0.35
    vol_15 = tf.get("15m", {}).get("current", {}).get("volume_ma_ratio", 1)
    if vol_15 < 1.2 and seq.get("seq_volume_energy_6", 0) >= 1.5:
        s += 2.5
    for label in ("1h", "2h"):
        cur = tf.get(label, {}).get("current", {})
        prev = tf.get(label, {}).get("previous", {})
        if cur.get("body_pct", 0) > prev.get("body_pct", 0) * 1.2:
            s += 1.0
        if cur.get("close_position", 0) >= 0.6:
            s += 0.5
    if lc.get("young_birth", 0) >= 1:
        s += 0.8
    return round(s, 4)


def build_reasons(seq: dict, tf: dict, compression: str, volume: str, momentum: str) -> list[str]:
    lines: list[str] = []
    if seq.get("seq_volume_energy_6", 0) >= 1.5:
        lines.append(f"5m seq volume energy elevated ({seq['seq_volume_energy_6']:.2f})")
    if seq.get("seq_positive_count_6", 0) >= 3:
        lines.append(f"5m sequence positive turn ({int(seq['seq_positive_count_6'])}/6 candles)")
    if seq.get("first_abnormal_candle_6", 0) >= 1:
        lines.append("First abnormal 5m candle detected (range/body/vol expansion)")
    h1 = tf.get("1h", {}).get("current", {})
    h1p = tf.get("1h", {}).get("previous", {})
    if h1.get("body_pct", 0) > h1p.get("body_pct", 0):
        lines.append(f"1h body expanding ({h1p.get('body_pct',0):.2f}% -> {h1.get('body_pct',0):.2f}%)")
    if "Compression" in compression and compression != "None":
        lines.append(f"Higher-TF compression context: {compression}")
    if volume == "Low_Volume_Energy_Build":
        lines.append("Low volume at scan with accumulating sequence energy (pre-explosion)")
    elif volume == "Volume_Surge":
        lines.append("Volume surge vs MA20 on 15m")
    if momentum in ("Accelerating_Up", "Sequence_Positive"):
        lines.append(f"Momentum: {momentum.replace('_', ' ')}")
    return lines[:5]


def process_symbol(symbol: str, end_ms: int) -> Candidate | None:
    try:
        k5 = fetch_klines(symbol, "5m", end_ms, 120)
        if len(k5) < 40:
            return None
        anchor = len(k5) - 1
        seq = window_seq(k5, anchor, 6)
        seq_out = {
            "seq_return_sum_6": seq.get("seq_return_sum_6", 0),
            "seq_volume_energy_6": seq.get("seq_volume_energy_6", 0),
            "seq_positive_count_6": seq.get("seq_positive_count_6", 0),
            "first_abnormal_candle_6": seq.get("first_abnormal_candle_6", 0),
        }

        tf: dict = {}
        k15 = fetch_klines(symbol, "15m", end_ms, 96)
        for label, interval, lb in TF_INTERVALS:
            kl = k15 if interval == "15m" else fetch_klines(symbol, interval, end_ms, lb)
            st = tf_candle_pair(kl)
            if st:
                tf[label] = st

        if not k15 or len(k15) < 30:
            return None
        o, h, l, c, vol = ohlcv(k15[-1])
        if not (t10.MIN_PRICE <= c <= t10.MAX_PRICE):
            return None
        rng = (h - l) / o * 100 if o else 0
        closes = [float(k[4]) for k in k15]
        ms = macd_sig(closes)
        if ms < -0.0016 or rng < 1.4768:
            return None
        vols = [ohlcv(k)[4] for k in k15[-25:-1]]
        vol_ma = statistics.mean(vols[-24:]) if vols else 0.0
        vol_ratio = vol / vol_ma if vol_ma else 0
        lc = lifecycle_15m(k15)
        rscore = h4_score(lc, vol_ratio)
        escore = explosion_score(seq, tf, lc)
        comp, vol_s, mom = classify_states(seq_out, tf, lc)
        reasons = build_reasons(seq_out, tf, comp, vol_s, mom)

        return Candidate(
            symbol=symbol,
            price=c,
            ranking_score=round(rscore, 4),
            explosion_score=escore,
            seq_5m=seq_out,
            tf_states=tf,
            compression=comp,
            volume_state=vol_s,
            momentum_state=mom,
            reasons=reasons,
        )
    except Exception:
        return None
    finally:
        time.sleep(API_SLEEP)


def eval_forward(symbol: str, entry: float, start_ms: int) -> dict:
    fwd = fetch_forward_5m(symbol, start_ms, 48)
    if entry <= 0 or not fwd:
        return {"return_4h": 0.0, "max_up_4h": 0.0}
    chunk = fwd[:48]
    max_h = max(ohlcv(k)[1] for k in chunk)
    close = float(chunk[-1][4])
    return {
        "return_4h": round((close - entry) / entry * 100, 4),
        "max_up_4h": round((max_h - entry) / entry * 100, 4),
    }


def scan_universe(end_ms: int) -> list[Candidate]:
    symbols = sorted(load_eligible_symbols(refresh=False, cache_only=False))
    print(f"  universe: {len(symbols)} symbols")
    rows: list[Candidate] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(process_symbol, s, end_ms): s for s in symbols}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                rows.append(r)
    rows.sort(key=lambda x: x.ranking_score, reverse=True)
    for i, r in enumerate(rows, 1):
        r.rank_h4 = i
    rows.sort(key=lambda x: x.explosion_score, reverse=True)
    for i, r in enumerate(rows, 1):
        r.rank_scout = i
    return rows


def find_missed(all_fwd: list[dict], picked: set[str], top_n: int = 10) -> list[dict]:
    missed = [
        r for r in all_fwd
        if r["symbol"] not in picked and r["max_up_4h"] >= MISS_THRESHOLD
    ]
    missed.sort(key=lambda x: x["max_up_4h"], reverse=True)
    return missed[:top_n]


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scan_dt = parse_kst(SCAN_KST)
    end_ms = int(scan_dt.timestamp() * 1000)
    start_ms = end_ms + 5 * 60 * 1000

    print(f"Phase 16 blind scan: {SCAN_KST} KST (no data after scan)")

    matches = scan_universe(end_ms)
    print(f"  Pattern B matches: {len(matches)}")

    top10 = sorted(matches, key=lambda x: x.explosion_score, reverse=True)[:10]
    top2 = top10[:2]

    lines = [
        "##############################################################",
        "SCOUT PHASE 16 — HUMAN VS SCOUT BLIND TEST",
        "##############################################################",
        "",
        f"Scan: {SCAN_KST} KST | Eval forward: {SCAN_KST} -> {FORWARD_END_KST}",
        f"Pattern B matches: {len(matches)}",
        "",
        "=" * 58,
        "BLIND PICK — TOP10 (Scout explosion score)",
        "=" * 58,
    ]

    for c in top10:
        lines.append(
            f"  #{c.rank_scout} {c.symbol} price={c.price} "
            f"scout={c.explosion_score} h4_rank=#{c.rank_h4} h4_score={c.ranking_score}"
        )
        lines.append(
            f"    5m: volE={c.seq_5m.get('seq_volume_energy_6')} ret6={c.seq_5m.get('seq_return_sum_6')} "
            f"pos={c.seq_5m.get('seq_positive_count_6')} abnormal={c.seq_5m.get('first_abnormal_candle_6')}"
        )
        for tf in ("15m", "30m", "1h", "2h", "4h"):
            cur = c.tf_states.get(tf, {}).get("current", {})
            if cur:
                lines.append(
                    f"    {tf} cur: body={cur.get('body_pct')}% range={cur.get('range_pct')}% "
                    f"vol/ma={cur.get('volume_ma_ratio')} close_pos={cur.get('close_position')}"
                )
        lines.append(f"    compression={c.compression} volume={c.volume_state} momentum={c.momentum_state}")

    lines.extend(["", "=" * 58, "FINAL TOP2 PICK", "=" * 58])
    for c in top2:
        lines.append(f"  ** {c.symbol} ** scout={c.explosion_score} h4=#{c.rank_h4}")
        for r in c.reasons:
            lines.append(f"    - {r}")

    # AFTER REPORT — forward only here
    lines.extend(["", "=" * 58, "AFTER REPORT (forward 4h to 15:00)", "=" * 58])

    picked = {c.symbol for c in top2}
    eval_rows: list[dict] = []
    for c in matches:
        fwd = eval_forward(c.symbol, c.price, start_ms)
        c.forward = fwd
        eval_rows.append({"symbol": c.symbol, "rank_scout": c.rank_scout, **fwd})
        time.sleep(0.15)

    eval_rows.sort(key=lambda x: x["max_up_4h"], reverse=True)
    for i, r in enumerate(eval_rows, 1):
        r["actual_rank"] = i

    lines.append("")
    lines.append("TOP10 actual by max_up_4h:")
    for r in eval_rows[:10]:
        mark = " <-- PICKED" if r["symbol"] in picked else ""
        lines.append(
            f"  #{r['actual_rank']} {r['symbol']} max_up={r['max_up_4h']}% "
            f"ret={r['return_4h']}% scout_rank=#{r['rank_scout']}{mark}"
        )

    lines.append("")
    lines.append("TOP2 pick results:")
    success = 0
    for c in top2:
        ar = next((r for r in eval_rows if r["symbol"] == c.symbol), {})
        hit = ar.get("max_up_4h", 0) >= MISS_THRESHOLD
        if hit:
            success += 1
        lines.append(
            f"  {c.symbol}: max_up={ar.get('max_up_4h',0)}% ret={ar.get('return_4h',0)}% "
            f"actual_rank=#{ar.get('actual_rank','?')} {'HIT' if hit else 'MISS'}"
        )

    missed = find_missed(eval_rows, picked)
    lines.extend(["", "Missed winners (>=7% max_up, not in TOP2):"])
    for m in missed[:5]:
        mc = next((x for x in matches if x.symbol == m["symbol"]), None)
        lines.append(
            f"  {m['symbol']} max_up={m['max_up_4h']}% scout_rank=#{m['rank_scout']} "
            f"h4_rank=#{mc.rank_h4 if mc else '?'}"
        )
        if mc:
            lines.append(
                f"    vs pick: volE6={mc.seq_5m.get('seq_volume_energy_6')} "
                f"abnormal={mc.seq_5m.get('first_abnormal_candle_6')}"
            )

    lines.extend(["", "=" * 58, "DIAGNOSIS", "=" * 58])
    t2_avg_rank = statistics.mean([r.get("actual_rank", 99) for r in eval_rows if r["symbol"] in picked]) if picked else 99
    h4_top2 = sorted(matches, key=lambda x: x.ranking_score, reverse=True)[:2]
    h4_syms = {x.symbol for x in h4_top2}
    lines.append(f"  TOP2 actual avg rank: {t2_avg_rank:.1f}")
    lines.append(f"  H4-only TOP2 would be: {', '.join(x.symbol for x in h4_top2)}")

    if success >= 1:
        lines.append("  TOP2 partial success: at least one hit >=7% max_up")
    else:
        lines.append("  TOP2 failed: neither reached 7% max_up in 4h")

    if missed:
        lines.append("  Weakness: scout score missed high movers outside Pattern B TOP10 explosion ranking")
        lines.append("  Without formula change: human can cross-check H4 rank + MTF on TOP30")
        lines.append("  If change needed: Ranking MODIFY (add seq_volume_energy aux) — not Filter")
    else:
        lines.append("  No major missed winners >=7% outside picks")

    report_path = OUT_DIR / f"blind_{SCAN_KST.replace(' ', '_').replace(':', '')}.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    write_csv(OUT_DIR / "top10_blind.csv", [
        {
            "symbol": c.symbol, "price": c.price, "scout_score": c.explosion_score,
            "h4_score": c.ranking_score, "h4_rank": c.rank_h4, "scout_rank": c.rank_scout,
            "compression": c.compression, "volume": c.volume_state, "momentum": c.momentum_state,
            **{f"5m_{k}": v for k, v in c.seq_5m.items()},
        }
        for c in top10
    ])
    write_csv(OUT_DIR / "forward_eval.csv", eval_rows)

    print("\n".join(lines[-30:]).encode("ascii", "replace").decode("ascii"))
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    run()
