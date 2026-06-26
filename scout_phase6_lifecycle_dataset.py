"""
Scout Phase 6 — Winner / Loser Lifecycle Dataset Construction

Builds reusable episode + phase records. NO trading rules. NO threshold optimization.
NO pattern discovery. Data storage only.

Usage:
  python scout_phase6_lifecycle_dataset.py build --from-cache
  python scout_phase6_lifecycle_dataset.py build --from-cache --max-scans 5
  python scout_phase6_lifecycle_dataset.py report
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10
from season2_p37_scout_decision_hierarchy import write_csv
from season2_p40_scout_transition_triggers import compute_obv, ema, public_get
from season2_universe_blind_test import (
    CACHE_DIR,
    gen_scan_times,
    load_cached_scan_times,
    load_eligible_symbols,
)
from season3_snapshot import build_snapshot_features, heikin_ashi_series, ma, ohlcv, vwap_distance

OUT_DIR = Path("logs") / "phase6_lifecycle"
EPISODES_JSONL = OUT_DIR / "episodes.jsonl"
PHASES_CSV = OUT_DIR / "phases.csv"
META_JSON = OUT_DIR / "dataset_meta.json"
REPORT_TXT = OUT_DIR / "dataset_report.txt"

INTERVAL_15M = "15m"
INTERVAL_15M_MS = 15 * 60 * 1000
LOOKBACK_15M = 96   # 24h history before anchor
FORWARD_15M = 48    # 12h forward
BIRTH_PCT = 3.0     # fixed structural constant — not optimized
ACCUM_RANGE_PCT = 3.0
MAX_ACCUM_CANDLES = 96
API_SLEEP = 0.04

PHASE_ORDER = (
    "Accumulation",
    "Ignition",
    "Birth",
    "Expansion",
    "Continuation",
    "Exhaustion",
    "Distribution",
    "End",
)

REQUIRED_PHASE_FIELDS = (
    "price", "volume", "dollar_volume", "volume_ma", "price_ma", "ema",
    "ma_slope", "atr", "atr_expansion", "obv", "vwap", "rsi", "macd",
    "macd_signal", "adx", "ha_open", "ha_close", "funding", "open_interest",
    "range_pct", "box_width_pct", "body_ratio", "upper_wick_ratio",
    "lower_wick_ratio", "higher_high", "higher_low", "btc_return_2h",
)

OPTIONAL_MISSING = ("orderbook_bid_depth", "orderbook_ask_depth")


def fetch_klines_15m(symbol: str, end_ms: int, limit: int) -> list[list]:
    params = urllib.parse.urlencode({
        "symbol": symbol,
        "interval": INTERVAL_15M,
        "endTime": end_ms,
        "limit": min(limit, t10.MAX_LIMIT),
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
        return json.loads(resp.read().decode())


def fetch_klines_forward_15m(symbol: str, start_ms: int, count: int) -> list[list]:
    params = urllib.parse.urlencode({
        "symbol": symbol,
        "interval": INTERVAL_15M,
        "startTime": start_ms,
        "limit": min(count, t10.MAX_LIMIT),
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
        return json.loads(resp.read().decode())


def rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(-period, 0):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd_values(closes: list[float]) -> tuple[float, float]:
    if len(closes) < 26:
        return 0.0, 0.0
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    line = ema12 - ema26
    # signal proxy from last 9 MACD samples
    macd_hist: list[float] = []
    for i in range(26, len(closes) + 1):
        chunk = closes[:i]
        macd_hist.append(ema(chunk, 12) - ema(chunk, 26))
    signal = ema(macd_hist, 9) if macd_hist else line
    return line, signal


def adx(klines: list[list], period: int = 14) -> float:
    if len(klines) < period + 2:
        return 0.0
    trs, plus_dm, minus_dm = [], [], []
    for i in range(1, len(klines)):
        o1, h1, l1, c1, _ = ohlcv(klines[i - 1])
        o2, h2, l2, c2, _ = ohlcv(klines[i])
        tr = max(h2 - l2, abs(h2 - c1), abs(l2 - c1))
        up = h2 - h1
        down = l1 - l2
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        trs.append(tr)
    if len(trs) < period:
        return 0.0
    atr_v = sum(trs[-period:]) / period
    if atr_v == 0:
        return 0.0
    pdi = 100 * (sum(plus_dm[-period:]) / period) / atr_v
    mdi = 100 * (sum(minus_dm[-period:]) / period) / atr_v
    if pdi + mdi == 0:
        return 0.0
    dx = abs(pdi - mdi) / (pdi + mdi) * 100
    return dx


def box_width_pct(klines: list[list], window: int = 8) -> float:
    if len(klines) < window:
        return 0.0
    chunk = klines[-window:]
    lows = [ohlcv(k)[2] for k in chunk]
    highs = [ohlcv(k)[1] for k in chunk]
    mid = statistics.mean([ohlcv(k)[4] for k in chunk])
    if mid <= 0:
        return 0.0
    return (max(highs) - min(lows)) / mid * 100


def higher_high(klines: list[list]) -> bool:
    if len(klines) < 3:
        return False
    h1 = ohlcv(klines[-1])[1]
    h2 = ohlcv(klines[-2])[1]
    h3 = ohlcv(klines[-3])[1]
    return h1 > h2 > h3


def higher_low(klines: list[list]) -> bool:
    if len(klines) < 3:
        return False
    l1 = ohlcv(klines[-1])[2]
    l2 = ohlcv(klines[-2])[2]
    l3 = ohlcv(klines[-3])[2]
    return l1 > l2 > l3


def fetch_funding_oi(symbol: str) -> tuple[float | None, float | None]:
    try:
        fr = public_get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": 1})
        funding = float(fr[-1]["fundingRate"]) if fr else None
        oi = public_get("/fapi/v1/openInterest", {"symbol": symbol})
        oi_val = float(oi.get("openInterest")) if oi else None
        return funding, oi_val
    except Exception:
        return None, None


def fetch_btc_metrics(end_ms: int) -> dict:
    try:
        k2 = t10.fetch_klines_before("BTCUSDT", t10.INTERVAL_2H, end_ms, t10.RANKING_KLINES_2H)
        if len(k2) < t10.RANKING_KLINES_2H:
            return {}
        c = float(k2[-1][4])
        prev24 = float(k2[-(t10.CANDLES_24H_2H + 1)][4])
        prev2 = float(k2[-2][4])
        prev_24 = k2[-(t10.CANDLES_24H_2H + 1):-1]
        return {
            "btc_return_24h": (c - prev24) / prev24 * 100 if prev24 else 0.0,
            "btc_return_2h": (c - prev2) / prev2 * 100 if prev2 else 0.0,
            "btc_atr_pct": t10.average_true_range_percent(prev_24, c),
        }
    except Exception:
        return {}


def kline_dt(k: list) -> datetime:
    return datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc)


def classify_outcome(entry: float, forward: list[list]) -> dict:
    if entry <= 0 or not forward:
        return {"label": "skip", "return_12h_pct": 0.0, "max_excursion_12h_pct": 0.0}

    max_high = entry
    close_12h = entry
    for i, k in enumerate(forward[:FORWARD_15M]):
        _, h, _, c, _ = ohlcv(k)
        max_high = max(max_high, h)
        close_12h = c

    max_exc = (max_high - entry) / entry * 100
    ret_12h = (close_12h - entry) / entry * 100

    if max_exc >= 5.0:
        label = "winner"
    elif ret_12h < 2.0 or ret_12h < 0:
        label = "loser"
    else:
        label = "neutral"

    return {
        "label": label,
        "return_12h_pct": round(ret_12h, 4),
        "max_excursion_12h_pct": round(max_exc, 4),
        "hit_5pct": max_exc >= 5.0,
        "hit_7pct": max_exc >= 7.0,
        "hit_10pct": max_exc >= 10.0,
    }


def find_lifecycle_bounds(history: list[list], forward: list[list], entry: float) -> dict | None:
    """Anchor at scan candle; lifecycle spans pre-scan accumulation through 12h forward."""
    if len(history) < 24 or len(forward) < 8:
        return None

    anchor_i = len(history) - 1
    all_c = history + forward
    forward_start = len(history)

    base_price = entry
    for i in range(max(0, anchor_i - 12), anchor_i + 1):
        base_price = min(base_price, ohlcv(all_c[i])[2])
    if base_price <= 0:
        return None

    birth_i = anchor_i
    for i in range(forward_start, min(len(all_c), forward_start + FORWARD_15M)):
        if ohlcv(all_c[i])[4] >= base_price * (1.0 + BIRTH_PCT / 100.0):
            birth_i = i
            break

    ignition_i = anchor_i
    for i in range(max(forward_start - 12, 0), birth_i):
        vols = [ohlcv(all_c[j])[4] for j in range(max(0, i - 8), i)]
        if not vols:
            continue
        v_ma = statistics.mean(vols)
        cur_v = ohlcv(all_c[i])[4]
        if v_ma > 0 and cur_v / v_ma >= 1.2:
            ignition_i = i
            break

    start_i = anchor_i
    for i in range(anchor_i, max(0, anchor_i - MAX_ACCUM_CANDLES), -1):
        window = all_c[max(0, i - 8): i + 1]
        if len(window) < 4:
            break
        if box_width_pct(window, len(window)) <= ACCUM_RANGE_PCT:
            start_i = i
        else:
            break

    end_i = min(len(all_c) - 1, forward_start + FORWARD_15M - 1)
    peak_i = birth_i
    peak_price = ohlcv(all_c[birth_i])[1]
    for i in range(birth_i, end_i + 1):
        h = ohlcv(all_c[i])[1]
        if h >= peak_price:
            peak_price = h
            peak_i = i

    return {
        "start_i": start_i,
        "ignition_i": ignition_i,
        "birth_i": birth_i,
        "peak_i": peak_i,
        "end_i": end_i,
        "base_price": base_price,
        "peak_price": peak_price,
        "all_c": all_c,
        "anchor_i": anchor_i,
    }


def phase_ranges(bounds: dict) -> dict[str, tuple[int, int]]:
    s = bounds["start_i"]
    ign = bounds["ignition_i"]
    birth = bounds["birth_i"]
    peak = bounds["peak_i"]
    end = bounds["end_i"]

    if peak <= birth:
        peak = min(end, birth + 1)

    span = max(1, peak - birth)
    b_exp = birth + max(1, int(span * 0.35))
    b_cont = birth + max(2, int(span * 0.65))
    exc_end = min(end, peak + max(1, (end - peak) // 2))

    return {
        "Accumulation": (s, max(s, ign - 1)),
        "Ignition": (max(s, ign), max(s, birth - 1)),
        "Birth": (birth, min(b_exp, peak)),
        "Expansion": (min(b_exp, peak), min(b_cont, peak)),
        "Continuation": (min(b_cont, peak), peak),
        "Exhaustion": (peak, exc_end),
        "Distribution": (min(exc_end + 1, end), end),
        "End": (end, end),
    }


def phase_metrics(
    klines_slice: list[list],
    btc: dict,
    funding: float | None,
    oi: float | None,
) -> dict:
    if len(klines_slice) < 10:
        return {}

    o, h, l, c, vol = ohlcv(klines_slice[-1])
    vols = [ohlcv(k)[4] for k in klines_slice[-25:-1]] if len(klines_slice) >= 25 else [ohlcv(k)[4] for k in klines_slice[:-1]]
    vol_ma = ma(vols[-24:]) if vols else 0.0
    closes = [float(k[4]) for k in klines_slice]
    price_ma = ma(closes[-20:]) if len(closes) >= 20 else ma(closes)
    ema20 = ema(closes, 20)
    atr_base = t10.average_true_range_percent(klines_slice[-15:-1], c) if len(klines_slice) >= 16 else 0.0
    atr_now = t10.true_range(klines_slice[-1], float(klines_slice[-2][4])) / c * 100 if c and len(klines_slice) >= 2 else 0.0
    obv_series = compute_obv(klines_slice)
    ha = heikin_ashi_series(klines_slice[-10:])
    ha_o, ha_c = ha[-1]
    rng = (h - l) / o * 100 if o else 0.0
    body = abs(c - o) / o * 100 if o else 0.0
    upper = (h - max(o, c)) / o * 100 if o else 0.0
    lower = (min(o, c) - l) / o * 100 if o else 0.0
    macd_line, macd_sig = macd_values(closes)

    return {
        "price": round(c, 8),
        "volume": round(vol, 4),
        "dollar_volume": round(c * vol, 4),
        "volume_ma": round(vol_ma, 4),
        "price_ma": round(price_ma, 8),
        "ema": round(ema20, 8),
        "ma_slope": round(
            (ma(closes[-6:-1]) - ma(closes[-12:-7])) / ma(closes[-12:-7]) * 100
            if len(closes) >= 12 and ma(closes[-12:-7]) else 0.0, 4
        ),
        "atr": round(atr_now, 4),
        "atr_expansion": round(atr_now / atr_base if atr_base > 0 else 0.0, 4),
        "obv": round(obv_series[-1], 4),
        "vwap": round(vwap_distance(klines_slice), 4),
        "rsi": round(rsi(closes), 4),
        "macd": round(macd_line, 6),
        "macd_signal": round(macd_sig, 6),
        "adx": round(adx(klines_slice), 4),
        "ha_open": round(ha_o, 8),
        "ha_close": round(ha_c, 8),
        "funding": funding if funding is not None else "",
        "open_interest": oi if oi is not None else "",
        "range_pct": round(rng, 4),
        "box_width_pct": round(box_width_pct(klines_slice), 4),
        "body_ratio": round(body / rng if rng > 0 else 0.0, 4),
        "upper_wick_ratio": round(upper / rng if rng > 0 else 0.0, 4),
        "lower_wick_ratio": round(lower / rng if rng > 0 else 0.0, 4),
        "higher_high": int(higher_high(klines_slice)),
        "higher_low": int(higher_low(klines_slice)),
        "btc_return_2h": round(btc.get("btc_return_2h", 0.0), 4),
        "orderbook_bid_depth": "",
        "orderbook_ask_depth": "",
    }


def compute_mdd(entry: float, klines: list[list]) -> float:
    if entry <= 0:
        return 0.0
    min_low = entry
    for k in klines:
        min_low = min(min_low, ohlcv(k)[2])
    return (entry - min_low) / entry * 100


def build_episode(
    symbol: str,
    scan_time_kst: str,
    scan_dt: datetime,
    end_ms: int,
    btc_cache: dict,
) -> dict | None:
    try:
        history = fetch_klines_15m(symbol, end_ms, LOOKBACK_15M)
        time.sleep(API_SLEEP)
        if len(history) < LOOKBACK_15M // 2:
            return None

        forward = fetch_klines_forward_15m(
            symbol,
            int(history[-1][0]) + INTERVAL_15M_MS,
            FORWARD_15M,
        )
        time.sleep(API_SLEEP)
        if len(forward) < 8:
            return None

        entry = float(history[-1][4])
        if not (t10.MIN_PRICE <= entry <= t10.MAX_PRICE):
            return None

        outcome = classify_outcome(entry, forward)
        if outcome["label"] == "neutral" or outcome["label"] == "skip":
            return None

        bounds = find_lifecycle_bounds(history, forward, entry)
        if bounds is None:
            return None

        all_c = bounds["all_c"]
        anchor_i = bounds["anchor_i"]
        start_i, peak_i, end_i = bounds["start_i"], bounds["peak_i"], bounds["end_i"]
        start_dt = kline_dt(all_c[start_i])
        peak_dt = kline_dt(all_c[peak_i])
        end_dt = kline_dt(all_c[end_i])

        btc_key = scan_time_kst[:13]
        if btc_key not in btc_cache:
            btc_cache[btc_key] = fetch_btc_metrics(end_ms)
        btc = btc_cache[btc_key]
        funding, oi = fetch_funding_oi(symbol)
        time.sleep(API_SLEEP)

        ranges = phase_ranges(bounds)
        phase_rows: list[dict] = []
        for phase_name in PHASE_ORDER:
            lo, hi = ranges[phase_name]
            mid = max(lo, (lo + hi) // 2)
            slice_end = mid + 1
            kslice = all_c[: slice_end]
            if len(kslice) < 10:
                kslice = all_c[: min(len(all_c), start_i + 10)]
            metrics = phase_metrics(kslice, btc, funding, oi)
            phase_rows.append({
                "phase": phase_name,
                "phase_start_utc": kline_dt(all_c[lo]).strftime("%Y-%m-%d %H:%M:%S"),
                "phase_end_utc": kline_dt(all_c[hi]).strftime("%Y-%m-%d %H:%M:%S"),
                "duration_min": max(15, (hi - lo + 1) * 15),
                **metrics,
            })

        end_close = float(all_c[end_i][4])
        if not (t10.MIN_PRICE <= end_close <= t10.MAX_PRICE * 5):
            end_close = entry * (1.0 + outcome["return_12h_pct"] / 100.0)
        total_return = (end_close - entry) / entry * 100 if entry else 0.0
        duration_min = max(15, (end_i - start_i + 1) * 15)
        mdd = compute_mdd(entry, all_c[anchor_i:end_i + 1])

        episode_id = f"{symbol}_{scan_time_kst.replace(' ', 'T').replace(':', '')}_{outcome['label']}"

        return {
            "episode_id": episode_id,
            "symbol": symbol,
            "scan_time_kst": scan_time_kst,
            "outcome": outcome["label"],
            "start_time_utc": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "peak_time_utc": peak_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time_utc": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "total_return_pct": round(total_return, 4),
            "max_excursion_12h_pct": outcome["max_excursion_12h_pct"],
            "return_12h_pct": outcome["return_12h_pct"],
            "hit_5pct": outcome["hit_5pct"],
            "hit_7pct": outcome["hit_7pct"],
            "hit_10pct": outcome["hit_10pct"],
            "duration_min": duration_min,
            "mdd_pct": round(mdd, 4),
            "entry_price": entry,
            "phases": phase_rows,
        }
    except Exception:
        return None


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_episodes() -> list[dict]:
    if not EPISODES_JSONL.exists():
        return []
    out: list[dict] = []
    for line in EPISODES_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def flatten_phases(episodes: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for ep in episodes:
        for ph in ep.get("phases", []):
            rows.append({
                "episode_id": ep["episode_id"],
                "symbol": ep["symbol"],
                "outcome": ep["outcome"],
                "scan_time_kst": ep["scan_time_kst"],
                **ph,
            })
    return rows


def build_dataset(
    from_cache: bool,
    max_scans: int | None,
    max_symbols_per_scan: int | None,
    refresh: bool,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if refresh or not EPISODES_JSONL.exists():
        EPISODES_JSONL.write_text("", encoding="utf-8")

    existing_ids = {json.loads(l)["episode_id"] for l in EPISODES_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()} if EPISODES_JSONL.exists() else set()

    eligible = load_eligible_symbols(refresh=False, cache_only=from_cache)
    if from_cache:
        scan_times = load_cached_scan_times()
        if not scan_times:
            scan_times = [(s, d) for s, d in gen_scan_times()[:1]]
    else:
        scan_times = gen_scan_times()

    if max_scans:
        scan_times = scan_times[:max_scans]

    btc_cache: dict = {}
    built = 0

    for scan_idx, (scan_kst, scan_dt) in enumerate(scan_times):
        print(f"Scan {scan_idx + 1}/{len(scan_times)}: {scan_kst}")
        cache_path = CACHE_DIR / f"{scan_kst.replace(' ', '_').replace(':', '')}.json"
        symbols: list[str] = []
        if from_cache and cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            symbols = [s["symbol"] for s in data.get("symbols", [])]
        if not symbols:
            symbols = sorted(eligible)

        if max_symbols_per_scan:
            symbols = symbols[:max_symbols_per_scan]

        end_ms = int(scan_dt.timestamp() * 1000)
        for sym_idx, symbol in enumerate(symbols, 1):
            if sym_idx % 50 == 0:
                print(f"  {sym_idx}/{len(symbols)} symbols...")
            ep = build_episode(symbol, scan_kst, scan_dt, end_ms, btc_cache)
            if ep and ep["episode_id"] not in existing_ids:
                append_jsonl(EPISODES_JSONL, ep)
                existing_ids.add(ep["episode_id"])
                built += 1

    episodes = load_episodes()
    write_csv(PHASES_CSV, flatten_phases(episodes))
    write_meta_and_report(episodes)
    print(f"Built {built} new episodes. Total={len(episodes)}")


def write_meta_and_report(episodes: list[dict]) -> None:
    winners = [e for e in episodes if e["outcome"] == "winner"]
    losers = [e for e in episodes if e["outcome"] == "loser"]

    phase_durations: dict[str, list[int]] = {p: [] for p in PHASE_ORDER}
    saved_fields: set[str] = set()
    missing_fields = set(OPTIONAL_MISSING)

    for ep in episodes:
        for ph in ep.get("phases", []):
            phase_durations[ph["phase"]].append(ph.get("duration_min", 0))
            for k, v in ph.items():
                if k not in ("phase", "phase_start_utc", "phase_end_utc", "duration_min") and v != "":
                    saved_fields.add(k)

    for req in REQUIRED_PHASE_FIELDS:
        if req not in saved_fields and req not in ("orderbook_bid_depth", "orderbook_ask_depth"):
            missing_fields.add(req)

    avg_lifecycle = statistics.mean([e["duration_min"] for e in episodes]) if episodes else 0.0
    avg_phase = {
        p: round(statistics.mean(v), 1) if v else 0.0
        for p, v in phase_durations.items()
    }

    min_episodes = 30
    min_winners = 10
    min_losers = 10
    # READY requires minimum counts; full production build should exceed 200+ episodes.
    ready = (
        len(episodes) >= min_episodes
        and len(winners) >= min_winners
        and len(losers) >= min_losers
        and not (missing_fields - set(OPTIONAL_MISSING))
    )

    meta = {
        "episode_count_total": len(episodes),
        "episode_count_winner": len(winners),
        "episode_count_loser": len(losers),
        "avg_lifecycle_min": round(avg_lifecycle, 1),
        "avg_phase_duration_min": avg_phase,
        "saved_fields": sorted(saved_fields),
        "missing_fields": sorted(missing_fields),
        "research_readiness": "READY" if ready else "NOT_READY",
        "quality_notes": [
            "Lifecycle starts from pre-birth accumulation search (fixed BIRTH_PCT=3.0).",
            "Funding/OI are point-in-time at build; historical funding not backfilled.",
            "Orderbook not available in historical API — fields empty.",
            "Neutral episodes (2-5% zone) excluded by design.",
        ],
    }
    META_JSON.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "===== SCOUT PHASE 6 — LIFECYCLE DATASET REPORT =====",
        "",
        "1. Episode Count",
        f"   Winner : {len(winners)}",
        f"   Loser  : {len(losers)}",
        f"   Total  : {len(episodes)}",
        "",
        "2. Average Lifecycle Length",
        f"   {avg_lifecycle:.1f} min",
        "",
        "3. Average Phase Duration (min)",
    ]
    for p in PHASE_ORDER:
        lines.append(f"   {p:14s} {avg_phase.get(p, 0):.1f}")
    lines.extend([
        "",
        "4. Saved Fields",
        f"   {', '.join(sorted(saved_fields))}",
        "",
        "5. Missing Fields",
        f"   {', '.join(sorted(missing_fields)) or '(none)'}",
        "",
        "6. Dataset Quality",
    ])
    for note in meta["quality_notes"]:
        lines.append(f"   - {note}")
    lines.extend([
        "",
        "7. Research Readiness",
        f"   {meta['research_readiness']}",
        "",
        f"Outputs: {OUT_DIR}",
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scout Phase 6 Lifecycle Dataset")
    sub = parser.add_subparsers(dest="cmd")

    b = sub.add_parser("build", help="Build lifecycle dataset")
    b.add_argument("--from-cache", action="store_true", help="Use cached universe snapshots")
    b.add_argument("--max-scans", type=int, default=None)
    b.add_argument("--max-symbols-per-scan", type=int, default=None)
    b.add_argument("--refresh", action="store_true", help="Clear and rebuild episodes.jsonl")

    sub.add_parser("report", help="Regenerate report from saved episodes")

    args = parser.parse_args()

    if args.cmd == "build":
        build_dataset(
            from_cache=args.from_cache,
            max_scans=args.max_scans,
            max_symbols_per_scan=args.max_symbols_per_scan,
            refresh=args.refresh,
        )
    elif args.cmd == "report":
        episodes = load_episodes()
        write_csv(PHASES_CSV, flatten_phases(episodes))
        write_meta_and_report(episodes)
        print(REPORT_TXT.read_text(encoding="utf-8"))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
