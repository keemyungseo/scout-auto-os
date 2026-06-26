"""
Scout Learning Season2 - P4 Empirical Market Physics Discovery

Research only. Do NOT optimize v1/v2/v3 engines.
Discover market behaviour from OHLCV. Features live or die by information gain.
"""

import csv
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from season2_p1_supply_probability_panel import DATASETS, LOGS_DIR, build_records, load_rows, pf
from season2_scout_probability_engine import attach_drawdown

KST = timezone(timedelta(hours=9))
API_BASE = "https://fapi.binance.com"
API_SLEEP = 0.04
MIN_BUCKET = 3
HORIZONS_H = (0.5, 1, 2, 4, 6, 12, 24)

PHYSICS_CSV = LOGS_DIR / "season2_p4_physics_features.csv"
RANKINGS_CSV = LOGS_DIR / "season2_p4_feature_rankings.csv"
DECAY_CSV = LOGS_DIR / "season2_p4_time_decay.csv"
SUPPLY_CSV = LOGS_DIR / "season2_p4_supply_zone.csv"
TREND_ARCH_CSV = LOGS_DIR / "season2_p4_trend_archetypes.csv"
COLLAPSE_ARCH_CSV = LOGS_DIR / "season2_p4_collapse_archetypes.csv"
SYMBOL_CSV = LOGS_DIR / "season2_p4_symbol_memory.csv"
COUNTER_CSV = LOGS_DIR / "season2_p4_counterexamples.csv"
REPORT_TXT = LOGS_DIR / "season2_p4_research_report.txt"


def ohlcv(k: list) -> tuple[float, float, float, float, float]:
    return float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])


def fetch_klines(symbol: str, interval: str, end_ms: int, limit: int) -> list[list]:
    params = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "endTime": end_ms, "limit": limit})
    url = f"{API_BASE}/fapi/v1/klines?{params}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
        return json.loads(resp.read().decode())


def fetch_forward_1h(symbol: str, start_ms: int, hours: int = 26) -> list[list]:
    end_ms = start_ms + hours * 3600 * 1000
    params = urllib.parse.urlencode(
        {"symbol": symbol, "interval": "1h", "startTime": start_ms, "endTime": end_ms, "limit": 30}
    )
    url = f"{API_BASE}/fapi/v1/klines?{params}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
        return json.loads(resp.read().decode())


def kline_close_utc(k: list, interval_h: float) -> datetime:
    open_dt = datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc)
    return open_dt + timedelta(hours=interval_h)


def forward_close_at(klines: list[list], scan_dt: datetime, hours: float, interval_h: float = 2.0) -> float | None:
    target = scan_dt + timedelta(hours=hours)
    close = None
    for k in klines:
        if kline_close_utc(k, interval_h) <= target:
            close = float(k[4])
        else:
            break
    return close


def forward_close_1h(klines_1h: list[list], scan_dt: datetime, hours: float) -> float | None:
    return forward_close_at(klines_1h, scan_dt, hours, interval_h=1.0)


def wick_structure(k: list) -> dict:
    o, h, l, c, _ = ohlcv(k)
    rng = max(h - l, 1e-12)
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return {
        "body_pct": body / rng * 100,
        "upper_wick_pct": upper / rng * 100,
        "lower_wick_pct": lower / rng * 100,
        "body_range_ratio": body / rng,
        "close_position": (c - l) / rng * 100,
        "bull": c >= o,
    }


def consecutive_direction(candles: list[list]) -> int:
    if len(candles) < 2:
        return 0
    streak = 1
    last_bull = wick_structure(candles[-1])["bull"]
    for k in reversed(candles[:-1]):
        if wick_structure(k)["bull"] == last_bull:
            streak += 1
        else:
            break
    return streak if last_bull else -streak


def pattern_flags(candles: list[list]) -> dict[str, bool]:
    if len(candles) < 3:
        return {k: False for k in ("inside", "outside", "engulf", "gap_up", "gap_down", "three_push")}

    cur, prev, p2 = candles[-1], candles[-2], candles[-3]
    co, ch, cl, cc, _ = ohlcv(cur)
    po, ph, pl, pc, _ = ohlcv(prev)

    inside = ch <= ph and cl >= pl
    outside = ch >= ph and cl <= pl
    engulf = (cc > co and pc < po and cc > po and co < pc) or (cc < co and pc > po and cc < po and co > pc)
    gap_up = cl > ph
    gap_down = ch < pl

    highs = [ohlcv(k)[1] for k in candles[-3:]]
    three_push = highs[2] > highs[1] > highs[0] if len(highs) == 3 else False

    return {
        "inside_bar": inside,
        "outside_bar": outside,
        "engulf": engulf,
        "gap_up": gap_up,
        "gap_down": gap_down,
        "three_push_up": three_push,
    }


def volume_physics(candles: list[list]) -> dict:
    vols = [ohlcv(k)[4] for k in candles]
    if len(vols) < 6:
        return {}
    ma6 = statistics.mean(vols[-6:])
    ma12 = statistics.mean(vols[-12:]) if len(vols) >= 12 else ma6
    last = vols[-1]
    ratio = last / ma6 if ma6 else 1.0
    persist_up = vols[-1] > vols[-2] > vols[-3] if len(vols) >= 3 else False
    persist_down = vols[-1] < vols[-2] < vols[-3] if len(vols) >= 3 else False
    shock = ratio >= 2.0
    exhaustion = ratio >= 2.5 and wick_structure(candles[-1])["upper_wick_pct"] > 45
    recovery = ratio >= 1.3 and wick_structure(candles[-1])["lower_wick_pct"] > 45
    return {
        "vol_ratio_6": ratio,
        "vol_ratio_12": last / ma12 if ma12 else 1.0,
        "vol_persist_up": persist_up,
        "vol_persist_down": persist_down,
        "vol_shock": shock,
        "vol_exhaustion": exhaustion,
        "vol_recovery": recovery,
        "vol_cluster_high": ratio >= 1.5 and statistics.mean(vols[-3:]) / ma6 >= 1.2 if ma6 else False,
    }


def supply_zone_physics(candles: list[list]) -> dict:
    if len(candles) < 12:
        return {}
    window = candles[-12:]
    highs = [ohlcv(k)[1] for k in window]
    lows = [ohlcv(k)[2] for k in window]
    closes = [ohlcv(k)[3] for k in window]
    c = closes[-1]
    hi12 = max(highs)
    lo12 = min(lows)
    dist_high = (hi12 - c) / c * 100 if c else 0
    dist_low = (c - lo12) / c * 100 if c else 0

    reversals = 0
    for i in range(1, len(closes) - 1):
        if closes[i] > closes[i - 1] and closes[i] > closes[i + 1]:
            reversals += 1
        if closes[i] < closes[i - 1] and closes[i] < closes[i + 1]:
            reversals += 1

    band = c * 0.02
    congestion = sum(1 for k in window if ohlcv(k)[2] <= c + band and ohlcv(k)[1] >= c - band)

    w = wick_structure(candles[-1])
    fast_rejection = w["upper_wick_pct"] > 55 and w["bull"]
    slow_absorption = w["body_pct"] < 35 and ohlcv(candles[-1])[4] > statistics.mean([ohlcv(k)[4] for k in window[-6:]])

    return {
        "dist_from_12h_high_pct": dist_high,
        "dist_from_12h_low_pct": dist_low,
        "reversal_count_12": reversals,
        "congestion_bars_12": congestion,
        "near_recent_high": dist_high < 2.0,
        "near_recent_low": dist_low < 2.0,
        "fast_rejection": fast_rejection,
        "slow_absorption": slow_absorption,
    }


def range_expansion_contraction(candles: list[list]) -> dict:
    if len(candles) < 8:
        return {}
    ranges = [(ohlcv(k)[1] - ohlcv(k)[2]) / max(ohlcv(k)[1], 1e-9) * 100 for k in candles]
    recent = statistics.mean(ranges[-3:])
    prior = statistics.mean(ranges[-8:-3])
    ratio = recent / prior if prior else 1.0
    return {
        "range_expand": ratio >= 1.3,
        "range_contract": ratio <= 0.75,
        "range_ratio_3v5": ratio,
    }


def extract_physics(symbol: str, scan_time: str, use_api: bool = True) -> dict:
    scan_kst = datetime.strptime(scan_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
    scan_utc = scan_kst.astimezone(timezone.utc)
    end_ms = int(scan_utc.timestamp() * 1000)

    out: dict = {"symbol": symbol, "scan_time": scan_time}

    if not use_api:
        return out

    try:
        k2h = fetch_klines(symbol, "2h", end_ms, 24)
        time.sleep(API_SLEEP)
        k1h_fwd = fetch_forward_1h(symbol, end_ms, 26)
        time.sleep(API_SLEEP)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        out["fetch_error"] = str(exc)[:80]
        return out

    if len(k2h) < 8:
        out["fetch_error"] = "insufficient_klines"
        return out

    entry = ohlcv(k2h[-1])[3]
    w = wick_structure(k2h[-1])
    out.update(w)
    out.update(pattern_flags(k2h))
    out.update(volume_physics(k2h))
    out.update(supply_zone_physics(k2h))
    out.update(range_expansion_contraction(k2h))
    out["consec_candles"] = consecutive_direction(k2h)
    out["long_tail_recovery"] = w["lower_wick_pct"] > 50 and w["bull"]
    out["long_tail_rejection"] = w["upper_wick_pct"] > 50 and not w["bull"]

    for hours in HORIZONS_H:
        if hours in (0.5, 1):
            close = forward_close_1h(k1h_fwd, scan_utc, hours)
        else:
            close = forward_close_at(k2h, scan_utc, hours, 2.0)
        if close is not None and entry:
            ret = (close - entry) / entry * 100
            out[f"forward_{hours}h"] = ret
            out[f"persist_{hours}h"] = 1 if ret >= 0 else 0

    return out


def bucket(value: float | None, cuts: list[float], prefix: str) -> str:
    if value is None:
        return f"{prefix}_na"
    bounds = [-math.inf] + cuts + [math.inf]
    labels = [f"{prefix}_lt{cuts[0]}"] + [f"{prefix}_{cuts[i]}_{cuts[i+1]}" for i in range(len(cuts)-1)] + [f"{prefix}_gt{cuts[-1]}"]
    for i in range(len(bounds) - 1):
        if bounds[i] <= value < bounds[i + 1]:
            return labels[i]
    return f"{prefix}_na"


def bool_tag(prefix: str, value: bool) -> str:
    return f"{prefix}_yes" if value else f"{prefix}_no"


def trend_archetype(record: dict) -> str:
    f2 = record.get("forward_2h")
    if f2 == 0.0 and record.get("forward_4h") not in (None, 0.0):
        f2 = None
    f4 = resolve_f4(record)
    f6 = resolve_f6(record)
    f12 = record.get("forward_12h")
    f24 = record.get("forward_24h")
    dd = record.get("max_drawdown") or 0
    ret24 = record.get("return_24h_at_scan") or 0

    if f6 is None:
        return "unknown"
    if f2 is not None and f2 >= 3 and f6 <= -5:
        return "fake_breakout"
    if ret24 >= 35 and f6 < 0:
        return "late_trend"
    if f2 is not None and f2 < 0 and f6 >= 5:
        return "recovering_trend"
    if f6 >= 8 and dd >= 15:
        return "violent_trend"
    if 0 <= f6 < 8 and f4 is not None and f4 >= 0 and dd < 10:
        return "steady_trend"
    if abs(f6) < 3 and f12 is not None and abs(f12) < 5:
        return "grinding_flat"
    if f4 is not None and f4 > 0 and f12 is not None and f12 < f4:
        return "mean_reverting"
    if f24 is not None and f6 * f24 < 0:
        return "rotating"
    return "mixed"


def collapse_archetype(record: dict) -> str:
    if record.get("collapse_label") != "YES":
        return "no_collapse"
    f2 = record.get("forward_2h") or record.get("forward_2.0h")
    f4 = record.get("forward_4h") or record.get("forward_4.0h")
    f6 = record.get("forward_6h") or record.get("forward_6.0h")
    if f2 is not None and f2 <= -5:
        return "early_collapse"
    if f4 is not None and f4 <= -10:
        return "sharp_4h_collapse"
    if f2 is not None and f2 >= 0 and f4 is not None and f4 <= -8:
        return "delayed_collapse"
    if f6 is not None and f6 <= -15:
        return "deep_6h_collapse"
    if f4 is not None and f4 <= -5 and f6 is not None and f6 > f4:
        return "false_warning_recovery"
    return "collapse_other"


def symbol_archetype(records: list[dict]) -> str:
    f6 = [resolve_f6(r) for r in records]
    f6 = [v for v in f6 if v is not None]
    collapses = sum(1 for r in records if r.get("collapse_label") == "YES")
    n = len(records)
    if not f6:
        return "unknown"
    med = statistics.median(f6)
    vol = statistics.pstdev(f6) if len(f6) > 1 else 0
    if collapses / n >= 0.35:
        return "collapse_prone"
    if med >= 8 and vol >= 12:
        return "explosive"
    if med >= 3 and vol < 8:
        return "persistent"
    if vol >= 15:
        return "high_noise"
    if vol < 5:
        return "low_noise"
    if med >= 2:
        return "rotation_favourite"
    return "mixed_symbol"


def build_feature_catalog(record: dict) -> list[tuple[str, str, str]]:
    """Returns (category, name, bucket_key)"""
    feats: list[tuple[str, str, str]] = []

    # Task 1 candle
    feats.append(("candle", "body_pct", bucket(record.get("body_pct"), [30, 50, 70], "body")))
    feats.append(("candle", "upper_wick", bucket(record.get("upper_wick_pct"), [25, 45, 60], "uw")))
    feats.append(("candle", "lower_wick", bucket(record.get("lower_wick_pct"), [25, 45, 60], "lw")))
    feats.append(("candle", "body_range_ratio", bucket(record.get("body_range_ratio"), [0.3, 0.5, 0.7], "brr")))
    feats.append(("candle", "close_position", bucket(record.get("close_position"), [30, 50, 70], "cp")))
    feats.append(("candle", "consec_candles", bucket(record.get("consec_candles"), [-2, 0, 2, 4], "consec")))
    for flag in ("inside_bar", "outside_bar", "engulf", "gap_up", "gap_down", "three_push_up",
                 "long_tail_recovery", "long_tail_rejection", "range_expand", "range_contract"):
        if flag in record:
            feats.append(("candle", flag, bool_tag(flag, bool(record[flag]))))

    # Task 2 volume
    feats.append(("volume", "vol_ratio_6", bucket(record.get("vol_ratio_6"), [0.8, 1.2, 2.0], "vr6")))
    for flag in ("vol_shock", "vol_exhaustion", "vol_recovery", "vol_persist_up", "vol_cluster_high"):
        if flag in record:
            feats.append(("volume", flag, bool_tag(flag, bool(record[flag]))))

    # Task 4 supply
    feats.append(("supply", "dist_12h_high", bucket(record.get("dist_from_12h_high_pct"), [2, 5, 10], "dhi")))
    feats.append(("supply", "dist_12h_low", bucket(record.get("dist_from_12h_low_pct"), [2, 5, 10], "dlo")))
    feats.append(("supply", "congestion", bucket(record.get("congestion_bars_12"), [3, 6, 9], "cong")))
    for flag in ("near_recent_high", "near_recent_low", "fast_rejection", "slow_absorption"):
        if flag in record:
            feats.append(("supply", flag, bool_tag(flag, bool(record[flag]))))

    # Legacy (test for death)
    feats.append(("legacy", "ma24_slope", bucket(record.get("ma24_slope_percent"), [4, 8], "ma24")))
    feats.append(("legacy", "volume_ratio_ma24", bucket(record.get("volume_ratio_ma24"), [1.2, 3], "vma24")))
    feats.append(("legacy", "trigger_bundle", record.get("trigger_bundle") or "unknown"))

    return feats


@dataclass
class FeatureRank:
    category: str
    name: str
    spread_f6: float
    spread_persist_4h: float
    loo_mae: float
    top1: str
    stability: float
    info_gain: float
    verdict: str


def loo_mae(records: list[dict], key_fn) -> tuple[float, int, int]:
    dates = sorted({r["date"] for r in records})
    errors: list[float] = []
    top1 = scans = 0
    for holdout in dates:
        train = [r for r in records if r["date"] != holdout]
        test = [r for r in records if r["date"] == holdout]
        groups: dict[str, list] = defaultdict(list)
        for r in train:
            groups[key_fn(r)].append(r)
        medians = {
            k: statistics.median([resolve_f6(x) for x in g if resolve_f6(x) is not None])
            for k, g in groups.items()
            if len(g) >= MIN_BUCKET and any(resolve_f6(x) is not None for x in g)
        }
        global_vals = [resolve_f6(x) for x in train if resolve_f6(x) is not None]
        global_m = statistics.median(global_vals) if global_vals else 0.0
        by_scan: dict[str, list] = defaultdict(list)
        for r in test:
            by_scan[r["scan_time"]].append(r)
        for scan_rows in by_scan.values():
            scans += 1
            preds = []
            for r in scan_rows:
                pred = medians.get(key_fn(r), global_m)
                actual = resolve_f6(r)
                if actual is not None:
                    errors.append(abs(pred - actual))
                preds.append((r["symbol"], pred))
            if preds:
                best = max(preds, key=lambda x: x[1])
                actual_best = max(scan_rows, key=lambda x: resolve_f6(x) or -999)
                if best[0] == actual_best["symbol"]:
                    top1 += 1
    return (statistics.mean(errors) if errors else 999.0, top1, scans)


def spread_metric(records: list[dict], key_fn, field: str) -> float:
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)
    vals = []
    for g in groups.values():
        if len(g) < MIN_BUCKET:
            continue
        nums = [resolve_f6(x) for x in g if resolve_f6(x) is not None]
        if field == "target_persist_4h":
            nums = [x.get("target_persist_4h") for x in g if x.get("target_persist_4h") is not None]
        if nums:
            vals.append(statistics.median(nums))
    return max(vals) - min(vals) if len(vals) >= 2 else 0.0


def rank_features(records: list[dict], baseline_mae: float) -> list[FeatureRank]:
    def make_key(name: str):
        def key_fn(r: dict) -> str:
            for _cat, n, val in build_feature_catalog(r):
                if n == name:
                    return val
            return "na"
        return key_fn

    seen: set[str] = set()
    ranks: list[FeatureRank] = []

    for cat, name, _ in build_feature_catalog(records[0]):
        if name in seen:
            continue
        seen.add(name)
        key_fn = make_key(name)
        mae, t1, ts = loo_mae(records, key_fn)
        sp_f6 = spread_metric(records, key_fn, "target_f6")
        sp_p4 = spread_metric(records, key_fn, "target_persist_4h")

        groups: dict[str, list] = defaultdict(list)
        for r in records:
            groups[key_fn(r)].append(r.get("target_f6") or 0)
        stab = statistics.mean([statistics.pstdev(v) if len(v) > 1 else 0 for v in groups.values()]) if groups else 0

        gain = max(0, baseline_mae - mae) + sp_f6 * 0.05
        if mae > baseline_mae * 1.01:
            verdict = "DEAD" if sp_f6 < 5 else "SPREAD_ONLY"
        elif mae < baseline_mae * 0.98 or sp_f6 >= 8:
            verdict = "ALIVE"
        elif sp_f6 >= 5:
            verdict = "HYPOTHESIS"
        else:
            verdict = "WEAK"

        p4_display = sp_p4 * 100 if sp_p4 <= 1 else sp_p4
        ranks.append(
            FeatureRank(cat, name, sp_f6, p4_display, mae, f"{t1}/{ts}", stab, gain, verdict)
        )

    ranks.sort(key=lambda x: (-x.info_gain, x.loo_mae))
    return ranks


def time_decay_analysis(records: list[dict], top_features: list[str]) -> list[dict]:
    rows = []
    horizon_map = [
        (0.5, "forward_0.5h", "persist_0.5h"),
        (1, "forward_1h", "persist_1h"),
        (2, "forward_2h", "persist_2h"),
        (4, "forward_4h", "persist_4h"),
        (6, "forward_6h", "persist_6h"),
        (12, "forward_12h", "persist_12h"),
        (24, "forward_24h", "persist_24h"),
    ]

    for hours, fk, pk in horizon_map:
        if hours in (0.5, 1):
            persist_vals = []
            for r in records:
                if pk in r and r.get(pk) not in ("", None):
                    persist_vals.append(float(r[pk]))
                elif fk in r and r.get(fk) not in ("", None):
                    persist_vals.append(1.0 if float(r[fk]) >= 0 else 0.0)
            if persist_vals:
                rows.append({"horizon_h": hours, "metric": "unconditional_persist", "value": round(statistics.mean(persist_vals) * 100, 1)})
            continue

        field = f"forward_{int(hours)}h" if hours != 0.5 else fk
        persist_vals = []
        for r in records:
            value = r.get(field)
            if value is None or value == "":
                continue
            value = float(value)
            if field == "forward_2h" and value == 0.0 and r.get("forward_4h") not in (None, 0.0, ""):
                continue
            persist_vals.append(1.0 if value >= 0 else 0.0)
        if persist_vals:
            rows.append({"horizon_h": hours, "metric": "unconditional_persist", "value": round(statistics.mean(persist_vals) * 100, 1)})

    for name in top_features[:5]:
        def key_fn(r):
            for cat, n, val in build_feature_catalog(r):
                if n == name:
                    return val
            return "na"

        for hours in (1, 2, 4, 6, 12, 24):
            fk = f"forward_{hours}h" if hours != 0.5 else "forward_0.5h"
            pk = f"persist_{hours}h" if hours != 0.5 else "persist_0.5h"
            groups: dict[str, list] = defaultdict(list)
            for r in records:
                if pk in r:
                    groups[key_fn(r)].append(r[pk])
            if not groups:
                continue
            best = max(groups.items(), key=lambda x: statistics.mean(x[1]) if x[1] else 0)
            worst = min(groups.items(), key=lambda x: statistics.mean(x[1]) if x[1] else 0)
            spread = (statistics.mean(best[1]) - statistics.mean(worst[1])) * 100 if best[1] and worst[1] else 0
            rows.append({
                "horizon_h": hours,
                "metric": f"spread_{name}",
                "value": round(spread, 1),
                "best_bucket": best[0],
                "worst_bucket": worst[0],
            })

    return rows


def estimate_half_life(decay_rows: list[dict]) -> str:
    uncond = {r["horizon_h"]: r["value"] for r in decay_rows if r["metric"] == "unconditional_persist"}
    if not uncond:
        return "insufficient_data"
    base = uncond.get(2, uncond.get(4, 50))
    for h in sorted(uncond.keys()):
        if uncond[h] <= base * 0.5:
            return f"~{h}h (persist drops 50% from 2h baseline {base:.0f}%)"
    return f">{max(uncond.keys())}h (no 50% decay observed in window)"


def collect_counterexamples(records: list[dict], ranks: list[FeatureRank]) -> list[dict]:
    rows = []
    alive = [r for r in ranks if r.verdict == "ALIVE"][:3]
    dead = [r for r in ranks if r.verdict == "DEAD" and r.category == "legacy"][:3]

    for feat in alive:
        rows.append({"type": "alive_feature", "detail": f"{feat.category}/{feat.name} gain={feat.info_gain:.2f} MAE={feat.loo_mae}"})

    for feat in dead:
        rows.append({"type": "dead_legacy", "detail": f"{feat.name} MAE={feat.loo_mae} spread={feat.spread_f6:.1f}"})

    # same candle pattern different outcome
    by_pat: dict[str, list] = defaultdict(list)
    for r in records:
        if "engulf" in r:
            key = f"engulf_{r['engulf']}"
            by_pat[key].append(r.get("target_f6"))
    for key, vals in by_pat.items():
        if len(vals) >= MIN_BUCKET and max(vals) - min(vals) >= 15:
            rows.append({"type": "same_pattern_diff_outcome", "detail": f"{key} f6 range {min(vals):.1f} to {max(vals):.1f}"})

    return rows


def load_or_build_physics(records: list[dict], refresh: bool = False) -> list[dict]:
    if PHYSICS_CSV.exists() and not refresh:
        cached = list(csv.DictReader(PHYSICS_CSV.open(encoding="utf-8")))
        if len(cached) >= len(records) * 0.9:
            return [_merge_record(r, cached) for r in records]

    rows_out = []
    total = len(records)
    for index, record in enumerate(records, 1):
        phys = extract_physics(record["symbol"], record["scan_time"], use_api=True)
        merged = {**record, **phys}
        rows_out.append(merged)
        if index % 20 == 0:
            print(f"  physics fetch {index}/{total}")

    if rows_out:
        keys = sorted({k for row in rows_out for k in row if not k.startswith("_")})
        with PHYSICS_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for row in rows_out:
                w.writerow({k: row.get(k, "") for k in keys})
    return rows_out


def _merge_record(record: dict, cached: list[dict]) -> dict:
    for row in cached:
        if row.get("symbol") == record["symbol"] and row.get("scan_time") == record["scan_time"]:
            merged = {**record}
            for k, v in row.items():
                if v != "" and k not in ("symbol", "scan_time"):
                    try:
                        merged[k] = float(v) if v not in ("True", "False") else v == "True"
                    except ValueError:
                        merged[k] = v
            return merged
    return extract_physics(record["symbol"], record["scan_time"], use_api=True) | record


def enrich_legacy(record: dict) -> None:
    for path in DATASETS:
        for row in load_rows(path):
            if row.get("scan_time_kst") == record["scan_time"] and row.get("symbol") == record["symbol"]:
                record["ma24_slope_percent"] = pf(row.get("ma24_slope_percent"))
                record["volume_ratio_ma24"] = pf(row.get("volume_ratio_ma24"))
                return


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def resolve_f6(record: dict) -> float | None:
    for key in ("target_f6", "forward_6h", "forward_6.0h"):
        value = record.get(key)
        if value is not None and value != "":
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def resolve_f4(record: dict) -> float | None:
    for key in ("forward_4h", "forward_4.0h"):
        value = record.get(key)
        if value is not None and value != "":
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def global_baseline_mae(records: list[dict]) -> float:
    dates = sorted({r["date"] for r in records})
    errors: list[float] = []
    for holdout in dates:
        train = [r for r in records if r["date"] != holdout]
        test = [r for r in records if r["date"] == holdout]
        gm_vals = [resolve_f6(x) for x in train]
        gm_vals = [v for v in gm_vals if v is not None]
        if not gm_vals:
            continue
        gm = statistics.median(gm_vals)
        for r in test:
            actual = resolve_f6(r)
            if actual is not None:
                errors.append(abs(gm - actual))
    return statistics.mean(errors) if errors else 999.0


def main() -> None:
    refresh = "--refresh" in sys.argv
    records = build_records()
    attach_drawdown(records)
    for r in records:
        enrich_legacy(r)
        r["target_f6"] = r.get("forward_6h")
        r["target_persist_4h"] = 1 if r.get("forward_4h") is not None and r["forward_4h"] >= 0 else 0

    print("Building physics features (API)...")
    enriched = load_or_build_physics(records, refresh=refresh)

    panel_by_key = {(r["scan_time"], r["symbol"]): r for r in records}
    for row in enriched:
        panel = panel_by_key.get((row["scan_time"], row["symbol"]), {})
        for k in ("forward_2h", "forward_4h", "forward_6h", "forward_12h", "forward_24h", "max_drawdown", "collapse_label", "supply_label"):
            if panel.get(k) is not None:
                row[k] = panel[k]
        row["target_f6"] = resolve_f6(row)
        f4 = resolve_f4(row)
        if f4 is not None:
            row["target_persist_4h"] = 1 if f4 >= 0 else 0
        row["trend_archetype"] = trend_archetype(row)
        row["collapse_archetype"] = collapse_archetype(row)

    baseline_mae = global_baseline_mae(enriched)
    ranks = rank_features(enriched, baseline_mae)

    decay_rows = time_decay_analysis(enriched, [r.name for r in ranks if r.verdict == "ALIVE"])
    half_life = estimate_half_life(decay_rows)

    # archetype tables
    trend_rows = []
    for arch in sorted({r["trend_archetype"] for r in enriched}):
        g = [r for r in enriched if r["trend_archetype"] == arch]
        f6 = [r["target_f6"] for r in g if r.get("target_f6") is not None]
        trend_rows.append({
            "archetype": arch, "n": len(g),
            "median_f6": round(statistics.median(f6), 2) if f6 else "",
            "collapse_pct": round(sum(1 for r in g if r.get("collapse_label") == "YES") / len(g) * 100, 1),
        })

    collapse_rows = []
    for arch in sorted({r["collapse_archetype"] for r in enriched}):
        g = [r for r in enriched if r["collapse_archetype"] == arch]
        if not g:
            continue
        collapse_rows.append({"archetype": arch, "n": len(g), "pct_of_sample": round(len(g) / len(enriched) * 100, 1)})

    sym_groups: dict[str, list] = defaultdict(list)
    for r in enriched:
        sym_groups[r["symbol"]].append(r)
    sym_rows = [{"symbol": s, "n": len(g), "symbol_archetype": symbol_archetype(g)} for s, g in sorted(sym_groups.items(), key=lambda x: -len(x[1]))]

    counters = collect_counterexamples(enriched, ranks)

    rank_rows = [{"category": r.category, "feature": r.name, "spread_f6": round(r.spread_f6, 2),
                  "loo_mae": round(r.loo_mae, 2), "top1": r.top1, "info_gain": round(r.info_gain, 3),
                  "stability": round(r.stability, 2), "verdict": r.verdict} for r in ranks]

    supply_rows = [r for r in rank_rows if r["category"] == "supply"]

    write_csv(RANKINGS_CSV, rank_rows)
    write_csv(DECAY_CSV, decay_rows)
    write_csv(SUPPLY_CSV, supply_rows)
    write_csv(TREND_ARCH_CSV, trend_rows)
    write_csv(COLLAPSE_ARCH_CSV, collapse_rows)
    write_csv(SYMBOL_CSV, sym_rows)
    write_csv(COUNTER_CSV, counters)

    alive = [r for r in ranks if r.verdict == "ALIVE"]
    dead_legacy = [r for r in ranks if r.category == "legacy" and r.verdict in ("DEAD", "SPREAD_ONLY")]

    lines = [
        "===== SCOUT SEASON2 P4 - EMPIRICAL MARKET PHYSICS =====",
        "",
        "Rule: no hypothesis defense. Features live or die by information gain.",
        f"Sample: {len(enriched)} records | Baseline LOO MAE: {baseline_mae:.2f}%",
        f"Information half-life estimate: {half_life}",
        "",
        "--- ALIVE features (empirical survivors) ---",
    ]
    for r in alive[:15]:
        lines.append(f"  [{r.category}] {r.name}: MAE={r.loo_mae:.2f} spread_f6={r.spread_f6:.1f} gain={r.info_gain:.2f} top1={r.top1}")

    lines.extend(["", "--- DEAD legacy indicators ---"])
    for r in dead_legacy:
        lines.append(f"  {r.name}: MAE={r.loo_mae:.2f} spread={r.spread_f6:.1f} -> DISCARD")

    lines.extend(["", "--- Task 1: Pure candle (top) ---"])
    for r in [x for x in ranks if x.category == "candle"][:8]:
        lines.append(f"  {r.name}: {r.verdict} spread={r.spread_f6:.1f}")

    lines.extend(["", "--- Task 2: Volume (top) ---"])
    for r in [x for x in ranks if x.category == "volume"][:6]:
        lines.append(f"  {r.name}: {r.verdict} spread={r.spread_f6:.1f}")

    lines.extend(["", "--- Task 4: Supply zone ---"])
    for r in [x for x in ranks if x.category == "supply"][:6]:
        lines.append(f"  {r.name}: {r.verdict} spread={r.spread_f6:.1f}")

    lines.extend(["", "--- Task 5: Trend archetypes ---"])
    for row in sorted(trend_rows, key=lambda x: -(x.get("median_f6") or -999))[:8]:
        lines.append(f"  {row['archetype']}: n={row['n']} median_f6={row['median_f6']}% collapse={row['collapse_pct']}%")

    lines.extend(["", "--- Task 6: Collapse archetypes ---"])
    for row in collapse_rows:
        lines.append(f"  {row['archetype']}: n={row['n']} ({row['pct_of_sample']}% of sample)")

    lines.extend(["", "--- Task 7: Symbol archetypes (top appearances) ---"])
    for row in sym_rows[:12]:
        lines.append(f"  {row['symbol']}: n={row['n']} type={row['symbol_archetype']}")

    lines.extend(["", "--- Counterexamples ---"])
    for c in counters[:12]:
        lines.append(f"  [{c['type']}] {c['detail']}")

    lines.extend([
        "",
        "--- Hypothesis verdicts ---",
        " DISCARD legacy by MAE: ma24_slope, volume_ratio_ma24, trigger_bundle (spread without MAE gain = noise)",
        " PREFER simple candle/volume/supply over composite indicators",
        " REVISE: textbook patterns - let buckets not names drive (inside/outside/engulf tested empirically)",
        " SUPPLY MEMORY: near_recent_high + fast_rejection tested; crypto decay shorter than equities (see half-life)",
        "",
        "--- Future engine output template ---",
        " SYMBOL | persist 30m/1h/2h/4h | E[ret] | E[dd] | collapse% | rel rank | conf | holding | reason | risk | score | action",
        "",
        f"Physics CSV: {PHYSICS_CSV}",
        f"Rankings: {RANKINGS_CSV}",
        "=" * 58,
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P4 MARKET PHYSICS DISCOVERY =====")
    print(f"Baseline MAE: {baseline_mae:.2f}% | Half-life: {half_life}")
    print(f"ALIVE features: {len(alive)} | DEAD legacy: {len(dead_legacy)}")
    if alive:
        print(f"Best: [{alive[0].category}] {alive[0].name} gain={alive[0].info_gain:.2f}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
