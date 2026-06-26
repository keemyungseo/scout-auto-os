"""
Scout Phase 28 - Acceleration Curve Lab

Observe early change as curves (derivatives, integrals, similarity).
No new features, formulas, triggers, thresholds, or ranking changes.

Input:
  logs/phase19_winner_dna/candidates.jsonl + kline_cache
  logs/phase23_formula_league/match_log.jsonl

Usage:
  python scout_phase28_acceleration_curve_lab.py
  python scout_phase28_acceleration_curve_lab.py --max-candidates 500
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import scout_phase16_human_blind_test as p16
import scout_phase20_winner_state_ranking as p20
from scout_phase13_5m_sequence_ignition import compression_length, window_seq
from scout_phase16_human_blind_test import fetch_klines, parse_kst
from season2_p37_scout_decision_hierarchy import write_csv
from season2_universe_blind_test import ohlcv

P19_CAND = Path("logs") / "phase19_winner_dna" / "candidates.jsonl"
P19_CACHE = Path("logs") / "phase19_winner_dna" / "kline_cache"
P23_MATCH = Path("logs") / "phase23_formula_league" / "match_log.jsonl"
OUT_DIR = Path("logs") / "phase28_curve_lab"
CURVE_CACHE = OUT_DIR / "curves_cache.jsonl"

FORMULAS = ("A", "A2", "A5", "A6")
FP_THRESHOLD = 2.0
TIME_MINUTES = (60, 30, 15, 10, 5)  # past -> near present (minutes before scan)
BAR_OFFSETS = (12, 6, 3, 2, 1)
CHANNELS = ("return", "ma_distance", "volume", "expansion", "compression", "energy")
BIRTH_WINDOWS = (
    ("0_15m", (60, 30, 15)),
    ("15_30m", (30, 15, 10)),
    ("30_60m", (60, 30, 10)),
)
MTF_INTERVALS = ("5m", "15m", "30m")
MTF_BAR_MAP = {
    "5m": {60: 12, 30: 6, 15: 3, 10: 2, 5: 1},
    "15m": {60: 4, 30: 2, 15: 1, 10: 1, 5: 1},
    "30m": {60: 2, 30: 1, 15: 1, 10: 1, 5: 1},
}


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def g(obj: dict, key: str, default: float = 0.0) -> float:
    return float(obj.get(key, default))


def entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def ig_binary(pos: int, pos_n: int, neg: int, neg_n: int) -> float:
    if pos_n == 0 or neg_n == 0:
        return 0.0
    p_y = pos_n / (pos_n + neg_n)
    parent = entropy(p_y)
    tot = pos + neg
    if tot == 0 or tot == pos_n + neg_n:
        return 0.0
    p_y_t = pos / tot if tot else 0
    rem = pos_n + neg_n - tot
    p_y_f = (pos_n - pos) / rem if rem else 0
    p_t = tot / (pos_n + neg_n)
    return parent - p_t * entropy(p_y_t) - (1 - p_t) * entropy(p_y_f)


def median_split(vals: list[float]) -> float:
    return statistics.median(vals) if vals else 0.0


def load_picks() -> dict[str, set[str]]:
    picks: dict[str, set[str]] = defaultdict(set)
    if not P23_MATCH.exists():
        return picks
    for line in P23_MATCH.open(encoding="utf-8"):
        m = json.loads(line)
        for fid in FORMULAS:
            picks[m["scan_kst"]].update(m.get(f"{fid}_top2", []))
    return picks


def snapshot_at_bar(kl: list, bar: int) -> dict[str, float]:
    """Metric snapshot at 5m bar index (phase19-compatible, no new indicators)."""
    if bar < 1 or bar >= len(kl):
        bar = max(1, min(bar, len(kl) - 1))
    o, h, l, c, vol = ohlcv(kl[bar])
    vols = [ohlcv(kl[j])[4] for j in range(max(0, bar - 24), bar)]
    vol_ma = statistics.mean(vols[-20:]) if vols else vol
    closes = [float(kl[j][4]) for j in range(max(0, bar - 21), bar + 1)]
    ma20 = statistics.mean(closes[-20:]) if len(closes) >= 20 else c
    ret = (c - o) / o * 100 if o else 0.0
    rng = (h - l) / o * 100 if o else 0.0
    comp = float(compression_length(kl, bar))
    seq6 = window_seq(kl, bar, 6)
    return {
        "return": round(ret, 4),
        "ma_distance": round((c - ma20) / ma20 * 100 if ma20 else 0, 4),
        "volume": round(vol / vol_ma if vol_ma else 0, 4),
        "expansion": round(rng, 4),
        "compression": comp,
        "energy": round(float(seq6.get("seq_volume_energy_6", 0)), 4),
    }


def build_mtf_curve(kl: list, interval: str) -> dict[str, list[float]]:
    """Curve channels at TIME_MINUTES for given kline interval."""
    offsets = MTF_BAR_MAP[interval]
    anchor = len(kl) - 1
    curve: dict[str, list[float]] = {ch: [] for ch in CHANNELS}
    for tm in TIME_MINUTES:
        off = offsets[tm]
        bar = max(1, anchor - off)
        snap = snapshot_at_bar(kl, bar)
        for ch in CHANNELS:
            curve[ch].append(snap[ch])
    return curve


def build_curves_for_candidate(symbol: str, scan_kst: str) -> dict | None:
    p16.CACHE_DIR = P19_CACHE
    end_ms = int(parse_kst(scan_kst).timestamp() * 1000)
    try:
        k5 = fetch_klines(symbol, "5m", end_ms, 120)
        k15 = fetch_klines(symbol, "15m", end_ms, 96)
        k30 = fetch_klines(symbol, "30m", end_ms, 72)
        if len(k5) < 20:
            return None
        curves = {"5m": build_mtf_curve(k5, "5m")}
        if len(k15) >= 10:
            curves["15m"] = build_mtf_curve(k15, "15m")
        if len(k30) >= 6:
            curves["30m"] = build_mtf_curve(k30, "30m")
        return {"curves": curves, "time_minutes": list(TIME_MINUTES)}
    except Exception:
        return None


def finite_derivatives(series: list[float], times: list[float]) -> dict[str, list[float]]:
    """Level, velocity, acceleration, jerk along uneven time axis."""
    n = len(series)
    if n < 2:
        return {"level": series, "velocity": [], "acceleration": [], "jerk": []}
    vel: list[float] = []
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        vel.append((series[i] - series[i - 1]) / dt if dt else 0.0)
    acc: list[float] = []
    for i in range(1, len(vel)):
        dt = times[i + 1] - times[i] if i + 1 < n else times[-1] - times[-2]
        acc.append((vel[i] - vel[i - 1]) / dt if dt else 0.0)
    jerk: list[float] = []
    for i in range(1, len(acc)):
        dt = times[min(i + 2, n - 1)] - times[min(i + 1, n - 1)]
        jerk.append((acc[i] - acc[i - 1]) / dt if dt else 0.0)
    return {"level": series, "velocity": vel, "acceleration": acc, "jerk": jerk}


def trapezoid_integral(series: list[float], times: list[float]) -> list[float]:
    out: list[float] = []
    cum = 0.0
    for i in range(1, len(series)):
        dt = times[i] - times[i - 1]
        cum += (series[i] + series[i - 1]) * 0.5 * dt
        out.append(cum)
    return out


def curve_vector(curves: dict[str, list[float]], channels: tuple[str, ...] = CHANNELS) -> list[float]:
    vec: list[float] = []
    for ch in channels:
        vec.extend(curves.get(ch, [0.0] * len(TIME_MINUTES)))
    return vec


def euclidean_dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine_dist(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)


def dtw_dist(a: list[float], b: list[float]) -> float:
    n, m = len(a), len(b)
    dp = [[1e18] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return math.sqrt(dp[n][m])


def avg_curve(rows: list[dict], mtf: str = "5m") -> dict[str, list[float]]:
    if not rows:
        return {ch: [0.0] * len(TIME_MINUTES) for ch in CHANNELS}
    acc: dict[str, list[float]] = {ch: [0.0] * len(TIME_MINUTES) for ch in CHANNELS}
    n = 0
    for r in rows:
        c = r["curves"].get(mtf)
        if not c:
            continue
        n += 1
        for ch in CHANNELS:
            for i, v in enumerate(c[ch]):
                acc[ch][i] += v
    if n == 0:
        return acc
    return {ch: [round(v / n, 4) for v in vals] for ch, vals in acc.items()}


def flatten_avg(curve: dict[str, list[float]]) -> list[float]:
    return curve_vector(curve)


def process_candidate(row: dict, picks: dict[str, set[str]]) -> dict | None:
    curve_data = build_curves_for_candidate(row["symbol"], row["scan_kst"])
    if not curve_data:
        return None
    scan, sym = row["scan_kst"], row["symbol"]
    rank, mu = row["outcome_rank"], row["max_up_4h"]
    picked = sym in picks.get(scan, set())
    cohort = "other"
    if rank <= 2:
        cohort = "winner"
    if picked and mu < FP_THRESHOLD:
        cohort = "false_positive"
    elif rank <= 2 and not picked:
        cohort = "top2_miss"

    times = list(reversed(TIME_MINUTES))  # 5,10,15,30,60 chronological toward scan
    mtf = "5m"
    ch_curves = curve_data["curves"][mtf]
    deriv: dict[str, dict[str, list[float]]] = {}
    integral: dict[str, dict[str, list[float]]] = {}
    for ch in CHANNELS:
        s = list(reversed(ch_curves[ch]))
        deriv[ch] = finite_derivatives(s, times)
        integ = trapezoid_integral(s, times)
        integ_vel = finite_derivatives(integ, times[1:])["velocity"] if len(integ) > 1 else []
        integral[ch] = {"cumulative": integ, "velocity": integ_vel}

    return {
        "scan_kst": scan,
        "symbol": sym,
        "outcome_rank": rank,
        "max_up_4h": mu,
        "cohort": cohort,
        "curves": curve_data["curves"],
        "derivatives": deriv,
        "integrals": integral,
        "times": times,
    }


def load_or_build_rows(max_candidates: int | None, workers: int = 8) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks = load_picks()
    raw = p20.load_candidates()
    if max_candidates:
        raw = raw[:max_candidates]

    cached: dict[tuple[str, str], dict] = {}
    if CURVE_CACHE.exists():
        for line in CURVE_CACHE.open(encoding="utf-8"):
            r = json.loads(line)
            cached[(r["scan_kst"], r["symbol"])] = r

    rows: list[dict] = []
    todo = [r for r in raw if (r["scan_kst"], r["symbol"]) not in cached]
    rows.extend(cached.values())

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_candidate, r, picks): r for r in todo}
            done = 0
            for fut in as_completed(futs):
                res = fut.result()
                done += 1
                if res:
                    rows.append(res)
                    if done % 200 == 0:
                        safe_print(f"  curves built: {done}/{len(todo)}")
        with CURVE_CACHE.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")
    return rows


def feature_sep_ig(
    winners: list[dict],
    fps: list[dict],
    extractor,
) -> float:
    pos_vals = [extractor(r) for r in winners]
    neg_vals = [extractor(r) for r in fps]
    if not pos_vals or not neg_vals:
        return 0.0
    cut = median_split(pos_vals + neg_vals)
    ph = sum(1 for v in pos_vals if v >= cut)
    nh = sum(1 for v in neg_vals if v >= cut)
    return ig_binary(ph, len(pos_vals), nh, len(neg_vals))


def derivative_importance(winners: list[dict], fps: list[dict]) -> list[dict]:
    out: list[dict] = []
    orders = ("level", "velocity", "acceleration", "jerk")
    for ch in CHANNELS:
        for order in orders:
            for mtf in MTF_INTERVALS:
                def ext(r, _ch=ch, _ord=order, _mtf=mtf):
                    c = r["curves"].get(_mtf)
                    if not c:
                        return 0.0
                    if _ord == "level":
                        return statistics.mean(c[_ch])
                    d = r["derivatives"][_ch]
                    arr = d.get(_ord, [])
                    return statistics.mean(arr) if arr else 0.0

                ig_wf = feature_sep_ig(winners, fps, ext)
                out.append({
                    "channel": ch,
                    "order": order,
                    "mtf": mtf,
                    "information_gain": round(ig_wf, 4),
                    "metric": f"{mtf}_{ch}_{order}",
                })
    out.sort(key=lambda x: x["information_gain"], reverse=True)
    return out


def integral_importance(winners: list[dict], fps: list[dict]) -> list[dict]:
    out: list[dict] = []
    for ch in CHANNELS:
        for kind in ("cumulative", "velocity"):
            def ext(r, _ch=ch, _k=kind):
                integ = r["integrals"][_ch]
                arr = integ.get(_k, [])
                return arr[-1] if arr else 0.0

            ig_wf = feature_sep_ig(winners, fps, ext)
            out.append({
                "channel": ch,
                "integral_kind": kind,
                "information_gain": round(ig_wf, 4),
                "metric": f"5m_{ch}_integral_{kind}",
            })
    out.sort(key=lambda x: x["information_gain"], reverse=True)
    return out


def similarity_rows(
    rows: list[dict],
    w_avg: dict[str, list[float]],
    f_avg: dict[str, list[float]],
    m_avg: dict[str, list[float]],
    mtf: str = "5m",
) -> tuple[list[dict], str]:
    wv = flatten_avg(w_avg)
    fv = flatten_avg(f_avg)
    mv = flatten_avg(m_avg)
    methods = {
        "euclidean": euclidean_dist,
        "cosine": cosine_dist,
        "dtw": dtw_dist,
    }
    best_method = "euclidean"
    best_ig = -1.0
    winners = [r for r in rows if r["cohort"] == "winner"]
    fps = [r for r in rows if r["cohort"] == "false_positive"]

    for name, fn in methods.items():
        ig = feature_sep_ig(
            winners,
            fps,
            lambda r, _fn=fn, _wv=wv, _mtf=mtf: -_fn(flatten_avg(r["curves"][_mtf]), _wv),
        )
        if ig > best_ig:
            best_ig = ig
            best_method = name

    fn = methods[best_method]
    out: list[dict] = []
    for r in rows:
        cv = flatten_avg(r["curves"][mtf])
        out.append({
            "scan_kst": r["scan_kst"],
            "symbol": r["symbol"],
            "cohort": r["cohort"],
            "best_method": best_method,
            "dist_winner": round(fn(cv, wv), 4),
            "dist_fp": round(fn(cv, fv), 4),
            "dist_miss": round(fn(cv, mv), 4),
            "similarity_winner": round(1.0 / (1.0 + fn(cv, wv)), 4),
        })
    return out, best_method


def global_earliest_split(winners: list[dict], fps: list[dict]) -> list[dict]:
    """Per-minute earliest winner/FP separation on 5m curves."""
    rows: list[dict] = []
    for ti, tm in enumerate(TIME_MINUTES):
        best_ch, best_ig = "", 0.0
        for ch in CHANNELS:
            def ext(r, _ti=ti, _ch=ch):
                return r["curves"]["5m"][_ch][_ti]

            ig = feature_sep_ig(winners, fps, ext)
            if ig > best_ig:
                best_ig = ig
                best_ch = ch
        rows.append({
            "minute_before_scan": tm,
            "best_channel": best_ch,
            "information_gain": round(best_ig, 4),
        })
    return rows


def birth_window_analysis(winners: list[dict], fps: list[dict]) -> list[dict]:
    out: list[dict] = []
    for win_label, minutes in BIRTH_WINDOWS:
        idxs = [TIME_MINUTES.index(m) for m in minutes if m in TIME_MINUTES]
        best_ch = ""
        best_ig = 0.0
        best_t = 0
        for ch in CHANNELS:
            for ti in idxs:
                def ext(r, _ch=ch, _ti=ti):
                    return r["curves"]["5m"][_ch][_ti]

                ig = feature_sep_ig(winners, fps, ext)
                if ig > best_ig:
                    best_ig = ig
                    best_ch = ch
                    best_t = TIME_MINUTES[ti]
        out.append({
            "window": win_label,
            "minutes_span": "-".join(str(m) for m in minutes),
            "first_split_minute": best_t,
            "first_split_channel": best_ch,
            "information_gain": round(best_ig, 4),
        })
    return out


def mtf_early_rank(winners: list[dict], fps: list[dict]) -> list[dict]:
    out: list[dict] = []
    for mtf in MTF_INTERVALS:
        ig = feature_sep_ig(
            winners,
            fps,
            lambda r, _m=mtf: -euclidean_dist(
                flatten_avg(r["curves"][_m]) if _m in r["curves"] else [0.0],
                flatten_avg(avg_curve(winners, _m)),
            ) if _m in r["curves"] else 0.0,
        )
        out.append({
            "mtf": mtf,
            "winner_fp_ig": round(ig, 4),
            "earliest_rank": 0,
        })
    out.sort(key=lambda x: x["winner_fp_ig"], reverse=True)
    for i, r in enumerate(out):
        r["earliest_rank"] = i + 1
    return out


def early_curve_score(
    rows: list[dict],
    sim_rows: list[dict],
    deriv_rows: list[dict],
    birth_rows: list[dict],
    minute_splits: list[dict],
) -> list[dict]:
    sim_map = {(r["scan_kst"], r["symbol"]): r for r in sim_rows}
    top_deriv = deriv_rows[0]["information_gain"] if deriv_rows else 0.01
    best_min_ig = max(r["information_gain"] for r in minute_splits) if minute_splits else 0.01

    out: list[dict] = []
    for r in rows:
        key = (r["scan_kst"], r["symbol"])
        sim = sim_map.get(key, {})
        sim_part = sim.get("similarity_winner", 0.5)
        # prefer acceleration / velocity over static level
        ma_acc = r["derivatives"].get("ma_distance", {}).get("acceleration", [0.0])
        ret_vel = r["derivatives"].get("return", {}).get("velocity", [0.0])
        acc_part = min(abs(statistics.mean(ma_acc)) / max(top_deriv, 0.005), 1.0) if ma_acc else 0.0
        vel_part = min(abs(statistics.mean(ret_vel)) / 2.0, 1.0) if ret_vel else 0.0
        integ_vel = r["integrals"].get("volume", {}).get("velocity", [0.0])
        vol_integ = min(abs(integ_vel[-1]) / 15.0, 1.0) if integ_vel else 0.0
        # penalize flat compression-heavy late curves (FP pattern)
        comp_now = r["curves"]["5m"]["compression"][-1]
        exp_now = r["curves"]["5m"]["expansion"][-1]
        flat_pen = min(comp_now / 15.0, 1.0) * 0.5 + (1.0 - min(exp_now / 5.0, 1.0)) * 0.3
        raw = (
            0.30 * sim_part
            + 0.25 * acc_part
            + 0.20 * vel_part
            + 0.15 * vol_integ
            - 0.10 * flat_pen
        )
        score = round(max(0.0, min(100.0, raw * 100)), 2)
        out.append({
            "scan_kst": r["scan_kst"],
            "symbol": r["symbol"],
            "cohort": r["cohort"],
            "early_curve_score": score,
            "similarity_winner": sim.get("similarity_winner", 0),
            "dist_winner": sim.get("dist_winner", 0),
            "accel_component": round(acc_part * 100, 2),
            "flat_penalty": round(flat_pen * 100, 2),
            "outcome_rank": r["outcome_rank"],
        })
    out.sort(key=lambda x: x["early_curve_score"], reverse=True)
    return out


def avg_curve_csv(curve: dict[str, list[float]], label: str) -> list[dict]:
    rows: list[dict] = []
    for i, tm in enumerate(TIME_MINUTES):
        row: dict = {"cohort": label, "minute_before_scan": tm}
        for ch in CHANNELS:
            row[ch] = curve[ch][i]
        rows.append(row)
    return rows


def run(max_candidates: int | None = None) -> None:
    safe_print("Building curve dataset (kline cache)...")
    rows = load_or_build_rows(max_candidates)
    if not rows:
        raise SystemExit("No curve rows built")

    winners = [r for r in rows if r["cohort"] == "winner"]
    fps = [r for r in rows if r["cohort"] == "false_positive"]
    misses = [r for r in rows if r["cohort"] == "top2_miss"]

    w_avg = avg_curve(winners, "5m")
    f_avg = avg_curve(fps, "5m")
    m_avg = avg_curve(misses, "5m")

    deriv_imp = derivative_importance(winners, fps)
    integ_imp = integral_importance(winners, fps)
    sim_rows, best_sim = similarity_rows(rows, w_avg, f_avg, m_avg, "5m")
    birth_rows = birth_window_analysis(winners, fps)
    minute_splits = global_earliest_split(winners, fps)
    mtf_rank = mtf_early_rank(winners, fps)
    ecs = early_curve_score(rows, sim_rows, deriv_imp, birth_rows, minute_splits)

    write_csv(OUT_DIR / "curve_similarity.csv", sim_rows)
    write_csv(OUT_DIR / "winner_average_curve.csv", avg_curve_csv(w_avg, "winner"))
    write_csv(OUT_DIR / "falsepositive_average_curve.csv", avg_curve_csv(f_avg, "false_positive"))
    write_csv(OUT_DIR / "miss_average_curve.csv", avg_curve_csv(m_avg, "top2_miss"))
    write_csv(OUT_DIR / "derivative_importance.csv", deriv_imp)
    write_csv(OUT_DIR / "integral_importance.csv", integ_imp)
    write_csv(OUT_DIR / "birth_window_analysis.csv", birth_rows + [
        {**r, "window": "per_minute"} for r in minute_splits
    ])
    write_csv(OUT_DIR / "early_curve_score.csv", ecs)

    # score separation
    w_scores = [r["early_curve_score"] for r in ecs if r["cohort"] == "winner"]
    f_scores = [r["early_curve_score"] for r in ecs if r["cohort"] == "false_positive"]
    w_mean = statistics.mean(w_scores) if w_scores else 0
    f_mean = statistics.mean(f_scores) if f_scores else 0

    lines = [
        "############################################################",
        "SCOUT PHASE 28 - ACCELERATION CURVE LAB",
        "############################################################",
        "",
        "Analysis only. No formula/trigger/threshold/weight/ranking changes.",
        f"Curve rows: {len(rows)} | winner={len(winners)} fp={len(fps)} miss={len(misses)}",
        f"Time axis (min before scan): {list(TIME_MINUTES)}",
        f"Channels: {', '.join(CHANNELS)}",
        "",
        "=" * 62,
        "STEP 1-2 — CURVE DATASET",
        "=" * 62,
        "  5m/15m/30m klines from phase19 cache; 6 channels x 5 time points",
        "",
        "=" * 62,
        "STEP 3 — DERIVATIVE LAB (winner vs FP IG)",
        "=" * 62,
    ]
    for r in deriv_imp[:8]:
        lines.append(f"  {r['metric']}: IG={r['information_gain']:.3f}")

    lines.extend(["", "=" * 62, "STEP 4 — INTEGRAL LAB", "=" * 62])
    for r in integ_imp[:6]:
        lines.append(f"  {r['metric']}: IG={r['information_gain']:.3f}")

    lines.extend(["", "=" * 62, "STEP 5 — CURVE SIMILARITY", "=" * 62])
    lines.append(f"  Best method: {best_sim}")
    for method in ("euclidean", "cosine", "dtw"):
        ig = feature_sep_ig(
            winners, fps,
            lambda r, m=method: -{"euclidean": euclidean_dist, "cosine": cosine_dist, "dtw": dtw_dist}[m](
                flatten_avg(r["curves"]["5m"]), flatten_avg(w_avg)),
        )
        lines.append(f"    {method}: IG={ig:.3f}")

    lines.extend(["", "=" * 62, "STEP 6 — BIRTH WINDOW (earliest split)", "=" * 62])
    for b in birth_rows:
        lines.append(
            f"  {b['window']}: split@{b['first_split_minute']}m ch={b['first_split_channel']} "
            f"IG={b['information_gain']:.3f}"
        )
    lines.append("  Per-minute split (5m curve):")
    for m in minute_splits:
        lines.append(f"    @{m['minute_before_scan']}m: {m['best_channel']} IG={m['information_gain']:.3f}")
    if minute_splits:
        strongest = max(minute_splits, key=lambda x: x["information_gain"])
        earliest_ig = [m for m in minute_splits if m["information_gain"] >= strongest["information_gain"] * 0.8]
        fastest = min(earliest_ig, key=lambda x: x["minute_before_scan"])
        lines.append(
            f"  Fastest branch: {fastest['minute_before_scan']}m ({fastest['best_channel']}) "
            f"IG={fastest['information_gain']:.3f}"
        )

    lines.extend(["", "=" * 62, "STEP 7 — MULTI-TF CURVE RANK", "=" * 62])
    for m in mtf_rank:
        lines.append(f"  {m['mtf']}: IG={m['winner_fp_ig']:.3f} rank=#{m['earliest_rank']}")

    lines.extend(["", "=" * 62, "STEP 8 — EARLY CURVE SCORE (0-100, not ranked)", "=" * 62])
    lines.append(f"  Winner mean score: {w_mean:.1f}")
    lines.append(f"  FP mean score:     {f_mean:.1f}")
    lines.append(f"  Separation delta:  {w_mean - f_mean:+.1f}")
    if ecs[:3]:
        lines.append("  Top3 by early_curve_score:")
        for r in ecs[:3]:
            lines.append(f"    {r['symbol']} score={r['early_curve_score']:.1f} cohort={r['cohort']}")

    lines.extend(["", "=" * 62, "TRACK INDEPENDENCE", "=" * 62])
    lines.append("  Track A Formula / B State / C Future / D Trigger — unchanged, no merge.")
    lines.append("")
    lines.append("DISCLAIMER: Curve dynamics descriptive — not price prediction.")

    report = OUT_DIR / "phase28_curve_lab_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for ln in lines:
        safe_print(ln)
    safe_print(f"\nSaved: {report}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run(args.max_candidates)
