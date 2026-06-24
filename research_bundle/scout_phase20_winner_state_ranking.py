"""
Scout Phase 20 - Winner State Ranking Engine

State-based ranking prototype from Phase 19 dataset.
NO filter/threshold/weight/rule changes to Pattern B.

Usage:
  python scout_phase20_winner_state_ranking.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from season2_p37_scout_decision_hierarchy import write_csv

P19_DIR = Path("logs") / "phase19_winner_dna"
CANDIDATES_PATH = P19_DIR / "candidates.jsonl"
OUT_DIR = Path("logs") / "phase20_state_ranking"

WINNER_TOP_N = 3
LOSER_BOTTOM_N = 3

STATES_5M = ("Quiet", "Normal", "SequenceStrong", "MomentumStrong", "Release")
STATES_15M = ("Weak", "VolumeSupport", "Expansion")
STATES_30M = ("Compression", "Neutral", "Expansion")
STATES_1H = ("Flat", "ExpansionStart", "Expansion", "Acceleration")
STATES_2H = ("Flat", "TrendAlive", "StrongTrend", "OverExtended")


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def load_candidates() -> list[dict]:
    rows: list[dict] = []
    for line in CANDIDATES_PATH.open(encoding="utf-8"):
        rows.append(json.loads(line))
    return rows


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = (len(s) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    w = idx - lo
    return s[lo] * (1 - w) + s[hi] * w


@dataclass
class Thresholds:
    """Data-derived cut points from winner pool (top-3 per scan)."""
    p25: dict[str, float] = field(default_factory=dict)
    p50: dict[str, float] = field(default_factory=dict)
    p75: dict[str, float] = field(default_factory=dict)
    p90: dict[str, float] = field(default_factory=dict)


def build_thresholds(winner_feats: list[dict]) -> Thresholds:
    keys = set()
    for f in winner_feats:
        keys.update(f.keys())
    th = Thresholds()
    for k in keys:
        vals = [float(f[k]) for f in winner_feats if k in f]
        if len(vals) < 20:
            continue
        th.p25[k] = percentile(vals, 0.25)
        th.p50[k] = percentile(vals, 0.50)
        th.p75[k] = percentile(vals, 0.75)
        th.p90[k] = percentile(vals, 0.90)
    return th


def g(f: dict, key: str, default: float = 0.0) -> float:
    return float(f.get(key, default))


def classify_5m(f: dict, th: Thresholds) -> str:
    release = g(f, "5m_release")
    comp = g(f, "5m_compression")
    mom = g(f, "5m_momentum")
    vol_e = g(f, "5m_seq_volume_energy_6")
    ret6 = g(f, "5m_seq_return_sum_6")
    rng_e = g(f, "5m_range_energy")

    if release > 0 or comp <= th.p25.get("5m_compression", 3):
        return "Release"
    if mom >= th.p75.get("5m_momentum", 0.5):
        return "MomentumStrong"
    if vol_e >= th.p75.get("5m_seq_volume_energy_6", 3) or ret6 >= th.p75.get("5m_seq_return_sum_6", 0.3):
        return "SequenceStrong"
    if vol_e <= th.p25.get("5m_seq_volume_energy_6", 1) and mom <= th.p25.get("5m_momentum", 0) and rng_e <= th.p25.get("5m_range_energy", 2):
        return "Quiet"
    return "Normal"


def classify_15m(f: dict, th: Thresholds, side: str = "current") -> str:
    vol = g(f, f"15m_{side}_volume_ratio")
    ret = g(f, f"15m_{side}_return_pct")
    body = g(f, f"15m_{side}_body_pct")
    rng = g(f, f"15m_{side}_range_pct")

    if ret >= th.p75.get(f"15m_{side}_return_pct", 1.0) or body >= th.p75.get(f"15m_{side}_body_pct", 1.5):
        return "Expansion"
    if vol < th.p25.get(f"15m_{side}_volume_ratio", 0.8) and ret <= 0:
        return "Weak"
    if vol >= th.p50.get(f"15m_{side}_volume_ratio", 1.0) or rng >= th.p50.get(f"15m_{side}_range_pct", 2.0):
        return "VolumeSupport"
    return "Weak" if ret < 0 else "VolumeSupport"


def classify_30m(f: dict, th: Thresholds, side: str = "current") -> str:
    comp = g(f, f"30m_{side}_compression")
    ret = g(f, f"30m_{side}_return_pct")
    rng = g(f, f"30m_{side}_range_pct")

    if comp >= th.p75.get(f"30m_{side}_compression", 4) or (
        rng <= th.p25.get(f"30m_{side}_range_pct", 2.0) and comp >= th.p50.get(f"30m_{side}_compression", 2)
    ):
        return "Compression"
    if ret > 0 and rng >= th.p75.get(f"30m_{side}_range_pct", 4.0):
        return "Expansion"
    return "Neutral"


def classify_1h(f: dict, th: Thresholds, side: str = "current") -> str:
    ret = g(f, f"1h_{side}_return_pct")
    prev_ret = g(f, "1h_previous_return_pct")
    rng = g(f, f"1h_{side}_range_pct")
    flat_cut = th.p25.get("1h_current_return_pct", 0.5)
    flat_cut = max(abs(flat_cut), 0.3)

    if side == "current" and ret > prev_ret and ret > 0 and prev_ret >= -flat_cut:
        if ret - prev_ret >= th.p50.get("1h_current_return_pct", 1.0) * 0.3:
            return "Acceleration"
    if ret >= th.p75.get(f"1h_{side}_return_pct", 2.0) and rng >= th.p50.get(f"1h_{side}_range_pct", 5.0):
        return "Expansion"
    if side == "current" and ret > 0 and abs(prev_ret) <= flat_cut:
        return "ExpansionStart"
    if abs(ret) <= flat_cut:
        return "Flat"
    if ret > 0:
        return "ExpansionStart"
    return "Flat"


def classify_2h(f: dict, th: Thresholds, side: str = "current") -> str:
    dist = g(f, f"2h_{side}_ma20_distance_pct")
    ret = g(f, f"2h_{side}_return_pct")
    rng = g(f, f"2h_{side}_range_pct")

    if dist >= th.p90.get(f"2h_{side}_ma20_distance_pct", 15):
        return "OverExtended"
    if rng >= th.p75.get(f"2h_{side}_range_pct", 8.0) and ret > 0:
        return "StrongTrend"
    if ret > 0 or dist >= th.p50.get(f"2h_{side}_ma20_distance_pct", 5.0):
        return "TrendAlive"
    return "Flat"


def infer_5m_prev(f: dict, th: Thresholds) -> str:
    comp = g(f, "5m_compression")
    vol_e = g(f, "5m_seq_volume_energy_6")
    if comp >= th.p75.get("5m_compression", 8):
        return "Quiet"
    if vol_e >= th.p50.get("5m_seq_volume_energy_6", 2):
        return "SequenceStrong"
    if vol_e <= th.p25.get("5m_seq_volume_energy_6", 1):
        return "Quiet"
    return "Normal"


def build_states(f: dict, th: Thresholds) -> dict[str, str]:
    return {
        "5m": classify_5m(f, th),
        "15m": classify_15m(f, th, "current"),
        "30m": classify_30m(f, th, "current"),
        "1h": classify_1h(f, th, "current"),
        "2h": classify_2h(f, th, "current"),
    }


def build_transitions(f: dict, th: Thresholds) -> dict[str, str]:
    s5_cur = classify_5m(f, th)
    s5_prev = infer_5m_prev(f, th)
    out = {"5m": f"{s5_prev}->{s5_cur}"}
    for tf, fn in (
        ("15m", classify_15m),
        ("30m", classify_30m),
        ("1h", classify_1h),
        ("2h", classify_2h),
    ):
        prev = fn(f, th, "previous")
        cur = fn(f, th, "current")
        out[tf] = f"{prev}->{cur}"
    return out


def combo_key(states: dict[str, str]) -> str:
    return "|".join(f"{k}={states[k]}" for k in ("5m", "15m", "30m", "1h", "2h"))


def entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def separation_row(
    label: str,
    w_count: int,
    w_total: int,
    l_count: int,
    l_total: int,
    all_count: int,
    all_total: int,
) -> dict:
    w_rate = w_count / w_total if w_total else 0
    l_rate = l_count / l_total if l_total else 0
    diff = w_rate - l_rate
    base = all_count / all_total if all_total else 0
    lift = w_rate / l_rate if l_rate > 0 else (99.0 if w_rate > 0 else 1.0)
    ow = w_rate / (1 - w_rate) if w_rate < 1 else 99.0
    ol = l_rate / (1 - l_rate) if l_rate < 1 else 99.0
    odds_ratio = ow / ol if ol > 0 else (99.0 if ow > 0 else 1.0)

    p_y = w_total / (w_total + l_total) if (w_total + l_total) else 0.5
    p_n = 1 - p_y
    parent_h = entropy(p_y)
    p_t = (w_count + l_count) / (w_total + l_total) if (w_total + l_total) else 0
    if w_count + l_count == 0 or w_count + l_count == w_total + l_total:
        ig = 0.0
    else:
        p_y_t = w_count / (w_count + l_count) if (w_count + l_count) else 0
        p_y_not = (w_total - w_count) / (w_total + l_total - w_count - l_count) if (w_total + l_total - w_count - l_count) else 0
        ig = parent_h - p_t * entropy(p_y_t) - (1 - p_t) * entropy(p_y_not)

    return {
        "label": label,
        "winner_count": w_count,
        "winner_rate": round(w_rate, 4),
        "loser_count": l_count,
        "loser_rate": round(l_rate, 4),
        "difference": round(diff, 4),
        "information_gain": round(ig, 4),
        "lift": round(lift, 4),
        "odds_ratio": round(odds_ratio, 4),
        "base_rate": round(base, 4),
    }


def annotate(rows: list[dict], th: Thresholds) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        f = r["features"]
        states = build_states(f, th)
        transitions = build_transitions(f, th)
        out.append({
            **r,
            "states": states,
            "transitions": transitions,
            "combo": combo_key(states),
            "transition_combo": "|".join(f"{k}:{transitions[k]}" for k in ("5m", "15m", "30m", "1h", "2h")),
        })
    return out


def winner_loser_sets(by_scan: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    winners: list[dict] = []
    losers: list[dict] = []
    for rows in by_scan.values():
        if len(rows) < 4:
            continue
        n = len(rows)
        winners.extend(rows[: min(WINNER_TOP_N, n)])
        losers.extend(rows[-min(LOSER_BOTTOM_N, n):])
    return winners, losers


def state_match_score(states: dict[str, str], trans: dict[str, str], profile: dict) -> float:
    """Empirical log-lift sum from winner vs all frequencies (no hand weights)."""
    score = 0.0
    for tf in ("5m", "15m", "30m", "1h", "2h"):
        s = states[tf]
        key = f"{tf}:{s}"
        lift = profile["state_lift"].get(key, 1.0)
        score += math.log(max(lift, 0.01))
        tkey = f"{tf}:{trans[tf]}"
        tlift = profile["trans_lift"].get(tkey, 1.0)
        score += 0.5 * math.log(max(tlift, 0.01))
    cluster = combo_key(states)
    clift = profile["cluster_lift"].get(cluster, 1.0)
    score += math.log(max(clift, 0.01))
    return score


def build_profile(winners: list[dict], all_rows: list[dict]) -> dict:
    w_total = len(winners)
    a_total = len(all_rows)
    state_w: Counter[str] = Counter()
    state_a: Counter[str] = Counter()
    trans_w: Counter[str] = Counter()
    trans_a: Counter[str] = Counter()
    cluster_w: Counter[str] = Counter()
    cluster_a: Counter[str] = Counter()

    for r in winners:
        for tf, s in r["states"].items():
            state_w[f"{tf}:{s}"] += 1
        for tf, t in r["transitions"].items():
            trans_w[f"{tf}:{t}"] += 1
        cluster_w[r["combo"]] += 1

    for r in all_rows:
        for tf, s in r["states"].items():
            state_a[f"{tf}:{s}"] += 1
        for tf, t in r["transitions"].items():
            trans_a[f"{tf}:{t}"] += 1
        cluster_a[r["combo"]] += 1

    def lifts(wc: Counter, ac: Counter) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in set(wc) | set(ac):
            wr = wc[k] / w_total if w_total else 0
            ar = ac[k] / a_total if a_total else 0
            out[k] = wr / ar if ar > 0 else (wr * a_total + 1)
        return out

    return {
        "state_lift": lifts(state_w, state_a),
        "trans_lift": lifts(trans_w, trans_a),
        "cluster_lift": lifts(cluster_w, cluster_a),
        "state_w_rate": {k: v / w_total for k, v in state_w.items()},
    }


def backtest_scan(rows: list[dict], profile: dict) -> dict:
    for r in rows:
        r["state_score"] = state_match_score(r["states"], r["transitions"], profile)

    by_outcome = sorted(rows, key=lambda x: x["outcome_rank"])
    actual_top2 = {r["symbol"] for r in by_outcome[:2]}
    actual_top5 = {r["symbol"] for r in by_outcome[:5]}

    h4_sorted = sorted(rows, key=lambda x: g(x["features"], "h4_score"), reverse=True)
    state_sorted = sorted(rows, key=lambda x: x["state_score"], reverse=True)

    h4_top2 = {r["symbol"] for r in h4_sorted[:2]}
    h4_top5 = {r["symbol"] for r in h4_sorted[:5]}
    st_top2 = {r["symbol"] for r in state_sorted[:2]}
    st_top5 = {r["symbol"] for r in state_sorted[:5]}

    def metrics(picked: list[dict]) -> dict:
        mus = [r["max_up_4h"] for r in picked]
        return {
            "avg_max_up": round(statistics.mean(mus), 4) if mus else 0,
            "median_max_up": round(statistics.median(mus), 4) if mus else 0,
        }

    return {
        "n": len(rows),
        "h4_top2_hit": len(h4_top2 & actual_top2),
        "state_top2_hit": len(st_top2 & actual_top2),
        "h4_top5_hit": len(h4_top5 & actual_top5),
        "state_top5_hit": len(st_top5 & actual_top5),
        "h4_top2": metrics(h4_sorted[:2]),
        "state_top2": metrics(state_sorted[:2]),
        "h4_top5": metrics(h4_sorted[:5]),
        "state_top5": metrics(state_sorted[:5]),
        "h4_rank_top1_actual": next((r["outcome_rank"] for r in rows if r["symbol"] == h4_sorted[0]["symbol"]), None),
        "state_rank_top1_actual": next((r["outcome_rank"] for r in rows if r["symbol"] == state_sorted[0]["symbol"]), None),
    }


def pairwise_scan(rows: list[dict]) -> list[dict]:
    if len(rows) < 3:
        return []
    top1, top2, bottom = rows[0], rows[1], rows[-1]
    pairs = [
        ("winner_vs_runnerup", top1, top2),
        ("winner_vs_bottom", top1, bottom),
        ("runnerup_vs_bottom", top2, bottom),
    ]
    out: list[dict] = []
    for tag, a, b in pairs:
        for tf in ("5m", "15m", "30m", "1h", "2h"):
            if a["states"][tf] != b["states"][tf]:
                out.append({
                    "pair": tag,
                    "tf": tf,
                    "winner_side": a["states"][tf],
                    "other_side": b["states"][tf],
                    "max_up_a": a["max_up_4h"],
                    "max_up_b": b["max_up_4h"],
                })
    return out


def run() -> list[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_candidates()
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by_scan[r["scan_kst"]].append(r)
    for scan in by_scan:
        by_scan[scan].sort(key=lambda x: x["outcome_rank"])

    winner_feats = [r["features"] for rows in by_scan.values() for r in rows[:WINNER_TOP_N] if len(rows) >= 4]
    th = build_thresholds(winner_feats)
    annotated = annotate(raw, th)
    ann_by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in annotated:
        ann_by_scan[r["scan_kst"]].append(r)
    for scan in ann_by_scan:
        ann_by_scan[scan].sort(key=lambda x: x["outcome_rank"])

    winners, losers = winner_loser_sets(ann_by_scan)
    all_rows = annotated
    profile = build_profile(winners, all_rows)

    # --- separation for atomic states ---
    sep_rows: list[dict] = []
    for tf in ("5m", "15m", "30m", "1h", "2h"):
        state_set = {
            "5m": STATES_5M,
            "15m": STATES_15M,
            "30m": STATES_30M,
            "1h": STATES_1H,
            "2h": STATES_2H,
        }[tf]
        for st in state_set:
            label = f"{tf}:{st}"
            wc = sum(1 for r in winners if r["states"][tf] == st)
            lc = sum(1 for r in losers if r["states"][tf] == st)
            ac = sum(1 for r in all_rows if r["states"][tf] == st)
            sep_rows.append(separation_row(label, wc, len(winners), lc, len(losers), ac, len(all_rows)))
    sep_rows.sort(key=lambda x: (x["information_gain"], x["difference"]), reverse=True)

    # --- transitions separation ---
    trans_sep: list[dict] = []
    trans_w: Counter[str] = Counter()
    trans_l: Counter[str] = Counter()
    trans_a: Counter[str] = Counter()
    for r in winners:
        for tf, t in r["transitions"].items():
            trans_w[f"{tf}:{t}"] += 1
    for r in losers:
        for tf, t in r["transitions"].items():
            trans_l[f"{tf}:{t}"] += 1
    for r in all_rows:
        for tf, t in r["transitions"].items():
            trans_a[f"{tf}:{t}"] += 1
    for key in trans_a:
        tf, _ = key.split(":", 1)
        wc = trans_w[key]
        lc = trans_l[key]
        ac = trans_a[key]
        trans_sep.append(separation_row(key, wc, len(winners), lc, len(losers), ac, len(all_rows)))
    trans_sep.sort(key=lambda x: (x["information_gain"], x["difference"]), reverse=True)

    # --- clusters ---
    w_cluster: Counter[str] = Counter(r["combo"] for r in winners)
    l_cluster: Counter[str] = Counter(r["combo"] for r in losers)

    # --- pairwise ---
    pw_diffs: Counter[str] = Counter()
    pw_n = 0
    for rows in ann_by_scan.values():
        if len(rows) < 4:
            continue
        for item in pairwise_scan(rows):
            pw_n += 1
            k = f"{item['pair']}|{item['tf']}|{item['winner_side']}>{item['other_side']}"
            pw_diffs[k] += 1

    # --- backtest (leave-one-scan-out profile for honest prototype) ---
    bt_scans: list[dict] = []
    for scan, rows in ann_by_scan.items():
        if len(rows) < 4:
            continue
        train = [r for r in annotated if r["scan_kst"] != scan]
        w_train, _ = winner_loser_sets(defaultdict(list, {s: [x for x in train if x["scan_kst"] == s] for s in {x["scan_kst"] for x in train}}))
        if len(w_train) < 30:
            prof = profile
        else:
            prof = build_profile(w_train, train)
        bt_scans.append(backtest_scan(rows, prof))

    def agg(key: str) -> float:
        return statistics.mean([b[key] for b in bt_scans]) if bt_scans else 0

    def agg_nested(a: str, b: str) -> float:
        return statistics.mean([x[a][b] for x in bt_scans]) if bt_scans else 0

    h4_top2_rate = agg("h4_top2_hit") / 2 * 100
    st_top2_rate = agg("state_top2_hit") / 2 * 100
    h4_top5_rate = agg("h4_top5_hit") / 5 * 100
    st_top5_rate = agg("state_top5_hit") / 5 * 100

    h4_rank1_actual = statistics.median([b["h4_rank_top1_actual"] for b in bt_scans if b["h4_rank_top1_actual"]])
    st_rank1_actual = statistics.median([b["state_rank_top1_actual"] for b in bt_scans if b["state_rank_top1_actual"]])

    improved_top2 = st_top2_rate > h4_top2_rate
    improved_top5 = st_top5_rate > h4_top5_rate
    if improved_top2 and improved_top5:
        verdict = "KEEP"
    elif improved_top2 or improved_top5 or st_rank1_actual < h4_rank1_actual:
        verdict = "MODIFY"
    else:
        verdict = "DISCARD"

    # --- winner/loser state tops ---
    def top_states(counter: Counter[str], total: int, n: int = 10) -> list[tuple[str, float]]:
        return [(k, round(v / total * 100, 1)) for k, v in counter.most_common(n)]

    w_state_ctr: Counter[str] = Counter()
    l_state_ctr: Counter[str] = Counter()
    for r in winners:
        for tf, s in r["states"].items():
            w_state_ctr[f"{tf}:{s}"] += 1
    for r in losers:
        for tf, s in r["states"].items():
            l_state_ctr[f"{tf}:{s}"] += 1

    lines = [
        "############################################################",
        "SCOUT PHASE 20 - WINNER STATE RANKING ENGINE",
        "############################################################",
        "",
        f"Input: Phase19 | {len(by_scan)} scans | {len(all_rows)} candidates",
        "Pattern B frozen | State ranking prototype only",
        "",
        "=" * 60,
        "1. WINNER STATE TOP10",
        "=" * 60,
    ]
    for k, pct in top_states(w_state_ctr, len(winners)):
        lines.append(f"  {k}: {pct}%")

    lines.extend(["", "=" * 60, "2. LOSER STATE TOP10", "=" * 60])
    for k, pct in top_states(l_state_ctr, len(losers)):
        lines.append(f"  {k}: {pct}%")

    lines.extend(["", "=" * 60, "3. STATE TRANSITION TOP10 (winner-enriched)", "=" * 60])
    for row in trans_sep[:10]:
        lines.append(
            f"  {row['label']}: win%={row['winner_rate']*100:.1f} "
            f"lose%={row['loser_rate']*100:.1f} IG={row['information_gain']:.3f} lift={row['lift']:.2f}"
        )

    lines.extend(["", "=" * 60, "4. WINNER CLUSTERS", "=" * 60])
    for combo, cnt in w_cluster.most_common(8):
        pct = cnt / len(winners) * 100
        parts = combo.replace("|", " | ")
        lines.append(f"  [{cnt}x / {pct:.1f}%] {parts}")

    lines.extend(["", "=" * 60, "5. LOSER CLUSTERS", "=" * 60])
    for combo, cnt in l_cluster.most_common(8):
        pct = cnt / len(losers) * 100
        lines.append(f"  [{cnt}x / {pct:.1f}%] {combo.replace('|', ' | ')}")

    lines.extend(["", "=" * 60, "6. STATE SEPARATION RANKING", "=" * 60])
    lines.append("  label | win_rate | lose_rate | diff | IG | lift | odds_ratio")
    for row in sep_rows[:15]:
        lines.append(
            f"  {row['label']} | {row['winner_rate']:.3f} | {row['loser_rate']:.3f} | "
            f"{row['difference']:+.3f} | {row['information_gain']:.3f} | {row['lift']:.2f} | {row['odds_ratio']:.2f}"
        )

    lines.extend(["", "=" * 60, "7. STATE MATCH SCORE DEFINITION", "=" * 60])
    lines.append("  score(candidate) = sum_tf log(lift(state_tf)) + 0.5*sum_tf log(lift(transition_tf)) + log(lift(cluster_combo))")
    lines.append("  lift(x) = P(x|winner_pool) / P(x|all_candidates)  [empirical frequencies only]")
    lines.append("  No hand-tuned weights. Thresholds = winner-pool percentiles (p25/p50/p75/p90).")
    top_lifts = sorted(profile["state_lift"].items(), key=lambda x: x[1], reverse=True)[:8]
    lines.append("  Top empirical state lifts (winner-enriched):")
    for k, v in top_lifts:
        lines.append(f"    {k}: lift={v:.2f}")

    lines.extend(["", "=" * 60, "8. BASELINE vs STATE RANKING (LOO backtest)", "=" * 60])
    lines.append(f"  Scans evaluated: {len(bt_scans)}")
    lines.append(f"  TOP2 hit rate:  H4={h4_top2_rate:.1f}%  State={st_top2_rate:.1f}%  delta={st_top2_rate-h4_top2_rate:+.1f}pp")
    lines.append(f"  TOP5 hit rate:  H4={h4_top5_rate:.1f}%  State={st_top5_rate:.1f}%  delta={st_top5_rate-h4_top5_rate:+.1f}pp")
    lines.append(f"  Avg max_up TOP2: H4={agg_nested('h4_top2','avg_max_up'):.2f}%  State={agg_nested('state_top2','avg_max_up'):.2f}%")
    lines.append(f"  Median max_up TOP2: H4={agg_nested('h4_top2','median_max_up'):.2f}%  State={agg_nested('state_top2','median_max_up'):.2f}%")
    lines.append(f"  Median actual rank of rank#1 pick: H4={h4_rank1_actual:.1f}  State={st_rank1_actual:.1f}")
    lines.append("  Note: return_4h / MDD not in Phase19 cache; max_up_4h used as outcome proxy.")

    lines.extend(["", "=" * 60, "9. BLIND RESULT (in-period LOO prototype)", "=" * 60])
    lines.append("  Same period 2026-06-01~15; profile trained on other scans per fold.")
    lines.append(f"  State ranking {'improves' if improved_top2 else 'does not improve'} TOP2 vs frozen H4.")
    lines.append(f"  State ranking {'improves' if improved_top5 else 'does not improve'} TOP5 vs frozen H4.")

    lines.extend(["", "=" * 60, "10. VERDICT", "=" * 60])
    lines.append(f"  {verdict}")
    if verdict == "KEEP":
        lines.append("  State match score improves leader selection vs H4 on this period.")
    elif verdict == "MODIFY":
        lines.append("  Partial separation signal; refine state boundaries or transition encoding before production.")
    else:
        lines.append("  State prototype does not beat H4 baseline on TOP2/TOP5 in LOO backtest.")

    lines.extend(["", "=" * 60, "PAIRWISE STATE DIFFS (why winner rose more)", "=" * 60])
    for k, cnt in pw_diffs.most_common(12):
        lines.append(f"  [{cnt}x] {k.replace('|', ' ')}")

    lines.extend(["", "DISCLAIMER: Descriptive state DNA + ranking prototype. No filter/rule changes."])

    write_csv(OUT_DIR / "state_separation.csv", sep_rows)
    write_csv(OUT_DIR / "transition_separation.csv", trans_sep)
    write_csv(OUT_DIR / "winner_clusters.csv", [
        {"cluster": k, "count": v, "pct": round(v / len(winners) * 100, 2)} for k, v in w_cluster.most_common(30)
    ])
    write_csv(OUT_DIR / "loser_clusters.csv", [
        {"cluster": k, "count": v, "pct": round(v / len(losers) * 100, 2)} for k, v in l_cluster.most_common(30)
    ])
    write_csv(OUT_DIR / "backtest_by_scan.csv", bt_scans)

    report = OUT_DIR / "phase20_state_ranking_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    return lines


def main() -> None:
    lines = run()
    for ln in lines[:20]:
        safe_print(ln)
    safe_print("...")
    for ln in lines[-25:]:
        safe_print(ln)
    safe_print(f"\nSaved: {OUT_DIR / 'phase20_state_ranking_report.txt'}")


if __name__ == "__main__":
    main()
