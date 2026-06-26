"""
A6 state classification and profile core.
Original source: scout_phase20_winner_state_ranking.py
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from export_a6_core_for_hong.a6_common import WINNER_TOP_N, g, percentile

STATES_5M = ("Quiet", "Normal", "SequenceStrong", "MomentumStrong", "Release")
STATES_15M = ("Weak", "VolumeSupport", "Expansion")
STATES_30M = ("Compression", "Neutral", "Expansion")
STATES_1H = ("Flat", "ExpansionStart", "Expansion", "Acceleration")
STATES_2H = ("Flat", "TrendAlive", "StrongTrend", "OverExtended")


@dataclass
class Thresholds:
    # Original: scout_phase20_winner_state_ranking.Thresholds
    p25: dict[str, float] = field(default_factory=dict)
    p50: dict[str, float] = field(default_factory=dict)
    p75: dict[str, float] = field(default_factory=dict)
    p90: dict[str, float] = field(default_factory=dict)


def build_thresholds(winner_feats: list[dict]) -> Thresholds:
    # Original: scout_phase20_winner_state_ranking.build_thresholds
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


def classify_5m(f: dict, th: Thresholds) -> str:
    # Original: scout_phase20_winner_state_ranking.classify_5m
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
    # Original: scout_phase20_winner_state_ranking.classify_15m
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
    # Original: scout_phase20_winner_state_ranking.classify_30m
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
    # Original: scout_phase20_winner_state_ranking.classify_1h
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
    # Original: scout_phase20_winner_state_ranking.classify_2h
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
    # Original: scout_phase20_winner_state_ranking.infer_5m_prev
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
    # Original: scout_phase20_winner_state_ranking.build_states
    return {
        "5m": classify_5m(f, th),
        "15m": classify_15m(f, th, "current"),
        "30m": classify_30m(f, th, "current"),
        "1h": classify_1h(f, th, "current"),
        "2h": classify_2h(f, th, "current"),
    }


def build_transitions(f: dict, th: Thresholds) -> dict[str, str]:
    # Original: scout_phase20_winner_state_ranking.build_transitions
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
    # Original: scout_phase20_winner_state_ranking.combo_key
    return "|".join(f"{k}={states[k]}" for k in ("5m", "15m", "30m", "1h", "2h"))


def annotate(rows: list[dict], th: Thresholds) -> list[dict]:
    # Original: scout_phase20_winner_state_ranking.annotate
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
    # Original: scout_phase20_winner_state_ranking.winner_loser_sets
    winners: list[dict] = []
    losers: list[dict] = []
    for rows in by_scan.values():
        if len(rows) < 4:
            continue
        n = len(rows)
        winners.extend(rows[: min(WINNER_TOP_N, n)])
        losers.extend(rows[-min(3, n):])
    return winners, losers


def state_match_score(states: dict[str, str], trans: dict[str, str], profile: dict) -> float:
    # Original: scout_phase20_winner_state_ranking.state_match_score
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
    # Original: scout_phase20_winner_state_ranking.build_profile
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


def separation_row(
    label: str,
    w_count: int,
    w_total: int,
    l_count: int,
    l_total: int,
    all_count: int,
    all_total: int,
) -> dict:
    # Original: scout_phase20_winner_state_ranking.separation_row (includes lift)
    w_rate = w_count / w_total if w_total else 0
    l_rate = l_count / l_total if l_total else 0
    diff = w_rate - l_rate
    base = all_count / all_total if all_total else 0
    lift = w_rate / l_rate if l_rate > 0 else (99.0 if w_rate > 0 else 1.0)
    return {
        "label": label,
        "winner_count": w_count,
        "winner_rate": round(w_rate, 4),
        "loser_count": l_count,
        "loser_rate": round(l_rate, 4),
        "difference": round(diff, 4),
        "lift": round(lift, 4),
        "base_rate": round(base, 4),
    }
