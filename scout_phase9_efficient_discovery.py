"""
Scout Phase 9 — Efficient Search Formula Discovery

Ranking-first research on Pattern B filter (fixed).
Hypothesis -> Validate -> KEEP/MODIFY/DISCARD. No brute-force combos.

Usage:
  python scout_phase9_efficient_discovery.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from season2_p37_scout_decision_hierarchy import pf, write_csv
from season2_scout_mission import evaluate_convergence, mission_summary_lines

LOGS6 = Path("logs") / "phase6_lifecycle"
LOGS8 = Path("logs") / "phase8_blind"
OUT_DIR = Path("logs") / "phase9_discovery"
REPORT_TXT = OUT_DIR / "phase9_report.txt"
RESULTS_CSV = OUT_DIR / "hypothesis_results.csv"

RANDOM_SEED = 42
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
MIN_HOLDOUT_N = 15
PERCENTILES = (50, 60, 70, 80, 85, 90, 95, 97, 99)

# Pattern B filter — frozen
PATTERN_B = (("macd_signal", "gte", -0.0016), ("range_pct", "gte", 1.4768))

# Phase 8 blind baseline (macd-heavy ranking)
PHASE8_BASELINE = {
    "name": "Phase8_macd_rank",
    "weights": {
        "macd_signal": 8000.0,
        "range_margin": 0.5,
        "ma_slope": 0.2,
        "volume_ma_ratio": 0.3,
    },
}

# Phase 8 blind TOP5 performance (reference)
PHASE8_BLIND_REF = {
    "top5_hit_5pct": 3,
    "top5_n": 5,
    "top5_avg_max_up": 6.91,
    "top5_return_per_hour": 0.58,
    "ranking_quality": 51.0,
}


@dataclass
class RankRow:
    episode_id: str
    symbol: str
    scan_time_kst: str
    outcome: str
    max_excursion_12h_pct: float
    return_per_hour: float
    mdd_pct: float
    hit_5pct: bool
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class Hypothesis:
    hid: str
    name: str
    priority: int
    why: str
    weight_features: list[str]
    decision: str = "pending"
    holdout: dict = field(default_factory=dict)
    blind: dict = field(default_factory=dict)
    vs_baseline: dict = field(default_factory=dict)


def passes_pattern(feats: dict) -> bool:
    for name, op, thr in PATTERN_B:
        v = feats.get(name)
        if v is None:
            return False
        if op == "gte" and v < thr:
            return False
        if op == "lte" and v > thr:
            return False
    return True


def load_phase_map() -> dict[str, dict[str, dict]]:
    """episode_id -> phase_name -> row dict."""
    out: dict[str, dict[str, dict]] = {}
    with (LOGS6 / "phases.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["episode_id"], {})[row["phase"]] = row
    return out


def build_features(ep: dict, phases: dict[str, dict]) -> dict[str, float]:
    birth = phases.get("Birth") or phases.get("Ignition") or phases.get("Accumulation")
    ign = phases.get("Ignition")
    acc = phases.get("Accumulation")
    if not birth:
        return {}

    def num(row: dict, k: str) -> float:
        return pf(row.get(k)) or 0.0

    vol = num(birth, "volume")
    vol_ma = max(num(birth, "volume_ma"), 1e-9)
    price = max(num(birth, "price"), 1e-9)
    vol_ratio = vol / vol_ma
    ma_slope = num(birth, "ma_slope")
    range_pct = num(birth, "range_pct")
    macd_signal = num(birth, "macd_signal")
    body_ratio = num(birth, "body_ratio")
    atr_exp = num(birth, "atr_expansion")
    btc_2h = num(birth, "btc_return_2h")
    hh = num(birth, "higher_high")
    birth_age = num(birth, "duration_min")
    ign_age = num(ign, "duration_min") if ign else 0.0
    acc_age = num(acc, "duration_min") if acc else 0.0
    ign_slope = num(ign, "ma_slope") if ign else ma_slope
    ign_body = num(ign, "body_ratio") if ign else body_ratio

    feats = {
        # Priority 1 — price action
        "range_pct": range_pct,
        "higher_high": hh,
        "body_ratio": body_ratio,
        "body_expansion": body_ratio - ign_body,
        "range_margin": max(0.0, range_pct - 1.4768),
        # Priority 2 — lifecycle age
        "birth_age_min": birth_age,
        "ignition_age_min": ign_age,
        "accumulation_age_min": acc_age,
        "young_birth": 1.0 if birth_age <= 45 else 0.0,
        # Priority 3 — volume
        "volume_ma_ratio": vol_ratio,
        "dollar_volume": num(birth, "dollar_volume"),
        # Priority 4 — momentum
        "ma_slope": ma_slope,
        "macd_signal": macd_signal,
        "rsi": num(birth, "rsi"),
        "atr_expansion": atr_exp,
        # Priority 5 — relative
        "rs_vs_btc": ma_slope - btc_2h,
        "ma_slope_accel": ma_slope - ign_slope,
        # Priority 6 — composite
        "range_volume": range_pct * vol_ratio,
        "slope_volume": ma_slope * vol_ratio,
    }
    return feats


def load_pattern_b_rows() -> list[RankRow]:
    phase_map = load_phase_map()
    rows: list[RankRow] = []
    for line in (LOGS6 / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ep = json.loads(line)
        eid = ep["episode_id"]
        phases = phase_map.get(eid, {})
        feats = build_features(ep, phases)
        if not feats or not passes_pattern(feats):
            continue
        max_ex = pf(ep.get("max_excursion_12h_pct")) or 0.0
        rows.append(RankRow(
            episode_id=eid,
            symbol=ep["symbol"],
            scan_time_kst=ep["scan_time_kst"],
            outcome=ep["outcome"],
            max_excursion_12h_pct=max_ex,
            return_per_hour=max_ex / 12.0,
            mdd_pct=pf(ep.get("mdd_pct")) or 0.0,
            hit_5pct=bool(ep.get("hit_5pct")),
            features=feats,
        ))
    return rows


def split_rows(rows: list[RankRow]) -> tuple[list[RankRow], list[RankRow], list[RankRow]]:
    rows = sorted(rows, key=lambda r: (r.scan_time_kst, r.episode_id))
    rng = random.Random(RANDOM_SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n = len(rows)
    n_tr = int(n * TRAIN_RATIO)
    n_va = int(n * VAL_RATIO)
    tr_i = set(idx[:n_tr])
    va_i = set(idx[n_tr:n_tr + n_va])
    ho_i = set(idx[n_tr + n_va:])
    return (
        [rows[i] for i in range(n) if i in tr_i],
        [rows[i] for i in range(n) if i in va_i],
        [rows[i] for i in range(n) if i in ho_i],
    )


def percentile_value(train_vals: list[float], pct: int) -> float:
    if not train_vals:
        return 0.0
    s = sorted(train_vals)
    idx = min(len(s) - 1, int(pct / 100 * (len(s) - 1)))
    return s[idx]


def train_effect_weights(train: list[RankRow], features: list[str]) -> dict[str, float]:
    """Winner-loser separation weights from TRAIN only."""
    weights: dict[str, float] = {}
    winners = [r for r in train if r.outcome == "winner"]
    losers = [r for r in train if r.outcome == "loser"]
    if len(winners) < 3 or len(losers) < 3:
        return {f: 1.0 for f in features}
    for feat in features:
        wv = [r.features.get(feat, 0) for r in winners]
        lv = [r.features.get(feat, 0) for r in losers]
        pooled = statistics.pstdev(wv + lv) or 1.0
        weights[feat] = (statistics.mean(wv) - statistics.mean(lv)) / pooled
    return weights


def rank_score(row: RankRow, weights: dict[str, float], within_scan_pct: dict[str, float] | None = None) -> float:
    sc = 0.0
    for feat, w in weights.items():
        v = row.features.get(feat, 0.0)
        if within_scan_pct and feat in within_scan_pct:
            v = within_scan_pct[feat]
        sc += w * v
    return sc


def phase8_score(row: RankRow) -> float:
    f = row.features
    return (
        f.get("macd_signal", 0) * PHASE8_BASELINE["weights"]["macd_signal"]
        + f.get("range_margin", 0) * PHASE8_BASELINE["weights"]["range_margin"]
        + max(0.0, f.get("ma_slope", 0)) * PHASE8_BASELINE["weights"]["ma_slope"]
        + max(0.0, f.get("volume_ma_ratio", 0) - 1.0) * PHASE8_BASELINE["weights"]["volume_ma_ratio"]
    )


def within_scan_percentiles(group: list[RankRow], feat: str) -> dict[str, float]:
    vals = [(r.episode_id, r.features.get(feat, 0)) for r in group]
    vals.sort(key=lambda x: x[1])
    n = len(vals)
    out: dict[str, float] = {}
    for i, (eid, _) in enumerate(vals):
        out[eid] = (i / max(n - 1, 1)) * 100
    return out


def eval_ranking(rows: list[RankRow], score_fn) -> dict:
    if not rows:
        return {"sample": 0}
    by_scan: dict[str, list[RankRow]] = {}
    for r in rows:
        by_scan.setdefault(r.scan_time_kst, []).append(r)

    top1_hits = 0
    top5_rets: list[float] = []
    top5_rph: list[float] = []
    top5_hits = 0
    top5_n = 0
    spearman_pairs: list[tuple[float, float]] = []

    for scan, group in by_scan.items():
        if len(group) < 2:
            continue
        scored = [(score_fn(r), r) for r in group]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_actual = max(group, key=lambda r: r.max_excursion_12h_pct)
        if scored[0][1].episode_id == best_actual.episode_id:
            top1_hits += 1
        top5 = [r for _, r in scored[:5]]
        top5_n += len(top5)
        top5_hits += sum(1 for r in top5 if r.hit_5pct)
        top5_rets.extend([r.max_excursion_12h_pct for r in top5])
        top5_rph.extend([r.return_per_hour for r in top5])
        for sc, r in scored:
            spearman_pairs.append((sc, r.max_excursion_12h_pct))

    n_scans = len(by_scan)
    rho = _spearman(spearman_pairs) if len(spearman_pairs) >= 5 else 0.0
    return {
        "sample": len(rows),
        "scans": n_scans,
        "top1_accuracy_pct": round(top1_hits / max(n_scans, 1) * 100, 1),
        "top5_hit_5pct": top5_hits,
        "top5_n": top5_n,
        "top5_hit_rate_pct": round(top5_hits / max(top5_n, 1) * 100, 1),
        "top5_avg_max_up": round(statistics.mean(top5_rets), 2) if top5_rets else 0.0,
        "top5_return_per_hour": round(statistics.mean(top5_rph), 3) if top5_rph else 0.0,
        "spearman": round(rho, 3),
    }


def _spearman(pairs: list[tuple[float, float]]) -> float:
    n = len(pairs)
    if n < 3:
        return 0.0
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n * n - 1))


def define_hypotheses() -> list[Hypothesis]:
    return [
        Hypothesis(
            "H0", "Phase8_macd_rank (baseline)", 4,
            "Phase 8 blind ranking used macd_signal x8000 — failed to rank CLO/MYX.",
            ["macd_signal", "range_margin", "ma_slope", "volume_ma_ratio"],
        ),
        Hypothesis(
            "H1", "Volume+Slope rank", 3,
            "Phase 8 failure: volume_ma_ratio and ma_slope separated winners.",
            ["volume_ma_ratio", "ma_slope", "range_pct"],
        ),
        Hypothesis(
            "H2", "Young Birth + Volume", 2,
            "Birth age <=45min is distinct state; early birth + volume precedes expansion.",
            ["young_birth", "volume_ma_ratio", "birth_age_min", "ma_slope"],
        ),
        Hypothesis(
            "H3", "Breakout structure", 1,
            "higher_high breakout with range expansion (price action priority).",
            ["higher_high", "range_pct", "body_expansion", "volume_ma_ratio"],
        ),
        Hypothesis(
            "H4", "Lifecycle composite", 2,
            "Ignition/Birth age + slope acceleration capture trend-start timing.",
            ["ignition_age_min", "birth_age_min", "ma_slope_accel", "volume_ma_ratio"],
        ),
        Hypothesis(
            "H5", "Relative strength vs BTC", 5,
            "Universe-relative outperformance vs BTC 2h return.",
            ["rs_vs_btc", "ma_slope", "volume_ma_ratio"],
        ),
        Hypothesis(
            "H6", "Range-Volume composite", 6,
            "Joint expansion: range_pct x volume_ma_ratio (minimal composite).",
            ["range_volume", "ma_slope", "young_birth"],
        ),
    ]


def decide_hypothesis(h: Hypothesis, baseline_rph: float, baseline_hit: float, hold_n: int) -> str:
    ho = h.holdout
    if hold_n < MIN_HOLDOUT_N:
        return "MODIFY"  # insufficient holdout for KEEP gate
    rph = ho.get("top5_return_per_hour", 0)
    hit = ho.get("top5_hit_rate_pct", 0)
    sp = ho.get("spearman", 0)
    improved = (
        rph > baseline_rph
        or hit > baseline_hit
        or sp > h.vs_baseline.get("baseline_spearman", 0) + 0.05
    )
    if not improved:
        return "DISCARD"
    if rph > baseline_rph * 1.05 and hit >= baseline_hit:
        return "KEEP"
    return "MODIFY"


def replay_phase8_blind(weights: dict[str, float], use_phase8: bool = False) -> dict:
    """Replay ranking on Phase 8 blind CSV."""
    csv_path = LOGS8 / "blind_2026-06-03_110000.csv"
    if not csv_path.exists():
        return {"available": False}
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return {"available": False}

    def score_row(row: dict) -> float:
        if use_phase8:
            ms = pf(row.get("macd_signal")) or 0
            rp = pf(row.get("range_pct")) or 0
            mslope = pf(row.get("ma_slope")) or 0
            vr = pf(row.get("volume_ma_ratio")) or 0
            return ms * 8000 + max(0, rp - 1.4768) * 0.5 + max(0, mslope) * 0.2 + max(0, vr - 1) * 0.3
        feats = {
            "volume_ma_ratio": pf(row.get("volume_ma_ratio")) or 0,
            "ma_slope": pf(row.get("ma_slope")) or 0,
            "range_pct": pf(row.get("range_pct")) or 0,
            "macd_signal": pf(row.get("macd_signal")) or 0,
            "range_margin": max(0, (pf(row.get("range_pct")) or 0) - 1.4768),
            "young_birth": 1.0,
            "birth_age_min": 30.0,
            "ignition_age_min": 60.0,
            "ma_slope_accel": pf(row.get("ma_slope")) or 0,
            "higher_high": 1.0 if (pf(row.get("range_pct")) or 0) > 2.0 else 0,
            "body_expansion": 0.1,
            "range_volume": (pf(row.get("range_pct")) or 0) * (pf(row.get("volume_ma_ratio")) or 0),
            "rs_vs_btc": pf(row.get("ma_slope")) or 0,
        }
        return sum(weights.get(k, 0) * feats.get(k, 0) for k in weights)

    scored = [(score_row(r), r) for r in rows]
    scored.sort(key=lambda x: x[0], reverse=True)
    top5 = [r for _, r in scored[:5]]
    top5_syms = [r["symbol"] for r in top5]
    max_col = "actual_actual_max_up_12h"
    hits = sum(1 for r in top5 if (pf(r.get(max_col)) or 0) >= 5.0)
    avg_max = statistics.mean([pf(r.get(max_col)) or 0 for r in top5])
    rph = avg_max / 12.0
    # Ranking accuracy: is best symbol in top5?
    best = max(rows, key=lambda r: pf(r.get(max_col)) or 0)
    best_sym = best["symbol"]
    best_in_top5 = best_sym in top5_syms
    best_rank = next((i + 1 for i, (_, r) in enumerate(scored) if r["symbol"] == best_sym), 99)
    return {
        "available": True,
        "top5_symbols": top5_syms,
        "top5_hit_5pct": hits,
        "top5_avg_max_up": round(avg_max, 2),
        "top5_return_per_hour": round(rph, 3),
        "best_symbol": best_sym,
        "best_max_up": round(pf(best.get(max_col)) or 0, 2),
        "best_in_top5": best_in_top5,
        "best_rank": best_rank,
    }


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_pattern_b_rows()
    if len(rows) < 20:
        raise SystemExit(f"Insufficient Pattern B episodes: {len(rows)}")

    train, val, hold = split_rows(rows)
    feature_names = sorted({k for r in rows for k in r.features})

    # TRAIN percentile reference (for report only)
    pct_ref: dict[str, dict[int, float]] = {}
    for feat in feature_names:
        vals = [r.features.get(feat, 0) for r in train]
        pct_ref[feat] = {p: percentile_value(vals, p) for p in PERCENTILES}

    hypotheses = define_hypotheses()

    # Baseline eval on holdout
    baseline_hold = eval_ranking(hold, phase8_score)
    baseline_rph = baseline_hold.get("top5_return_per_hour", 0)
    baseline_hit = baseline_hold.get("top5_hit_rate_pct", 0)
    baseline_sp = baseline_hold.get("spearman", 0)

    results: list[dict] = []
    for h in hypotheses:
        if h.hid == "H0":
            w = {
                "macd_signal": 8000.0,
                "range_margin": 0.5,
                "ma_slope": 0.2,
                "volume_ma_ratio": 0.3,
            }
            ho_fn = phase8_score
        else:
            w = train_effect_weights(train, h.weight_features)
            ho_fn = lambda r, ww=w: rank_score(r, ww)

        ho = eval_ranking(hold, ho_fn)
        va = eval_ranking(val, ho_fn)
        h.holdout = ho
        h.vs_baseline = {
            "delta_return_per_hour": round(ho.get("top5_return_per_hour", 0) - baseline_rph, 3),
            "delta_hit_rate": round(ho.get("top5_hit_rate_pct", 0) - baseline_hit, 1),
            "baseline_spearman": baseline_sp,
            "delta_spearman": round(ho.get("spearman", 0) - baseline_sp, 3),
        }
        h.blind = replay_phase8_blind(w, use_phase8=(h.hid == "H0"))
        h.decision = decide_hypothesis(h, baseline_rph, baseline_hit, ho.get("sample", 0))

        results.append({
            "hypothesis_id": h.hid,
            "name": h.name,
            "priority": h.priority,
            "decision": h.decision,
            "holdout_n": ho.get("sample", 0),
            "holdout_return_per_hour": ho.get("top5_return_per_hour", 0),
            "holdout_hit_rate_pct": ho.get("top5_hit_rate_pct", 0),
            "holdout_top1_accuracy": ho.get("top1_accuracy_pct", 0),
            "holdout_spearman": ho.get("spearman", 0),
            "delta_rph_vs_baseline": h.vs_baseline["delta_return_per_hour"],
            "blind_top5_hit": h.blind.get("top5_hit_5pct", ""),
            "blind_rph": h.blind.get("top5_return_per_hour", ""),
            "blind_best_rank": h.blind.get("best_rank", ""),
        })

    # Pick best KEEP/MODIFY by blind return/hour then holdout
    ranked = sorted(
        [h for h in hypotheses if h.decision != "DISCARD"],
        key=lambda h: (
            h.blind.get("top5_return_per_hour", 0) if h.blind.get("available") else 0,
            h.holdout.get("top5_return_per_hour", 0),
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else hypotheses[0]

    if best.hid == "H0":
        best_weights = {"macd_signal": 8000, "range_margin": 0.5, "ma_slope": 0.2, "volume_ma_ratio": 0.3}
    else:
        best_weights = train_effect_weights(train, best.weight_features)

    conv = evaluate_convergence(
        "interaction_mining",
        improves=["relative_ranking_between_candidates"],
        sample_size=len(hold),
        confidence="medium" if len(hold) >= MIN_HOLDOUT_N else "hypothesis",
    )

    lines = [
        "############################################################",
        "SCOUT PHASE 9 — EFFICIENT SEARCH FORMULA DISCOVERY",
        "############################################################",
        "",
        "Focus: RANKING within Pattern B filter (filter frozen).",
        f"Pattern B: macd_signal >= -0.0016 AND range_pct >= 1.4768",
        f"Episodes (Pattern B): {len(rows)} | Train {len(train)} | Val {len(val)} | Hold {len(hold)}",
        f"Holdout n gate: {MIN_HOLDOUT_N} (current hold={len(hold)} — note sample limit)",
        "",
        mission_summary_lines()[0] if mission_summary_lines() else "",
        f"Convergence tier: {conv['tier']} | criteria: {conv.get('convergence_criteria_met', [])}",
        "",
        "=" * 60,
        "1. KEEP / MODIFY / DISCARD",
        "=" * 60,
    ]
    for h in hypotheses:
        lines.append(
            f"  {h.hid} {h.name}: {h.decision} | "
            f"hold r/h={h.holdout.get('top5_return_per_hour', 0)} "
            f"hit={h.holdout.get('top5_hit_rate_pct', 0)}% "
            f"blind r/h={h.blind.get('top5_return_per_hour', 'n/a')}"
        )

    lines.extend([
        "",
        "=" * 60,
        "2. NEW FEATURES (generated)",
        "=" * 60,
        "  body_expansion, young_birth, birth_age_min, ignition_age_min,",
        "  accumulation_age_min, rs_vs_btc, ma_slope_accel, range_volume, slope_volume",
        "",
        "=" * 60,
        "3. WHY CREATED",
        "=" * 60,
    ])
    for h in hypotheses:
        if h.hid == "H0":
            continue
        lines.append(f"  {h.hid}: {h.why}")

    lines.extend([
        "",
        "=" * 60,
        "4. HOLDOUT RESULTS (episode split)",
        "=" * 60,
        f"  Baseline H0: r/h={baseline_rph} hit={baseline_hit}% spearman={baseline_sp}",
    ])
    for h in hypotheses:
        ho = h.holdout
        lines.append(
            f"  {h.hid}: n={ho.get('sample')} top1_acc={ho.get('top1_accuracy_pct')}% "
            f"top5_hit={ho.get('top5_hit_rate_pct')}% r/h={ho.get('top5_return_per_hour')} "
            f"spearman={ho.get('spearman')}"
        )

    lines.extend([
        "",
        "=" * 60,
        "5. BLIND REPLAY (Phase 8: 2026-06-03 11:00 KST)",
        "=" * 60,
        f"  Phase 8 actual TOP5: hit_5pct={PHASE8_BLIND_REF['top5_hit_5pct']}/5 "
        f"r/h={PHASE8_BLIND_REF['top5_return_per_hour']}",
    ])
    for h in hypotheses:
        b = h.blind
        if not b.get("available"):
            continue
        lines.append(
            f"  {h.hid}: TOP5={b.get('top5_symbols')} hit={b.get('top5_hit_5pct')}/5 "
            f"r/h={b.get('top5_return_per_hour')} best={b.get('best_symbol')} "
            f"(+{b.get('best_max_up')}%) rank={b.get('best_rank')}"
        )

    lines.extend([
        "",
        "=" * 60,
        "6. RETURN/HOUR CHANGE vs Baseline",
        "=" * 60,
    ])
    for h in hypotheses:
        lines.append(
            f"  {h.hid}: hold delta_rph={h.vs_baseline.get('delta_return_per_hour', 0):+.3f} "
            f"blind r/h={h.blind.get('top5_return_per_hour', 'n/a')}"
        )

    lines.extend([
        "",
        "=" * 60,
        "7. RANKING CHANGE",
        "=" * 60,
        f"  Phase 8 missed CLOUSDT(#9 +42%), MYXUSDT(#7 +23%) due to macd weight.",
    ])
    b_best = best.blind
    if b_best.get("available"):
        lines.append(
            f"  Best hypothesis {best.hid}: best_actual={b_best.get('best_symbol')} "
            f"rank={b_best.get('best_rank')} in_top5={b_best.get('best_in_top5')}"
        )

    lines.extend([
        "",
        "=" * 60,
        "8. vs EXISTING FORMULA",
        "=" * 60,
        "  Filter (Pattern B): KEEP — 53.3% universe hit at Phase 8 blind.",
        f"  Ranking: {'IMPROVED' if best.hid != 'H0' else 'NOT IMPROVED'} — best={best.hid} ({best.name})",
    ])

    ops_ready = (
        best.hid != "H0"
        and best.blind.get("top5_return_per_hour", 0) > PHASE8_BLIND_REF["top5_return_per_hour"]
        and best.blind.get("top5_hit_5pct", 0) >= PHASE8_BLIND_REF["top5_hit_5pct"]
    )
    lines.extend([
        "",
        "=" * 60,
        "9. OPERATIONAL RECOMMENDATION",
        "=" * 60,
        f"  Filter: KEEP Pattern B",
        f"  Ranking: {'DEPLOY ' + best.hid if ops_ready else 'MODIFY — holdout n<' + str(MIN_HOLDOUT_N) + ' or blind not beating Phase 8'}",
        "",
        "=" * 60,
        "10. RECOMMENDED FORMULA",
        "=" * 60,
        "  FILTER (frozen):",
        "    macd_signal >= -0.0016 AND range_pct >= 1.4768",
        f"  RANK (best={best.hid}):",
    ])
    for feat, w in sorted(best_weights.items(), key=lambda x: abs(x[1]), reverse=True):
        lines.append(f"    score += {w:.4f} * {feat}")
    lines.append("")
    lines.append(
        f"VERDICT: Filter=KEEP | Ranking={'KEEP ' + best.hid if best.decision == 'KEEP' else 'MODIFY ' + best.hid}"
    )
    lines.append(
        f"Evidence: blind r/h {best.blind.get('top5_return_per_hour')} vs Phase8 {PHASE8_BLIND_REF['top5_return_per_hour']}, "
        f"hold delta_rph {best.vs_baseline.get('delta_return_per_hour', 0):+.3f}"
    )

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    write_csv(RESULTS_CSV, results)
    print("\n".join(lines[-30:]).encode("ascii", "replace").decode("ascii"))
    print(f"\nSaved: {REPORT_TXT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
