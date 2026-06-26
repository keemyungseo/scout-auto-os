"""
Scout Phase 7 — Pattern Discovery & Search Formula Generation

Uses ONLY logs/phase6_lifecycle/ data.
Train / Validation / Holdout split. Thresholds fit on TRAIN only.

Usage:
  python scout_phase7_pattern_discovery.py
  python scout_phase7_pattern_discovery.py --phase Birth
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from season2_p37_scout_decision_hierarchy import pf, write_csv

LOGS_DIR = Path("logs") / "phase6_lifecycle"
EPISODES_JSONL = LOGS_DIR / "episodes.jsonl"
PHASES_CSV = LOGS_DIR / "phases.csv"
OUT_DIR = Path("logs") / "phase7_patterns"
REPORT_TXT = OUT_DIR / "pattern_discovery_report.txt"
PATTERNS_CSV = OUT_DIR / "pattern_candidates.csv"

RANDOM_SEED = 42
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
MIN_TRAIN_SAMPLE = 12
MIN_HOLDOUT_SAMPLE = 8
MAX_COMBO_SIZE = 4
TOP_FEATURES = 12

# Trend-start search uses pre-expansion phases
START_PHASES = ("Accumulation", "Ignition", "Birth")
PROGRESS_PHASES = ("Expansion", "Continuation")
END_PHASES = ("Exhaustion", "Distribution")

NUMERIC_FEATURES = (
    "ma_slope",
    "atr",
    "atr_expansion",
    "obv",
    "vwap",
    "rsi",
    "macd",
    "macd_signal",
    "adx",
    "range_pct",
    "box_width_pct",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "higher_high",
    "higher_low",
    "btc_return_2h",
    "volume_ma_ratio",
    "dollar_volume_ratio",
    "price_ema_dist_pct",
)

DERIVED = {
    "volume_ma_ratio": lambda r: (pf(r.get("volume")) or 0) / (pf(r.get("volume_ma")) or 1),
    "dollar_volume_ratio": lambda r: (pf(r.get("dollar_volume")) or 0) / max(pf(r.get("volume_ma")) or 1, 1) / max(pf(r.get("price")) or 1, 1e-9),
    "price_ema_dist_pct": lambda r: ((pf(r.get("price")) or 0) - (pf(r.get("ema")) or 0)) / max(pf(r.get("ema")) or 1, 1e-9) * 100,
}


@dataclass
class EpisodeRow:
    episode_id: str
    symbol: str
    outcome: str
    scan_time_kst: str
    return_12h_pct: float
    max_excursion_12h_pct: float
    hit_5pct: bool
    hit_7pct: bool
    hit_10pct: bool
    duration_min: float
    mdd_pct: float
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class Condition:
    feature: str
    op: str  # gte | lte
    threshold: float

    @staticmethod
    def fmt_thr(v: float) -> str:
        if abs(v) >= 1000:
            return f"{v:.4g}"
        return f"{v:.4f}".rstrip("0").rstrip(".")

    def label(self) -> str:
        return f"{self.feature} {self.op} {self.fmt_thr(self.threshold)}"

    def passes(self, row: EpisodeRow) -> bool:
        v = row.features.get(self.feature)
        if v is None:
            return False
        if self.op == "gte":
            return v >= self.threshold
        return v <= self.threshold


@dataclass
class Pattern:
    pattern_id: str
    phase_group: str
    conditions: list[Condition]
    train_stats: dict
    val_stats: dict
    holdout_stats: dict
    score: float
    confidence_pct: float
    reason: str
    decision: str


def load_episodes() -> list[dict]:
    rows: list[dict] = []
    for line in EPISODES_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_phase_features(phase_names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    import csv

    # Prefer Birth, fallback Ignition, then Accumulation per episode
    priority = {p: i for i, p in enumerate(phase_names)}
    best: dict[str, tuple[int, dict]] = {}

    with PHASES_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = row["episode_id"]
            phase = row["phase"]
            if phase not in priority:
                continue
            rank = priority[phase]
            if eid in best and best[eid][0] <= rank:
                continue
            feats: dict[str, float] = {}
            for k in NUMERIC_FEATURES:
                if k in DERIVED:
                    feats[k] = DERIVED[k](row)
                else:
                    v = pf(row.get(k))
                    if v is not None:
                        feats[k] = v
            best[eid] = (rank, feats)
    return {eid: feats for eid, (_, feats) in best.items()}


def build_rows(phase_group: str) -> list[EpisodeRow]:
    if phase_group == "start":
        phases = START_PHASES
    elif phase_group == "progress":
        phases = PROGRESS_PHASES
    else:
        phases = END_PHASES

    phase_feats = load_phase_features(phases)
    rows: list[EpisodeRow] = []
    for ep in load_episodes():
        eid = ep["episode_id"]
        if eid not in phase_feats:
            continue
        rows.append(EpisodeRow(
            episode_id=eid,
            symbol=ep["symbol"],
            outcome=ep["outcome"],
            scan_time_kst=ep["scan_time_kst"],
            return_12h_pct=pf(ep.get("return_12h_pct")) or 0.0,
            max_excursion_12h_pct=pf(ep.get("max_excursion_12h_pct")) or 0.0,
            hit_5pct=bool(ep.get("hit_5pct")),
            hit_7pct=bool(ep.get("hit_7pct")),
            hit_10pct=bool(ep.get("hit_10pct")),
            duration_min=pf(ep.get("duration_min")) or 0.0,
            mdd_pct=pf(ep.get("mdd_pct")) or 0.0,
            features=phase_feats[eid],
        ))
    return rows


def split_rows(rows: list[EpisodeRow]) -> tuple[list[EpisodeRow], list[EpisodeRow], list[EpisodeRow]]:
    rows = sorted(rows, key=lambda r: (r.scan_time_kst, r.episode_id))
    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    n = len(rows)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train_i = set(indices[:n_train])
    val_i = set(indices[n_train:n_train + n_val])
    hold_i = set(indices[n_train + n_val:])
    train = [rows[i] for i in range(n) if i in train_i]
    val = [rows[i] for i in range(n) if i in val_i]
    hold = [rows[i] for i in range(n) if i in hold_i]
    return train, val, hold


def cohort_stats(rows: list[EpisodeRow]) -> dict:
    if not rows:
        return {"sample": 0}
    winners = [r for r in rows if r.outcome == "winner"]
    losers = [r for r in rows if r.outcome == "loser"]
    n = len(rows)
    return {
        "sample": n,
        "winners": len(winners),
        "losers": len(losers),
        "rate_5pct": round(sum(1 for r in rows if r.hit_5pct) / n * 100, 1),
        "rate_7pct": round(sum(1 for r in rows if r.hit_7pct) / n * 100, 1),
        "rate_10pct": round(sum(1 for r in rows if r.hit_10pct) / n * 100, 1),
        "expected_return": round(statistics.mean([r.max_excursion_12h_pct for r in rows]), 2),
        "median_return": round(statistics.median([r.max_excursion_12h_pct for r in rows]), 2),
        "avg_duration": round(statistics.mean([r.duration_min for r in rows]), 0),
        "avg_mdd": round(statistics.mean([r.mdd_pct for r in rows]), 2),
        "winner_rate": round(len(winners) / n * 100, 1),
        "return_per_hour": round(statistics.mean([r.max_excursion_12h_pct / 12.0 for r in rows]), 2),
    }


def apply_conditions(rows: list[EpisodeRow], conds: list[Condition]) -> list[EpisodeRow]:
    out: list[EpisodeRow] = []
    for r in rows:
        if all(c.passes(r) for c in conds):
            out.append(r)
    return out


def candidate_thresholds(train: list[EpisodeRow], feature: str) -> list[tuple[str, float]]:
    vals = [r.features.get(feature) for r in train if r.features.get(feature) is not None]
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) < 8:
        return []
    vals_sorted = sorted(vals)
    pts = [0.1, 0.25, 0.5, 0.75, 0.9]
    thresholds: list[tuple[str, float]] = []
    for p in pts:
        idx = min(len(vals_sorted) - 1, int(p * (len(vals_sorted) - 1)))
        t = vals_sorted[idx]
        thresholds.append(("gte", t))
        thresholds.append(("lte", t))
    return thresholds


def separation_score(train: list[EpisodeRow], conds: list[Condition]) -> float:
    matched = apply_conditions(train, conds)
    if len(matched) < MIN_TRAIN_SAMPLE:
        return -999.0
    base_wr = sum(1 for r in train if r.outcome == "winner") / len(train)
    wr = sum(1 for r in matched if r.outcome == "winner") / len(matched)
    avg_ret = statistics.mean([r.max_excursion_12h_pct for r in matched])
    ret_per_h = statistics.mean([r.max_excursion_12h_pct / 12.0 for r in matched])
    # Primary objective: expected return per hour with winner lift over base
    return ret_per_h + (wr - base_wr) * 20 + avg_ret * 0.05


def discover_univariate(train: list[EpisodeRow]) -> list[tuple[str, Condition, float]]:
    found: list[tuple[str, Condition, float]] = []
    for feat in NUMERIC_FEATURES:
        for op, thr in candidate_thresholds(train, feat):
            cond = Condition(feat, op, thr)
            sc = separation_score(train, [cond])
            if sc > -900:
                found.append((feat, cond, sc))
    found.sort(key=lambda x: x[2], reverse=True)
    return found


def discover_combos(train: list[EpisodeRow], seeds: list[tuple[str, Condition, float]]) -> list[list[Condition]]:
    top_feats = []
    seen = set()
    for feat, cond, _ in seeds:
        if feat not in seen:
            top_feats.append(cond)
            seen.add(feat)
        if len(top_feats) >= TOP_FEATURES:
            break

    combos: list[list[Condition]] = []
    # singles
    for c in top_feats[:TOP_FEATURES]:
        combos.append([c])
    # pairs
    for i in range(len(top_feats)):
        for j in range(i + 1, min(len(top_feats), TOP_FEATURES)):
            combos.append([top_feats[i], top_feats[j]])
    # triples (limit)
    for i in range(min(6, len(top_feats))):
        for j in range(i + 1, min(7, len(top_feats))):
            for k in range(j + 1, min(8, len(top_feats))):
                combos.append([top_feats[i], top_feats[j], top_feats[k]])
    # quads (top 5 only)
    for i in range(min(5, len(top_feats))):
        for j in range(i + 1, min(5, len(top_feats))):
            for k in range(j + 1, min(5, len(top_feats))):
                for m in range(k + 1, min(5, len(top_feats))):
                    combos.append([top_feats[i], top_feats[j], top_feats[k], top_feats[m]])
    return combos


def best_feature_separation(train: list[EpisodeRow], conds: list[Condition]) -> str:
    w = apply_conditions(train, conds)
    w_win = [r for r in w if r.outcome == "winner"]
    w_los = [r for r in w if r.outcome == "loser"]
    if not w_win or not w_los:
        return "insufficient split"
    best_feat = ""
    best_gap = -1.0
    for feat in NUMERIC_FEATURES:
        wv = [r.features.get(feat, 0) for r in w_win if feat in r.features]
        lv = [r.features.get(feat, 0) for r in w_los if feat in r.features]
        if len(wv) < 3 or len(lv) < 3:
            continue
        gap = abs(statistics.mean(wv) - statistics.mean(lv))
        if gap > best_gap:
            best_gap = gap
            best_feat = feat
    return best_feat or "combined"


def confidence_stars(conf_pct: float) -> str:
    if conf_pct >= 85:
        return "★★★★★"
    if conf_pct >= 70:
        return "★★★★☆"
    if conf_pct >= 55:
        return "★★★☆☆"
    if conf_pct >= 40:
        return "★★☆☆☆"
    return "★☆☆☆☆"


def decide_pattern(hold: dict, val: dict, train: dict) -> str:
    if hold.get("sample", 0) < MIN_HOLDOUT_SAMPLE:
        return "MODIFY"
    if hold.get("rate_5pct", 0) >= train.get("rate_5pct", 0) * 0.7 and hold.get("expected_return", 0) > 3:
        return "KEEP"
    if hold.get("sample", 0) >= MIN_HOLDOUT_SAMPLE and hold.get("expected_return", 0) > 0:
        return "MODIFY"
    return "DISCARD"


def run_discovery(phase_group: str = "start") -> list[Pattern]:
    rows = build_rows(phase_group)
    if len(rows) < 40:
        raise SystemExit(f"Insufficient episodes: {len(rows)}")

    train, val, hold = split_rows(rows)
    uni = discover_univariate(train)
    combos = discover_combos(train, uni)

    # Prefer interpretable trend-start features when scoring ties
    interpretable = {"ma_slope", "volume_ma_ratio", "box_width_pct", "atr_expansion", "rsi", "vwap", "range_pct"}

    scored: list[tuple[list[Condition], float]] = []
    for conds in combos:
        sc = separation_score(train, conds)
        if sc > -900:
            if any(c.feature in interpretable for c in conds):
                sc += 0.5
            if len(apply_conditions(hold, conds)) >= MIN_HOLDOUT_SAMPLE:
                sc += 1.0
            scored.append((conds, sc))
    scored.sort(key=lambda x: x[1], reverse=True)

    patterns: list[Pattern] = []
    seen_sigs: set[str] = set()
    for conds, tr_sc in scored:
        sig = "|".join(c.label() for c in conds)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)

        tr_m = apply_conditions(train, conds)
        va_m = apply_conditions(val, conds)
        ho_m = apply_conditions(hold, conds)
        tr_s = cohort_stats(tr_m)
        va_s = cohort_stats(va_m)
        ho_s = cohort_stats(ho_m)

        if tr_s.get("sample", 0) < MIN_TRAIN_SAMPLE:
            continue

        # Confidence: holdout winner rate vs base + val agreement
        base_wr = sum(1 for r in hold if r.outcome == "winner") / max(len(hold), 1)
        ho_wr = ho_s.get("winner_rate", 0) / 100 if ho_s.get("sample") else 0
        val_agree = 1.0 - abs(tr_s.get("rate_5pct", 0) - va_s.get("rate_5pct", 0)) / 100 if va_s.get("sample") else 0
        conf = max(0, min(100, (ho_wr / max(base_wr, 0.01)) * 35 + val_agree * 25 + min(tr_s["sample"], 40) * 0.4))
        if ho_s.get("sample", 0) >= MIN_HOLDOUT_SAMPLE:
            conf += min(25, ho_s.get("rate_5pct", 0) * 0.25)
        conf = min(100, conf)

        sep_feat = best_feature_separation(train, conds)
        reason = (
            f"Loser vs Winner largest mean gap on `{sep_feat}` in TRAIN matched cohort. "
            f"Thresholds fixed from TRAIN percentiles only."
        )

        pat = Pattern(
            pattern_id=f"Pattern_{chr(65 + len(patterns))}",
            phase_group=phase_group,
            conditions=conds,
            train_stats=tr_s,
            val_stats=va_s,
            holdout_stats=ho_s,
            score=tr_sc,
            confidence_pct=round(conf, 1),
            reason=reason,
            decision=decide_pattern(ho_s, va_s, tr_s),
        )
        patterns.append(pat)
        if len(patterns) >= 8:
            break

    return patterns


def format_pattern(p: Pattern) -> str:
    lines = [
        "=" * 50,
        p.pattern_id,
        "",
        "조건",
    ]
    for i, c in enumerate(p.conditions):
        prefix = "AND" if i else ""
        if prefix:
            lines.append(prefix)
        lines.append(c.label())
    lines.extend([
        "",
        "-" * 50,
        "",
        "Sample",
        str(p.holdout_stats.get("sample", p.train_stats.get("sample", 0))),
        "",
        "Winner",
        str(p.holdout_stats.get("winners", p.train_stats.get("winners", 0))),
        "",
        "Loser",
        str(p.holdout_stats.get("losers", p.train_stats.get("losers", 0))),
        "",
        "-" * 50,
        "",
        "12h max +5%",
        f"{p.holdout_stats.get('rate_5pct', p.train_stats.get('rate_5pct', 0))}%",
        "",
        "12h max +7%",
        f"{p.holdout_stats.get('rate_7pct', p.train_stats.get('rate_7pct', 0))}%",
        "",
        "12h max +10%",
        f"{p.holdout_stats.get('rate_10pct', p.train_stats.get('rate_10pct', 0))}%",
        "",
        "-" * 50,
        "",
        "Expected Return (12h max excursion, holdout)",
        f"+{p.holdout_stats.get('expected_return', 0)}%",
        "",
        "Median Return",
        f"+{p.holdout_stats.get('median_return', 0)}%",
        "",
        "-" * 50,
        "",
        "Average Duration",
        f"{p.holdout_stats.get('avg_duration', 0)} min",
        "",
        "Average MDD",
        f"-{abs(p.holdout_stats.get('avg_mdd', 0))}%",
        "",
        "-" * 50,
        "",
        "Expected Return / Hour (holdout)",
        f"+{p.holdout_stats.get('return_per_hour', 0)}%/h",
        "",
        "Confidence",
        f"{p.confidence_pct}%",
        "",
        "-" * 50,
        "",
        "Reason",
        p.reason,
        "",
        "Decision",
        p.decision,
        "",
        "=" * 50,
        "",
    ])
    return "\n".join(lines)


def binance_search_line(p: Pattern) -> str:
    parts = []
    for c in p.conditions:
        parts.append(c.label())
    return " AND ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-group", default="start", choices=("start", "progress", "end"))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patterns = run_discovery(args.phase_group)
    patterns.sort(key=lambda p: p.holdout_stats.get("return_per_hour", 0), reverse=True)

    top5 = patterns[:5]

    csv_rows = []
    for p in patterns:
        csv_rows.append({
            "pattern_id": p.pattern_id,
            "phase_group": p.phase_group,
            "conditions": " AND ".join(c.label() for c in p.conditions),
            "train_sample": p.train_stats.get("sample", 0),
            "holdout_sample": p.holdout_stats.get("sample", 0),
            "holdout_rate_5pct": p.holdout_stats.get("rate_5pct", 0),
            "holdout_expected_return": p.holdout_stats.get("expected_return", 0),
            "holdout_return_per_hour": p.holdout_stats.get("return_per_hour", 0),
            "confidence_pct": p.confidence_pct,
            "decision": p.decision,
            "score": round(p.score, 4),
        })
    write_csv(PATTERNS_CSV, csv_rows)

    lines = [
        "SCOUT PHASE 7 — PATTERN DISCOVERY REPORT",
        f"Phase group: {args.phase_group} ({', '.join(START_PHASES if args.phase_group=='start' else PROGRESS_PHASES if args.phase_group=='progress' else END_PHASES)})",
        f"Episodes source: {EPISODES_JSONL}",
        f"Split: Train {int(TRAIN_RATIO*100)}% / Val {int(VAL_RATIO*100)}% / Holdout {int((1-TRAIN_RATIO-VAL_RATIO)*100)}% (seed={RANDOM_SEED})",
        "Thresholds: TRAIN percentiles only. No post-hoc tuning.",
        "",
        "NOTE: Phase 6 labels are 12h forward max excursion (+5/+7/+10). 2h/4h horizons not in dataset.",
        "",
    ]

    for p in patterns[:3]:
        lines.append(format_pattern(p))

    lines.extend([
        "",
        "=" * 60,
        "1. TOP 5 SEARCH FORMULAS (by holdout return/hour)",
        "=" * 60,
        "",
    ])
    for i, p in enumerate(top5, 1):
        lines.append(f"{i}. {p.pattern_id} | score={p.score:.2f} | ret/h={p.holdout_stats.get('return_per_hour',0)}%/h | conf={p.confidence_pct}% | {p.decision}")
        lines.append(f"   {binance_search_line(p)}")
        lines.append("")

    if top5:
        best = top5[0]
        ho = best.holdout_stats
        lines.extend([
            "=" * 60,
            "2. BINANCE SEARCH CONDITIONS (numeric, copy-ready)",
            "=" * 60,
            "",
            binance_search_line(best),
            "",
            "=" * 60,
            "3. EXPECTED SEARCH RESULTS (holdout cohort)",
            "=" * 60,
            "",
            f"Average matched symbols per scan (holdout): {ho.get('sample', 0)}",
            f"Hit rate 12h max +5%: {ho.get('rate_5pct', 0)}%",
            f"Expected max excursion: +{ho.get('expected_return', 0)}%",
            f"Average duration: {ho.get('avg_duration', 0)} min",
            f"Average MDD: -{abs(ho.get('avg_mdd', 0))}%",
            f"Return per hour: +{ho.get('return_per_hour', 0)}%/h",
            "",
            "=" * 60,
            "4. CONFIDENCE",
            "=" * 60,
            "",
        ])
        for p in top5:
            lines.append(f"{p.pattern_id}: {confidence_stars(p.confidence_pct)} ({p.confidence_pct}%)")
        lines.extend([
            "",
            "=" * 60,
            "5. DECISIONS",
            "=" * 60,
            "",
        ])
        for p in top5:
            lines.append(f"{p.pattern_id}: {p.decision} — holdout n={p.holdout_stats.get('sample',0)}, +5%={p.holdout_stats.get('rate_5pct',0)}%, ret/h={p.holdout_stats.get('return_per_hour',0)}")
    else:
        lines.append("No patterns passed minimum sample gates.")

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_TXT.read_text(encoding="utf-8").replace("\u2014", "-"))
    print(f"\nSaved: {REPORT_TXT}")
    print(f"Saved: {PATTERNS_CSV}")


if __name__ == "__main__":
    main()
