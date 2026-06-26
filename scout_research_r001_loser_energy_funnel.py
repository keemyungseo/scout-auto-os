"""
Scout Research R001 — Loser Funnel + Energy Gate + Frozen Formula Top2

Tracks: B (transition), Loser Research, partial C (energy dynamics)
Formula frozen (h4_score ranking only). No weight changes.

Hypothesis:
  Reject high-confidence loser transition patterns + low dynamic energy
  before h4_score Top2 selection improves blind performance vs raw formula.

Usage:
  python scout_research_r001_loser_energy_funnel.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import scout_phase20_winner_state_ranking as p20
from season2_scout_mission import mission_summary_lines
from season2_p37_scout_decision_hierarchy import write_csv

P19_CAND = Path("logs") / "phase19_winner_dna" / "candidates.jsonl"
P23_MATCH = Path("logs") / "phase23_formula_league" / "match_log.jsonl"
OUT_DIR = Path("logs") / "research_r001_loser_energy_funnel"

LOSER_MAX_UP = 2.0
MIN_COMBO_N = 15
LOSER_ODDS_MIN = 3.0
TARGET_POOL = 30
HOLDOUT_FRAC = 0.30


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def g(f: dict, key: str, default: float = 0.0) -> float:
    return float(f.get(key, default))


def energy_score(f: dict) -> float:
    """Dynamic energy: price + volume + momentum deltas + persistence."""
    price_delta = (
        abs(g(f, "15m_current_return_pct"))
        + abs(g(f, "30m_current_return_pct"))
        + abs(g(f, "1h_current_return_pct"))
    )
    vol_delta = (
        g(f, "5m_volume_ma_ratio")
        + g(f, "15m_current_volume_ratio")
        + g(f, "30m_current_volume_ratio")
    )
    mom_delta = abs(g(f, "5m_momentum")) + abs(
        g(f, "15m_current_return_pct") - g(f, "15m_previous_return_pct")
    )
    persistence = g(f, "5m_seq_positive_count_6") / 6.0
    release_bonus = 2.0 if g(f, "5m_release") > 0 else 0.0
    compression_penalty = g(f, "5m_compression") * 1.5
    return price_delta + 0.4 * vol_delta + mom_delta + persistence * 4.0 + release_bonus - compression_penalty


def death_transition(states: dict[str, str]) -> bool:
    """1h Flat + 2h Flat/OverExtended after weak 5m — expansion unlikely."""
    return (
        states.get("5m") in ("Quiet", "Normal")
        and states.get("15m") == "Weak"
        and states.get("1h") == "Flat"
        and states.get("2h") in ("Flat", "OverExtended")
    )


def load_annotated() -> tuple[list[dict], dict[str, list[dict]]]:
    raw = p20.load_candidates()
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by_scan[r["scan_kst"]].append(r)
    for scan in by_scan:
        by_scan[scan].sort(key=lambda x: x["outcome_rank"])

    winner_feats = [
        r["features"]
        for rows in by_scan.values()
        for r in rows[: p20.WINNER_TOP_N]
        if len(rows) >= 4
    ]
    th = p20.build_thresholds(winner_feats)
    annotated = p20.annotate(raw, th)
    for r in annotated:
        r["energy"] = energy_score(r["features"])
    ann_by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in annotated:
        ann_by_scan[r["scan_kst"]].append(r)
    for scan in ann_by_scan:
        ann_by_scan[scan].sort(key=lambda x: x["outcome_rank"])
    return annotated, ann_by_scan


def learn_reject_rules(train: list[dict]) -> dict:
    """Train-only loser pattern discovery."""
    losers = [r for r in train if r["max_up_4h"] < LOSER_MAX_UP or r["outcome_rank"] > len(
        [x for x in train if x["scan_kst"] == r["scan_kst"]]
    ) - 3]
    winners = [r for r in train if r["outcome_rank"] <= 2]
    if not losers:
        losers = sorted(train, key=lambda x: x["max_up_4h"])[: max(len(train) // 10, 1)]

    combo_l: Counter[str] = Counter(r["combo"] for r in losers)
    combo_w: Counter[str] = Counter(r["combo"] for r in winners)
    combo_a: Counter[str] = Counter(r["combo"] for r in train)

    reject_combos: set[str] = set()
    for combo, lc in combo_l.items():
        ac = combo_a[combo]
        if ac < MIN_COMBO_N:
            continue
        wr = combo_w[combo] / max(len(winners), 1)
        lr = lc / ac
        if lr >= 0.35 and wr <= 0.05:
            reject_combos.add(combo)
        elif lr / max(wr, 0.01) >= LOSER_ODDS_MIN and ac >= MIN_COMBO_N:
            reject_combos.add(combo)

    loser_energy = [r["energy"] for r in losers]
    energy_cut = (
        statistics.quantiles(loser_energy, n=4)[0]
        if len(loser_energy) >= 8
        else (min(loser_energy) if loser_energy else -999.0)
    )

    return {
        "reject_combos": reject_combos,
        "energy_cut": energy_cut,
        "n_loser_combos": len(reject_combos),
    }


def should_reject(r: dict, rules: dict) -> tuple[bool, str]:
    if r["combo"] in rules["reject_combos"]:
        return True, "loser_combo"
    if death_transition(r["states"]):
        return True, "death_transition"
    if r["energy"] < rules["energy_cut"]:
        return True, "low_energy"
    return False, ""


def apply_funnel(rows: list[dict], rules: dict) -> tuple[list[dict], dict]:
    survivors: list[dict] = []
    reject_reasons: Counter[str] = Counter()
    for r in rows:
        rej, reason = should_reject(r, rules)
        if rej:
            reject_reasons[reason] += 1
        else:
            survivors.append(r)

    if len(survivors) < 4:
        return rows, {"fallback": True, "rejected": 0, "reasons": {}}

    survivors.sort(key=lambda x: g(x["features"], "h4_score"), reverse=True)
    if len(survivors) > TARGET_POOL:
        survivors = survivors[:TARGET_POOL]
    return survivors, {
        "fallback": False,
        "rejected": len(rows) - len(survivors),
        "reasons": dict(reject_reasons),
        "pool_size": len(survivors),
    }


def pick_top2(rows: list[dict]) -> list[dict]:
    ranked = sorted(rows, key=lambda x: g(x["features"], "h4_score"), reverse=True)
    return ranked[:2]


def eval_scan(rows: list[dict], picks: list[dict]) -> dict:
    actual_top2 = {r["symbol"] for r in sorted(rows, key=lambda x: x["outcome_rank"])[:2]}
    picked_syms = [r["symbol"] for r in picks]
    hits = len(set(picked_syms) & actual_top2)
    recall = hits / 2.0
    avg_ret = statistics.mean([r["max_up_4h"] for r in picks]) if picks else 0.0
    return {
        "hits": hits,
        "recall": recall,
        "avg_max_up": round(avg_ret, 4),
        "picked": picked_syms,
        "actual": sorted(actual_top2),
    }


def aggregate(scan_rows: list[dict]) -> dict:
    if not scan_rows:
        return {}
    n = len(scan_rows)
    total_hits = sum(r["hits"] for r in scan_rows)
    top2_hit_pct = total_hits / (n * 2) * 100
    both_hit_pct = sum(1 for r in scan_rows if r["hits"] == 2) / n * 100
    avg_recall = statistics.mean([r["recall"] for r in scan_rows]) * 100
    avg_return = statistics.mean([r["avg_max_up"] for r in scan_rows])
    recalls = [r["recall"] for r in scan_rows]
    stability = 100 - statistics.stdev(recalls) * 100 if len(recalls) > 1 else 0.0
    return {
        "scans": n,
        "top2_hit_pct": round(top2_hit_pct, 2),
        "both_hit_pct": round(both_hit_pct, 2),
        "winner_recall_pct": round(avg_recall, 2),
        "avg_return_pct": round(avg_return, 4),
        "blind_stability": round(max(0, stability), 2),
    }


def load_a6_baseline(ann_by_scan: dict[str, list[dict]]) -> list[dict]:
    if not P23_MATCH.exists():
        return []
    idx = {(r["scan_kst"], r["symbol"]): r for rows in ann_by_scan.values() for r in rows}
    rows_out: list[dict] = []
    for ln in P23_MATCH.open(encoding="utf-8"):
        m = json.loads(ln)
        scan = m["scan_kst"]
        scan_rows = ann_by_scan.get(scan, [])
        if len(scan_rows) < 4:
            continue
        picks = [idx[(scan, s)] for s in m.get("A6_top2", []) if (scan, s) in idx]
        if len(picks) < 2:
            continue
        ev = eval_scan(scan_rows, picks[:2])
        rows_out.append({"scan_kst": scan, **ev})
    return rows_out


def temporal_holdout(scans: list[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(scans)
    cut = int(len(ordered) * (1 - HOLDOUT_FRAC))
    return ordered[:cut], ordered[cut:]


def run_loo(ann_by_scan: dict[str, list[dict]], annotated: list[dict]) -> tuple[list[dict], list[dict]]:
    funnel_rows: list[dict] = []
    baseline_rows: list[dict] = []

    for scan, rows in ann_by_scan.items():
        if len(rows) < 4:
            continue
        train = [r for r in annotated if r["scan_kst"] != scan]
        rules = learn_reject_rules(train)
        survivors, meta = apply_funnel(rows, rules)
        picks = pick_top2(survivors)
        ev = eval_scan(rows, picks)
        funnel_rows.append({
            "scan_kst": scan,
            "split": "loo",
            **ev,
            "pool_size": meta.get("pool_size", len(rows)),
            "rejected": meta.get("rejected", 0),
            "fallback": meta.get("fallback", False),
            "loser_combos": rules["n_loser_combos"],
        })
        baseline_rows.append({"scan_kst": scan, **eval_scan(rows, pick_top2(rows))})

    return funnel_rows, baseline_rows


def run_holdout(ann_by_scan: dict[str, list[dict]], annotated: list[dict]) -> tuple[list[dict], list[dict]]:
    scans = [s for s, rows in ann_by_scan.items() if len(rows) >= 4]
    train_scans, test_scans = temporal_holdout(scans)
    train = [r for r in annotated if r["scan_kst"] in train_scans]
    rules = learn_reject_rules(train)

    funnel_rows: list[dict] = []
    baseline_rows: list[dict] = []
    for scan in test_scans:
        rows = ann_by_scan[scan]
        survivors, meta = apply_funnel(rows, rules)
        picks = pick_top2(survivors)
        funnel_rows.append({
            "scan_kst": scan,
            "split": "holdout",
            **eval_scan(rows, picks),
            "pool_size": meta.get("pool_size", len(rows)),
            "rejected": meta.get("rejected", 0),
            "fallback": meta.get("fallback", False),
        })
        baseline_rows.append({"scan_kst": scan, **eval_scan(rows, pick_top2(rows))})
    return funnel_rows, baseline_rows


def write_report(
    loo_funnel: dict,
    loo_baseline: dict,
    loo_a6: dict,
    ho_funnel: dict,
    ho_baseline: dict,
    rules_sample: dict,
) -> str:
    delta_loo = round(loo_funnel["top2_hit_pct"] - loo_baseline["top2_hit_pct"], 2)
    delta_ho = round(ho_funnel["top2_hit_pct"] - ho_baseline["top2_hit_pct"], 2)
    delta_a6 = round(loo_funnel["top2_hit_pct"] - loo_a6.get("top2_hit_pct", 0), 2)

    verdict = "REJECT"
    if delta_loo > 1.0 and delta_ho >= 0 and loo_funnel["avg_return_pct"] >= loo_baseline["avg_return_pct"]:
        verdict = "HOLDOUT_CANDIDATE"
    if delta_loo <= 0 and delta_ho <= 0:
        verdict = "REJECT"

    lines = [
        "############################################################",
        "SCOUT RESEARCH R001 — LOSER FUNNEL + ENERGY GATE",
        "############################################################",
        "",
        "Tracks: B (transition) + Loser Research | Formula: FROZEN (h4_score)",
        f"Dataset: Phase19 | LOO scans: {loo_funnel.get('scans', 0)} | Holdout scans: {ho_funnel.get('scans', 0)}",
        "",
        "=" * 62,
        "HYPOTHESIS",
        "=" * 62,
        "  Loser combo rejection + death transition + low energy gate",
        "  narrows universe before frozen h4_score Top2 selection.",
        "",
        "=" * 62,
        "LOO BLIND (leave-one-scan-out rules)",
        "=" * 62,
        f"  {'Model':<22} {'Top2Hit%':>9} {'BothHit%':>9} {'Recall%':>8} {'AvgRet%':>8} {'Stability':>9}",
        f"  {'h4 raw Top2':<22} {loo_baseline['top2_hit_pct']:>8.1f}% {loo_baseline['both_hit_pct']:>8.1f}% "
        f"{loo_baseline['winner_recall_pct']:>7.1f}% {loo_baseline['avg_return_pct']:>7.2f}% {loo_baseline['blind_stability']:>8.1f}",
        f"  {'A6 formula (P23)':<22} {loo_a6.get('top2_hit_pct', 0):>8.1f}% {loo_a6.get('both_hit_pct', 0):>8.1f}% "
        f"{loo_a6.get('winner_recall_pct', 0):>7.1f}% {loo_a6.get('avg_return_pct', 0):>7.2f}% {loo_a6.get('blind_stability', 0):>8.1f}",
        f"  {'R001 funnel+h4':<22} {loo_funnel['top2_hit_pct']:>8.1f}% {loo_funnel['both_hit_pct']:>8.1f}% "
        f"{loo_funnel['winner_recall_pct']:>7.1f}% {loo_funnel['avg_return_pct']:>7.2f}% {loo_funnel['blind_stability']:>8.1f}",
        f"  Delta vs h4 raw: {delta_loo:+.1f}pp | Delta vs A6: {delta_a6:+.1f}pp",
        "",
        "=" * 62,
        "TEMPORAL HOLDOUT (last 30% scans, train-on-rest rules)",
        "=" * 62,
        f"  {'h4 raw Top2':<22} {ho_baseline['top2_hit_pct']:>8.1f}% recall={ho_baseline['winner_recall_pct']:.1f}% "
        f"avg={ho_baseline['avg_return_pct']:.2f}%",
        f"  {'R001 funnel+h4':<22} {ho_funnel['top2_hit_pct']:>8.1f}% recall={ho_funnel['winner_recall_pct']:.1f}% "
        f"avg={ho_funnel['avg_return_pct']:.2f}%",
        f"  Holdout delta: {delta_ho:+.1f}pp",
        "",
        "=" * 62,
        "TRAIN RULES (sample from full-data profile)",
        "=" * 62,
        f"  Loser combos learned: {rules_sample['n_loser_combos']}",
        f"  Energy cutoff (p25 losers): {rules_sample['energy_cut']:.2f}",
        "",
        "=" * 62,
        "EXPERIMENT CHECKLIST",
        "=" * 62,
        "  1. Hypothesis: loser funnel before frozen formula improves Top2 blind",
        f"  2. Blind improved (LOO): {'YES' if delta_loo > 0 else 'NO'} ({delta_loo:+.1f}pp)",
        f"  3. Holdout maintained: {'YES' if delta_ho >= 0 else 'NO'} ({delta_ho:+.1f}pp)",
        "  4. Overfit risk: MEDIUM — combo rules from training; LOO mitigates",
        f"  5. Formula integration: {'NO — reject layer only' if verdict == 'REJECT' else 'CANDIDATE — reject gate only, not formula change'}",
        "  6. Next highest ROI: Trigger Survival Curve on focal Release->Expansion path",
        "",
        "=" * 62,
        f"VERDICT: {verdict}",
        "=" * 62,
        "Learning recommendation: NO_ACTION on formula. Reject gate stored as hypothesis.",
        "",
    ]
    lines.extend(mission_summary_lines())
    return "\n".join(lines)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated, ann_by_scan = load_annotated()

    loo_funnel_list, loo_baseline_list = run_loo(ann_by_scan, annotated)
    ho_funnel_list, ho_baseline_list = run_holdout(ann_by_scan, annotated)
    a6_list = load_a6_baseline(ann_by_scan)

    loo_funnel = aggregate(loo_funnel_list)
    loo_baseline = aggregate(loo_baseline_list)
    loo_a6 = aggregate(a6_list)
    ho_funnel = aggregate(ho_funnel_list)
    ho_baseline = aggregate(ho_baseline_list)

    rules_sample = learn_reject_rules(annotated)

    write_csv(OUT_DIR / "loo_funnel_scans.csv", loo_funnel_list)
    write_csv(OUT_DIR / "loo_baseline_scans.csv", loo_baseline_list)
    write_csv(OUT_DIR / "holdout_funnel_scans.csv", ho_funnel_list)
    write_csv(OUT_DIR / "summary.csv", [
        {"split": "loo", "model": "h4_raw", **loo_baseline},
        {"split": "loo", "model": "A6", **loo_a6},
        {"split": "loo", "model": "R001_funnel", **loo_funnel},
        {"split": "holdout", "model": "h4_raw", **ho_baseline},
        {"split": "holdout", "model": "R001_funnel", **ho_funnel},
    ])

    report = write_report(loo_funnel, loo_baseline, loo_a6, ho_funnel, ho_baseline, rules_sample)
    (OUT_DIR / "research_r001_report.txt").write_text(report, encoding="utf-8")
    safe_print(report)


if __name__ == "__main__":
    run()
