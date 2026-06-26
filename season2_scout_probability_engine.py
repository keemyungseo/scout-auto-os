"""
Scout Learning Season2 - Probability Engine (Research Prototype)

Scout final form: per-symbol probability output, NOT a condition sccreener.
Uses empirical cohort lookup from integrated panel (State + Trigger + Regime
+ Symbol Behaviour + Relative Market Position).

Research only. No trading conditions.
"""

import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from season2_p1_supply_probability_panel import (
    DATASETS,
    LOGS_DIR,
    build_records,
    load_rows,
    pf,
)

TARGET_DATES = ["2026-06-13", "2026-06-14", "2026-06-15", "2026-06-16"]
MIN_TRAINING_ROWS = 120
PERSIST_THRESHOLD = 0.0  # forward return >= 0 counts as trend persistence

SCORES_CSV = LOGS_DIR / "season2_scout_probability_scores.csv"
COHORTS_CSV = LOGS_DIR / "season2_scout_engine_cohorts.csv"
REPORT_TXT = LOGS_DIR / "season2_scout_engine_report.txt"

DATE_DATA_MAP = {
    "2026-06-13": LOGS_DIR / "top10_gainer_learning_20260613.csv",
    "2026-06-14": LOGS_DIR / "top3_gainers_20260614_enriched.csv",
    "2026-06-15": LOGS_DIR / "top10_gainer_learning_20260615.csv",
    "2026-06-16": LOGS_DIR / "top10_gainer_learning_20260616.csv",
}

COLLECTOR_SCRIPTS = {
    "2026-06-13": "top10_gainer_learning_20260613.py",
    "2026-06-15": "top10_gainer_learning_20260615.py",
}


@dataclass
class CohortStats:
    n: int = 0
    persist_2h: list[float] = field(default_factory=list)
    persist_4h: list[float] = field(default_factory=list)
    persist_6h: list[float] = field(default_factory=list)
    forward_2h: list[float] = field(default_factory=list)
    forward_4h: list[float] = field(default_factory=list)
    forward_6h: list[float] = field(default_factory=list)
    forward_12h: list[float] = field(default_factory=list)
    drawdown: list[float] = field(default_factory=list)
    collapse: list[int] = field(default_factory=list)

    def add(self, record: dict) -> None:
        self.n += 1
        for horizon, attr in [
            ("forward_2h", "persist_2h"),
            ("forward_4h", "persist_4h"),
            ("forward_6h", "persist_6h"),
        ]:
            value = record.get(horizon)
            if value is None:
                continue
            if horizon == "forward_2h" and value == 0.0 and record.get("forward_4h") not in (None, 0.0):
                continue
            getattr(self, attr).append(1.0 if value >= PERSIST_THRESHOLD else 0.0)
            getattr(self, horizon).append(value)

        for horizon in ("forward_12h",):
            value = record.get(horizon)
            if value is not None:
                getattr(self, horizon).append(value)

        dd = record.get("max_drawdown")
        if dd is not None:
            self.drawdown.append(dd)

        self.collapse.append(1 if record.get("collapse_label") == "YES" else 0)

    def rate(self, values: list[float]) -> float | None:
        return statistics.mean(values) * 100 if values else None

    def median(self, values: list[float]) -> float | None:
        return statistics.median(values) if values else None


def attach_drawdown(records: list[dict]) -> None:
    by_key: dict[tuple[str, str], float] = {}
    for path in DATASETS:
        for row in load_rows(path):
            scan = row.get("scan_time_kst", "")
            symbol = row.get("symbol", "")
            dd = pf(row.get("max_drawdown"))
            if scan and symbol and dd is not None:
                by_key[(scan, symbol)] = dd

    for record in records:
        record["max_drawdown"] = by_key.get((record["scan_time"], record["symbol"]))


def rank_tier(rank: int) -> str:
    if rank <= 3:
        return "top3"
    if rank <= 7:
        return "mid"
    return "bottom"


def symbol_behaviour(symbol: str, history: dict[str, list[dict]]) -> str:
    prior = history.get(symbol, [])
    if not prior:
        return "new"
    ranks = [record["rank"] for record in prior if record.get("rank")]
    collapses = sum(1 for record in prior if record.get("collapse_label") == "YES")
    if ranks and statistics.mean(ranks) <= 3:
        return "repeat_leader"
    if collapses >= 2:
        return "collapse_prone"
    if len(prior) >= 3:
        return "frequent"
    return "seen_before"


def relative_position_tier(record: dict, scan_peers: list[dict]) -> str:
    returns = sorted(
        peer["return_24h_at_scan"]
        for peer in scan_peers
        if peer.get("return_24h_at_scan") is not None
    )
    value = record.get("return_24h_at_scan")
    if not returns or value is None:
        return "mid_return"
    pct = sum(1 for item in returns if item <= value) / len(returns)
    if pct >= 0.8:
        return "return_top"
    if pct <= 0.3:
        return "return_bottom"
    return "return_mid"


def cohort_keys(record: dict, symbol_tier: str, rel_tier: str) -> list[str]:
    base = [
        (
            f"{record['state_scan']}|{record['trigger_bundle']}|{record['market_regime']}"
            f"|{rank_tier(record['rank'])}|{symbol_tier}|{rel_tier}"
        ),
        (
            f"{record['state_scan']}|{record['trigger_bundle']}|{record['market_regime']}"
            f"|{rank_tier(record['rank'])}|{symbol_tier}"
        ),
        f"{record['state_scan']}|{record['trigger_bundle']}|{record['market_regime']}|{rank_tier(record['rank'])}",
        f"{record['state_scan']}|{record['trigger_bundle']}|{record['market_regime']}",
        f"{record['state_scan']}|{record['trigger_bundle']}",
        f"{record['state_scan']}|{record['market_regime']}",
        record["state_scan"],
        "__global__",
    ]
    return base


def confidence_label(n: int, key_level: int) -> str:
    effective = n * (1.0 + 0.15 * key_level)
    if effective >= 20:
        return "high"
    if effective >= 10:
        return "medium"
    if n >= 3:
        return "low"
    return "hypothesis"


def lookup_cohort(
    cohorts: dict[str, CohortStats], keys: list[str]
) -> tuple[CohortStats | None, str, int]:
    for level, key in enumerate(keys):
        stats = cohorts.get(key)
        if stats and stats.n >= 3:
            return stats, key, level
    for level, key in enumerate(keys):
        stats = cohorts.get(key)
        if stats and stats.n >= 1:
            return stats, key, level
    return None, "__none__", -1


def holding_recommendation(p2: float | None, p4: float | None, p6: float | None, collapse: float | None) -> str:
    p2 = p2 or 0.0
    p4 = p4 or 0.0
    p6 = p6 or 0.0
    collapse = collapse or 0.0

    if collapse >= 30:
        return "Short"
    if p6 >= 55 and p4 >= 50 and collapse < 15:
        return "Long"
    if p4 >= 45 and collapse < 20:
        return "Medium"
    if p2 >= 50 and p6 < 40:
        return "Short"
    return "Medium"


def recommendation_score(
    persist_4h: float | None,
    expected_return: float | None,
    collapse_prob: float | None,
    market_relative_pct: float,
    confidence: str,
) -> float:
    p4 = (persist_4h or 0.0) / 100
    ret = max(min((expected_return or 0.0) / 12.0, 1.0), -1.0)
    ret_norm = (ret + 1.0) / 2.0
    collapse = (collapse_prob or 0.0) / 100
    rel = market_relative_pct / 100
    conf_map = {"high": 1.0, "medium": 0.75, "low": 0.5, "hypothesis": 0.25}
    conf = conf_map.get(confidence, 0.25)

    raw = 0.25 * p4 + 0.25 * ret_norm + 0.25 * (1 - collapse) + 0.15 * rel + 0.10 * conf
    return round(max(min(raw * 10, 10.0), 0.0), 1)


def build_training_cohorts(train: list[dict]) -> dict[str, CohortStats]:
    history: dict[str, list[dict]] = defaultdict(list)
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for record in train:
        by_scan[record["scan_time"]].append(record)

    cohorts: dict[str, CohortStats] = defaultdict(CohortStats)
    for record in sorted(train, key=lambda item: item["scan_time"]):
        peers = by_scan[record["scan_time"]]
        sym_tier = symbol_behaviour(record["symbol"], history)
        rel_tier = relative_position_tier(record, peers)
        for key in cohort_keys(record, sym_tier, rel_tier):
            cohorts[key].add(record)
        history[record["symbol"]].append(record)

    return cohorts


def predict_record(
    record: dict,
    cohorts: dict[str, CohortStats],
    history: dict[str, list[dict]],
    scan_peers: list[dict],
) -> dict:
    sym_tier = symbol_behaviour(record["symbol"], history)
    rel_tier = relative_position_tier(record, scan_peers)
    keys = cohort_keys(record, sym_tier, rel_tier)
    stats, matched_key, level = lookup_cohort(cohorts, keys)

    if stats is None:
        stats = CohortStats()
        matched_key = "__none__"
        level = -1

    persist_2h = stats.rate(stats.persist_2h)
    persist_4h = stats.rate(stats.persist_4h)
    persist_6h = stats.rate(stats.persist_6h)
    expected_return = stats.median(stats.forward_6h) or stats.median(stats.forward_4h)
    expected_drawdown = stats.median(stats.drawdown)
    collapse_prob = stats.rate([float(value) for value in stats.collapse])

    conf = confidence_label(stats.n, max(0, 7 - level))

    return {
        "date": record["date"],
        "symbol": record["symbol"],
        "scan_time": record["scan_time"],
        "rank": record["rank"],
        "state_scan": record["state_scan"],
        "trigger_bundle": record["trigger_bundle"],
        "market_regime": record["market_regime"],
        "symbol_behaviour": sym_tier,
        "relative_return_tier": rel_tier,
        "cohort_key": matched_key,
        "cohort_n": stats.n,
        "trend_persistence_2h_pct": round(persist_2h, 1) if persist_2h is not None else "",
        "trend_persistence_4h_pct": round(persist_4h, 1) if persist_4h is not None else "",
        "trend_persistence_6h_pct": round(persist_6h, 1) if persist_6h is not None else "",
        "expected_return_pct": round(expected_return, 2) if expected_return is not None else "",
        "expected_drawdown_pct": round(expected_drawdown, 2) if expected_drawdown is not None else "",
        "collapse_probability_pct": round(collapse_prob, 1) if collapse_prob is not None else "",
        "confidence": conf,
        "actual_forward_6h": record.get("forward_6h"),
        "actual_collapse": record.get("collapse_label"),
        "actual_supply_label": record.get("supply_label"),
        "_expected_return_raw": expected_return,
        "_persist_4h_raw": persist_4h,
        "_collapse_raw": collapse_prob,
        "_confidence_raw": conf,
    }


def pre_rank_score(row: dict) -> float:
    return recommendation_score(
        row.get("_persist_4h_raw"),
        row.get("_expected_return_raw"),
        row.get("_collapse_raw"),
        50.0,
        row.get("_confidence_raw", "hypothesis"),
    )


def rank_within_scans(scored: list[dict]) -> None:
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_scan[row["scan_time"]].append(row)

    for rows in by_scan.values():
        for row in rows:
            row["_pre_score"] = pre_rank_score(row)

        rows.sort(
            key=lambda item: (
                item.get("_pre_score", 0),
                item.get("_expected_return_raw") or -999,
            ),
            reverse=True,
        )
        total = len(rows)
        for index, row in enumerate(rows):
            rank_pct = (total - index) / total * 100
            row["market_relative_rank_pct"] = round(rank_pct, 1)
            row["market_relative_rank_label"] = f"Top {rank_pct:.0f}%"
            row["recommendation_score"] = recommendation_score(
                row.get("_persist_4h_raw"),
                row.get("_expected_return_raw"),
                row.get("_collapse_raw"),
                rank_pct,
                row.get("_confidence_raw", "hypothesis"),
            )
            row["recommended_holding"] = holding_recommendation(
                pf(row.get("trend_persistence_2h_pct")),
                pf(row.get("trend_persistence_4h_pct")),
                pf(row.get("trend_persistence_6h_pct")),
                pf(row.get("collapse_probability_pct")),
            )


def leave_one_date_out(records: list[dict]) -> list[str]:
    lines: list[str] = []
    dates = sorted({record["date"] for record in records})
    for holdout in dates:
        train = [record for record in records if record["date"] != holdout]
        test = [record for record in records if record["date"] == holdout]
        if not test:
            continue
        cohorts = build_training_cohorts(train)
        history: dict[str, list[dict]] = defaultdict(list)
        by_scan: dict[str, list[dict]] = defaultdict(list)
        for record in test:
            by_scan[record["scan_time"]].append(record)

        scored: list[dict] = []
        for record in sorted(test, key=lambda item: item["scan_time"]):
            scored.append(
                predict_record(record, cohorts, history, by_scan[record["scan_time"]])
            )
            history[record["symbol"]].append(record)

        rank_within_scans(scored)

        hits = 0
        total = 0
        for row in scored:
            actual = row.get("actual_forward_6h")
            if actual is None:
                continue
            total += 1
            if row["recommendation_score"] >= 5 and actual >= 3:
                hits += 1
            elif row["recommendation_score"] < 5 and actual < 3:
                hits += 1

        top_pick_correct = 0
        scan_groups: dict[str, list[dict]] = defaultdict(list)
        for row in scored:
            scan_groups[row["scan_time"]].append(row)
        for rows in scan_groups.values():
            best_pred = max(rows, key=lambda item: item["recommendation_score"])
            actual_best = max(
                rows,
                key=lambda item: item.get("actual_forward_6h") or -9999,
            )
            if best_pred["symbol"] == actual_best["symbol"]:
                top_pick_correct += 1

        lines.append(
            f"  holdout {holdout}: n={len(test)} score-alignment={hits}/{total} "
            f"top1-scan-match={top_pick_correct}/{len(scan_groups)}"
        )
    return lines


def find_counterexamples(scored: list[dict]) -> list[str]:
    lines: list[str] = []
    by_cohort: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_cohort[row["cohort_key"]].append(row)

    shown = 0
    for key, rows in sorted(by_cohort.items(), key=lambda item: -len(item[1])):
        if len(rows) < 2:
            continue
        scores = {round(row["recommendation_score"], 0) for row in rows}
        outcomes = {row.get("actual_supply_label") for row in rows}
        if len(scores) <= 1 and len(outcomes) <= 1:
            continue
        if max(scores) >= 6 and min(scores) <= 3:
            lines.append(f"  cohort {key}:")
            for row in rows[:4]:
                lines.append(
                    f"    {row['symbol']} {row['scan_time'][11:16]} "
                    f"score={row['recommendation_score']} "
                    f"actual_f6={row.get('actual_forward_6h')} "
                    f"label={row.get('actual_supply_label')}"
                )
            shown += 1
        if shown >= 5:
            break
    return lines


def ensure_datasets() -> list[str]:
    missing: list[str] = []
    for date, path in DATE_DATA_MAP.items():
        if path.exists():
            continue
        missing.append(date)
        script = COLLECTOR_SCRIPTS.get(date)
        if script and Path(script).exists():
            print(f"[auto-expand] collecting {date} via {script} ...")
            result = subprocess.run([sys.executable, script], capture_output=False)
            if result.returncode != 0:
                print(f"[auto-expand] failed for {date}")
            elif path.exists():
                missing.remove(date)
    return missing


def load_all_records() -> list[dict]:
    missing = ensure_datasets()
    records = build_records()
    attach_drawdown(records)
    if missing:
        print(f"[warn] missing dates (no collector): {', '.join(missing)}")
    return records


def save_scores(scored: list[dict]) -> None:
    fields = [
        "date",
        "symbol",
        "scan_time",
        "rank",
        "state_scan",
        "trigger_bundle",
        "market_regime",
        "symbol_behaviour",
        "relative_return_tier",
        "trend_persistence_2h_pct",
        "trend_persistence_4h_pct",
        "trend_persistence_6h_pct",
        "expected_return_pct",
        "expected_drawdown_pct",
        "market_relative_rank_pct",
        "market_relative_rank_label",
        "collapse_probability_pct",
        "confidence",
        "recommendation_score",
        "recommended_holding",
        "cohort_key",
        "cohort_n",
        "actual_forward_6h",
        "actual_collapse",
        "actual_supply_label",
    ]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with SCORES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in scored:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_cohorts(cohorts: dict[str, CohortStats]) -> None:
    with COHORTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cohort_key",
                "n",
                "persist_2h_pct",
                "persist_4h_pct",
                "persist_6h_pct",
                "median_return_6h",
                "median_drawdown",
                "collapse_pct",
            ],
        )
        writer.writeheader()
        for key, stats in sorted(cohorts.items(), key=lambda item: -item[1].n):
            if stats.n == 0:
                continue
            writer.writerow(
                {
                    "cohort_key": key,
                    "n": stats.n,
                    "persist_2h_pct": f"{stats.rate(stats.persist_2h) or 0:.1f}",
                    "persist_4h_pct": f"{stats.rate(stats.persist_4h) or 0:.1f}",
                    "persist_6h_pct": f"{stats.rate(stats.persist_6h) or 0:.1f}",
                    "median_return_6h": f"{stats.median(stats.forward_6h) or 0:.2f}",
                    "median_drawdown": f"{stats.median(stats.drawdown) or 0:.2f}",
                    "collapse_pct": f"{stats.rate([float(v) for v in stats.collapse]) or 0:.1f}",
                }
            )


def format_symbol_card(row: dict) -> str:
    return (
        f"{row['symbol']} @ {row['scan_time']}\n"
        f"  Trend Persistence 2h: {row['trend_persistence_2h_pct']}%\n"
        f"  Trend Persistence 4h: {row['trend_persistence_4h_pct']}%\n"
        f"  Trend Persistence 6h: {row['trend_persistence_6h_pct']}%\n"
        f"  Expected Return: {row['expected_return_pct']}%\n"
        f"  Expected Drawdown: {row['expected_drawdown_pct']}%\n"
        f"  Market Relative Rank: {row['market_relative_rank_label']}\n"
        f"  Collapse Probability: {row['collapse_probability_pct']}%\n"
        f"  Confidence: {row['confidence']} (cohort n={row['cohort_n']})\n"
        f"  Recommendation Score: {row['recommendation_score']}/10\n"
        f"  Recommended Holding: {row['recommended_holding']}"
    )


def write_report(records: list[dict], scored: list[dict], cohorts: dict[str, CohortStats]) -> None:
    lines: list[str] = []
    lines.append("===== SCOUT PROBABILITY ENGINE (Season2 Research) =====")
    lines.append("")
    lines.append("Architecture: empirical cohort lookup (NOT condition screener)")
    lines.append("Priority features: State + Trigger + Regime + Symbol Behaviour + Relative Position")
    lines.append("Goal: relative best candidate within each scan, not absolute HIGH/LOW label")
    lines.append("")
    lines.append(f"Training rows: {len(records)}")
    lines.append(f"Cohort buckets: {sum(1 for stats in cohorts.values() if stats.n > 0)}")
    lines.append("")
    lines.append("--- Leave-one-date-out validation ---")
    lines.extend(leave_one_date_out(records))
    lines.append("")
    lines.append("--- Counterexamples (same cohort, different score/outcome) ---")
    counter = find_counterexamples(scored)
    lines.extend(counter or ["  (none in top groups)"])
    lines.append("")
    lines.append("--- Example outputs (latest scan, top 3 by recommendation) ---")
    latest_scan = max(row["scan_time"] for row in scored)
    latest_rows = sorted(
        [row for row in scored if row["scan_time"] == latest_scan],
        key=lambda item: item["recommendation_score"],
        reverse=True,
    )[:3]
    for row in latest_rows:
        lines.append(format_symbol_card(row))
        lines.append("")

    lines.append("--- Research notes ---")
    lines.append(" forward_2h=0 artifact excluded from 2h persistence cohort stats")
    lines.append(" REVISE: absolute supply_label tiers -> use relative rank within scan")
    lines.append(" KEEP: +4h checkpoint still needed for live update (engine v1 is scan-time prior)")
    lines.append(" DROP: single-feature dominant trigger as primary score driver")
    lines.append(" NEXT: +4h state update layer; true intrabar drawdown; expand dates to 06-16+")
    lines.append(" NEXT: full-universe scan (not only TOP10) for market-relative rank")
    lines.append("")
    lines.append(f"Scores CSV: {SCORES_CSV}")
    lines.append(f"Cohorts CSV: {COHORTS_CSV}")
    lines.append("=" * 52)

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    records = load_all_records()
    if len(records) < MIN_TRAINING_ROWS:
        print(f"[warn] training rows {len(records)} < target {MIN_TRAINING_ROWS}")

    cohorts = build_training_cohorts(records)
    history: dict[str, list[dict]] = defaultdict(list)
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_scan[record["scan_time"]].append(record)

    scored: list[dict] = []
    for record in sorted(records, key=lambda item: (item["scan_time"], item["rank"])):
        row = predict_record(record, cohorts, history, by_scan[record["scan_time"]])
        scored.append(row)
        history[record["symbol"]].append(record)

    rank_within_scans(scored)
    save_scores(scored)
    save_cohorts(cohorts)
    write_report(records, scored, cohorts)

    print("===== SCOUT PROBABILITY ENGINE =====")
    print(f"Scored rows: {len(scored)}")
    print(f"Output: {SCORES_CSV}")
    print(f"Cohorts: {COHORTS_CSV}")
    print(f"Report: {REPORT_TXT}")
    print("")
    latest_scan = max(row["scan_time"] for row in scored)
    top = max(
        (row for row in scored if row["scan_time"] == latest_scan),
        key=lambda item: item["recommendation_score"],
    )
    print(format_symbol_card(top))
    print("=" * 36)


if __name__ == "__main__":
    main()
