"""
Scout Learning Season2 - P2 Prediction Error Reduction & Intelligence Upgrade

Research only. No new trading conditions.
Goal: reduce v1 prediction error via failure analysis, not feature sprawl.
"""

import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from season2_p1_supply_probability_panel import DATASETS, LOGS_DIR, build_records, load_rows, pf
from season2_scout_probability_engine import (
    CohortStats,
    attach_drawdown,
    build_training_cohorts,
    cohort_keys,
    confidence_label,
    holding_recommendation,
    lookup_cohort,
    predict_record,
    rank_within_scans,
    rank_tier,
    recommendation_score,
    relative_position_tier,
)

LOGS_DIR.mkdir(parents=True, exist_ok=True)

ERRORS_CSV = LOGS_DIR / "season2_p2_prediction_errors.csv"
BEHAVIOUR_CSV = LOGS_DIR / "season2_p2_symbol_behaviour.csv"
PERSISTENCE_CSV = LOGS_DIR / "season2_p2_persistence_curves.csv"
CALIBRATION_CSV = LOGS_DIR / "season2_p2_return_calibration.csv"
DRAWDOWN_CSV = LOGS_DIR / "season2_p2_drawdown_analysis.csv"
COUNTER_CSV = LOGS_DIR / "season2_p2_counterexamples.csv"
REGIME_CSV = LOGS_DIR / "season2_p2_regime_behaviour.csv"
SCORES_V2_CSV = LOGS_DIR / "season2_scout_probability_scores_v2.csv"
REPORT_TXT = LOGS_DIR / "season2_p2_research_report.txt"

HORIZONS = ("30m", "1h", "2h", "4h", "6h", "12h", "24h")
PERSIST_THRESHOLD = 0.0
MIN_N_CONFIRM = 3


def enrich_records(records: list[dict]) -> None:
    extra: dict[tuple[str, str], dict] = {}
    for path in DATASETS:
        for row in load_rows(path):
            scan = row.get("scan_time_kst", "")
            symbol = row.get("symbol", "")
            if not scan or not symbol:
                continue
            extra[(scan, symbol)] = {
                "forward_24h": pf(row.get("forward_24h")),
                "max_drawdown": pf(row.get("max_drawdown")),
                "atr_ratio": pf(row.get("atr_ratio")),
                "atr_percent": pf(row.get("atr_percent")),
                "return_prev_2h": pf(row.get("return_prev_2h_percent")),
            }

    for record in records:
        data = extra.get((record["scan_time"], record["symbol"]), {})
        for key, value in data.items():
            if value is not None:
                record[key] = value


def forward_at_horizon(record: dict, horizon: str) -> float | None:
    mapping = {
        "2h": "forward_2h",
        "4h": "forward_4h",
        "6h": "forward_6h",
        "12h": "forward_12h",
        "24h": "forward_24h",
    }
    if horizon in mapping:
        value = record.get(mapping[horizon])
        if horizon == "2h" and value == 0.0 and record.get("forward_4h") not in (None, 0.0):
            return None
        return value

    f2 = record.get("forward_2h")
    if f2 is None or (f2 == 0.0 and record.get("forward_4h") not in (None, 0.0)):
        return None
    if horizon == "1h":
        return f2 / 2.0
    if horizon == "30m":
        return f2 / 4.0
    return None


def refined_regime(day_records: list[dict]) -> str:
    from season2_p1_supply_probability_panel import classify_regime

    base = classify_regime(day_records)
    atrs = [record["atr_ratio"] for record in day_records if record.get("atr_ratio") is not None]
    avg_atr = statistics.mean(atrs) if atrs else 1.0
    vol_tag = "High Volatility" if avg_atr >= 1.5 else "Low Volatility" if avg_atr <= 1.1 else ""

    if base == "Rotation" and vol_tag:
        return f"Rotation ({vol_tag})"
    if "Leader Concentration" in base and vol_tag:
        return f"{base} ({vol_tag})"
    if base == "Broad Rally":
        return base
    if vol_tag and base in ("Mixed", "Slow Market"):
        return f"{base} ({vol_tag})"
    return base


def apply_refined_regimes(records: list[dict]) -> None:
    by_date: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_date[record["date"]].append(record)
    regimes = {date: refined_regime(rows) for date, rows in by_date.items()}
    for record in records:
        record["market_regime_v2"] = regimes[record["date"]]
        record["market_regime_base"] = record["market_regime"]


@dataclass
class SymbolMemory:
    symbol: str
    appearances: int = 0
    days_seen: set = field(default_factory=set)
    ranks: list = field(default_factory=list)
    forward_6h: list = field(default_factory=list)
    collapses: int = 0
    high_supply: int = 0
    regimes: list = field(default_factory=list)
    drawdowns: list = field(default_factory=list)
    scan_times: list = field(default_factory=list)

    def observe(self, record: dict) -> None:
        self.appearances += 1
        self.days_seen.add(record["date"])
        self.scan_times.append(record["scan_time"])
        if record.get("rank"):
            self.ranks.append(record["rank"])
        f6 = record.get("forward_6h")
        if f6 is not None:
            self.forward_6h.append(f6)
        if record.get("collapse_label") == "YES":
            self.collapses += 1
        if record.get("supply_label") == "HIGH_SUPPLY":
            self.high_supply += 1
        self.regimes.append(record.get("market_regime_v2") or record.get("market_regime", ""))
        if record.get("max_drawdown") is not None:
            self.drawdowns.append(record["max_drawdown"])


def classify_symbol_behaviour(memory: SymbolMemory, current_regime: str) -> str:
    if memory.appearances == 0:
        return "new"

    avg_rank = statistics.mean(memory.ranks) if memory.ranks else 99
    collapse_rate = memory.collapses / memory.appearances
    high_rate = memory.high_supply / memory.appearances
    median_f6 = statistics.median(memory.forward_6h) if memory.forward_6h else 0.0

    rotation_hits = sum(1 for regime in memory.regimes if regime.startswith("Rotation"))
    leader_hits = sum(1 for regime in memory.regimes if "Leader Concentration" in regime)

    had_collapse = memory.collapses > 0
    recent_recovery = had_collapse and high_rate >= 0.25

    if collapse_rate >= 0.35 and memory.appearances >= 2:
        return "collapse_prone"
    if recent_recovery:
        return "recovery"
    if avg_rank <= 3 and memory.appearances >= 2:
        if median_f6 < 0:
            return "leader_fade"
        return "repeat_leader"
    if rotation_hits >= 2 and median_f6 >= 3:
        return "rotation_favorite"
    if leader_hits >= 2 and high_rate >= 0.2:
        return "leader_favorite"
    if memory.appearances >= 4:
        return "frequent"
    if memory.appearances >= 2:
        return "seen_before"
    return "new"


def behaviour_window_stats(memory: SymbolMemory, scan_time: str, days: int) -> dict:
    scan_dt = datetime.strptime(scan_time, "%Y-%m-%d %H:%M:%S")
    cutoff = scan_dt.timestamp() - days * 86400
    # chronological prior only - caller passes prior records
    return {
        "window_days": days,
        "note": "calendar span limited to 3 study dates; window uses prior scans only",
    }


def build_symbol_memories(records: list[dict]) -> dict[str, SymbolMemory]:
    memories: dict[str, SymbolMemory] = {}
    for record in sorted(records, key=lambda item: item["scan_time"]):
        symbol = record["symbol"]
        if symbol not in memories:
            memories[symbol] = SymbolMemory(symbol=symbol)
        memories[symbol].observe(record)
    return memories


def behaviour_at_scan(symbol: str, history: list[dict], regime: str) -> str:
    memory = SymbolMemory(symbol=symbol)
    for record in history:
        memory.observe(record)
    return classify_symbol_behaviour(memory, regime)


@dataclass
class PersistenceCurve:
    n: int = 0
    by_horizon: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, record: dict) -> None:
        self.n += 1
        for horizon in HORIZONS:
            value = forward_at_horizon(record, horizon)
            if value is not None:
                self.by_horizon[horizon].append(1.0 if value >= PERSIST_THRESHOLD else 0.0)

    def curve(self) -> dict[str, float | None]:
        return {
            horizon: statistics.mean(values) * 100 if values else None
            for horizon, values in self.by_horizon.items()
        }


@dataclass
class CohortStatsV2(CohortStats):
    curves: PersistenceCurve = field(default_factory=PersistenceCurve)
    forward_24h: list[float] = field(default_factory=list)

    def add(self, record: dict) -> None:
        super().add(record)
        self.curves.add(record)
        value = record.get("forward_24h")
        if value is not None:
            self.forward_24h.append(value)


def build_cohorts_v2(train: list[dict]) -> dict[str, CohortStatsV2]:
    history: dict[str, list[dict]] = defaultdict(list)
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for record in train:
        by_scan[record["scan_time"]].append(record)

    cohorts: dict[str, CohortStatsV2] = defaultdict(CohortStatsV2)
    for record in sorted(train, key=lambda item: item["scan_time"]):
        peers = by_scan[record["scan_time"]]
        regime = record.get("market_regime_v2") or record["market_regime"]
        sym_tier = behaviour_at_scan(record["symbol"], history[record["symbol"]], regime)
        rel_tier = relative_position_tier(record, peers)
        record_for_keys = {**record, "market_regime": regime, "_behaviour": sym_tier}
        keys = cohort_keys_v2(record_for_keys, sym_tier, rel_tier)
        for key in keys:
            cohorts[key].add(record)
        history[record["symbol"]].append(record)
    return cohorts


def cohort_keys_v2(record: dict, symbol_tier: str, rel_tier: str) -> list[str]:
    regime = record.get("market_regime_v2") or record["market_regime"]
    patched = {**record, "market_regime": regime}
    keys = cohort_keys(patched, symbol_tier, rel_tier)
    behaviour_key = (
        f"{record['state_scan']}|{record['trigger_bundle']}|{regime}|{symbol_tier}"
    )
    return [behaviour_key] + keys


def confidence_v2(n: int, key_level: int) -> str:
    effective = n * (1.0 + 0.12 * key_level)
    if n >= 30 and effective >= 35:
        return "production"
    if effective >= 20:
        return "high"
    if effective >= 10:
        return "medium"
    if n >= 3:
        return "low"
    return "hypothesis"


def calibrate_return(
    raw: float | None,
    confidence: str,
    behaviour: str,
    regime: str,
    global_median: float,
) -> float | None:
    if raw is None:
        return None
    adj = raw
    if behaviour == "collapse_prone":
        adj *= 0.65
    elif behaviour == "leader_fade":
        adj *= 0.55
    elif behaviour == "recovery":
        adj *= 1.08
    elif behaviour == "rotation_favorite":
        adj *= 1.05
    if "Leader Concentration" in regime and behaviour in ("repeat_leader", "leader_fade"):
        adj *= 0.70
    shrink = {"production": 0.05, "high": 0.15, "medium": 0.30, "low": 0.50, "hypothesis": 0.70}
    adj = adj * (1 - shrink[confidence]) + global_median * shrink[confidence]
    return adj


def calibrate_collapse(
    raw: float | None,
    behaviour: str,
    regime: str,
    drawdown: float | None,
) -> float | None:
    if raw is None:
        return None
    adj = raw
    if behaviour == "collapse_prone":
        adj += 12
    elif behaviour == "recovery":
        adj -= 8
    if "Leader Concentration (extreme)" in regime:
        adj += 5
    if drawdown is not None and drawdown >= 25:
        adj += 5
    return min(max(adj, 0), 95)


def calibrate_drawdown(
    cohort_dd: float | None,
    behaviour: str,
    actual_high_supply_dds: list[float],
) -> float | None:
    if behaviour == "collapse_prone" and cohort_dd is not None:
        return cohort_dd * 1.15
    if behaviour == "recovery" and cohort_dd is not None:
        return cohort_dd * 0.90
    if actual_high_supply_dds:
        return statistics.median(actual_high_supply_dds)
    return cohort_dd


def score_v2(
    persist_4h: float | None,
    expected_return: float | None,
    collapse_prob: float | None,
    market_relative_pct: float,
    confidence: str,
    behaviour: str,
) -> float:
    base = recommendation_score(
        persist_4h, expected_return, collapse_prob, market_relative_pct, confidence
    )
    penalty = 0.0
    if behaviour in ("collapse_prone", "leader_fade"):
        penalty += 1.2
    if behaviour == "recovery":
        penalty -= 0.4
    if confidence in ("hypothesis", "low"):
        penalty += 0.5
    return round(max(min(base - penalty, 10.0), 0.0), 1)


def action_label(score: float, collapse: float | None, confidence: str) -> str:
    collapse = collapse or 0
    if collapse >= 35:
        return "Avoid"
    if score >= 7.5 and collapse < 20 and confidence in ("medium", "high", "production"):
        return "Prioritize"
    if score >= 6 and collapse < 25:
        return "Consider"
    if score >= 4:
        return "Watch"
    return "Avoid"


def reason_and_caution(
    behaviour: str,
    regime: str,
    state: str,
    collapse: float | None,
    confidence: str,
) -> tuple[str, str]:
    reasons: list[str] = []
    cautions: list[str] = []

    if behaviour == "rotation_favorite":
        reasons.append("symbol history strong in Rotation regime")
    if behaviour == "repeat_leader":
        reasons.append("repeat top-rank appearances with positive prior")
    if behaviour == "recovery":
        reasons.append("prior collapse followed by recovery pattern")
    if state in ("Warm-up", "Compression"):
        reasons.append(f"scan state {state} historically less exhausted")
    if not reasons:
        reasons.append("cohort empirical persistence and return prior")

    if behaviour in ("collapse_prone", "leader_fade"):
        cautions.append("symbol collapse-prone or leader-fade history")
    if "Leader Concentration (extreme)" in regime:
        cautions.append("extreme leader concentration day: higher collapse tail")
    if confidence in ("hypothesis", "low"):
        cautions.append(f"confidence {confidence}: small cohort sample")
    if (collapse or 0) >= 25:
        cautions.append("elevated collapse probability")
    if not cautions:
        cautions.append("monitor +4h state before commitment")

    return "; ".join(reasons), "; ".join(cautions)


def predict_v2(
    record: dict,
    cohorts: dict[str, CohortStatsV2],
    history: dict[str, list[dict]],
    scan_peers: list[dict],
    global_median_f6: float,
    high_supply_dds_by_behaviour: dict[str, list[float]],
) -> dict:
    regime = record.get("market_regime_v2") or record["market_regime"]
    sym_tier = behaviour_at_scan(record["symbol"], history[record["symbol"]], regime)
    rel_tier = relative_position_tier(record, scan_peers)
    patched = {**record, "market_regime": regime}
    keys = cohort_keys_v2(patched, sym_tier, rel_tier)
    stats, matched_key, level = lookup_cohort(cohorts, keys)

    if stats is None:
        stats = CohortStatsV2()
        matched_key = "__none__"
        level = -1

    curve = stats.curves.curve() if isinstance(stats, CohortStatsV2) else {}
    conf = confidence_v2(stats.n, max(0, 8 - level))

    raw_return = stats.median(stats.forward_6h) or stats.median(stats.forward_4h)
    expected_return = calibrate_return(raw_return, conf, sym_tier, regime, global_median_f6)
    raw_collapse = stats.rate([float(v) for v in stats.collapse])
    raw_dd = stats.median(stats.drawdown)
    expected_dd = calibrate_drawdown(
        raw_dd, sym_tier, high_supply_dds_by_behaviour.get(sym_tier, [])
    )
    collapse_prob = calibrate_collapse(raw_collapse, sym_tier, regime, expected_dd)

    row = {
        "date": record["date"],
        "symbol": record["symbol"],
        "scan_time": record["scan_time"],
        "rank": record["rank"],
        "state_scan": record["state_scan"],
        "trigger_bundle": record["trigger_bundle"],
        "market_regime": regime,
        "symbol_behaviour": sym_tier,
        "relative_return_tier": rel_tier,
        "cohort_key": matched_key,
        "cohort_n": stats.n,
        "confidence": conf,
        "expected_return_pct": round(expected_return, 2) if expected_return is not None else "",
        "expected_drawdown_pct": round(expected_dd, 2) if expected_dd is not None else "",
        "collapse_probability_pct": round(collapse_prob, 1) if collapse_prob is not None else "",
        "actual_forward_6h": record.get("forward_6h"),
        "actual_forward_24h": record.get("forward_24h"),
        "actual_drawdown": record.get("max_drawdown"),
        "actual_collapse": record.get("collapse_label"),
        "actual_supply_label": record.get("supply_label"),
        "_persist_4h_raw": stats.rate(stats.persist_4h),
        "_expected_return_raw": expected_return,
        "_collapse_raw": collapse_prob,
        "_confidence_raw": conf,
        "_behaviour": sym_tier,
    }
    for horizon in HORIZONS:
        value = curve.get(horizon)
        row[f"persist_{horizon}"] = round(value, 1) if value is not None else ""

    reason, caution = reason_and_caution(
        sym_tier, regime, record["state_scan"], collapse_prob, conf
    )
    row["recommendation_reason"] = reason
    row["caution_notes"] = caution
    return row


def run_loo_scoring(records: list[dict], version: str) -> list[dict]:
    dates = sorted({record["date"] for record in records})
    all_scored: list[dict] = []

    global_f6 = [
        record["forward_6h"] for record in records if record.get("forward_6h") is not None
    ]
    global_median = statistics.median(global_f6) if global_f6 else 0.0

    high_dds: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.get("supply_label") == "HIGH_SUPPLY" and record.get("max_drawdown") is not None:
            mem = SymbolMemory(symbol=record["symbol"])
            # approximate behaviour from full history up to that point - simplified for calibration pool
            high_dds["all"].append(record["max_drawdown"])

    for holdout in dates:
        train = [record for record in records if record["date"] != holdout]
        test = [record for record in records if record["date"] == holdout]
        if version == "v1":
            cohorts = build_training_cohorts(train)
        else:
            cohorts = build_cohorts_v2(train)

        history: dict[str, list[dict]] = defaultdict(list)
        by_scan: dict[str, list[dict]] = defaultdict(list)
        for record in test:
            by_scan[record["scan_time"]].append(record)

        scored: list[dict] = []
        for record in sorted(test, key=lambda item: item["scan_time"]):
            if version == "v1":
                row = predict_record(record, cohorts, history, by_scan[record["scan_time"]])
            else:
                row = predict_v2(
                    record,
                    cohorts,
                    history,
                    by_scan[record["scan_time"]],
                    global_median,
                    high_dds,
                )
            row["_holdout"] = holdout
            row["_version"] = version
            scored.append(row)
            history[record["symbol"]].append(record)

        rank_within_scans_v2(scored) if version == "v2" else rank_within_scans(scored)
        all_scored.extend(scored)

    return all_scored


def rank_within_scans_v2(scored: list[dict]) -> None:
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_scan[row["scan_time"]].append(row)

    for rows in by_scan.values():
        for row in rows:
            row["_pre_score"] = score_v2(
                row.get("_persist_4h_raw"),
                row.get("_expected_return_raw"),
                row.get("_collapse_raw"),
                50.0,
                row.get("_confidence_raw", "hypothesis"),
                row.get("_behaviour", "new"),
            )
        rows.sort(
            key=lambda item: (item.get("_pre_score", 0), item.get("_expected_return_raw") or -999),
            reverse=True,
        )
        total = len(rows)
        for index, row in enumerate(rows):
            rank_pct = (total - index) / total * 100
            row["market_relative_rank_pct"] = round(rank_pct, 1)
            row["market_relative_rank_label"] = f"Top {rank_pct:.0f}%"
            row["recommendation_score"] = score_v2(
                row.get("_persist_4h_raw"),
                row.get("_expected_return_raw"),
                row.get("_collapse_raw"),
                rank_pct,
                row.get("_confidence_raw", "hypothesis"),
                row.get("_behaviour", "new"),
            )
            row["recommended_holding"] = holding_recommendation(
                pf(row.get("persist_2h")),
                pf(row.get("persist_4h")),
                pf(row.get("persist_6h")),
                pf(row.get("collapse_probability_pct")),
            )
            row["action"] = action_label(
                row["recommendation_score"],
                pf(row.get("collapse_probability_pct")),
                row.get("confidence", "hypothesis"),
            )


def failure_tags(row: dict) -> list[str]:
    tags: list[str] = []
    score = row.get("recommendation_score", 0)
    actual = row.get("actual_forward_6h")
    expected = pf(row.get("expected_return_pct"))
    collapse_p = pf(row.get("collapse_probability_pct"))
    actual_col = row.get("actual_collapse") == "YES"

    if actual is None:
        return ["missing_actual"]

    if score >= 7 and actual < 3:
        tags.append("high_score_failure")
    if score <= 4 and actual >= 10:
        tags.append("low_score_success")
    if expected is not None and expected - actual >= 8:
        tags.append("overestimated_return")
    if expected is not None and actual - expected >= 8:
        tags.append("underestimated_return")
    if collapse_p is not None and collapse_p < 15 and actual_col:
        tags.append("missed_collapse")
    if collapse_p is not None and collapse_p >= 30 and not actual_col and actual >= 3:
        tags.append("false_collapse_fear")
    if row.get("symbol_behaviour") in ("repeat_leader", "leader_fade") and actual < 0:
        tags.append("repeat_leader_fade")
    if row.get("confidence") in ("hypothesis", "low") and abs(actual) >= 10:
        tags.append("low_confidence_extreme_outcome")
    if row.get("cohort_n", 0) < 5 and score >= 6:
        tags.append("small_cohort_high_score")
    if row.get("state_scan") == "Exhaustion" and actual >= 10:
        tags.append("exhaustion_scan_success")
    if row.get("state_scan") == "Exhaustion" and actual_col:
        tags.append("exhaustion_scan_collapse")
    return tags or ["within_tolerance"]


def analyze_errors(scored_v1: list[dict], scored_v2: list[dict]) -> list[dict]:
    rows: list[dict] = []
    v2_by_key = {(r["scan_time"], r["symbol"]): r for r in scored_v2}
    for row in scored_v1:
        key = (row["scan_time"], row["symbol"])
        v2 = v2_by_key.get(key, {})
        actual = row.get("actual_forward_6h")
        expected_v1 = pf(row.get("expected_return_pct"))
        expected_v2 = pf(v2.get("expected_return_pct"))
        tags = failure_tags(row)
        rows.append(
            {
                "date": row["date"],
                "symbol": row["symbol"],
                "scan_time": row["scan_time"],
                "state_scan": row.get("state_scan"),
                "symbol_behaviour": row.get("symbol_behaviour"),
                "market_regime": row.get("market_regime"),
                "v1_score": row.get("recommendation_score"),
                "v2_score": v2.get("recommendation_score"),
                "v1_expected_return": expected_v1,
                "v2_expected_return": expected_v2,
                "actual_forward_6h": actual,
                "return_error_v1": round(expected_v1 - actual, 2) if expected_v1 is not None and actual is not None else "",
                "return_error_v2": round(expected_v2 - actual, 2) if expected_v2 is not None and actual is not None else "",
                "actual_collapse": row.get("actual_collapse"),
                "failure_tags": "|".join(tags),
                "primary_failure_reason": tags[0],
            }
        )
    return rows


def metrics(scored: list[dict]) -> dict:
    errors: list[float] = []
    hits = 0
    total = 0
    high_fail = 0
    high_total = 0
    low_success = 0
    low_total = 0
    top1 = 0
    scans = 0

    by_scan: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_scan[row["scan_time"]].append(row)
        actual = row.get("actual_forward_6h")
        expected = pf(row.get("expected_return_pct"))
        if actual is not None and expected is not None:
            errors.append(abs(expected - actual))
        if actual is not None:
            total += 1
            score = row.get("recommendation_score", 0)
            if score >= 5 and actual >= 3:
                hits += 1
            elif score < 5 and actual < 3:
                hits += 1
            if score >= 7:
                high_total += 1
                if actual < 3:
                    high_fail += 1
            if score <= 4:
                low_total += 1
                if actual >= 10:
                    low_success += 1

    for rows in by_scan.values():
        scans += 1
        best = max(rows, key=lambda item: item.get("recommendation_score", 0))
        actual_best = max(rows, key=lambda item: item.get("actual_forward_6h") or -9999)
        if best["symbol"] == actual_best["symbol"]:
            top1 += 1

    return {
        "mae_return": round(statistics.mean(errors), 2) if errors else None,
        "alignment": f"{hits}/{total}",
        "top1": f"{top1}/{scans}",
        "high_score_fail_rate": round(high_fail / high_total * 100, 1) if high_total else 0,
        "low_score_success_rate": round(low_success / low_total * 100, 1) if low_total else 0,
    }


def collect_counterexamples(scored_v1: list[dict], records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    record_by_key = {(r["scan_time"], r["symbol"]): r for r in records}

    for row in scored_v1:
        tags = failure_tags(row)
        if "high_score_failure" in tags or "low_score_success" in tags:
            rows.append(
                {
                    "type": "prediction_error",
                    "symbol": row["symbol"],
                    "scan_time": row["scan_time"],
                    "detail": f"score={row.get('recommendation_score')} actual_f6={row.get('actual_forward_6h')}",
                    "tags": "|".join(tags),
                }
            )

    state_groups: dict[str, list] = defaultdict(list)
    sym_groups: dict[str, list] = defaultdict(list)
    for record in records:
        state_groups[record["state_scan"]].append(record)
        sym_groups[record["symbol"]].append(record)

    for state, group in state_groups.items():
        labels = {record["supply_label"] for record in group}
        if "HIGH_SUPPLY" in labels and "COLLAPSE" in labels:
            rows.append(
                {
                    "type": "same_state_different_outcome",
                    "symbol": state,
                    "scan_time": "",
                    "detail": f"state_scan={state} mixed HIGH/COLLAPSE n={len(group)}",
                    "tags": "same_state",
                }
            )

    for symbol, group in sym_groups.items():
        if len(group) < 2:
            continue
        labels = {record["supply_label"] for record in group}
        if len(labels) >= 2:
            f6_range = [record.get("forward_6h") for record in group if record.get("forward_6h") is not None]
            if f6_range and max(f6_range) - min(f6_range) >= 15:
                rows.append(
                    {
                        "type": "same_symbol_different_outcome",
                        "symbol": symbol,
                        "scan_time": "",
                        "detail": f"f6 range {min(f6_range):.1f} to {max(f6_range):.1f} n={len(group)}",
                        "tags": "same_symbol",
                    }
                )

    return rows


def regime_behaviour_matrix(records: list[dict]) -> list[dict]:
    history: dict[str, list[dict]] = defaultdict(list)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for record in sorted(records, key=lambda item: item["scan_time"]):
        regime = record.get("market_regime_v2") or record["market_regime"]
        behaviour = behaviour_at_scan(record["symbol"], history[record["symbol"]], regime)
        cells[(regime, behaviour)].append(record)
        history[record["symbol"]].append(record)

    rows: list[dict] = []
    for (regime, behaviour), group in sorted(cells.items(), key=lambda item: -len(item[1])):
        n = len(group)
        if n < 2:
            continue
        f6 = [record["forward_6h"] for record in group if record.get("forward_6h") is not None]
        collapses = sum(1 for record in group if record.get("collapse_label") == "YES")
        highs = sum(1 for record in group if record.get("supply_label") == "HIGH_SUPPLY")
        rows.append(
            {
                "market_regime": regime,
                "symbol_behaviour": behaviour,
                "n": n,
                "median_forward_6h": round(statistics.median(f6), 2) if f6 else "",
                "high_supply_pct": round(highs / n * 100, 1),
                "collapse_pct": round(collapses / n * 100, 1),
                "confidence": "confirmed" if n >= MIN_N_CONFIRM else "hypothesis_only",
            }
        )
    return rows


def persistence_curve_table(cohorts: dict[str, CohortStatsV2]) -> list[dict]:
    rows: list[dict] = []
    for key, stats in sorted(cohorts.items(), key=lambda item: -item[1].n):
        if stats.n < MIN_N_CONFIRM:
            continue
        curve = stats.curves.curve()
        rows.append(
            {
                "cohort_key": key,
                "n": stats.n,
                **{f"persist_{h}": round(curve[h], 1) if curve.get(h) is not None else "" for h in HORIZONS},
                "note_30m_1h": "interpolated from 2h when 2h valid",
            }
        )
    return rows[:80]


def drawdown_analysis(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for label in ("HIGH_SUPPLY", "MID_SUPPLY", "LOW_SUPPLY", "COLLAPSE"):
        group = [record for record in records if record.get("supply_label") == label]
        dds = [record["max_drawdown"] for record in group if record.get("max_drawdown") is not None]
        if not dds:
            continue
        rows.append(
            {
                "supply_label": label,
                "n": len(group),
                "median_drawdown": round(statistics.median(dds), 2),
                "p75_drawdown": round(sorted(dds)[int(len(dds) * 0.75)], 2),
                "low_dd_high_supply_n": sum(
                    1 for record in group
                    if record.get("max_drawdown") is not None and record["max_drawdown"] < 15
                ),
                "high_dd_high_supply_n": sum(
                    1 for record in group
                    if record.get("max_drawdown") is not None and record["max_drawdown"] >= 25
                ),
            }
        )

    high = [record for record in records if record.get("supply_label") == "HIGH_SUPPLY"]
    by_state: dict[str, list] = defaultdict(list)
    for record in high:
        by_state[record["state_scan"]].append(record.get("max_drawdown"))

    for state, dds in by_state.items():
        dds = [value for value in dds if value is not None]
        if len(dds) < 2:
            continue
        rows.append(
            {
                "supply_label": f"HIGH@{state}",
                "n": len(dds),
                "median_drawdown": round(statistics.median(dds), 2),
                "p75_drawdown": round(sorted(dds)[int(len(dds) * 0.75)], 2),
                "low_dd_high_supply_n": sum(1 for value in dds if value < 15),
                "high_dd_high_supply_n": sum(1 for value in dds if value >= 25),
            }
        )
    return rows


def symbol_behaviour_table(memories: dict[str, SymbolMemory]) -> list[dict]:
    rows: list[dict] = []
    for symbol, memory in sorted(memories.items(), key=lambda item: -item[1].appearances):
        regime = memory.regimes[-1] if memory.regimes else ""
        behaviour = classify_symbol_behaviour(memory, regime)
        rows.append(
            {
                "symbol": symbol,
                "appearances": memory.appearances,
                "days_seen": len(memory.days_seen),
                "avg_rank": round(statistics.mean(memory.ranks), 1) if memory.ranks else "",
                "median_forward_6h": round(statistics.median(memory.forward_6h), 2) if memory.forward_6h else "",
                "collapse_rate_pct": round(memory.collapses / memory.appearances * 100, 1),
                "high_supply_rate_pct": round(memory.high_supply / memory.appearances * 100, 1),
                "median_drawdown": round(statistics.median(memory.drawdowns), 2) if memory.drawdowns else "",
                "symbol_behaviour": behaviour,
                "window_3d_note": f"seen on {sorted(memory.days_seen)}",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_v2_scores(scored: list[dict]) -> None:
    fields = [
        "date", "symbol", "scan_time", "rank", "state_scan", "trigger_bundle",
        "market_regime", "symbol_behaviour", "relative_return_tier",
        "persist_30m", "persist_1h", "persist_2h", "persist_4h", "persist_6h",
        "persist_12h", "persist_24h",
        "expected_return_pct", "expected_drawdown_pct",
        "market_relative_rank_pct", "market_relative_rank_label",
        "collapse_probability_pct", "confidence", "recommendation_score",
        "recommended_holding", "action", "recommendation_reason", "caution_notes",
        "cohort_key", "cohort_n",
        "actual_forward_6h", "actual_collapse", "actual_supply_label",
    ]
    with SCORES_V2_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in scored:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_report(
    records: list[dict],
    errors: list[dict],
    m_v1: dict,
    m_v2: dict,
    counters: list[dict],
    regime_rows: list[dict],
) -> None:
    tag_counts = Counter()
    for row in errors:
        for tag in row.get("failure_tags", "").split("|"):
            if tag and tag != "within_tolerance":
                tag_counts[tag] += 1

    lines = [
        "===== SCOUT SEASON2 P2 - PREDICTION ERROR REDUCTION =====",
        "",
        "Goal: reduce v1 error without condition sprawl",
        f"Records: {len(records)} | Dates: {sorted({r['date'] for r in records})}",
        "",
        "--- v1 vs v2 LOO metrics ---",
        f"  v1 return MAE: {m_v1['mae_return']}% | alignment: {m_v1['alignment']} | top1: {m_v1['top1']}",
        f"  v2 return MAE: {m_v2['mae_return']}% | alignment: {m_v2['alignment']} | top1: {m_v2['top1']}",
        f"  v1 high-score fail rate: {m_v1['high_score_fail_rate']}%",
        f"  v2 high-score fail rate: {m_v2['high_score_fail_rate']}%",
        f"  v1 low-score success rate: {m_v1['low_score_success_rate']}%",
        f"  v2 low-score success rate: {m_v2['low_score_success_rate']}%",
        "",
        "--- Top failure tags (v1 LOO) ---",
    ]
    for tag, count in tag_counts.most_common(12):
        lines.append(f"  {tag}: {count}")

    lines.extend(["", "--- Why high scores failed (patterns) ---"])
    high_fail = [row for row in errors if "high_score_failure" in row.get("failure_tags", "")]
    hf_beh = Counter(row.get("symbol_behaviour") for row in high_fail)
    hf_state = Counter(row.get("state_scan") for row in high_fail)
    for key, count in hf_beh.most_common(5):
        lines.append(f"  behaviour {key}: {count}")
    for key, count in hf_state.most_common(5):
        lines.append(f"  state_scan {key}: {count}")

    lines.extend(["", "--- Why low scores succeeded (patterns) ---"])
    low_ok = [row for row in errors if "low_score_success" in row.get("failure_tags", "")]
    for row in sorted(low_ok, key=lambda item: -(item.get("actual_forward_6h") or 0))[:8]:
        lines.append(
            f"  {row['symbol']} {row['scan_time'][11:16]} "
            f"v1={row['v1_score']} actual_f6={row['actual_forward_6h']} "
            f"beh={row['symbol_behaviour']} state={row['state_scan']}"
        )

    lines.extend(["", "--- Regime x Behaviour (favourable combos, n>=3) ---"])
    for row in regime_rows:
        if row.get("confidence") != "confirmed":
            continue
        if row.get("median_forward_6h") != "" and float(row["median_forward_6h"]) >= 3:
            lines.append(
                f"  {row['market_regime']} + {row['symbol_behaviour']}: "
                f"median_f6={row['median_forward_6h']}% collapse={row['collapse_pct']}%"
            )

    lines.extend(["", "--- Counterexamples ---"])
    for row in counters[:15]:
        lines.append(f"  [{row['type']}] {row['symbol']} {row.get('detail', '')}")

    lines.extend(["", "--- Hypothesis revisions ---"])
    lines.extend([
        " REVISE: repeat_leader trust -> add leader_fade; penalize in extreme regime",
        " REVISE: raw cohort median return -> shrink toward global when low n",
        " REVISE: market_regime date-level -> add volatility sub-tag via atr_ratio",
        " KEEP: relative rank within scan (not absolute supply tier)",
        " DROP: high score on small cohort without confidence penalty",
        " ADD: leader_fade behaviour (repeat rank but negative median f6)",
        "",
        "--- Next research ---",
        " +4h state posterior update (biggest remaining error source)",
        " true forward 30m/1h from klines (replace interpolation proxy)",
        " expand to 06-16+ dates for 7d/14d symbol memory",
        " drawdown split: HIGH_SUPPLY low-DD vs high-DD archetypes",
        "",
        f"Errors CSV: {ERRORS_CSV}",
        f"V2 scores CSV: {SCORES_V2_CSV}",
        "=" * 54,
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    records = build_records()
    attach_drawdown(records)
    enrich_records(records)
    apply_refined_regimes(records)

    scored_v1 = run_loo_scoring(records, "v1")
    scored_v2 = run_loo_scoring(records, "v2")

    errors = analyze_errors(scored_v1, scored_v2)
    counters = collect_counterexamples(scored_v1, records)
    memories = build_symbol_memories(records)
    regime_rows = regime_behaviour_matrix(records)
    cohorts_full = build_cohorts_v2(records)
    persist_rows = persistence_curve_table(cohorts_full)
    dd_rows = drawdown_analysis(records)
    behaviour_rows = symbol_behaviour_table(memories)

    cal_rows = [
        {
            "symbol": row["symbol"],
            "scan_time": row["scan_time"],
            "v1_error": row["return_error_v1"],
            "v2_error": row["return_error_v2"],
            "bias_v1": "overestimate" if pf(row["return_error_v1"]) and pf(row["return_error_v1"]) > 3 else (
                "underestimate" if pf(row["return_error_v1"]) and pf(row["return_error_v1"]) < -3 else "ok"
            ),
            "bias_v2": "overestimate" if pf(row["return_error_v2"]) and pf(row["return_error_v2"]) > 3 else (
                "underestimate" if pf(row["return_error_v2"]) and pf(row["return_error_v2"]) < -3 else "ok"
            ),
        }
        for row in errors
        if row.get("return_error_v1") != "" or row.get("return_error_v2") != ""
    ]

    write_csv(ERRORS_CSV, errors)
    write_csv(BEHAVIOUR_CSV, behaviour_rows)
    write_csv(PERSISTENCE_CSV, persist_rows)
    write_csv(CALIBRATION_CSV, cal_rows)
    write_csv(DRAWDOWN_CSV, dd_rows)
    write_csv(COUNTER_CSV, counters)
    write_csv(REGIME_CSV, regime_rows)
    save_v2_scores(scored_v2)

    m_v1 = metrics(scored_v1)
    m_v2 = metrics(scored_v2)
    write_report(records, errors, m_v1, m_v2, counters, regime_rows)

    print("===== SCOUT P2 PREDICTION ERROR REDUCTION =====")
    print(f"v1 LOO MAE={m_v1['mae_return']}% top1={m_v1['top1']} high-fail={m_v1['high_score_fail_rate']}%")
    print(f"v2 LOO MAE={m_v2['mae_return']}% top1={m_v2['top1']} high-fail={m_v2['high_score_fail_rate']}%")
    print(f"Report: {REPORT_TXT}")
    print(f"V2 scores: {SCORES_V2_CSV}")


if __name__ == "__main__":
    main()
