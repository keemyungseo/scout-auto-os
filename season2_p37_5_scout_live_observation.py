"""
Scout Learning Season2 - P37.5 Live Observation & Delayed Reality Validation

Observe current market, assign institutional grades, lock exactly two symbols,
then validate reasoning against reality after ~10 hours.

Research only. No prediction. No Buy/Sell. No target prices. No stop losses.
P25-P37 institutions and hierarchy are read-only — never modified here.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
import urllib.error
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10
from season2_p37_scout_decision_hierarchy import (
    L5_CAP,
    hierarchical_decide,
    institution_votes,
    load_csv,
    pf,
    pi,
    pbool,
)
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

LIVE_SELECTION_CSV = LOGS_DIR / "season2_p37_live_selection.csv"
OBSERVATION_LOG_CSV = LOGS_DIR / "season2_p37_observation_log.csv"
DELAYED_VALIDATION_CSV = LOGS_DIR / "season2_p37_delayed_validation.csv"
REASONING_SCORE_CSV = LOGS_DIR / "season2_p37_reasoning_score.csv"
PERSISTENT_LESSONS_CSV = LOGS_DIR / "season2_p37_persistent_lessons.csv"
TEMPORARY_LESSONS_CSV = LOGS_DIR / "season2_p37_temporary_lessons.csv"
REALITY_REPORT_TXT = LOGS_DIR / "season2_p37_reality_report.txt"

VALIDATION_DELAY_HOURS = 10
KST = timezone(timedelta(hours=9))
MODE = "hybrid"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
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


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def format_kst(dt: datetime) -> str:
    return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")


def format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_locked_selection() -> list[dict]:
    if not LIVE_SELECTION_CSV.exists():
        return []
    rows = load_csv(LIVE_SELECTION_CSV)
    return [r for r in rows if str(r.get("lock_status", "")).upper() == "LOCKED"]


def fetch_current_price(symbol: str) -> float | None:
    try:
        data = t10.public_get("/fapi/v1/ticker/price", {"symbol": symbol})
        return pf(data.get("price"))
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, TypeError):
        return None


def fetch_volatility_pct(symbol: str, end_ms: int) -> float:
    try:
        klines = t10.fetch_klines_before(symbol, t10.INTERVAL_2H, end_ms, 6)
        if len(klines) < 2:
            return 0.0
        ranges = []
        for candle in klines:
            open_p, high_p, low_p, close_p, _ = t10.ohlcv(candle)
            base = open_p or close_p or 1.0
            ranges.append((high_p - low_p) / base * 100)
        return round(statistics.mean(ranges), 2)
    except urllib.error.HTTPError:
        return 0.0


def scan_live_universe(scan_dt: datetime) -> list[dict]:
    end_ms = int(scan_dt.timestamp() * 1000)
    eligible = sorted(t10.get_eligible_symbols())
    if not eligible:
        raise RuntimeError("no eligible symbols from exchange")

    print(f"Scanning {len(eligible)} symbols at {format_utc(scan_dt)}")
    rows: list[dict] = []
    for index, symbol in enumerate(eligible, start=1):
        if index % 50 == 0:
            print(f"  progress {index}/{len(eligible)}")
        try:
            klines = t10.fetch_klines_before(symbol, t10.INTERVAL_2H, end_ms, t10.RANKING_KLINES_2H)
            ranking = t10.compute_24h_ranking(klines)
            if ranking is None:
                continue
            vol = fetch_volatility_pct(symbol, end_ms)
            rows.append({
                "symbol": symbol,
                "price_at_scan": ranking["price_at_scan"],
                "return_24h_percent": round(ranking["return_24h_percent"], 2),
                "volatility_pct": vol,
            })
        except urllib.error.HTTPError:
            continue
        time.sleep(t10.API_SLEEP_SEC)

    rows.sort(key=lambda item: item["return_24h_percent"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_24h"] = rank
    print(f"Scored {len(rows)} symbols with valid 24h data")
    return rows


def infer_market_regime(scanned: list[dict]) -> str:
    if not scanned:
        return "Mixed"
    top10 = scanned[:10]
    avg_top = statistics.mean(r["return_24h_percent"] for r in top10)
    positive = sum(1 for r in scanned if r["return_24h_percent"] > 0)
    breadth = positive / len(scanned)
    if avg_top >= 8 and breadth >= 0.45:
        return "Healthy Expansion"
    if avg_top <= -2 or breadth <= 0.25:
        return "Panic"
    if avg_top >= 3 and breadth >= 0.35:
        return "Rotation"
    if avg_top <= 1 and breadth <= 0.4:
        return "Compression"
    return "Conflict"


def build_memory_index(library: list[dict]) -> dict[str, list[dict]]:
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for case in library:
        by_symbol[case["symbol"]].append(case)
    for cases in by_symbol.values():
        cases.sort(key=lambda c: c.get("scan_time", ""), reverse=True)
    return by_symbol


def memory_outcome_distribution(cases: list[dict]) -> Counter:
    return Counter(c.get("empirical_outcome") or c.get("outcome") or "unknown" for c in cases)


def top_memory_outcome(cases: list[dict]) -> str:
    if not cases:
        return ""
    dist = memory_outcome_distribution(cases)
    if not dist:
        return ""
    return dist.most_common(1)[0][0]


def latest_case_fields(cases: list[dict]) -> dict:
    if not cases:
        return {}
    return cases[0]


def structure_key_from_case(case: dict) -> str:
    return "|".join([
        str(case.get("situation", "Unknown")),
        str(case.get("playbook_match", "none")),
        str(case.get("priority_tier", "?")),
        str(case.get("field_coherence", "unknown")),
        str(case.get("supply_context", "UNKNOWN")),
    ])


def live_false_convergence(rank: int, case: dict) -> bool:
    if pbool(case.get("false_convergence_flagged")):
        return True
    persist = pi(case.get("convergence_persist_scans"))
    seedbed = case.get("seedbed_quality", "Unknown")
    if rank <= 3 and persist < 2 and seedbed in ("Unknown", ""):
        return True
    if rank <= 2 and pi(case.get("independent_conflict")) >= 3:
        return True
    return False


def build_live_record(
    market_row: dict,
    scan_time: str,
    regime: str,
    memory_by_symbol: dict[str, list[dict]],
) -> dict:
    symbol = market_row["symbol"]
    cases = memory_by_symbol.get(symbol, [])
    latest = latest_case_fields(cases)
    rank = pi(market_row.get("rank_24h"))

    persist = pi(latest.get("convergence_persist_scans")) if latest else 0
    unknown = latest.get("unknown_audit", "unknown_active") if latest else "unknown_active"
    if not cases:
        unknown = "honest_unknown"

    false_conv = live_false_convergence(rank, latest) if latest else (rank <= 3)

    record = {
        "date": scan_time[:10],
        "symbol": symbol,
        "scan_time": scan_time,
        "situation": latest.get("situation", "Unknown"),
        "field_ecology": (
            f"rank={rank}|{latest.get('field_verdict', 'mixed')}|"
            f"{latest.get('field_coherence', 'unknown')}"
            if latest else f"rank={rank}|live|unknown"
        ),
        "seedbed_quality": latest.get("seedbed_quality", "Unknown"),
        "convergence_state": latest.get("convergence_state", "unknown"),
        "playbook": latest.get("playbook_match", "none"),
        "priority_tier": latest.get("priority_tier", "D"),
        "persistence_scans": persist,
        "promotion_path": latest.get("promotion_path", ""),
        "demotion_path": latest.get("demotion_path", ""),
        "promotion_count": latest.get("promotion_count", 0),
        "demotion_count": latest.get("demotion_count", 0),
        "supply": latest.get("supply_context", "UNKNOWN"),
        "interaction": latest.get("interaction", "absent"),
        "collapse_risk_pct": latest.get("collapse_risk_pct", 10),
        "false_convergence": false_conv,
        "unknown_honesty": unknown,
        "independent_support": latest.get("independent_support", 2 if latest else 1),
        "independent_conflict": latest.get("independent_conflict", 2 if latest else 1),
        "support_families": latest.get("support_families", "live_scan"),
        "market_regime": regime,
        "confidence_score": latest.get("attention_score", 25) if latest else 20,
        "p25_corrections": latest.get("p25_corrections", "P25_R1_persistence_or_coherence"),
        "memory_top_outcome": top_memory_outcome(cases),
        "memory_cases": len(cases),
        "structure_key": structure_key_from_case(latest) if latest else f"Unknown|none|D|unknown|UNKNOWN",
        "structure_duplicate_index": 0,
        "observation_allocation": "Ignore",
        "return_24h_percent": market_row["return_24h_percent"],
        "price_at_scan": market_row["price_at_scan"],
        "volatility_pct": market_row["volatility_pct"],
        "rank_24h": rank,
        "_attn_weight": 0,
        "_gov": {"market_regime": regime},
    }
    return record


def assign_duplicate_index(records: list[dict]) -> None:
    key_counts = Counter(r["structure_key"] for r in records)
    for record in records:
        key = record["structure_key"]
        record["structure_duplicate_index"] = max(0, key_counts[key] - 1)


def l5_led(decision: dict) -> bool:
    for step in decision.get("trace", []):
        if step.startswith("L5:") and "+" in step:
            return True
        if step.startswith("L1:") or step.startswith("L2:") or step.startswith("L3:"):
            if "=" in step and not step.endswith("=baseline") and "+1" in step:
                return False
            if "VETO" in step:
                return False
    return False


def component_scores(record: dict, gov: dict, decision: dict) -> dict[str, float]:
    votes = decision["votes"]
    memory = 70.0 if votes.get("memory") == 1 else (35.0 if votes.get("memory") == 0 else 15.0)
    if not pi(record.get("memory_cases")):
        memory = 45.0 if record.get("unknown_honesty") in ("honest_unknown", "unknown_active") else 30.0

    diversification = 85.0 if votes.get("diversification") == 1 else (50.0 if votes.get("diversification") == 0 else 20.0)
    false_conv = 90.0 if votes.get("false_convergence_protection") == 1 and not decision["vetoed"] else 25.0
    unknown = 80.0 if record.get("unknown_honesty") in ("honest_unknown", "unknown_active") else 55.0
    confidence_penalty = 0.0
    if votes.get("confidence", 0) > 0:
        confidence_penalty += 15.0
    if l5_led(decision):
        confidence_penalty += 25.0
    if pi(record.get("confidence_score")) >= 55 and pi(record.get("persistence_scans")) < 2:
        confidence_penalty += 10.0

    return {
        "memory_score": round(memory, 1),
        "diversification_score": round(diversification, 1),
        "false_convergence_protection_score": round(false_conv, 1),
        "unknown_honesty_score": round(unknown, 1),
        "confidence_penalty": round(confidence_penalty, 1),
    }


def institution_grade(scores: dict, decision: dict) -> str:
    if decision["vetoed"]:
        return "C"
    core_avg = statistics.mean([
        scores["memory_score"],
        scores["diversification_score"],
        scores["false_convergence_protection_score"],
        scores["unknown_honesty_score"],
    ])
    core_avg -= scores["confidence_penalty"] * 0.5
    if l5_led(decision):
        core_avg -= 15
    if core_avg >= 75 and scores["memory_score"] >= 60 and not l5_led(decision):
        return "A+"
    if core_avg >= 62:
        return "A"
    if core_avg >= 45:
        return "B"
    return "C"


def discipline_score(record: dict, decision: dict, scores: dict) -> float:
    score = decision["score"] * 100.0
    if decision["vetoed"]:
        score -= 120
    if l5_led(decision):
        score -= 80
    score += scores["memory_score"] * 0.15
    score += scores["diversification_score"] * 0.12
    score += scores["false_convergence_protection_score"] * 0.15
    score += scores["unknown_honesty_score"] * 0.10
    score -= scores["confidence_penalty"]
    if decision["stance"] == "watch_default":
        score += 8
    if pi(record.get("persistence_scans")) >= 2:
        score += 6
    if pbool(record.get("false_convergence")):
        score -= 40
    return round(score, 2)


def institutional_strengths(record: dict, decision: dict, scores: dict) -> str:
    strengths = []
    if scores["memory_score"] >= 60:
        strengths.append("memory")
    if scores["diversification_score"] >= 60:
        strengths.append("diversification")
    if scores["false_convergence_protection_score"] >= 70:
        strengths.append("false_convergence_protection")
    if scores["unknown_honesty_score"] >= 70:
        strengths.append("unknown_honesty")
    if decision.get("watch_signal"):
        strengths.append("watch_default")
    if pi(record.get("persistence_scans")) >= 2:
        strengths.append("persistence")
    if not strengths:
        strengths.append("baseline_observation")
    return "|".join(strengths)


def institutional_weaknesses(record: dict, decision: dict, scores: dict) -> str:
    weaknesses = []
    if scores["memory_score"] < 45:
        weaknesses.append("memory_thin")
    if scores["diversification_score"] < 45:
        weaknesses.append("structure_overlap")
    if pbool(record.get("false_convergence")):
        weaknesses.append("false_convergence_risk")
    if "conflicting" in str(record.get("field_ecology")):
        weaknesses.append("field_conflict")
    if l5_led(decision):
        weaknesses.append("l5_pressure")
    if scores["confidence_penalty"] >= 15:
        weaknesses.append("confidence_overreach")
    if pi(record.get("persistence_scans")) < 2:
        weaknesses.append("low_persistence")
    if not weaknesses:
        weaknesses.append("none_notable")
    return "|".join(weaknesses)


def survival_reason(record: dict, decision: dict, scores: dict, rank_position: int) -> str:
    parts = [
        f"discipline_rank={rank_position}",
        f"stance={decision['stance']}",
        f"grade_components=mem{scores['memory_score']:.0f}|div{scores['diversification_score']:.0f}|fc{scores['false_convergence_protection_score']:.0f}",
    ]
    if not decision["vetoed"] and not l5_led(decision):
        parts.append("hierarchy_clean")
    if decision.get("watch_signal"):
        parts.append("unknown_honesty_watch")
    return "; ".join(parts)


def evaluate_universe(scanned: list[dict], scan_time: str, regime: str) -> list[dict]:
    library = load_csv(LOGS_DIR / "season2_p23_case_library.csv")
    memory_by_symbol = build_memory_index(library)
    records = [build_live_record(row, scan_time, regime, memory_by_symbol) for row in scanned]
    assign_duplicate_index(records)

    evaluated: list[dict] = []
    for record in records:
        gov = record["_gov"]
        decision = hierarchical_decide(record, gov, MODE)
        scores = component_scores(record, gov, decision)
        grade = institution_grade(scores, decision)
        disc = discipline_score(record, decision, scores)
        evaluated.append({
            **record,
            "decision": decision,
            "scores": scores,
            "institution_grade": grade,
            "discipline_score": disc,
            "l5_led": l5_led(decision),
            "decision_trace": "|".join(decision["trace"]),
            "hierarchy_stance": decision["stance"],
            "hierarchy_score": round(decision["score"], 4),
            "vetoed": decision["vetoed"],
        })

    evaluated.sort(key=lambda item: item["discipline_score"], reverse=True)
    return evaluated


def select_two_symbols(evaluated: list[dict]) -> list[dict]:
    candidates = [
        item for item in evaluated
        if not item["vetoed"] and not item["l5_led"] and item["institution_grade"] in ("A+", "A", "B")
    ]
    if len(candidates) < 2:
        candidates = [item for item in evaluated if not item["vetoed"] and not item["l5_led"]]
    if len(candidates) < 2:
        candidates = [item for item in evaluated if not item["vetoed"]][:10]

    selected: list[dict] = []
    used_keys: set[str] = set()
    for item in candidates:
        key = item["structure_key"]
        if key in used_keys and pi(item.get("structure_duplicate_index")) >= 1:
            continue
        selected.append(item)
        used_keys.add(key)
        if len(selected) == 2:
            break

    if len(selected) < 2:
        for item in candidates:
            if item not in selected:
                selected.append(item)
            if len(selected) == 2:
                break
    return selected[:2]


def build_observation_log_rows(evaluated: list[dict], observation_id: str) -> list[dict]:
    rows = []
    for item in evaluated:
        scores = item["scores"]
        rows.append({
            "observation_id": observation_id,
            "symbol": item["symbol"],
            "rank_24h": item["rank_24h"],
            "return_24h_percent": item["return_24h_percent"],
            "institution_grade": item["institution_grade"],
            "discipline_score": item["discipline_score"],
            "memory_score": scores["memory_score"],
            "diversification_score": scores["diversification_score"],
            "false_convergence_protection_score": scores["false_convergence_protection_score"],
            "unknown_honesty_score": scores["unknown_honesty_score"],
            "confidence_penalty": scores["confidence_penalty"],
            "hierarchy_stance": item["hierarchy_stance"],
            "hierarchy_score": item["hierarchy_score"],
            "vetoed": item["vetoed"],
            "l5_led": item["l5_led"],
            "market_regime": item["market_regime"],
            "decision_trace": item["decision_trace"],
        })
    return rows


def build_selection_rows(
    selected: list[dict],
    observation_id: str,
    scan_dt: datetime,
    validation_due: datetime,
    regime: str,
) -> list[dict]:
    rows = []
    for index, item in enumerate(selected, start=1):
        scores = item["scores"]
        rows.append({
            "observation_id": observation_id,
            "lock_status": "LOCKED",
            "selection_rank": index,
            "symbol": item["symbol"],
            "institution_grade": item["institution_grade"],
            "memory_score": scores["memory_score"],
            "diversification_score": scores["diversification_score"],
            "false_convergence_protection_score": scores["false_convergence_protection_score"],
            "unknown_honesty_score": scores["unknown_honesty_score"],
            "confidence_penalty": scores["confidence_penalty"],
            "institutional_strengths": institutional_strengths(item, item["decision"], scores),
            "institutional_weaknesses": institutional_weaknesses(item, item["decision"], scores),
            "survival_reason": survival_reason(item, item["decision"], scores, index),
            "discipline_score": item["discipline_score"],
            "hierarchy_stance": item["hierarchy_stance"],
            "decision_trace": item["decision_trace"],
            "market_regime": regime,
            "price_at_observation": item["price_at_scan"],
            "volatility_at_observation_pct": item["volatility_pct"],
            "return_24h_at_observation_pct": item["return_24h_percent"],
            "observation_timestamp_utc": format_utc(scan_dt),
            "observation_timestamp_kst": format_kst(scan_dt),
            "validation_due_timestamp_utc": format_utc(validation_due),
            "validation_due_timestamp_kst": format_kst(validation_due),
            "validation_status": "SCHEDULED",
        })
    return rows


def write_scheduled_validation_placeholders(selection_rows: list[dict]) -> None:
    validation_rows = []
    reasoning_rows = []
    for row in selection_rows:
        validation_rows.append({
            "observation_id": row["observation_id"],
            "symbol": row["symbol"],
            "validation_status": "SCHEDULED",
            "validation_due_timestamp_utc": row["validation_due_timestamp_utc"],
            "observation_price": row["price_at_observation"],
            "validation_price": "",
            "price_change_pct": "",
            "volatility_change_pct": "",
            "decision_robustness": "PENDING",
            "institutions_helped": "",
            "institutions_failed": "",
            "reasoning_survived_reality": "PENDING",
            "success_type": "PENDING",
            "failure_type": "PENDING",
        })
        reasoning_rows.append({
            "observation_id": row["observation_id"],
            "symbol": row["symbol"],
            "process_verdict": "PENDING",
            "reasoning_quality_score": "",
            "financial_outcome_note": "Not primary judge",
            "process_over_outcome": "PENDING",
            "institution_grade_at_observation": row["institution_grade"],
            "validation_status": "SCHEDULED",
        })
    write_csv(DELAYED_VALIDATION_CSV, validation_rows)
    write_csv(REASONING_SCORE_CSV, reasoning_rows)
    write_csv(PERSISTENT_LESSONS_CSV, [{
        "observation_id": selection_rows[0]["observation_id"],
        "lesson_status": "SCHEDULED",
        "lesson": "Await delayed validation — reasoning judged after ~10 hours",
        "source": "P37.5",
    }])
    write_csv(TEMPORARY_LESSONS_CSV, [{
        "observation_id": selection_rows[0]["observation_id"],
        "lesson_status": "SCHEDULED",
        "lesson": "Live observation locked — no modifications permitted",
        "source": "P37.5",
    }])
    REALITY_REPORT_TXT.write_text(
        "\n".join([
            "===== SCOUT SEASON2 P37.5 - LIVE OBSERVATION LOCKED =====",
            "",
            f"Observation ID: {selection_rows[0]['observation_id']}",
            f"Locked symbols: {', '.join(r['symbol'] for r in selection_rows)}",
            f"Validation scheduled: {selection_rows[0]['validation_due_timestamp_utc']}",
            "",
            "Status: SCHEDULED — run with --validate after validation window.",
            "",
            "Core principle: Judge reasoning quality more than financial outcome.",
            "",
            *mission_summary_lines(),
        ]),
        encoding="utf-8",
    )


def observe(force: bool = False) -> None:
    locked = load_locked_selection()
    if locked and not force:
        print(f"Observation already LOCKED ({locked[0]['observation_id']}). No modifications allowed.")
        print("Use --validate when validation window opens.")
        return

    scan_dt = now_utc()
    scan_time = format_kst(scan_dt)
    observation_id = f"P37LIVE_{scan_dt.strftime('%Y%m%d_%H%M%S')}"
    validation_due = scan_dt + timedelta(hours=VALIDATION_DELAY_HOURS)

    scanned = scan_live_universe(scan_dt)
    regime = infer_market_regime(scanned)
    print(f"Inferred live regime: {regime}")

    evaluated = evaluate_universe(scanned, scan_time, regime)
    selected = select_two_symbols(evaluated)
    if len(selected) < 2:
        raise RuntimeError("Could not select two symbols from live universe")

    selection_rows = build_selection_rows(selected, observation_id, scan_dt, validation_due, regime)
    observation_rows = build_observation_log_rows(evaluated, observation_id)

    write_csv(LIVE_SELECTION_CSV, selection_rows)
    write_csv(OBSERVATION_LOG_CSV, observation_rows)
    write_scheduled_validation_placeholders(selection_rows)

    print(f"Locked observation {observation_id}")
    for row in selection_rows:
        print(
            f"  #{row['selection_rank']} {row['symbol']} grade={row['institution_grade']} "
            f"discipline={row['discipline_score']}"
        )
    print(f"Validation due: {format_utc(validation_due)}")


def process_verdict(selection: dict, validation: dict) -> tuple[str, float, str]:
    score = 50.0
    notes = []

    if selection.get("institution_grade") in ("A+", "A"):
        score += 10
    if "memory" in str(selection.get("institutional_strengths", "")):
        score += 8
        notes.append("memory_supported")
    if "false_convergence_protection" in str(selection.get("institutional_strengths", "")):
        score += 8
        notes.append("false_convergence_guard")
    if "l5_pressure" in str(selection.get("institutional_weaknesses", "")):
        score -= 20
        notes.append("l5_weakness")
    if pi(selection.get("confidence_penalty")) >= 20:
        score -= 10
        notes.append("confidence_penalty_applied")
    if validation.get("decision_robustness") == "robust":
        score += 12
    elif validation.get("decision_robustness") == "fragile":
        score -= 8

    if validation.get("reasoning_survived_reality") == "yes":
        score += 15
    elif validation.get("reasoning_survived_reality") == "no":
        score -= 10

    # Process over outcome: good process can pass with bad price move
    price_change = pf(validation.get("price_change_pct"), 0.0)
    if score >= 65:
        verdict = "PASS"
    elif score >= 50 and validation.get("reasoning_survived_reality") != "no":
        verdict = "PASS"
    elif score < 45 or "l5_pressure" in str(selection.get("institutional_weaknesses", "")):
        verdict = "FAIL"
    else:
        verdict = "MARGINAL"

    if price_change is not None and price_change < -5 and score >= 60:
        notes.append("bad_outcome_good_process")
    if price_change is not None and price_change > 5 and score < 50:
        notes.append("good_outcome_bad_process")

    return verdict, round(max(0.0, min(100.0, score)), 1), "|".join(notes) or "baseline"


def validate(force: bool = False) -> None:
    locked = load_locked_selection()
    if not locked:
        print("No locked observation found. Run observe first.")
        return

    observation_id = locked[0]["observation_id"]
    due_str = locked[0].get("validation_due_timestamp_utc", "")
    due_dt = datetime.strptime(due_str.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    now = now_utc()

    if now < due_dt and not force:
        print(f"Validation not due until {format_utc(due_dt)}")
        print("Use --force-validate to run early (research only).")
        return

    validation_rows = []
    reasoning_rows = []
    persistent_lessons = []
    temporary_lessons = []

    for row in locked:
        symbol = row["symbol"]
        obs_price = pf(row.get("price_at_observation"))
        val_price = fetch_current_price(symbol)
        if val_price is None or obs_price is None:
            raise RuntimeError(f"Could not fetch validation price for {symbol}")

        price_change = (val_price - obs_price) / obs_price * 100
        obs_vol = pf(row.get("volatility_at_observation_pct"), 0.0)
        val_vol = fetch_volatility_pct(symbol, int(now.timestamp() * 1000))
        vol_change = val_vol - obs_vol if obs_vol else val_vol

        strengths = str(row.get("institutional_strengths", ""))
        weaknesses = str(row.get("institutional_weaknesses", ""))
        helped = []
        failed = []
        if "memory" in strengths:
            helped.append("memory")
        if "diversification" in strengths:
            helped.append("diversification")
        if "false_convergence_protection" in strengths:
            helped.append("false_convergence_protection")
        if "memory_thin" in weaknesses:
            failed.append("memory")
        if "structure_overlap" in weaknesses:
            failed.append("diversification")
        if "false_convergence_risk" in weaknesses:
            failed.append("false_convergence_protection")
        if "l5_pressure" in weaknesses:
            failed.append("confidence|council|field_ecology")

        abs_move = abs(price_change)
        if abs_move <= obs_vol * 1.5 + 2:
            robustness = "robust"
        elif abs_move <= obs_vol * 3 + 5:
            robustness = "moderate"
        else:
            robustness = "fragile"

        reasoning_survived = "yes" if robustness in ("robust", "moderate") and "l5_pressure" not in weaknesses else "no"
        if pf(row.get("confidence_penalty")) >= 20 and abs_move > 8:
            reasoning_survived = "no"

        if price_change >= 0 and reasoning_survived == "yes":
            success_type = "discipline" if row.get("institution_grade") in ("A+", "A") else "mixed"
        elif price_change >= 0:
            success_type = "luck"
        else:
            success_type = "n/a"

        if price_change < 0 and reasoning_survived == "yes":
            failure_type = "randomness"
        elif price_change < 0:
            failure_type = "process"
        else:
            failure_type = "n/a"

        validation_row = {
            "observation_id": observation_id,
            "symbol": symbol,
            "validation_status": "COMPLETE",
            "validation_timestamp_utc": format_utc(now),
            "validation_due_timestamp_utc": due_str,
            "observation_price": round(obs_price, 8),
            "validation_price": round(val_price, 8),
            "price_change_pct": round(price_change, 2),
            "volatility_at_observation_pct": obs_vol,
            "volatility_at_validation_pct": val_vol,
            "volatility_change_pct": round(vol_change, 2),
            "decision_robustness": robustness,
            "institutions_helped": "|".join(helped) or "watch_default",
            "institutions_failed": "|".join(failed) or "none",
            "reasoning_survived_reality": reasoning_survived,
            "success_type": success_type,
            "failure_type": failure_type,
        }
        validation_rows.append(validation_row)

        verdict, quality, note = process_verdict(row, validation_row)
        reasoning_rows.append({
            "observation_id": observation_id,
            "symbol": symbol,
            "process_verdict": verdict,
            "reasoning_quality_score": quality,
            "process_notes": note,
            "financial_outcome_note": f"price_change={price_change:.2f}% (not primary judge)",
            "process_over_outcome": "PASS" if verdict == "PASS" and price_change < 0 else (
                "PASS" if verdict == "PASS" else "FAIL"
            ),
            "institution_grade_at_observation": row.get("institution_grade"),
            "validation_status": "COMPLETE",
        })

    for lesson, tier in [
        ("memory and diversification must lead; L5 never leads", "persistent"),
        ("unknown honesty watch default survives volatile validation windows", "persistent"),
        ("false convergence protection veto is process-safe even when price moves", "persistent"),
    ]:
        persistent_lessons.append({
            "observation_id": observation_id,
            "lesson_status": "ACTIVE",
            "lesson": lesson,
            "source": "P37.5_validation",
        })

    pass_count = sum(1 for r in reasoning_rows if r["process_verdict"] == "PASS")
    temporary_lessons.append({
        "observation_id": observation_id,
        "lesson_status": "ACTIVE",
        "lesson": f"Live window pass rate {pass_count}/{len(reasoning_rows)} — revisit after next observation cycle",
        "source": "P37.5_validation",
    })
    if any("luck" in r.get("success_type", "") for r in validation_rows):
        temporary_lessons.append({
            "observation_id": observation_id,
            "lesson_status": "ACTIVE",
            "lesson": "Good price move with weak process — do not promote confidence or council",
            "source": "P37.5_validation",
        })
    temporary_lessons.append({
        "observation_id": observation_id,
        "lesson_status": "ACTIVE",
        "lesson": "Avoid: selecting symbols where L5 institutions would lead",
        "source": "P37.5_validation",
    })

    write_csv(DELAYED_VALIDATION_CSV, validation_rows)
    write_csv(REASONING_SCORE_CSV, reasoning_rows)
    write_csv(PERSISTENT_LESSONS_CSV, persistent_lessons)
    write_csv(TEMPORARY_LESSONS_CSV, temporary_lessons)

    updated_selection = []
    for row in locked:
        updated_selection.append({**row, "validation_status": "COMPLETE"})
    write_csv(LIVE_SELECTION_CSV, updated_selection)

    report_lines = [
        "===== SCOUT SEASON2 P37.5 - DELAYED REALITY VALIDATION =====",
        "",
        f"Observation ID: {observation_id}",
        f"Validation time: {format_utc(now)}",
        "",
        "=== Selection (locked at observation) ===",
    ]
    for row in locked:
        report_lines.append(
            f"  {row['symbol']} grade={row.get('institution_grade')} "
            f"discipline={row.get('discipline_score')} strengths={row.get('institutional_strengths')}"
        )
    report_lines.extend(["", "=== Reality comparison ==="])
    for val in validation_rows:
        report_lines.extend([
            f"  {val['symbol']}: price {val['price_change_pct']:+.2f}% | "
            f"volatility delta {val['volatility_change_pct']:+.2f}% | "
            f"robustness={val['decision_robustness']} | reasoning_survived={val['reasoning_survived_reality']}",
            f"    helped={val['institutions_helped']} | failed={val['institutions_failed']} | "
            f"success={val['success_type']} | failure={val['failure_type']}",
        ])
    report_lines.extend(["", "=== Reasoning quality (primary judge) ==="])
    for reason in reasoning_rows:
        report_lines.append(
            f"  {reason['symbol']}: {reason['process_verdict']} "
            f"(quality={reason['reasoning_quality_score']}) — {reason['process_notes']}"
        )
    report_lines.extend([
        "",
        "Good process with bad outcome may PASS. Bad process with good outcome may FAIL.",
        "",
        "=== New observation note ===",
        "Scout observed, committed honestly, waited, and compared reasoning to reality.",
        "",
        *mission_summary_lines(),
    ])
    REALITY_REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Validation complete for {observation_id}")
    for reason in reasoning_rows:
        print(f"  {reason['symbol']}: {reason['process_verdict']} (quality={reason['reasoning_quality_score']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="P37.5 Scout Live Observation & Delayed Validation")
    parser.add_argument("--observe", action="store_true", help="Run live market observation and lock two symbols")
    parser.add_argument("--validate", action="store_true", help="Run delayed validation for locked observation")
    parser.add_argument("--force-validate", action="store_true", help="Validate before 10h window (research)")
    parser.add_argument("--force-observe", action="store_true", help="Replace locked observation (NOT recommended)")
    args = parser.parse_args()

    if args.validate or args.force_validate:
        validate(force=args.force_validate)
        return

    if args.observe or args.force_observe:
        observe(force=args.force_observe)
        return

    locked = load_locked_selection()
    if locked:
        due_str = locked[0].get("validation_due_timestamp_utc", "")
        print(f"Locked observation: {locked[0]['observation_id']} | symbols: {[r['symbol'] for r in locked]}")
        print(f"Validation due: {due_str}")
        print("Run --validate when due, or --observe only if starting fresh (blocked while locked).")
    else:
        observe(force=False)


if __name__ == "__main__":
    main()
