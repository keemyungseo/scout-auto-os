"""
Scout Learning Season2 - Autonomous Research

Transition Trigger + Market Regime + Counterexample analysis.
Research only. Does not output Scout trading conditions.
"""

import csv
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "season2_transition_trigger_research.csv"
REPORT_PATH = LOGS_DIR / "season2_autonomous_research_report.txt"

TOP10_PATH = LOGS_DIR / "top10_gainer_learning_20260613.csv"
STATE_PATH = LOGS_DIR / "state_transition_learning_season2.csv"
TOP3_ENRICHED = LOGS_DIR / "top3_gainers_20260614_enriched.csv"
TOP3_PATH = LOGS_DIR / "top3_gainers_20260614.csv"


def pf(value: str) -> float | None:
    value = (value or "").strip()
    return float(value) if value else None


def pb(value: str) -> bool:
    return (value or "").strip().upper() == "YES"


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def join_records() -> list[dict]:
    top10 = { (r["scan_time_kst"], r["symbol"]): r for r in load_csv(TOP10_PATH) }
    states = load_csv(STATE_PATH)
    records: list[dict] = []

    for state_row in states:
        key = (state_row["scan_time_kst"], state_row["symbol"])
        feat = top10.get(key, {})
        records.append({**feat, **state_row})

    top3_source = TOP3_ENRICHED if TOP3_ENRICHED.exists() else TOP3_PATH
    for row in load_csv(top3_source):
        key = (row["scan_time_kst"], row["symbol"])
        if key in top10:
            continue
        records.append(
            {
                "scan_time_kst": row["scan_time_kst"],
                "symbol": row["symbol"],
                "rank_24h": row.get("market_rank", ""),
                "return_24h_percent": row.get("return_24h_percent", ""),
                "position_7d_percent": row.get("position_7d_percent", ""),
                "body_expansion_ratio": row.get("body_expansion_ratio", ""),
                "volume_ratio_ma24": row.get("volume_ratio_ma24", ""),
                "ma24_slope_percent": row.get("ma24_slope_percent", ""),
                "distance_from_ma24_percent": row.get("distance_from_ma24_percent", ""),
                "break_24h_highest_close": "",
                "pre6_tight_range": "",
                "forward_2h": row.get("forward_return_2h", ""),
                "forward_4h": row.get("forward_return_4h", ""),
                "forward_6h": row.get("forward_return_6h", ""),
                "forward_12h": row.get("forward_return_12h", ""),
                "forward_24h": row.get("forward_return_24h", ""),
                "state_scan": "",
                "state_h4": "",
                "state_h6": "",
                "state_h12": "",
                "outcome_ref_12h": "",
                "study_date": row["scan_time_kst"][:10],
                "source_file": top3_source.name,
            }
        )
    return records


def classify_regime(date: str, day_records: list[dict]) -> str:
    symbols = {r["symbol"] for r in day_records}
    ranks = [int(r.get("rank_24h") or 99) for r in day_records if r.get("rank_24h")]
    top1_returns = [
        pf(r.get("return_24h_percent", ""))
        for r in day_records
        if str(r.get("rank_24h")) == "1" and pf(r.get("return_24h_percent", "")) is not None
    ]
    avg_top1 = statistics.mean(top1_returns) if top1_returns else 0.0
    unique = len(symbols)
    leader_repeat = len(day_records) / max(unique, 1)

    favorable = sum(1 for r in day_records if r.get("outcome_ref_12h") == "favorable")
    total_out = sum(1 for r in day_records if r.get("outcome_ref_12h") in {"favorable", "unfavorable", "mixed"})

    if avg_top1 >= 50:
        return "Leader Concentration (extreme)"
    if leader_repeat >= 2.5 and unique <= 12:
        return "Leader Concentration"
    if favorable >= total_out * 0.35 and avg_top1 < 25:
        return "Broad Rally"
    if avg_top1 < 20 and favorable < total_out * 0.2:
        return "Slow Market"
    if unique >= 15:
        return "Rotation"
    return "Mixed Regime"


def supply_score(record: dict) -> str:
    f6 = pf(record.get("forward_6h", ""))
    f12 = pf(record.get("forward_12h", ""))
    if f6 is not None and f6 >= 5 and f12 is not None and f12 >= 5:
        return "high"
    if f6 is not None and f6 >= 3:
        return "medium"
    if f12 is not None and f12 < 0:
        return "low"
    return "unknown"


@dataclass
class TriggerRow:
    study_date: str
    market_regime: str
    symbol: str
    scan_time_kst: str
    state_scan: str
    state_h4: str
    state_h6: str
    transition_scan_h4: str
    trigger_ma24_slope: float | None
    trigger_volume_ratio: float | None
    trigger_position_7d: float | None
    trigger_body_expansion: float | None
    trigger_rank_24h: int | None
    trigger_f4_return: float | None
    supply_score: str
    outcome_ref_12h: str
    counterexample_key: str


def build_trigger_rows(records: list[dict], regimes: dict[str, str]) -> list[TriggerRow]:
    rows: list[TriggerRow] = []
    for record in records:
        state_scan = record.get("state_scan", "")
        state_h4 = record.get("state_h4", "")
        if not state_scan:
            continue
        date = record.get("study_date") or record["scan_time_kst"][:10]
        rank = record.get("rank_24h")
        rows.append(
            TriggerRow(
                study_date=date,
                market_regime=regimes.get(date, "unknown"),
                symbol=record["symbol"],
                scan_time_kst=record["scan_time_kst"],
                state_scan=state_scan,
                state_h4=state_h4,
                state_h6=record.get("state_h6", ""),
                transition_scan_h4=f"{state_scan}->{state_h4}",
                trigger_ma24_slope=pf(record.get("ma24_slope_percent", "")),
                trigger_volume_ratio=pf(record.get("volume_ratio_ma24", "")),
                trigger_position_7d=pf(record.get("position_7d_percent", "")),
                trigger_body_expansion=pf(record.get("body_expansion_ratio", "")),
                trigger_rank_24h=int(rank) if str(rank).isdigit() else None,
                trigger_f4_return=pf(record.get("forward_4h", "")),
                supply_score=supply_score(record),
                outcome_ref_12h=record.get("outcome_ref_12h", "unknown"),
                counterexample_key=f"{date}|{state_scan}|{state_h4}",
            )
        )
    return rows


def mean_feature(rows: list[TriggerRow], attr: str) -> float | None:
    values = [getattr(row, attr) for row in rows]
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def find_counterexamples(rows: list[TriggerRow]) -> list[tuple[str, list[TriggerRow]]]:
    groups: dict[str, list[TriggerRow]] = defaultdict(list)
    for row in rows:
        key = f"{row.state_scan}|{row.transition_scan_h4}"
        groups[key].append(row)

    results: list[tuple[str, list[TriggerRow]]] = []
    for key, group in groups.items():
        supplies = {row.supply_score for row in group}
        if "high" in supplies and "low" in supplies and len(group) >= 2:
            results.append((key, group))
    results.sort(key=lambda item: len(item[1]), reverse=True)
    return results


def save_rows(rows: list[TriggerRow]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in fields(TriggerRow)]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = {field.name: getattr(row, field.name) for field in fields(TriggerRow)}
            for key, value in data.items():
                if isinstance(value, float):
                    data[key] = f"{value:.4f}"
                elif value is None:
                    data[key] = ""
            writer.writerow(data)


def write_report(lines: list[str]) -> None:
    text = "\n".join(lines)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    records = join_records()
    by_date: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        date = record.get("study_date") or record["scan_time_kst"][:10]
        by_date[date].append(record)

    regimes = {date: classify_regime(date, day_rows) for date, day_rows in by_date.items()}
    trigger_rows = build_trigger_rows(records, regimes)
    save_rows(trigger_rows)

    lines: list[str] = []
    lines.append("===== SEASON2 AUTONOMOUS RESEARCH REPORT =====")
    lines.append("")
    lines.append("1. Today's research topic")
    lines.append("   Transition Trigger + Market Regime + counterexample-first analysis")
    lines.append("   (linking scan-state, +4h path, pretrend triggers, 2-6h supply score)")
    lines.append("")
    lines.append("2. New data acquisition reason")
    if TOP3_ENRICHED.exists():
        lines.append("   - Used enriched 06-14 TOP3 with forward +2h/+4h")
    else:
        lines.append("   - MISSING: run enrich_forward_20260614.py (06-14 lacked +2h/+4h)")
    lines.append("   - Still missing: 3rd date TOP10, near-miss log, per-checkpoint features")
    lines.append("")
    lines.append("3. Observations")
    for date, regime in sorted(regimes.items()):
        day = [row for row in trigger_rows if row.study_date == date]
        high = sum(1 for row in day if row.supply_score == "high")
        lines.append(f"   - {date} regime={regime} | cases={len(day)} | high_supply={high}")

    warm = [row for row in trigger_rows if row.state_scan == "Warm-up"]
    warm_high = [row for row in warm if row.supply_score == "high"]
    warm_low = [row for row in warm if row.supply_score == "low"]
    if warm_high and warm_low:
        lines.append(
            f"   - Warm-up scan: high_supply ma24_slope avg={mean_feature(warm_high, 'trigger_ma24_slope'):.2f} "
            f"vs low_supply {mean_feature(warm_low, 'trigger_ma24_slope'):.2f}"
        )
        lines.append(
            f"   - Warm-up scan: high_supply vol_ratio avg={mean_feature(warm_high, 'trigger_volume_ratio'):.2f} "
            f"vs low_supply {mean_feature(warm_low, 'trigger_volume_ratio'):.2f}"
        )

    exh = [row for row in trigger_rows if row.state_scan == "Exhaustion"]
    if exh:
        collapse_h4 = sum(1 for row in exh if row.state_h4 == "Collapse")
        lines.append(
            f"   - Exhaustion at scan -> Collapse at +4h: {collapse_h4}/{len(exh)} "
            f"({collapse_h4 / len(exh) * 100:.0f}%)"
        )

    lines.append("")
    lines.append("4. Counterexamples (priority)")
    for key, group in find_counterexamples(trigger_rows)[:5]:
        lines.append(f"   - Same transition {key}:")
        for row in group[:4]:
            time_label = row.scan_time_kst[11:16]
            lines.append(
                f"       {row.symbol} @{time_label} supply={row.supply_score} "
                f"slope={row.trigger_ma24_slope} vol={row.trigger_volume_ratio} pos7d={row.trigger_position_7d}"
            )

    lines.append("")
    lines.append("5. New hypotheses")
    hypotheses = [
        "H1: +4h forward return (trigger_f4) separates supply better than scan-state alone",
        "H2: Warm-up + MA24 slope>5 + volume_ratio 1.5-4.0 -> higher 6h supply than Warm-up without slope",
        "H3: Exhaustion scan in Leader Concentration regime -> Collapse at +4h more often",
        "H4: rank_24h >3 with Warm-up scan outperforms rank<=2 Exhaustion for 6h supply",
        "H5: Transition trigger bundle (slope+vol+compression) beats any single feature",
    ]
    for hypothesis in hypotheses:
        lines.append(f"   - {hypothesis}")

    lines.append("")
    lines.append("6. Existing hypothesis revisions")
    revisions = [
        "REVISE: 'Compression at scan is good' -> only when +4h NOT Collapse (COAI vs ESPORTS)",
        "REVISE: 'Exhaustion always bad' -> COAI@17 Exhaustion still reached high supply (late parabolic)",
        "KEEP: scan-only decisions insufficient; +4h checkpoint remains critical",
        "DROP as primary: body_expansion alone (weak separator in Warm-up counterexamples)",
    ]
    for item in revisions:
        lines.append(f"   - {item}")

    lines.append("")
    lines.append("7. Next research needed because")
    needs = [
        "Only 2 dates; need 2026-06-15 TOP10 for regime replication",
        "Need near-miss file (TOP10 symbols failing one pretrend gate)",
        "Need observable features AT +4h candle (not just cumulative return)",
        "Need HUSDT/COAI/RIF cross-day panel with identical state labels",
        "forward_2h=0 artifact on 06-13 may distort Choppy->* transitions",
    ]
    for item in needs:
        lines.append(f"   - {item}")

    lines.append("")
    lines.append("8. Scout ideas (research only)")
    ideas = [
        "State machine output: current_state + dominant_trigger + regime_tag",
        "Alert tier-1: Warm-up/Compression in Broad Rally or Rotation regime",
        "Alert tier-2 (+4h): confirm Expansion/Continuation path before automation hook",
        "Anti-trigger research: rank#1 AND return_24h>40% -> label Exhaustion regime",
        "Log transition_pair counts daily as market maturity indicator",
    ]
    for idea in ideas:
        lines.append(f"   - {idea}")

    lines.append("")
    lines.append("9. Most promising research direction")
    lines.append(
        "   Build +4h Transition Trigger panel: for each scan-state, measure which OBSERVABLE "
        "triggers at scan (MA slope, volume ratio, compression, rank) predict "
        "Expansion/Continuation at +6h vs Collapse - across 3+ dates. "
        "This directly serves 2-6h auto-trade supply without inventing Scout filters yet."
    )
    lines.append("==============================================")
    lines.append(f"CSV: {OUTPUT_CSV}")

    write_report(lines)


if __name__ == "__main__":
    main()
