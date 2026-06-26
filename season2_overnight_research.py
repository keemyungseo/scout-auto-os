"""
Scout Learning Season2 - Overnight Autonomous Research Orchestrator

Research only. Does NOT output Scout trading conditions.
Runs multiple analysis cycles and appends to a research log.
Re-runnable: each run appends new cycle blocks.
"""

import csv
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

LOGS_DIR = Path("logs")
CYCLE_LOG = LOGS_DIR / "season2_overnight_research_log.txt"
SUPPLY_PANEL_CSV = LOGS_DIR / "season2_supply_probability_panel.csv"
NEAR_MISS_CSV = LOGS_DIR / "season2_near_miss_v3.csv"
TRIGGER_BUNDLE_CSV = LOGS_DIR / "season2_trigger_bundle_rates.csv"

DATASETS = [
    LOGS_DIR / "top10_gainer_learning_20260613.csv",
    LOGS_DIR / "top3_gainers_20260614_enriched.csv",
    LOGS_DIR / "top10_gainer_learning_20260615.csv",
]

V3 = {
    "pos7d_max": 85.0,
    "body_min": 1.3,
    "body_max": 4.0,
    "vol_min": 1.2,
    "vol_max": 4.0,
    "ret_min": 10.0,
    "ret_max": 100.0,
    "slope_min": -2.0,
    "slope_max": 8.0,
}


def pf(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    value = str(value).strip()
    return float(value) if value else None


def pb(value: str) -> bool:
    return (value or "").strip().upper() == "YES"


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize(row: dict[str, str], source: str) -> dict | None:
    scan = row.get("scan_time_kst", "")
    symbol = row.get("symbol", "")
    if not scan or not symbol:
        return None
    return {
        "source": source,
        "study_date": scan[:10],
        "scan_time_kst": scan,
        "symbol": symbol,
        "rank_24h": int(row.get("rank_24h") or row.get("market_rank") or 0),
        "return_24h_percent": pf(row.get("return_24h_percent")),
        "forward_2h": pf(row.get("forward_2h") or row.get("forward_return_2h")),
        "forward_4h": pf(row.get("forward_4h") or row.get("forward_return_4h")),
        "forward_6h": pf(row.get("forward_6h") or row.get("forward_return_6h")),
        "forward_12h": pf(row.get("forward_12h") or row.get("forward_return_12h")),
        "forward_24h": pf(row.get("forward_24h") or row.get("forward_return_24h")),
        "max_profit": pf(row.get("max_profit") or row.get("max_profit_24h")),
        "position_7d_percent": pf(row.get("position_7d_percent")),
        "position_24h_percent": pf(row.get("position_24h_percent")),
        "body_expansion_ratio": pf(row.get("body_expansion_ratio")),
        "volume_ratio_ma24": pf(row.get("volume_ratio_ma24")),
        "ma24_slope_percent": pf(row.get("ma24_slope_percent")),
        "distance_ma24_percent": pf(row.get("distance_from_ma24_percent")),
        "return_prev_24h_percent": pf(row.get("return_prev_24h_percent")),
        "break_24h": pb(row.get("break_24h_highest_close", "")),
        "pre6_tight_range": pb(row.get("pre6_tight_range", "")),
        "pre6_volatility_compression": pb(row.get("pre6_volatility_compression", "")),
        "volume_before_price": pb(row.get("volume_before_price", "")),
    }


def load_all() -> list[dict]:
    records: list[dict] = []
    for path in DATASETS:
        for row in load_rows(path):
            normalized = normalize(row, path.name)
            if normalized:
                records.append(normalized)
    return records


def supply_tier(record: dict) -> str:
    f6 = record["forward_6h"]
    f12 = record["forward_12h"]
    if f6 is not None and f6 >= 5 and f12 is not None and f12 >= 5:
        return "high"
    if f6 is not None and f6 >= 3:
        return "medium"
    if f12 is not None and f12 < 0:
        return "low"
    return "mixed"


def state_at_scan(record: dict) -> str:
    pos7d = record["position_7d_percent"]
    pos24 = record["position_24h_percent"]
    body = record["body_expansion_ratio"]
    vol = record["volume_ratio_ma24"]
    slope = record["ma24_slope_percent"]
    ret_rank = record["return_24h_percent"]
    rank = record["rank_24h"]

    compressed = record["pre6_tight_range"] or record["pre6_volatility_compression"]
    if compressed and (body is None or body < 1.5):
        return "Compression"
    if rank <= 2 and ret_rank is not None and ret_rank >= 35:
        return "Exhaustion"
    if pos7d is not None and pos7d >= 95:
        return "Exhaustion"
    if pos24 is not None and pos24 >= 100:
        return "Exhaustion"
    if record["break_24h"] and ret_rank is not None and ret_rank >= 20:
        return "Exhaustion"
    if slope is not None and slope >= 4 and vol is not None and vol >= 1.2:
        if pos7d is None or pos7d < 80:
            return "Warm-up"
    if body is not None and body >= 1.8 and vol is not None and vol >= 2.0:
        return "Expansion"
    if record["return_prev_24h_percent"] is not None:
        prev = record["return_prev_24h_percent"]
        if 8 <= prev <= 25 and (pos7d is None or pos7d < 75):
            return "Warm-up"
    return "Choppy" if ret_rank is not None and ret_rank < 15 else "Transition"


def state_at_forward(ret: float | None, prev_ret: float | None) -> str:
    if ret is None:
        return "Unknown"
    prev = prev_ret if prev_ret is not None else 0.0
    delta = ret - prev
    if ret <= -8:
        return "Collapse"
    if ret <= -3 and delta >= 2:
        return "Recovery"
    if ret < 0 and delta < -2:
        return "Collapse"
    if ret < 0:
        return "Exhaustion"
    if ret >= 12 and delta >= 5:
        return "Acceleration"
    if ret >= 6 and delta >= 2:
        return "Continuation"
    if ret >= 3 and delta > 0:
        return "Expansion"
    if ret > 0 and delta < -3:
        return "Exhaustion"
    if abs(ret) < 2 and abs(delta) < 2:
        return "Choppy"
    if ret > 0 and prev < 0:
        return "Recovery"
    if ret >= 2:
        return "Transition"
    return "Choppy"


def classify_regime(date: str, day_records: list[dict]) -> str:
    symbols = {record["symbol"] for record in day_records}
    top1_returns = [
        record["return_24h_percent"]
        for record in day_records
        if record["rank_24h"] == 1 and record["return_24h_percent"] is not None
    ]
    avg_top1 = statistics.mean(top1_returns) if top1_returns else 0.0
    unique = len(symbols)
    leader_repeat = len(day_records) / max(unique, 1)
    high = sum(1 for record in day_records if supply_tier(record) == "high")
    total = len(day_records)

    if avg_top1 >= 50:
        return "Leader Concentration (extreme)"
    if leader_repeat >= 2.5 and unique <= 12:
        return "Leader Concentration"
    if high >= total * 0.35 and avg_top1 < 25:
        return "Broad Rally"
    if avg_top1 < 20 and high < total * 0.2:
        return "Slow Market"
    if unique >= 15:
        return "Rotation"
    return "Mixed"


def trigger_bundle(record: dict) -> str:
    slope = record["ma24_slope_percent"]
    vol = record["volume_ratio_ma24"]
    pos = record["position_7d_percent"]
    slope_tag = "slope_high" if slope is not None and slope >= 5 else "slope_low"
    vol_tag = "vol_mid" if vol is not None and 1.2 <= vol <= 4.0 else "vol_extreme"
    pos_tag = "pos_low" if pos is not None and pos < 75 else "pos_high"
    compress = "compress_yes" if record["pre6_tight_range"] else "compress_no"
    return f"{slope_tag}|{vol_tag}|{pos_tag}|{compress}"


def v3_failures(record: dict) -> list[str]:
    failures: list[str] = []
    pos = record["position_7d_percent"]
    body = record["body_expansion_ratio"]
    vol = record["volume_ratio_ma24"]
    ret = record["return_prev_24h_percent"]
    slope = record["ma24_slope_percent"]
    if pos is not None and pos > V3["pos7d_max"]:
        failures.append("position_7d")
    if body is not None and body < V3["body_min"]:
        failures.append("body_low")
    if body is not None and body > V3["body_max"]:
        failures.append("body_high")
    if vol is not None and vol < V3["vol_min"]:
        failures.append("volume_low")
    if vol is not None and vol > V3["vol_max"]:
        failures.append("volume_high")
    if ret is not None and ret < V3["ret_min"]:
        failures.append("return_low")
    if ret is not None and ret > V3["ret_max"]:
        failures.append("return_high")
    if slope is not None and slope < V3["slope_min"]:
        failures.append("slope_low")
    if slope is not None and slope > V3["slope_max"]:
        failures.append("slope_high")
    return failures


def enrich_records(records: list[dict], regimes: dict[str, str]) -> list[dict]:
    enriched: list[dict] = []
    for record in records:
        item = dict(record)
        item["state_scan"] = state_at_scan(record)
        item["state_h4"] = state_at_forward(record["forward_4h"], record["forward_2h"])
        item["state_h6"] = state_at_forward(record["forward_6h"], record["forward_4h"])
        item["supply_tier"] = supply_tier(record)
        item["market_regime"] = regimes.get(record["study_date"], "unknown")
        item["trigger_bundle"] = trigger_bundle(record)
        item["transition_scan_h4"] = f"{item['state_scan']}->{item['state_h4']}"
        item["v3_pass"] = len(v3_failures(record)) == 0
        item["v3_failures"] = ",".join(v3_failures(record))
        item["v3_near_miss"] = len(v3_failures(record)) == 1
        enriched.append(item)
    return enriched


def append_log(lines: list[str]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    block = [f"\n{'=' * 60}", f"CYCLE {stamp}", "=" * 60] + lines
    with CYCLE_LOG.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(block) + "\n")
    print("\n".join(block))


def save_supply_panel(records: list[dict]) -> None:
    keys = [
        "study_date",
        "market_regime",
        "state_scan",
        "state_h4",
        "transition_scan_h4",
        "trigger_bundle",
        "symbol",
        "scan_time_kst",
        "supply_tier",
        "forward_4h",
        "forward_6h",
        "forward_12h",
    ]
    with SUPPLY_PANEL_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in keys})


def save_near_miss(records: list[dict]) -> None:
    near = [record for record in records if record["v3_near_miss"]]
    fields = [
        "study_date",
        "symbol",
        "scan_time_kst",
        "rank_24h",
        "v3_failures",
        "supply_tier",
        "forward_6h",
        "forward_12h",
        "state_scan",
        "ma24_slope_percent",
        "position_7d_percent",
    ]
    with NEAR_MISS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in near:
            writer.writerow({field: record.get(field, "") for field in fields})


def save_trigger_rates(records: list[dict]) -> None:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        key = f"{record['market_regime']}|{record['state_scan']}|{record['trigger_bundle']}"
        groups[key].append(record["supply_tier"])

    rows: list[dict] = []
    for key, tiers in groups.items():
        regime, state, bundle = key.split("|", 2)
        n = len(tiers)
        high = sum(1 for tier in tiers if tier == "high")
        rows.append(
            {
                "market_regime": regime,
                "state_scan": state,
                "trigger_bundle": bundle,
                "n": n,
                "high_supply_rate": f"{high / n * 100:.1f}" if n else "0",
                "high_supply_count": high,
            }
        )
    rows.sort(key=lambda row: (float(row["high_supply_rate"]), int(row["n"])), reverse=True)

    with TRIGGER_BUNDLE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def run_cycle(records: list[dict]) -> None:
    by_date: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_date[record["study_date"]].append(record)

    regimes = {date: classify_regime(date, rows) for date, rows in by_date.items()}
    enriched = enrich_records(records, regimes)
    save_supply_panel(enriched)
    save_near_miss(enriched)
    save_trigger_rates(enriched)

    lines: list[str] = []
    lines.append("1. Research topic: Multi-date Transition Trigger + Regime + Near-Miss v3")
    lines.append("2. Data acquired:")
    for path in DATASETS:
        status = "OK" if path.exists() else "MISSING"
        lines.append(f"   - {path.name}: {status}")
    lines.append(f"   Total unified records: {len(enriched)}")

    lines.append("3. Observations by date/regime:")
    for date in sorted(regimes):
        day = [record for record in enriched if record["study_date"] == date]
        high = sum(1 for record in day if record["supply_tier"] == "high")
        lines.append(
            f"   - {date} regime={regimes[date]} n={len(day)} high_supply={high} "
            f"({high / len(day) * 100:.0f}%)"
        )

    trans_counts = Counter(record["transition_scan_h4"] for record in enriched)
    lines.append("4. Top transitions (scan->+4h):")
    for transition, count in trans_counts.most_common(8):
        highs = sum(
            1
            for record in enriched
            if record["transition_scan_h4"] == transition and record["supply_tier"] == "high"
        )
        lines.append(f"   - {transition}: n={count} high_supply={highs}")

    lines.append("5. Counterexamples (same scan-state + same trigger_bundle, different supply):")
    bundle_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in enriched:
        bundle_groups[(record["state_scan"], record["trigger_bundle"])].append(record)
    cx_shown = 0
    for (state, bundle), group in sorted(bundle_groups.items(), key=lambda item: -len(item[1])):
        tiers = {record["supply_tier"] for record in group}
        if "high" in tiers and "low" in tiers and len(group) >= 2:
            lines.append(f"   - state={state} bundle={bundle}:")
            for record in group[:3]:
                lines.append(
                    f"       {record['symbol']} {record['study_date']} "
                    f"supply={record['supply_tier']} f4={record['forward_4h']}"
                )
            cx_shown += 1
        if cx_shown >= 5:
            break

    near_high = [
        record
        for record in enriched
        if record["v3_near_miss"] and record["supply_tier"] == "high"
    ]
    near_low = [
        record
        for record in enriched
        if record["v3_near_miss"] and record["supply_tier"] == "low"
    ]
    lines.append("6. Near-miss Scout v3 (failed exactly 1 condition):")
    lines.append(f"   - total near-miss: {sum(1 for r in enriched if r['v3_near_miss'])}")
    lines.append(f"   - near-miss + high supply: {len(near_high)}")
    lines.append(f"   - near-miss + low supply: {len(near_low)}")
    if near_high:
        fail_counts = Counter(record["v3_failures"] for record in near_high)
        lines.append(f"   - top failing gate among near-miss winners: {fail_counts.most_common(3)}")

    repeated = Counter(record["symbol"] for record in enriched)
    repeat_symbols = [symbol for symbol, count in repeated.items() if count >= 4]
    lines.append("7. Repeated symbols (4+ appearances):")
    lines.append(f"   {', '.join(sorted(repeat_symbols)[:15])}")
    for symbol in ["HUSDT", "COAIUSDT", "RIFUSDT", "EVAAUSDT", "CLOUSDT"]:
        sym_records = [record for record in enriched if record["symbol"] == symbol]
        if len(sym_records) >= 2:
            states = Counter(record["state_scan"] for record in sym_records)
            tiers = Counter(record["supply_tier"] for record in sym_records)
            lines.append(f"   - {symbol}: scans={len(sym_records)} states={dict(states)} supply={dict(tiers)}")

    lines.append("8. New hypotheses:")
    lines.append("   H1: trigger_bundle slope_high|vol_mid|pos_low predicts high supply better than v3 pass")
    lines.append("   H2: +4h Expansion/Continuation state dominates supply; scan-state is prior only")
    lines.append("   H3: Near-miss on return_low may hide valid leaders in Leader Concentration regime")
    lines.append("   H4: EVAAUSDT panel shows state matters more than symbol identity across days")

    lines.append("9. Revised / dropped hypotheses:")
    lines.append("   REVISE: v3 position_7d<=85 - EVAA high supply at pos7d>90 on 06-15")
    lines.append("   REVISE: rank#1 always Exhaustion - EVAA rank1 can still high supply if f4 positive")
    lines.append("   DROP: volume_ratio alone as primary trigger (counterexamples in Warm-up group)")
    lines.append("   KEEP: +4h checkpoint mandatory before automation supply judgment")

    lines.append("10. Most promising next research:")
    lines.append("    Empirical Supply Probability table:")
    lines.append("    P(high | state_scan, state_h4, regime, trigger_bundle) with 4+ dates")
    lines.append("11. Next research reason:")
    lines.append("    Only 3 dates; need 06-16 TOP10 + near-miss on full universe sample")
    lines.append("12. Scout output ideas:")
    lines.append("    current_state, dominant_trigger, market_regime, supply_probability, confidence")

    if TRIGGER_BUNDLE_CSV.exists():
        with TRIGGER_BUNDLE_CSV.open(encoding="utf-8") as handle:
            top_bundles = list(csv.DictReader(handle))[:5]
        lines.append("13. Top trigger bundles by high_supply_rate:")
        for row in top_bundles:
            if int(row["n"]) >= 3:
                lines.append(
                    f"    {row['state_scan']} | {row['trigger_bundle']} | "
                    f"rate={row['high_supply_rate']}% n={row['n']}"
                )

    append_log(lines)


def main() -> None:
    print("Season2 overnight research orchestrator starting...")
    records = load_all()
    if not records:
        print("No records found. Check DATASETS paths.")
        return
    run_cycle(records)
    print(f"\nLog appended: {CYCLE_LOG}")
    print(f"Panel: {SUPPLY_PANEL_CSV}")
    print(f"Near-miss: {NEAR_MISS_CSV}")
    print(f"Trigger rates: {TRIGGER_BUNDLE_CSV}")


if __name__ == "__main__":
    main()
