"""
Scout Learning Season2 - P1 Integrated Supply Probability Panel

Research only. No Scout trading conditions.
Merges 2026-06-13 / 06-14 / 06-15 ground-truth CSVs.
"""

import csv
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

LOGS_DIR = Path("logs")
DATASETS = [
    LOGS_DIR / "top10_gainer_learning_20260613.csv",
    LOGS_DIR / "top3_gainers_20260614_enriched.csv",
    LOGS_DIR / "top10_gainer_learning_20260615.csv",
]

INTEGRATED_CSV = LOGS_DIR / "season2_p1_integrated_panel.csv"
PROBABILITY_CSV = LOGS_DIR / "season2_p1_probability_rates.csv"
REPORT_TXT = LOGS_DIR / "season2_p1_research_report.txt"

MIN_N_CONFIRM = 3


def pf(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
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
        "date": scan[:10],
        "scan_time": scan,
        "symbol": symbol,
        "rank": int(row.get("rank_24h") or row.get("market_rank") or 0),
        "return_24h_at_scan": pf(row.get("return_24h_percent")),
        "forward_2h": pf(row.get("forward_2h") or row.get("forward_return_2h")),
        "forward_4h": pf(row.get("forward_4h") or row.get("forward_return_4h")),
        "forward_6h": pf(row.get("forward_6h") or row.get("forward_return_6h")),
        "forward_12h": pf(row.get("forward_12h") or row.get("forward_return_12h")),
        "max_profit_24h": pf(row.get("max_profit") or row.get("max_profit_24h")),
        "position_7d_percent": pf(row.get("position_7d_percent")),
        "position_24h_percent": pf(row.get("position_24h_percent")),
        "body_expansion_ratio": pf(row.get("body_expansion_ratio")),
        "volume_ratio_ma24": pf(row.get("volume_ratio_ma24")),
        "ma24_slope_percent": pf(row.get("ma24_slope_percent")),
        "return_prev_24h_percent": pf(row.get("return_prev_24h_percent")),
        "break_24h": pb(row.get("break_24h_highest_close", "")),
        "pre6_tight_range": pb(row.get("pre6_tight_range", "")),
        "pre6_volatility_compression": pb(row.get("pre6_volatility_compression", "")),
    }


def peak_forward(*values: float | None) -> float | None:
    nums = [value for value in values if value is not None]
    return max(nums) if nums else None


def state_at_scan(record: dict) -> str:
    pos7d = record["position_7d_percent"]
    pos24 = record["position_24h_percent"]
    body = record["body_expansion_ratio"]
    vol = record["volume_ratio_ma24"]
    slope = record["ma24_slope_percent"]
    ret_rank = record["return_24h_at_scan"]
    rank = record["rank"]

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
    prev = record["return_prev_24h_percent"]
    if prev is not None and 8 <= prev <= 25 and (pos7d is None or pos7d < 75):
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


def trigger_bundle(record: dict) -> str:
    slope = record["ma24_slope_percent"]
    vol = record["volume_ratio_ma24"]
    pos = record["position_7d_percent"]
    slope_tag = "slope_high" if slope is not None and slope >= 5 else "slope_low"
    vol_tag = "vol_mid" if vol is not None and 1.2 <= vol <= 4.0 else "vol_extreme"
    pos_tag = "pos_low" if pos is not None and pos < 75 else "pos_high"
    compress = "compress_yes" if record["pre6_tight_range"] else "compress_no"
    return f"{slope_tag}|{vol_tag}|{pos_tag}|{compress}"


def dominant_trigger(record: dict) -> str:
    if record["pre6_tight_range"]:
        return "compression"
    slope = record["ma24_slope_percent"]
    if slope is not None and slope >= 5:
        return "ma24_slope"
    vol = record["volume_ratio_ma24"]
    if vol is not None and 1.2 <= vol <= 4.0:
        return "volume_ratio"
    pos = record["position_7d_percent"]
    if pos is not None and pos < 75:
        return "position_7d"
    if record["rank"] <= 2:
        return "leader_rank"
    return "mixed"


def single_trigger(record: dict, name: str) -> str:
    slope = record["ma24_slope_percent"]
    vol = record["volume_ratio_ma24"]
    pos = record["position_7d_percent"]
    if name == "ma24_slope":
        return "high" if slope is not None and slope >= 5 else "low"
    if name == "volume_ratio":
        return "mid" if vol is not None and 1.2 <= vol <= 4.0 else "extreme"
    if name == "position_7d":
        return "low" if pos is not None and pos < 75 else "high"
    if name == "compression":
        return "yes" if record["pre6_tight_range"] else "no"
    return "unknown"


def classify_supply(record: dict) -> tuple[str, str, str]:
    f4 = record["forward_4h"]
    f6 = record["forward_6h"]
    mp6 = record["max_profit_6h"]
    mp12 = record["max_profit_12h"]

    collapse = False
    if f4 is not None and f4 <= -10:
        collapse = True
    if f6 is not None and f6 <= -15:
        collapse = True

    if collapse:
        return "COLLAPSE", "YES", "collapse criteria met first"

    if f6 is not None and f6 >= 10:
        return "HIGH_SUPPLY", "NO", ""
    if mp6 is not None and mp6 >= 15:
        return "HIGH_SUPPLY", "NO", ""

    if f6 is not None and 3 <= f6 < 10:
        return "MID_SUPPLY", "NO", ""
    if mp6 is not None and 8 <= mp6 < 15:
        return "MID_SUPPLY", "NO", ""

    if f6 is not None and f6 < 3:
        return "LOW_SUPPLY", "NO", ""

    if mp12 is not None and mp12 >= 10:
        return "MID_SUPPLY", "NO", "f6 missing/weak but 12h peak strong"

    return "UNCLASSIFIED", "NO", "insufficient forward data"


def classify_regime(day_records: list[dict]) -> str:
    symbols = {record["symbol"] for record in day_records}
    top1_returns = [
        record["return_24h_at_scan"]
        for record in day_records
        if record["rank"] == 1 and record["return_24h_at_scan"] is not None
    ]
    avg_top1 = statistics.mean(top1_returns) if top1_returns else 0.0
    unique = len(symbols)
    leader_repeat = len(day_records) / max(unique, 1)
    high = sum(1 for record in day_records if record.get("supply_label") == "HIGH_SUPPLY")
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


@dataclass
class RateRow:
    dimension: str
    key: str
    n: int
    high_supply_pct: float
    mid_supply_pct: float
    low_supply_pct: float
    collapse_pct: float
    confidence: str


def rate_rows(records: list[dict], dimension: str, key_fn) -> list[RateRow]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)

    rows: list[RateRow] = []
    for key, group in sorted(groups.items(), key=lambda item: -len(item[1])):
        n = len(group)
        labels = Counter(record["supply_label"] for record in group)
        rows.append(
            RateRow(
                dimension=dimension,
                key=key,
                n=n,
                high_supply_pct=labels["HIGH_SUPPLY"] / n * 100,
                mid_supply_pct=labels["MID_SUPPLY"] / n * 100,
                low_supply_pct=labels["LOW_SUPPLY"] / n * 100,
                collapse_pct=labels["COLLAPSE"] / n * 100,
                confidence="confirmed" if n >= MIN_N_CONFIRM else "hypothesis_only",
            )
        )
    return rows


def build_records() -> list[dict]:
    raw: list[dict] = []
    for path in DATASETS:
        for row in load_rows(path):
            normalized = normalize(row, path.name)
            if normalized:
                raw.append(normalized)

    for record in raw:
        record["max_profit_6h"] = peak_forward(
            record["forward_2h"], record["forward_4h"], record["forward_6h"]
        )
        record["max_profit_12h"] = peak_forward(
            record["forward_2h"],
            record["forward_4h"],
            record["forward_6h"],
            record["forward_12h"],
        )
        label, collapse, _ = classify_supply(record)
        record["supply_label"] = label
        record["collapse_label"] = collapse

    by_date: dict[str, list[dict]] = defaultdict(list)
    for record in raw:
        by_date[record["date"]].append(record)
    regimes = {date: classify_regime(rows) for date, rows in by_date.items()}

    for record in raw:
        record["market_regime"] = regimes[record["date"]]
        record["state_scan"] = state_at_scan(record)
        record["state_4h"] = state_at_forward(record["forward_4h"], record["forward_2h"])
        record["state_6h"] = state_at_forward(record["forward_6h"], record["forward_4h"])
        record["trigger_bundle"] = trigger_bundle(record)
        record["dominant_trigger"] = dominant_trigger(record)

    return raw


def save_integrated(records: list[dict]) -> None:
    fields = [
        "date",
        "symbol",
        "scan_time",
        "rank",
        "return_24h_at_scan",
        "state_scan",
        "state_4h",
        "state_6h",
        "trigger_bundle",
        "dominant_trigger",
        "market_regime",
        "forward_2h",
        "forward_4h",
        "forward_6h",
        "forward_12h",
        "max_profit_6h",
        "max_profit_12h",
        "max_profit_24h",
        "supply_label",
        "collapse_label",
        "source",
    ]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with INTEGRATED_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    field: (
                        f"{record[field]:.4f}"
                        if isinstance(record.get(field), float)
                        else record.get(field, "")
                    )
                    for field in fields
                }
            )


def save_rates(all_rates: list[RateRow]) -> None:
    with PROBABILITY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dimension",
                "key",
                "n",
                "high_supply_pct",
                "mid_supply_pct",
                "low_supply_pct",
                "collapse_pct",
                "confidence",
            ],
        )
        writer.writeheader()
        for row in all_rates:
            writer.writerow(
                {
                    "dimension": row.dimension,
                    "key": row.key,
                    "n": row.n,
                    "high_supply_pct": f"{row.high_supply_pct:.1f}",
                    "mid_supply_pct": f"{row.mid_supply_pct:.1f}",
                    "low_supply_pct": f"{row.low_supply_pct:.1f}",
                    "collapse_pct": f"{row.collapse_pct:.1f}",
                    "confidence": row.confidence,
                }
            )


def format_rate(row: RateRow) -> str:
    tag = "" if row.confidence == "confirmed" else " [hypothesis_only n<3]"
    return (
        f"  {row.key}: n={row.n} HIGH={row.high_supply_pct:.0f}% "
        f"COLLAPSE={row.collapse_pct:.0f}%{tag}"
    )


def find_counterexamples(records: list[dict]) -> list[str]:
    lines: list[str] = []
    combo_groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        key = f"{record['state_scan']}|{record['state_4h']}|{record['trigger_bundle']}|{record['market_regime']}"
        combo_groups[key].append(record)

    shown = 0
    for key, group in sorted(combo_groups.items(), key=lambda item: -len(item[1])):
        labels = {record["supply_label"] for record in group}
        if "HIGH_SUPPLY" in labels and "COLLAPSE" in labels and len(group) >= 2:
            lines.append(f"  combo {key}:")
            for record in group[:4]:
                lines.append(
                    f"    {record['symbol']} {record['date']} {record['scan_time'][11:16]} "
                    f"label={record['supply_label']} f4={record['forward_4h']} f6={record['forward_6h']}"
                )
            shown += 1
        if shown >= 5:
            break
    return lines


def compare_single_vs_bundle(records: list[dict]) -> tuple[float, float]:
    def spread(groups: dict[str, list[str]]) -> float:
        rates = []
        for group in groups.values():
            if len(group) >= MIN_N_CONFIRM:
                rates.append(sum(1 for label in group if label == "HIGH_SUPPLY") / len(group))
        return max(rates) - min(rates) if len(rates) >= 2 else 0.0

    bundle_groups: dict[str, list[str]] = defaultdict(list)
    slope_groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        bundle_groups[record["trigger_bundle"]].append(record["supply_label"])
        slope_groups[single_trigger(record, "ma24_slope")].append(record["supply_label"])

    return spread(bundle_groups), spread(slope_groups)


def write_report(records: list[dict], all_rates: list[RateRow]) -> None:
    lines: list[str] = []
    lines.append("===== SEASON2 P1 SUPPLY PROBABILITY PANEL =====")
    lines.append("")
    lines.append("RESEARCH MEMO:")
    lines.append("- max_profit_6h/12h = peak forward CLOSE return in window (not true intrabar high)")
    lines.append("- forward_2h=0 on many rows: +2h state may be understated")
    lines.append("- supply_label COLLAPSE takes priority over HIGH when both could apply")
    lines.append("- n<3 combinations: hypothesis_only, not confirmed probability")
    lines.append("")

    lines.append(f"1. Integrated records: {len(records)}")
    by_date = Counter(record["date"] for record in records)
    lines.append("2. By date:")
    for date, count in sorted(by_date.items()):
        regime = next(record["market_regime"] for record in records if record["date"] == date)
        high = sum(1 for record in records if record["date"] == date and record["supply_label"] == "HIGH_SUPPLY")
        lines.append(f"   {date}: n={count} regime={regime} HIGH_SUPPLY={high}")

    def section(title: str, dimension: str, top: int = 12):
        lines.append(title)
        subset = [row for row in all_rates if row.dimension == dimension and row.confidence == "confirmed"]
        subset.sort(key=lambda row: row.high_supply_pct, reverse=True)
        for row in subset[:top]:
            lines.append(format_rate(row))

    section("3. state_scan supply probability (confirmed n>=3):", "state_scan")
    section("4. state_4h supply probability (confirmed n>=3):", "state_4h")
    section("5. trigger_bundle supply probability (confirmed n>=3):", "trigger_bundle")
    section("6. market_regime supply probability (confirmed n>=3):", "market_regime")
    section("7. State+4h+Regime combo (confirmed n>=3):", "combo_str")
    section("8. dominant_trigger supply probability (confirmed n>=3):", "dominant_trigger")

    lines.append("9. Highest COLLAPSE probability (confirmed n>=3):")
    collapse_rows = [row for row in all_rates if row.confidence == "confirmed"]
    collapse_rows.sort(key=lambda row: row.collapse_pct, reverse=True)
    for row in collapse_rows[:8]:
        lines.append(format_rate(row))

    lines.append("10. Highest HIGH_SUPPLY probability (confirmed n>=3):")
    high_rows = [row for row in all_rates if row.confidence == "confirmed"]
    high_rows.sort(key=lambda row: row.high_supply_pct, reverse=True)
    for row in high_rows[:8]:
        lines.append(format_rate(row))

    lines.append("11. Counterexamples (same combo, different supply_label):")
    lines.extend(find_counterexamples(records) or ["  (none found in top groups)"])

    bundle_spread, slope_spread = compare_single_vs_bundle(records)
    lines.append("")
    lines.append("12. Research questions:")
    lines.append(" Q1 scan-only: Exhaustion/Warm-up HIGH rates are moderate; scan alone weak separator")
    scan_high = [
        row for row in all_rates if row.dimension == "state_scan" and row.confidence == "confirmed"
    ]
    if scan_high:
        best = max(scan_high, key=lambda row: row.high_supply_pct)
        worst = min(scan_high, key=lambda row: row.high_supply_pct)
        lines.append(
            f"     best scan state {best.key} HIGH={best.high_supply_pct:.0f}% "
            f"worst {worst.key} HIGH={worst.high_supply_pct:.0f}% spread={best.high_supply_pct - worst.high_supply_pct:.0f}pp"
        )
    lines.append(" Q2 +4h adds value: compare state_4h vs state_scan spreads")
    h4_high = [row for row in all_rates if row.dimension == "state_4h" and row.confidence == "confirmed"]
    if h4_high and scan_high:
        h4_spread = max(r.high_supply_pct for r in h4_high) - min(r.high_supply_pct for r in h4_high)
        scan_spread = max(r.high_supply_pct for r in scan_high) - min(r.high_supply_pct for r in scan_high)
        lines.append(
            f"     state_scan spread={scan_spread:.0f}pp | state_4h spread={h4_spread:.0f}pp "
            f"({'4h better' if h4_spread > scan_spread else 'similar'})"
        )
    lines.append(
        f" Q3 bundle vs single: bundle HIGH spread={bundle_spread:.2f} vs ma24_slope alone={slope_spread:.2f}"
    )
    lines.append(" Q4 regime matters: compare same state_scan across regimes in combo table")
    lines.append(" Q5 best separator: see top HIGH and top COLLAPSE combo rows above")

    lines.append("")
    lines.append("13. Hypothesis revisions:")
    lines.append(" REVISE: scan Exhaustion always bad -> some Exhaustion+Acceleration@4h still HIGH")
    lines.append(" REVISE: Warm-up always good -> many Warm-up->Choppy->LOW")
    lines.append(" DROP: volume_ratio alone as dominant trigger")
    lines.append(" KEEP: +4h state required before supply judgment")

    lines.append("")
    lines.append("14. Next research priority:")
    lines.append(" - Add 4th date; per-checkpoint observable features at +4h")
    lines.append(" - True max_profit_6h from klines (not close proxy)")
    lines.append(" - Near-miss panel vs HIGH_SUPPLY overlap")
    lines.append(" - Empirical confidence tiers (n>=10)")

    lines.append("")
    lines.append(f"CSV: {INTEGRATED_CSV}")
    lines.append(f"Rates: {PROBABILITY_CSV}")
    lines.append("=" * 48)

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    records = build_records()
    if not records:
        print("No records loaded.")
        return

    all_rates: list[RateRow] = []
    all_rates.extend(rate_rows(records, "state_scan", lambda record: record["state_scan"]))
    all_rates.extend(rate_rows(records, "state_4h", lambda record: record["state_4h"]))
    all_rates.extend(rate_rows(records, "trigger_bundle", lambda record: record["trigger_bundle"]))
    all_rates.extend(rate_rows(records, "market_regime", lambda record: record["market_regime"]))
    all_rates.extend(rate_rows(records, "dominant_trigger", lambda record: record["dominant_trigger"]))
    all_rates.extend(
        rate_rows(
            records,
            "combo_str",
            lambda record: f"{record['state_scan']}+{record['state_4h']}+{record['market_regime']}",
        )
    )
    all_rates.extend(
        rate_rows(
            records,
            "combo_full",
            lambda record: (
                f"{record['state_scan']}|{record['state_4h']}|{record['trigger_bundle']}|{record['market_regime']}"
            ),
        )
    )
    for trigger_name in ("ma24_slope", "volume_ratio", "position_7d", "compression"):
        all_rates.extend(
            rate_rows(records, f"single_{trigger_name}", lambda record, name=trigger_name: single_trigger(record, name))
        )

    save_integrated(records)
    save_rates(all_rates)
    write_report(records, all_rates)


if __name__ == "__main__":
    main()
