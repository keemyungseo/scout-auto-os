"""
Scout Learning Season2 - Trend Case Learning

Research only. Classifies trend SHAPE from forward path (+2h/+4h/+6h/+12h).
Does NOT validate Scout filters. Does NOT output trading rules.
"""

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "trend_case_learning_season2.csv"

SOURCES = [
    LOGS_DIR / "top10_gainer_learning_20260613.csv",
    LOGS_DIR / "top3_gainers_20260614.csv",
]


@dataclass
class TrendCase:
    source_file: str
    study_date: str
    scan_time_kst: str
    symbol: str
    rank_24h: int
    return_24h_percent: float
    forward_2h: float | None
    forward_4h: float | None
    forward_6h: float | None
    forward_12h: float | None
    forward_24h: float | None
    max_profit: float
    max_drawdown: float
    trend_type: str
    lifecycle_stages: str
    auto_entry_zone: str
    auto_late_zone: str
    auto_reentry_zone: str
    auto_exit_signal: str
    notes: str


def parse_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_row(path: Path, row: dict[str, str]) -> dict | None:
    scan = row.get("scan_time_kst") or row.get("search_time_kst", "")
    symbol = row.get("symbol", "")
    if not scan or not symbol:
        return None

    rank = int(row.get("rank_24h") or row.get("market_rank") or row.get("rank") or 0)
    ret24 = parse_float(row.get("return_24h_percent") or row.get("change_24h_percent", ""))
    if ret24 is None:
        return None

    f2 = parse_float(row.get("forward_2h") or row.get("forward_return_2h", ""))
    f4 = parse_float(row.get("forward_4h") or row.get("forward_return_4h", ""))
    f6 = parse_float(row.get("forward_6h") or row.get("forward_return_6h", ""))
    f12 = parse_float(row.get("forward_12h") or row.get("forward_return_12h", ""))
    f24 = parse_float(row.get("forward_24h") or row.get("forward_return_24h", ""))
    mp = parse_float(row.get("max_profit") or row.get("max_profit_24h", "")) or 0.0
    md = parse_float(row.get("max_drawdown") or row.get("max_drawdown_24h", "")) or 0.0

    return {
        "source_file": path.name,
        "study_date": scan[:10],
        "scan_time_kst": scan,
        "symbol": symbol,
        "rank_24h": rank,
        "return_24h_percent": ret24,
        "forward_2h": f2,
        "forward_4h": f4,
        "forward_6h": f6,
        "forward_12h": f12,
        "forward_24h": f24,
        "max_profit": mp,
        "max_drawdown": md,
        "position_7d_percent": parse_float(row.get("position_7d_percent", "")),
        "break_24h": (row.get("break_24h_highest_close") or "").upper() == "YES",
    }


def classify_trend(case: dict) -> tuple[str, str, str]:
    f4 = case["forward_4h"]
    f6 = case["forward_6h"]
    f12 = case["forward_12h"]
    f24 = case["forward_24h"]
    mp = case["max_profit"]
    ret_scan = case["return_24h_percent"]
    rank = case["rank_24h"]

    stages: list[str] = ["Birth"]

    if f4 is None and f6 is None:
        trend = "Incomplete Data"
        return trend, "Birth", "Need +2h/+4h forward columns"

    f4v = f4 if f4 is not None else 0.0
    f6v = f6 if f6 is not None else 0.0
    f12v = f12 if f12 is not None else 0.0
    f24v = f24 if f24 is not None else 0.0

    if f4v <= -5 or (f6 is not None and f6v <= -8):
        if mp >= 15:
            trend = "Early Leader Collapse"
            stages += ["Warm-up", "Death"]
        else:
            trend = "False Breakout"
            stages += ["Expansion", "Death"]
    elif f4v < 0 and f12 is not None and f12v >= 5:
        trend = "Second Wave"
        stages += ["Warm-up", "Exhaustion", "Expansion", "Continuation"]
    elif f4v >= 0 and f6 is not None and f6v >= 5 and f12 is not None and f12v >= f6v:
        trend = "Slow Builder"
        stages += ["Warm-up", "Expansion", "Acceleration", "Continuation"]
    elif f6 is not None and f6v >= 10 and f12 is not None and f12v < f6v * 0.5:
        trend = "Explosive Spike"
        stages += ["Expansion", "Acceleration", "Exhaustion"]
    elif f6 is not None and f6v >= 5 and f12 is not None and f12v >= 5 and f24v >= 0:
        trend = "Sustained Runner"
        stages += ["Expansion", "Continuation"]
    elif rank <= 3 and ret_scan >= 30 and f6 is not None and f6v < 0:
        trend = "Early Leader Collapse"
        stages += ["Expansion", "Exhaustion", "Death"]
    elif mp >= 10 and f12 is not None and f12v < 0:
        trend = "Spike Then Fade"
        stages += ["Expansion", "Exhaustion", "Death"]
    elif f6 is not None and 0 <= f6v < 5:
        trend = "Weak Drift"
        stages += ["Warm-up", "Exhaustion"]
    else:
        trend = "Choppy / Unclear"
        stages += ["Warm-up"]

    if mp >= 20 and f12 is not None and f12v >= 10:
        if "Continuation" not in stages:
            stages.append("Continuation")
    if f24 is not None and f24v < 0 and "Death" not in stages:
        stages.append("Death")

    lifecycle = " -> ".join(dict.fromkeys(stages))
    return trend, lifecycle, ""


def auto_trading_notes(trend_type: str) -> tuple[str, str, str, str]:
    mapping = {
        "Slow Builder": (
            "After shallow dip in first 4h or prior compression",
            "When 24h rank #1 with +30% already at scan",
            "Second wave after -5% to -10% flush if volume holds",
            "6h forward stalls while max profit already captured",
        ),
        "Second Wave": (
            "4h to 6h recovery after early flush",
            "Scan candle close when already top-3 gainer",
            "If first dip holds above MA24 and 12h turns green",
            "12h fails to exceed 6h high",
        ),
        "Sustained Runner": (
            "Warm-up through +4h if slope positive",
            "After +12h when already extended in 7d range",
            "Rare; only if pullback holds prior 6h low",
            "12h momentum decelerates vs 6h",
        ),
        "Early Leader Collapse": (
            "Avoid at scan; research pre-spike only",
            "Any entry when rank #1 and 24h change >40%",
            "Generally none",
            "Immediate negative 4h forward",
        ),
        "Explosive Spike": (
            "Only first 2-4h if volume expansion confirmed",
            "After +6h vertical move",
            "High risk",
            "Sharp drop from 6h to 12h",
        ),
        "False Breakout": (
            "Not recommended at scan",
            "Breakout candle close",
            "None",
            "4h forward negative",
        ),
        "Spike Then Fade": (
            "First 4h only with tight stop",
            "6h peak area",
            "Unreliable",
            "12h closes below 6h",
        ),
        "Weak Drift": (
            "Skip for auto-trade horizon",
            "Scan time",
            "Low probability",
            "No follow-through by 6h",
        ),
    }
    return mapping.get(
        trend_type,
        ("Case-by-case study", "When already top gainer", "Unclear", "Forward 6h negative"),
    )


def build_case(raw: dict) -> TrendCase:
    trend, lifecycle, note = classify_trend(raw)
    entry, late, reentry, exit_sig = auto_trading_notes(trend)
    return TrendCase(
        source_file=raw["source_file"],
        study_date=raw["study_date"],
        scan_time_kst=raw["scan_time_kst"],
        symbol=raw["symbol"],
        rank_24h=raw["rank_24h"],
        return_24h_percent=raw["return_24h_percent"],
        forward_2h=raw["forward_2h"],
        forward_4h=raw["forward_4h"],
        forward_6h=raw["forward_6h"],
        forward_12h=raw["forward_12h"],
        forward_24h=raw["forward_24h"],
        max_profit=raw["max_profit"],
        max_drawdown=raw["max_drawdown"],
        trend_type=trend,
        lifecycle_stages=lifecycle,
        auto_entry_zone=entry,
        auto_late_zone=late,
        auto_reentry_zone=reentry,
        auto_exit_signal=exit_sig,
        notes=note,
    )


def save_cases(cases: list[TrendCase]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(TrendCase)]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            row = {field.name: getattr(case, field.name) for field in fields(TrendCase)}
            for key, value in row.items():
                if isinstance(value, float):
                    row[key] = f"{value:.4f}"
                elif value is None:
                    row[key] = ""
            writer.writerow(row)


def print_report(cases: list[TrendCase]) -> None:
    type_counts = Counter(case.trend_type for case in cases)
    print("\n===== SEASON2 TREND CASE LEARNING =====")
    print(f"Total cases: {len(cases)}")
    print("Trend type distribution:")
    for trend_type, count in type_counts.most_common():
        print(f"  {trend_type}: {count}")

    print("\n--- 1. Trend types discovered ---")
    for trend_type in type_counts:
        sample = next(case for case in cases if case.trend_type == trend_type)
        print(f"  {trend_type}: e.g. {sample.symbol} @{sample.scan_time_kst[11:16]}")

    print("\n--- 2. Type characteristics ---")
    characteristics = {
        "Slow Builder": "f4->f6->f12 stair-step positive; often mid TOP10 rank, not always #1",
        "Second Wave": "negative first 4h then recovery by 12h; entry timing critical",
        "Sustained Runner": "holds positive through 6h and 12h; best auto-trade supply",
        "Early Leader Collapse": "rank #1 + high 24h% at scan; forward 4-6h negative",
        "Explosive Spike": "large max_profit but 12h << 6h; short-lived",
        "False Breakout": "no meaningful max profit; early death",
        "Spike Then Fade": "good peak, bad 12h close",
    }
    for trend_type, text in characteristics.items():
        if trend_type in type_counts:
            print(f"  {trend_type}: {text}")

    print("\n--- 3. Auto-trading zones (research notes, not rules) ---")
    for trend_type in ["Slow Builder", "Second Wave", "Sustained Runner", "Early Leader Collapse"]:
        if trend_type not in type_counts:
            continue
        entry, late, _, exit_sig = auto_trading_notes(trend_type)
        print(f"  {trend_type}:")
        print(f"    good entry: {entry}")
        print(f"    late entry: {late}")
        print(f"    exit signal: {exit_sig}")

    print("\n--- 4. Counterexamples ---")
    counterexamples = [
        (
            "COAIUSDT 06-13 19:00 Slow Builder vs 21:00/23:00 Spike Fade",
            "Same symbol; exhaustion after parabolic scan candle (+27% body at 21:00)",
        ),
        (
            "RIFUSDT 06-13 13:00 Second Wave success vs 17:00 Early Leader",
            "Similar rank #1; 17:00 already extended (pos7d>115%) then flat",
        ),
        (
            "HUSDT 06-13 21:00 Sustained Runner vs 06-14 17:00 Collapse",
            "Same symbol different day; 06-14 leader already +140% at scan",
        ),
        (
            "CLOUSDT 06-14 multiple scans Slow Builder",
            "Not top 24h gainer early but steady continuation across scans",
        ),
    ]
    for title, reason in counterexamples:
        print(f"  - {title}: {reason}")

    print("\n--- 5. New hypotheses ---")
    hypotheses = [
        "H1: Trend TYPE at +4h is more stable than pretrend filters at scan",
        "H2: Rank #1 with 24h change >40% predicts Early Leader Collapse, not continuation",
        "H3: Slow Builder / Second Wave need 4-6h confirmation before Scout should care",
        "H4: Repeated symbol same day shifts from Builder to Spike Fade (COAI, RIF)",
        "H5: CLOUSDT-style lower 24h rank with steady f6/f12 beats HUSDT-style leaderboard #1",
    ]
    for index, hypothesis in enumerate(hypotheses, start=1):
        print(f"  {index}. {hypothesis}")

    print("\n--- 6. Cases needing more validation ---")
    symbols_multi = Counter(case.symbol for case in cases)
    repeated = [symbol for symbol, count in symbols_multi.items() if count >= 4]
    print(f"  Repeated symbols across scans: {', '.join(repeated)}")
    print("  Need forward_2h/4h on 06-14 TOP3 for full shape classification")
    print("  Need 3+ dates to confirm Slow Builder vs Collapse split")

    print("\n--- 7. Scout research ideas (not conditions) ---")
    ideas = [
        "Scout should detect trend PHASE (compression/warm-up) not leaderboard rank",
        "Delay signal until +4h path classification or proxy (volume + MA slope)",
        "Block re-alert same symbol within 6h when first case was Spike/Exhaustion",
        "Study CLOUSDT-like 'mid-rank steady builder' as target archetype",
        "Leaderboard #1 flag as anti-pattern filter research item",
    ]
    for index, idea in enumerate(ideas, start=1):
        print(f"  {index}. {idea}")

    print("\n--- Key cases (Sustained / Builder) ---")
    good = [case for case in cases if case.trend_type in {"Slow Builder", "Sustained Runner", "Second Wave"} and (case.forward_12h or 0) >= 10]
    for case in sorted(good, key=lambda item: item.forward_12h or 0, reverse=True)[:8]:
        print(
            f"  {case.trend_type} {case.symbol} @{case.scan_time_kst[11:16]} "
            f"f4={case.forward_4h} f6={case.forward_6h} f12={case.forward_12h} mp={case.max_profit:.1f}"
        )

    print("\n--- Key cases (Collapse / False) ---")
    bad = [case for case in cases if case.trend_type in {"Early Leader Collapse", "False Breakout", "Spike Then Fade"}]
    for case in sorted(bad, key=lambda item: item.return_24h_percent, reverse=True)[:8]:
        print(
            f"  {case.trend_type} {case.symbol} @{case.scan_time_kst[11:16]} "
            f"ret24h={case.return_24h_percent:.1f}% f6={case.forward_6h} f12={case.forward_12h}"
        )

    print("========================================")


def main() -> None:
    cases: list[TrendCase] = []
    for path in SOURCES:
        for row in load_rows(path):
            normalized = normalize_row(path, row)
            if normalized is None:
                continue
            cases.append(build_case(normalized))

    save_cases(cases)
    print_report(cases)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
