"""
Scout Learning Season2 - State Transition Learning

Research only.
- State labels use ONLY information observable at each checkpoint.
- Forward outcomes (+12h/+24h) are stored separately for evaluation, NOT for state definition.
"""

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

LOGS_DIR = Path("logs")
OUTPUT_CSV = LOGS_DIR / "state_transition_learning_season2.csv"

SOURCES = [
    LOGS_DIR / "top10_gainer_learning_20260613.csv",
    LOGS_DIR / "top3_gainers_20260614_enriched.csv"
    if (LOGS_DIR / "top3_gainers_20260614_enriched.csv").exists()
    else LOGS_DIR / "top3_gainers_20260614.csv",
]

CHECKPOINTS = ("scan", "h2", "h4", "h6", "h12")

STATES = (
    "Compression",
    "Warm-up",
    "Expansion",
    "Continuation",
    "Acceleration",
    "Exhaustion",
    "Collapse",
    "Choppy",
    "Recovery",
    "Transition",
)


@dataclass
class StateCase:
    source_file: str
    study_date: str
    scan_time_kst: str
    symbol: str
    rank_24h: int
    state_scan: str
    state_h2: str
    state_h4: str
    state_h6: str
    state_h12: str
    transition_chain: str
    transition_pairs: str
    outcome_ref_12h: str
    outcome_ref_24h: str
    forward_2h: float | None
    forward_4h: float | None
    forward_6h: float | None
    forward_12h: float | None
    forward_24h: float | None
    counterexample_group: str
    research_note: str


def parse_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def parse_bool(value: str) -> bool:
    return (value or "").strip().upper() == "YES"


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

    ret24 = parse_float(row.get("return_24h_percent") or row.get("change_24h_percent", ""))
    if ret24 is None:
        return None

    return {
        "source_file": path.name,
        "study_date": scan[:10],
        "scan_time_kst": scan,
        "symbol": symbol,
        "rank_24h": int(row.get("rank_24h") or row.get("market_rank") or row.get("rank") or 0),
        "return_24h_percent": ret24,
        "forward_2h": parse_float(row.get("forward_2h") or row.get("forward_return_2h", "")),
        "forward_4h": parse_float(row.get("forward_4h") or row.get("forward_return_4h", "")),
        "forward_6h": parse_float(row.get("forward_6h") or row.get("forward_return_6h", "")),
        "forward_12h": parse_float(row.get("forward_12h") or row.get("forward_return_12h", "")),
        "forward_24h": parse_float(row.get("forward_24h") or row.get("forward_return_24h", "")),
        "position_7d_percent": parse_float(row.get("position_7d_percent", "")),
        "position_24h_percent": parse_float(row.get("position_24h_percent", "")),
        "body_expansion_ratio": parse_float(row.get("body_expansion_ratio", "")),
        "volume_ratio_ma24": parse_float(row.get("volume_ratio_ma24", "")),
        "ma24_slope_percent": parse_float(row.get("ma24_slope_percent", "")),
        "distance_ma24_percent": parse_float(row.get("distance_from_ma24_percent", "")),
        "break_24h": parse_bool(row.get("break_24h_highest_close", "")),
        "break_7d": parse_bool(row.get("break_7d_highest_close", "")),
        "pre6_tight_range": parse_bool(row.get("pre6_tight_range", "")),
        "pre6_body_compression": parse_bool(row.get("pre6_body_compression", "")),
        "pre6_volatility_compression": parse_bool(row.get("pre6_volatility_compression", "")),
        "pre6_volume_contraction": parse_bool(row.get("pre6_volume_contraction", "")),
        "pre12_tight_range": parse_bool(row.get("pre12_tight_range", "")),
        "return_prev_24h_percent": parse_float(row.get("return_prev_24h_percent", "")),
    }


def state_at_scan(row: dict) -> str:
    """Pretrend observables only (data available at scan candle close)."""
    pos7d = row["position_7d_percent"]
    pos24 = row["position_24h_percent"]
    body = row["body_expansion_ratio"]
    vol = row["volume_ratio_ma24"]
    slope = row["ma24_slope_percent"]
    dist_ma24 = row["distance_ma24_percent"]
    ret_rank = row["return_24h_percent"]
    rank = row["rank_24h"]
    prev24 = row["return_prev_24h_percent"]

    compressed = (
        row["pre6_tight_range"]
        or row["pre6_volatility_compression"]
        or row["pre12_tight_range"]
    )
    if compressed and (body is None or body < 1.5):
        return "Compression"

    if rank <= 2 and ret_rank >= 35:
        return "Exhaustion"
    if pos7d is not None and pos7d >= 95:
        return "Exhaustion"
    if pos24 is not None and pos24 >= 100:
        return "Exhaustion"
    if row["break_24h"] and ret_rank >= 20:
        return "Exhaustion"

    if slope is not None and slope >= 4 and vol is not None and vol >= 1.2:
        if pos7d is None or pos7d < 80:
            return "Warm-up"

    if body is not None and body >= 1.8 and vol is not None and vol >= 2.0:
        return "Expansion"

    if prev24 is not None and 8 <= prev24 <= 25 and (pos7d is None or pos7d < 75):
        return "Warm-up"

    if dist_ma24 is not None and 5 <= dist_ma24 <= 25 and slope is not None and slope > 0:
        return "Expansion"

    if ret_rank >= 15 and rank <= 5:
        return "Transition"

    return "Choppy"


def state_at_forward(ret: float | None, prev_ret: float | None) -> str:
    """Observable at checkpoint: cumulative return since scan and change since prior checkpoint."""
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


def outcome_label(ret: float | None, threshold_strong: float = 10.0) -> str:
    if ret is None:
        return "unknown"
    if ret >= threshold_strong:
        return "favorable"
    if ret < 0:
        return "unfavorable"
    return "mixed"


def build_transitions(states: dict[str, str]) -> tuple[str, str]:
    chain_parts = [states[cp] for cp in CHECKPOINTS if states.get(cp) != "Unknown"]
    chain = " -> ".join(chain_parts)

    pairs: list[str] = []
    for index in range(len(CHECKPOINTS) - 1):
        left = CHECKPOINTS[index]
        right = CHECKPOINTS[index + 1]
        if states[left] == "Unknown" or states[right] == "Unknown":
            continue
        pairs.append(f"{states[left]}->{states[right]}")

    return chain, "; ".join(pairs)


def auto_trading_memo() -> None:
    print("\n--- 6. Auto-trading research memos (not rules) ---")
    memos = {
        "Compression": ("watch", "observe", "if -> Warm-up", "if -> Collapse without warm-up"),
        "Warm-up": ("favorable watch", "early Expansion", "after shallow dip", "failed Expansion"),
        "Expansion": ("possible entry research", "late if Exhaustion next", "after pullback", "-> Exhaustion"),
        "Continuation": ("hold research zone", "avoid new chase", "if pullback holds", "deceleration"),
        "Acceleration": ("late chase risk", "avoid new", "rare", "sharp -> Exhaustion"),
        "Exhaustion": ("avoid entry", "exit research", "only if -> Recovery confirmed", "-> Collapse"),
        "Collapse": ("no entry", "already late", "none", "stay out"),
        "Recovery": ("cautious re-entry research", "if unconfirmed", "after Recovery at +4h", "second Collapse"),
        "Choppy": ("wait", "no entry", "low priority", "prolonged chop"),
        "Transition": ("need next checkpoint", "wait", "case-by-case", "unclear path"),
    }
    for state, (entry, wait, reentry, exit_note) in memos.items():
        print(f"  {state}: entry={entry} | wait={wait} | reentry={reentry} | exit={exit_note}")


def build_case(row: dict) -> StateCase:
    states = {
        "scan": state_at_scan(row),
        "h2": state_at_forward(row["forward_2h"], 0.0),
        "h4": state_at_forward(row["forward_4h"], row["forward_2h"]),
        "h6": state_at_forward(row["forward_6h"], row["forward_4h"]),
        "h12": state_at_forward(row["forward_12h"], row["forward_6h"]),
    }

    chain, pairs = build_transitions(states)
    out12 = outcome_label(row["forward_12h"])
    out24 = outcome_label(row["forward_24h"])

    return StateCase(
        source_file=row["source_file"],
        study_date=row["study_date"],
        scan_time_kst=row["scan_time_kst"],
        symbol=row["symbol"],
        rank_24h=row["rank_24h"],
        state_scan=states["scan"],
        state_h2=states["h2"],
        state_h4=states["h4"],
        state_h6=states["h6"],
        state_h12=states["h12"],
        transition_chain=chain,
        transition_pairs=pairs,
        outcome_ref_12h=out12,
        outcome_ref_24h=out24,
        forward_2h=row["forward_2h"],
        forward_4h=row["forward_4h"],
        forward_6h=row["forward_6h"],
        forward_12h=row["forward_12h"],
        forward_24h=row["forward_24h"],
        counterexample_group="",
        research_note="",
    )


def mark_counterexamples(cases: list[StateCase]) -> None:
    groups: dict[tuple[str, str], list[StateCase]] = defaultdict(list)
    for case in cases:
        if case.state_scan == "Unknown":
            continue
        groups[(case.study_date, case.state_scan)].append(case)

    for group_cases in groups.values():
        outcomes = {case.outcome_ref_12h for case in group_cases}
        if len(outcomes) > 1 and "unknown" not in outcomes:
            group_id = f"{group_cases[0].study_date}|{group_cases[0].state_scan}"
            for case in group_cases:
                case.counterexample_group = group_id
                case.research_note = (
                    f"Same scan-state {case.state_scan} on {case.study_date} "
                    f"but outcome_12h={case.outcome_ref_12h}"
                )


def save_cases(cases: list[StateCase]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(StateCase)]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            row = {field.name: getattr(case, field.name) for field in fields(StateCase)}
            for key, value in row.items():
                if isinstance(value, float):
                    row[key] = f"{value:.4f}"
                elif value is None:
                    row[key] = ""
            writer.writerow(row)


def print_report(cases: list[StateCase]) -> None:
    print("\n===== SEASON2 STATE TRANSITION LEARNING =====")
    print(f"Cases: {len(cases)}")
    print("State labels: observable at each checkpoint only.")
    print("outcome_ref_*: evaluation only, NOT used in state labeling.")

    scan_states = Counter(case.state_scan for case in cases)
    print("\n--- 1. States observed at scan ---")
    for state, count in scan_states.most_common():
        print(f"  {state}: {count}")

    print("\n--- 2. Example transition chains ---")
    interesting = sorted(
        cases,
        key=lambda item: abs(item.forward_12h or 0),
        reverse=True,
    )[:10]
    for case in interesting:
        time_label = case.scan_time_kst[11:16]
        print(f"  {case.symbol} @{time_label} [{case.study_date}]")
        print(f"    {case.transition_chain}")
        print(f"    outcome_12h(ref)={case.outcome_ref_12h}")

    pair_counts: Counter[str] = Counter()
    for case in cases:
        for pair in case.transition_pairs.split("; "):
            if pair:
                pair_counts[pair] += 1

    print("\n--- 3. Frequent transitions ---")
    for pair, count in pair_counts.most_common(15):
        print(f"  {pair}: {count}")

    print("\n--- 4. Counterexamples (same scan-state, different +12h outcome) ---")
    cx_groups: dict[str, list[StateCase]] = defaultdict(list)
    for case in cases:
        if case.counterexample_group:
            cx_groups[case.counterexample_group].append(case)

    shown = 0
    for group_id, group_cases in cx_groups.items():
        if shown >= 6:
            break
        date, scan_state = group_id.split("|", 1)
        print(f"  Group {scan_state} on {date}:")
        for case in group_cases[:4]:
            time_label = case.scan_time_kst[11:16]
            print(
                f"    {case.symbol} @{time_label} -> h12={case.state_h12} "
                f"(f12={case.forward_12h}, ref={case.outcome_ref_12h})"
            )
        shown += 1

    if not cx_groups:
        print("  (none marked - broaden dates or states)")

    print("\n--- 5. New hypotheses ---")
    hypotheses = [
        "H1: Compression at scan -> Warm-up at +4h is a higher-value path than Expansion at scan",
        "H2: Scan-state Exhaustion often -> Collapse at +4h regardless of 24h rank momentum",
        "H3: Expansion at +4h with Continuation at +6h separates tradeable supply better than scan labels",
        "H4: Recovery at +4h after early Exhaustion is rare; treat as counter-trend research only",
        "H5: Transition at scan needs +4h resolution; Scout should not act on scan alone",
    ]
    for index, text in enumerate(hypotheses, start=1):
        print(f"  {index}. {text}")

    auto_trading_memo()

    print("\n--- 7. States needing more validation ---")
    unknown_h2 = sum(1 for case in cases if case.state_h2 == "Unknown")
    print(f"  Unknown at +2h (missing forward_2h data): {unknown_h2} cases")
    print("  Exhaustion at scan: validate on 3+ dates")
    print("  Recovery path: small sample, needs dedicated labeling review")
    print("  Compression -> Warm-up: count per date vs market regime")

    print("\n--- 8. Scout research ideas ---")
    ideas = [
        "Output current STATE not buy/sell; state machine over repeated 2h observations",
        "First Scout pass: classify Compression/Warm-up/Exhaustion at scan",
        "Second pass (+4h): confirm Expansion vs Collapse before alerting automation",
        "Track transition_pair frequencies per date as market regime indicator",
        "Store counterexample_group ids to prioritize manual review",
    ]
    for index, idea in enumerate(ideas, start=1):
        print(f"  {index}. {idea}")

    print("\n--- Symbol cross-day note (HUSDT) ---")
    h_cases = [case for case in cases if case.symbol == "HUSDT"]
    for case in sorted(h_cases, key=lambda item: item.scan_time_kst)[:6]:
        print(f"  {case.study_date} @{case.scan_time_kst[11:16]} scan={case.state_scan} chain={case.transition_chain[:60]}...")

    print("==============================================")


def main() -> None:
    cases: list[StateCase] = []
    for path in SOURCES:
        for row in load_rows(path):
            normalized = normalize_row(path, row)
            if normalized is None:
                continue
            cases.append(build_case(normalized))

    mark_counterexamples(cases)
    save_cases(cases)
    print_report(cases)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
