"""
Scout Research OS

Idea -> Metric -> Historical Validation -> Probability -> Scanner -> Trading Rule

Research Scientist workflow. Code implements validated research only.

Usage:
  python scout_research_os.py research --idea phase_transition
  python scout_research_os.py scan --idea phase_transition
  python scout_research_os.py list-ideas
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from season2_p37_scout_decision_hierarchy import load_csv, pf, write_csv

LOGS_DIR = Path("logs")
OUT_DIR = LOGS_DIR / "research_os"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KST = timezone(timedelta(hours=9))

MetricFn = Callable[[dict], float | None]
FilterFn = Callable[[dict], bool]


@dataclass
class MetricSpec:
    metric_id: str
    label: str
    extract: MetricFn
    direction: str = "gte"  # gte | lte | eq_yes


@dataclass
class IdeaSpec:
    idea_id: str
    name: str
    phenomenon: str
    metrics: list[MetricSpec] = field(default_factory=list)
    combos: list[tuple[str, ...]] = field(default_factory=list)


@dataclass
class NumericDefinition:
    parts: list[str]
    filters: list[tuple[str, str, float | str]]  # metric_id, op, threshold


@dataclass
class ValidationResult:
    definition: NumericDefinition
    sample: int
    hit_5_rate: float
    hit_7_rate: float
    hit_10_rate: float
    win_rate: float
    expected_return: float
    avg_duration_min: float
    avg_mdd: float
    holdout_hit_5_rate: float
    score: float


@dataclass
class SimulationResult:
    trades: int
    win_rate: float
    expected_return: float
    avg_mdd: float
    avg_hold_min: float
    stop_loss_pct: float
    trailing_pct: float


@dataclass
class ResearchOutput:
    idea: IdeaSpec
    definition: NumericDefinition
    validation: ValidationResult
    simulation: SimulationResult
    candidates: list[dict]
    explanation: str
    confidence: str
    self_check: dict[str, str]


def yes_flag(row: dict, col: str) -> float | None:
    v = row.get(col, "")
    if v in ("YES", "yes", "1", 1, True):
        return 1.0
    if v in ("NO", "no", "0", 0, False):
        return 0.0
    return None


def num(row: dict, col: str) -> float | None:
    v = pf(row.get(col))
    return v if v is not None else None


def est_duration_min(row: dict) -> float:
    mp = pf(row.get("max_profit")) or 0.0
    f4 = pf(row.get("forward_4h")) or 0.0
    f6 = pf(row.get("forward_6h")) or 0.0
    f12 = pf(row.get("forward_12h")) or 0.0
    if mp <= 0:
        return 60.0
    if f12 >= mp * 0.85:
        return 720.0
    if f6 >= mp * 0.85:
        return 360.0
    if f4 >= mp * 0.85:
        return 240.0
    return 120.0


def load_research_dataset() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(LOGS_DIR.glob("top10_gainer_learning_*.csv")):
        for row in load_csv(path):
            if row.get("symbol") and row.get("scan_time_kst"):
                row["_source"] = path.name
                rows.append(row)
    return rows


def build_phase_transition_idea() -> IdeaSpec:
    metrics = [
        MetricSpec("ma24_slope", "MA24 Slope %", lambda r: num(r, "ma24_slope_percent")),
        MetricSpec("ma48_slope", "MA48 Slope %", lambda r: num(r, "ma48_slope_percent")),
        MetricSpec("slope_accel", "MA Slope Acceleration", lambda r: (
            (num(r, "ma24_slope_percent") or 0) - (num(r, "ma48_slope_percent") or 0)
            if num(r, "ma24_slope_percent") is not None else None
        )),
        MetricSpec("volume_ratio", "Volume / MA24", lambda r: num(r, "volume_ratio_ma24")),
        MetricSpec("volume_accel", "Volume Acceleration", lambda r: num(r, "volume_acceleration_ratio")),
        MetricSpec("atr_ratio", "ATR Ratio", lambda r: num(r, "atr_ratio")),
        MetricSpec("range_expansion", "Range Expansion", lambda r: num(r, "range_expansion_ratio")),
        MetricSpec("atr_compression", "ATR Compression (inverse expansion)", lambda r: (
            1.0 / (num(r, "range_expansion_ratio") or 1.0) if num(r, "range_expansion_ratio") else None
        ), direction="gte"),
        MetricSpec("body_expansion", "Body Expansion Ratio", lambda r: num(r, "body_expansion_ratio")),
        MetricSpec("return_2h", "Return Prev 2h %", lambda r: num(r, "return_prev_2h_percent")),
        MetricSpec("return_4h", "Return Prev 4h %", lambda r: num(r, "return_prev_4h_percent")),
        MetricSpec("pre6_tight", "Pre6 Tight Range", lambda r: yes_flag(r, "pre6_tight_range"), direction="eq_yes"),
        MetricSpec("pre6_vol_compress", "Pre6 Volatility Compression", lambda r: yes_flag(r, "pre6_volatility_compression"), direction="eq_yes"),
        MetricSpec("pre6_vol_contract", "Pre6 Volume Contraction", lambda r: yes_flag(r, "pre6_volume_contraction"), direction="eq_yes"),
        MetricSpec("slope_x_volume", "MA24 Slope x Volume Ratio", lambda r: (
            (num(r, "ma24_slope_percent") or 0) * (num(r, "volume_ratio_ma24") or 0)
            if num(r, "ma24_slope_percent") is not None and num(r, "volume_ratio_ma24") is not None else None
        )),
        MetricSpec("compress_x_volume", "Tight Range x Volume Ratio", lambda r: (
            (num(r, "volume_ratio_ma24") or 0) if yes_flag(r, "pre6_tight_range") == 1.0 else None
        )),
        MetricSpec("atr_x_volume", "ATR Ratio x Volume Ratio", lambda r: (
            (num(r, "atr_ratio") or 0) * (num(r, "volume_ratio_ma24") or 0)
            if num(r, "atr_ratio") is not None else None
        )),
    ]
    combos = [
        ("ma24_slope", "volume_ratio"),
        ("slope_accel", "volume_ratio"),
        ("pre6_tight", "volume_ratio"),
        ("pre6_vol_compress", "volume_accel"),
        ("ma24_slope", "volume_ratio", "pre6_tight"),
        ("slope_accel", "volume_ratio", "atr_compression"),
        ("ma24_slope", "volume_ratio", "return_2h"),
    ]
    return IdeaSpec(
        idea_id="phase_transition",
        name="Phase Transition",
        phenomenon=(
            "Market shifts from compression/equilibrium to directional expansion: "
            "range tightens, volatility compresses, then slope and volume accelerate together."
        ),
        metrics=metrics,
        combos=combos,
    )


IDEA_REGISTRY: dict[str, Callable[[], IdeaSpec]] = {
    "phase_transition": build_phase_transition_idea,
}


def metric_map(idea: IdeaSpec) -> dict[str, MetricSpec]:
    return {m.metric_id: m for m in idea.metrics}


def passes_filter(row: dict, spec: MetricSpec, threshold: float | str) -> bool:
    val = spec.extract(row)
    if val is None:
        return False
    if spec.direction == "eq_yes":
        return val >= 1.0
    if spec.direction == "lte":
        return float(val) <= float(threshold)
    return float(val) >= float(threshold)


def build_filter_fn(idea: IdeaSpec, filters: list[tuple[str, str, float | str]]) -> FilterFn:
    mmap = metric_map(idea)

    def _fn(row: dict) -> bool:
        for mid, op, thr in filters:
            spec = mmap.get(mid)
            if not spec:
                return False
            if op == "eq_yes":
                if not passes_filter(row, spec, thr):
                    return False
            elif op == "lte":
                v = spec.extract(row)
                if v is None or float(v) > float(thr):
                    return False
            else:
                v = spec.extract(row)
                if v is None or float(v) < float(thr):
                    return False
        return True

    return _fn


def format_definition(idea: IdeaSpec, filters: list[tuple[str, str, float | str]]) -> NumericDefinition:
    mmap = metric_map(idea)
    parts: list[str] = []
    for mid, op, thr in filters:
        spec = mmap[mid]
        if op == "eq_yes":
            parts.append(f"{spec.label} = YES")
        elif op == "lte":
            parts.append(f"{spec.label} <= {thr}")
        else:
            parts.append(f"{spec.label} >= {thr}")
    return NumericDefinition(parts=parts, filters=filters)


def percentile_thresholds(values: list[float], ps: list[float] | None = None) -> list[float]:
    ps = ps or [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    if not values:
        return []
    vals = sorted(values)
    out: list[float] = []
    for p in ps:
        idx = min(len(vals) - 1, int(p * len(vals)))
        out.append(vals[idx])
    return sorted(set(round(v, 4) for v in out))


def evaluate_subset(rows: list[dict], filt: FilterFn) -> dict:
    matched = [r for r in rows if filt(r)]
    n = len(matched)
    if n == 0:
        return {"n": 0}
    profits = [pf(r.get("max_profit")) or 0.0 for r in matched]
    mdds = [pf(r.get("max_drawdown")) or 0.0 for r in matched]
    durs = [est_duration_min(r) for r in matched]
    hit5 = sum(1 for p in profits if p >= 5)
    hit7 = sum(1 for p in profits if p >= 7)
    hit10 = sum(1 for p in profits if p >= 10)
    wins = sum(1 for p in profits if p > 0)
    return {
        "n": n,
        "hit_5_rate": hit5 / n,
        "hit_7_rate": hit7 / n,
        "hit_10_rate": hit10 / n,
        "win_rate": wins / n,
        "expected_return": statistics.mean(profits),
        "avg_duration_min": statistics.mean(durs),
        "avg_mdd": statistics.mean(mdds),
    }


def generate_candidate_definitions(idea: IdeaSpec, train: list[dict]) -> list[list[tuple[str, str, float | str]]]:
    mmap = metric_map(idea)
    candidates: list[list[tuple[str, str, float | str]]] = []
    min_filters = 2 if idea.idea_id == "phase_transition" else 1

    for spec in idea.metrics:
        if min_filters > 1:
            continue
        if spec.direction == "eq_yes":
            candidates.append([(spec.metric_id, "eq_yes", "YES")])
            continue
        vals = [v for r in train if (v := spec.extract(r)) is not None]
        for thr in percentile_thresholds(vals):
            op = "lte" if spec.direction == "lte" else "gte"
            candidates.append([(spec.metric_id, op, thr)])

    for combo in idea.combos:
        for p_combo in _combo_threshold_grid(idea, train, combo):
            if len(p_combo) >= min_filters:
                candidates.append(p_combo)

    return candidates


def _combo_threshold_grid(
    idea: IdeaSpec,
    train: list[dict],
    combo: tuple[str, ...],
) -> list[list[tuple[str, str, float | str]]]:
    """Generate threshold variants for a metric combination."""
    mmap = metric_map(idea)
    per_metric_opts: list[list[tuple[str, str, float | str]]] = []

    for mid in combo:
        spec = mmap[mid]
        opts: list[tuple[str, str, float | str]] = []
        if spec.direction == "eq_yes":
            opts.append((mid, "eq_yes", "YES"))
        else:
            vals = [v for r in train if (v := spec.extract(r)) is not None]
            if len(vals) < 15:
                return []
            for thr in percentile_thresholds(vals, [0.55, 0.65, 0.75]):
                op = "lte" if spec.direction == "lte" else "gte"
                opts.append((mid, op, thr))
        per_metric_opts.append(opts)

    out: list[list[tuple[str, str, float | str]]] = [[]]
    for opts in per_metric_opts:
        merged: list[list[tuple[str, str, float | str]]] = []
        for base in out:
            for o in opts:
                merged.append(base + [o])
        out = merged
    return out


def score_validation(train_stats: dict, hold_stats: dict, n_filters: int) -> float:
    if train_stats.get("n", 0) < 15:
        return -1.0
    if n_filters < 2:
        return -1.0
    sample_bonus = min(1.0, train_stats["n"] / 200)
    hold_penalty = 0.0
    if hold_stats.get("n", 0) >= 5:
        hold_penalty = abs(train_stats["hit_5_rate"] - hold_stats.get("hit_5_rate", 0)) * 0.5
    # Penalize tiny samples that overfit gainer list
    if train_stats["n"] < 25:
        sample_bonus *= 0.5
    return (
        train_stats["hit_5_rate"] * 0.30
        + train_stats["hit_7_rate"] * 0.20
        + train_stats["hit_10_rate"] * 0.10
        + min(train_stats["expected_return"] / 25.0, 1.0) * 0.15
        + sample_bonus * 0.15
        + min(n_filters / 4.0, 1.0) * 0.10
        - hold_penalty
    )


def historical_validation(idea: IdeaSpec, rows: list[dict]) -> ValidationResult:
    scans = sorted(set(r["scan_time_kst"] for r in rows))
    random.seed(42)
    holdout_scans = set(random.sample(scans, max(1, len(scans) // 5)))
    train = [r for r in rows if r["scan_time_kst"] not in holdout_scans]
    hold = [r for r in rows if r["scan_time_kst"] in holdout_scans]

    best_score = -1.0
    best_filters: list[tuple[str, str, float | str]] = []
    best_train: dict = {"n": 0}
    best_hold: dict = {"n": 0}

    for filters in generate_candidate_definitions(idea, train):
        filt = build_filter_fn(idea, filters)
        tr = evaluate_subset(train, filt)
        ho = evaluate_subset(hold, filt)
        sc = score_validation(tr, ho, len(filters))
        if sc > best_score and tr.get("n", 0) >= 15:
            best_score = sc
            best_filters = filters
            best_train = tr
            best_hold = ho

    definition = format_definition(idea, best_filters)
    return ValidationResult(
        definition=definition,
        sample=best_train.get("n", 0),
        hit_5_rate=best_train.get("hit_5_rate", 0) * 100,
        hit_7_rate=best_train.get("hit_7_rate", 0) * 100,
        hit_10_rate=best_train.get("hit_10_rate", 0) * 100,
        win_rate=best_train.get("win_rate", 0) * 100,
        expected_return=best_train.get("expected_return", 0),
        avg_duration_min=best_train.get("avg_duration_min", 0),
        avg_mdd=best_train.get("avg_mdd", 0),
        holdout_hit_5_rate=best_hold.get("hit_5_rate", 0) * 100,
        score=best_score,
    )


def run_simulation(
    idea: IdeaSpec,
    rows: list[dict],
    validation: ValidationResult,
    stop_loss_pct: float = 3.0,
    trailing_pct: float = 2.0,
) -> SimulationResult:
    filt = build_filter_fn(idea, validation.definition.filters)
    trades = [r for r in rows if filt(r)]
    if not trades:
        return SimulationResult(0, 0, 0, 0, 0, stop_loss_pct, trailing_pct)

    outcomes: list[float] = []
    mdds: list[float] = []
    holds: list[float] = []
    wins = 0

    for row in trades:
        mp = pf(row.get("max_profit")) or 0.0
        mdd = pf(row.get("max_drawdown")) or 0.0
        if mdd >= stop_loss_pct and mp < stop_loss_pct:
            ret = -stop_loss_pct
        else:
            ret = mp * 0.90 - trailing_pct * 0.25
        outcomes.append(ret)
        mdds.append(mdd)
        holds.append(est_duration_min(row))
        if ret > 0:
            wins += 1

    return SimulationResult(
        trades=len(trades),
        win_rate=wins / len(trades) * 100,
        expected_return=statistics.mean(outcomes),
        avg_mdd=statistics.mean(mdds),
        avg_hold_min=statistics.mean(holds),
        stop_loss_pct=stop_loss_pct,
        trailing_pct=trailing_pct,
    )


def live_scan(idea: IdeaSpec, validation: ValidationResult, rows: list[dict]) -> list[dict]:
    latest_scan = max(r["scan_time_kst"] for r in rows)
    filt = build_filter_fn(idea, validation.definition.filters)
    candidates = []
    for row in rows:
        if row["scan_time_kst"] != latest_scan:
            continue
        if not filt(row):
            continue
        sym = row["symbol"].replace("USDT", "")
        candidates.append({
            "symbol": sym,
            "full_symbol": row["symbol"],
            "scan_time_kst": latest_scan,
            "max_profit_hist_label": pf(row.get("max_profit")),
            "return_24h": pf(row.get("return_24h_percent")),
        })
    candidates.sort(key=lambda c: c.get("return_24h") or 0, reverse=True)
    return candidates


def build_explanation(idea: IdeaSpec, validation: ValidationResult) -> str:
    lines = [
        "Past cases matching this numeric definition showed:",
        f"- {validation.hit_5_rate:.1f}% reached +5% max excursion",
        f"- {validation.hit_7_rate:.1f}% reached +7%",
        f"- Win rate {validation.win_rate:.1f}% with mean max excursion {validation.expected_return:+.1f}%",
        "",
        f"Phenomenon ({idea.name}): {idea.phenomenon}",
        "",
        "When slope, volume, and compression metrics align as defined,",
        "historical data shows a higher rate of directional expansion after the scan point.",
    ]
    if validation.holdout_hit_5_rate > 0:
        lines.append(f"Holdout 5%+ rate: {validation.holdout_hit_5_rate:.1f}% (sanity check).")
    return "\n".join(lines)


def confidence_level(validation: ValidationResult, simulation: SimulationResult) -> str:
    if validation.sample >= 100 and validation.hit_5_rate >= 65 and simulation.expected_return >= 3:
        return "High"
    if validation.sample >= 40 and validation.hit_5_rate >= 55:
        return "Medium"
    return "Low"


def self_check(output: ResearchOutput) -> dict[str, str]:
    v, s, c = output.validation, output.simulation, output.candidates
    sim_ok = s.trades >= 15 and s.expected_return > 0
    return {
        "can_search_symbols": "YES" if c else "NO",
        "can_build_entry": "YES" if len(v.definition.filters) >= 2 else "NO",
        "probability_calculated": "YES" if v.sample >= 25 else "NO",
        "expected_return_calculated": "YES" if sim_ok else "NO",
        "explainable": "YES" if output.explanation else "NO",
    }


def trading_rule_text(simulation: SimulationResult) -> str:
    return (
        f"Entry: all numeric definition conditions at scan close\n"
        f"Stop Loss: -{simulation.stop_loss_pct:.1f}%\n"
        f"Trailing: {simulation.trailing_pct:.1f}% from peak (sim proxy)\n"
        f"Hold: until max excursion window (~{simulation.avg_hold_min:.0f} min historical avg)"
    )


def format_research_output(output: ResearchOutput) -> str:
    v = output.validation
    s = output.simulation
    lines = [
        "Idea",
        output.idea.name,
        "",
        "--------------------------------",
        "",
        "Numeric Definition",
        "",
    ]
    for part in v.definition.parts:
        lines.append(part)
    lines.extend([
        "",
        "--------------------------------",
        "",
        "Historical Result",
        "",
        f"Sample",
        f"{v.sample:,}",
        "",
        f"5%+ rise rate",
        f"{v.hit_5_rate:.1f}%",
        "",
        f"7%+ rise rate",
        f"{v.hit_7_rate:.1f}%",
        "",
        f"10%+ rise rate",
        f"{v.hit_10_rate:.1f}%",
        "",
        f"Win Rate",
        f"{v.win_rate:.1f}%",
        "",
        f"Expected Return",
        f"{v.expected_return:+.1f}%",
        "",
        f"Average Duration",
        f"{v.avg_duration_min:.0f} min",
        "",
        f"Average MDD",
        f"{v.avg_mdd:.1f}%",
        "",
        "--------------------------------",
        "",
        "Simulation",
        "",
        f"Trades: {s.trades}",
        f"Sim Win Rate: {s.win_rate:.1f}%",
        f"Sim Expected Return: {s.expected_return:+.2f}%",
        f"Sim Avg MDD: {s.avg_mdd:.1f}%",
        "",
        "Trading Rule",
        trading_rule_text(s),
        "",
        "--------------------------------",
        "",
        "Current Candidates",
        "",
    ])
    if output.candidates:
        for c in output.candidates[:10]:
            lines.append(c["symbol"])
    else:
        lines.append("(none at latest scan)")
    lines.extend([
        "",
        "--------------------------------",
        "",
        "Explanation",
        "",
        output.explanation,
        "",
        "--------------------------------",
        "",
        "Confidence",
        "",
        output.confidence,
        "",
        "--------------------------------",
        "",
        "Self Check",
        "",
    ])
    for q, ans in output.self_check.items():
        lines.append(f"{q}: {ans}")
    all_yes = all(a == "YES" for a in output.self_check.values())
    lines.extend([
        "",
        "Universe Note: Top10 gainer CSV (Jun 6-15). Base rates are NOT market-wide.",
        "",
        f"Research Complete: {'YES' if all_yes else 'NO'}",
        f"Learning recommendation: {'ADOPT_CANDIDATE' if all_yes and output.confidence != 'Low' else 'NO_ACTION'}",
    ])
    return "\n".join(lines)


def save_artifacts(idea_id: str, output: ResearchOutput, rows: list[dict]) -> None:
    prefix = OUT_DIR / idea_id
    prefix.mkdir(parents=True, exist_ok=True)

    (prefix / "research_output.txt").write_text(format_research_output(output), encoding="utf-8")
    (prefix / "definition.json").write_text(json.dumps({
        "idea": output.idea.name,
        "phenomenon": output.idea.phenomenon,
        "parts": output.validation.definition.parts,
        "filters": output.validation.definition.filters,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    write_csv(prefix / "candidates.csv", output.candidates)
    write_csv(prefix / "matched_history.csv", [
        {
            "scan_time_kst": r["scan_time_kst"],
            "symbol": r["symbol"],
            "max_profit": pf(r.get("max_profit")),
            "max_drawdown": pf(r.get("max_drawdown")),
            "group": r.get("group", ""),
        }
        for r in rows if build_filter_fn(output.idea, output.validation.definition.filters)(r)
    ])


def run_research(idea_id: str) -> ResearchOutput:
    if idea_id not in IDEA_REGISTRY:
        raise SystemExit(f"Unknown idea: {idea_id}. Use list-ideas.")
    rows = load_research_dataset()
    if len(rows) < 30:
        raise SystemExit(f"Insufficient data: {len(rows)} rows")

    idea = IDEA_REGISTRY[idea_id]()
    validation = historical_validation(idea, rows)
    simulation = run_simulation(idea, rows, validation)
    candidates = live_scan(idea, validation, rows)
    explanation = build_explanation(idea, validation)
    confidence = confidence_level(validation, simulation)

    output = ResearchOutput(
        idea=idea,
        definition=validation.definition,
        validation=validation,
        simulation=simulation,
        candidates=candidates,
        explanation=explanation,
        confidence=confidence,
        self_check={},
    )
    output.self_check = self_check(output)
    save_artifacts(idea_id, output, rows)
    return output


def scan_only(idea_id: str) -> None:
    path = OUT_DIR / idea_id / "definition.json"
    if not path.exists():
        raise SystemExit(f"No saved research for {idea_id}. Run research first.")
    rows = load_research_dataset()
    idea = IDEA_REGISTRY[idea_id]()
    saved = json.loads(path.read_text(encoding="utf-8"))
    filters = [tuple(f) for f in saved["filters"]]
    definition = NumericDefinition(parts=saved["parts"], filters=filters)  # type: ignore[arg-type]
    validation = ValidationResult(
        definition=definition, sample=0, hit_5_rate=0, hit_7_rate=0, hit_10_rate=0,
        win_rate=0, expected_return=0, avg_duration_min=0, avg_mdd=0,
        holdout_hit_5_rate=0, score=0,
    )
    candidates = live_scan(idea, validation, rows)
    print(f"Latest scan candidates for {idea.name}:")
    for c in candidates:
        print(f"  {c['symbol']} ({c['full_symbol']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scout Research OS")
    sub = parser.add_subparsers(dest="cmd")

    r1 = sub.add_parser("research", help="Run full 7-step research pipeline")
    r1.add_argument("--idea", required=True, help="e.g. phase_transition")

    s1 = sub.add_parser("scan", help="Scan current market with saved definition")
    s1.add_argument("--idea", required=True)

    sub.add_parser("list-ideas", help="List registered research ideas")

    args = parser.parse_args()

    if args.cmd == "research":
        output = run_research(args.idea)
        print(format_research_output(output))
        print(f"\nSaved to {OUT_DIR / args.idea}/")

    elif args.cmd == "scan":
        scan_only(args.idea)

    elif args.cmd == "list-ideas":
        for iid, builder in IDEA_REGISTRY.items():
            spec = builder()
            print(f"  {iid}: {spec.name} - {spec.phenomenon[:80]}...")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
