"""Zero-Base Validation V1 — long-term blind validation report (research only)."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.research.zero_base.candidates import (
    SCORERS,
    rank_all_engines,
    rank_engine,
    scan_context,
)
from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics
from scout_auto_os.engine.research.zero_base.ranking import (
    MIN_CHAMPION_SAMPLES,
    aggregate_candidate_metrics,
    build_champion_board,
    champion_eligible,
    compare_vs_baseline,
)
from scout_auto_os.engine.research.zero_base.random_baseline import generate_random_draws
from scout_auto_os.engine.research.zero_base.runner import (
    TRAIN_CUTOFF,
    load_candidates_jsonl,
    load_forward_klines,
    _is_validation,
)

KST = timezone(timedelta(hours=9))

# User-facing engine names → zero_base / league scorers (no new rules)
VALIDATION_ENGINES: dict[str, str] = {
    "RANDOM": "RANDOM_BASELINE",
    "A6": "A6_CURRENT",
    "FORMULA_LEAGUE": "PURE_MOMENTUM_1H",  # formula_league MOMENTUM_ONLY proxy
    "FEATURE_LEAGUE": "HIGH_VOLUME_GREEN_CANDLE",  # feature-league volume+momentum proxy
    "STATE_LEAGUE": "MULTI_TIMEFRAME_ALIGNMENT",  # state-league MTF alignment proxy
    "MOMENTUM": "PURE_MOMENTUM_1H",
    "BREAKOUT": "BREAKOUT_1H_HIGH",
    "COMPRESSION": "COMPRESSION_RELEASE",
    "RANGE_EXPANSION": "RANGE_EXPANSION",
    "VWAP": "VWAP_RECLAIM",
    "PULLBACK": "PULLBACK_CONTINUATION",
    "RELATIVE_STRENGTH": "BTC_RELATIVE_STRENGTH",
}


def _feature_league_rank(rows: list[dict], top_k: int = 5) -> list[str]:
    """Feature league pick: momentum_1h>0 filtered, ranked by volume_ratio."""
    filt = [r for r in rows if float(r["features"].get("1h_current_return_pct", 0)) > 0]
    if not filt:
        filt = rows
    ranked = sorted(
        filt,
        key=lambda r: float(r["features"].get("15m_current_volume_ratio", 0)),
        reverse=True,
    )
    return [r["symbol"] for r in ranked[:top_k]]


def rank_validation_engine(rows: list[dict], engine_name: str, top_k: int = 5) -> list[str]:
    if engine_name == "FEATURE_LEAGUE":
        return _feature_league_rank(rows, top_k)
    scorer_key = VALIDATION_ENGINES.get(engine_name, engine_name)
    if scorer_key == "RANDOM_BASELINE":
        return rank_engine(rows, "RANDOM_BASELINE", top_k, scan_context(rows))
    key = scorer_key if scorer_key in SCORERS else engine_name
    if key in SCORERS:
        ctx = scan_context(rows)
        ranked = sorted(rows, key=lambda r: SCORERS[key](r, ctx), reverse=True)
        return [r["symbol"] for r in ranked[:top_k]]
    picks = rank_all_engines(rows, top_k=top_k)
    return picks.get(scorer_key, [])


def random_stats_with_ci(
    symbols: list[str],
    metric_fn,
    n_draws: int = 100,
    seed: int = 42,
) -> dict:
    draw_means: list[float] = []
    all_samples: list[dict] = []
    for i, pick in enumerate(generate_random_draws(symbols, 5, n_draws, seed)):
        rets: list[float] = []
        for sym in pick:
            m = metric_fn(sym)
            if m:
                all_samples.append(m)
                rets.append(float(m["return_2h"]))
        if rets:
            draw_means.append(statistics.mean(rets))
    agg = aggregate_candidate_metrics(all_samples)
    if len(draw_means) < 2:
        ci_lo, ci_hi = agg.get("avg_return_2h", 0), agg.get("avg_return_2h", 0)
    else:
        mu = statistics.mean(draw_means)
        se = statistics.stdev(draw_means) / math.sqrt(len(draw_means))
        ci_lo = round(mu - 1.96 * se, 4)
        ci_hi = round(mu + 1.96 * se, 4)
    agg["draw_means_std"] = round(statistics.pstdev(draw_means), 4) if len(draw_means) > 1 else 0
    agg["ci95_low"] = ci_lo
    agg["ci95_high"] = ci_hi
    agg["draw_count"] = len(draw_means)
    return agg


def significance_vs_random(engine_returns: list[float], random_draw_means: list[float]) -> dict:
    """Welch-style: engine per-pick returns vs random draw means."""
    if len(engine_returns) < 30 or len(random_draw_means) < 10:
        return {"significant": False, "p_approx": None, "reason": "insufficient_sample"}
    mu_e = statistics.mean(engine_returns)
    mu_r = statistics.mean(random_draw_means)
    se_e = statistics.stdev(engine_returns) / math.sqrt(len(engine_returns)) if len(engine_returns) > 1 else 0
    se_r = statistics.stdev(random_draw_means) / math.sqrt(len(random_draw_means)) if len(random_draw_means) > 1 else 0
    se_diff = math.sqrt(se_e ** 2 + se_r ** 2) or 1e-9
    z = (mu_e - mu_r) / se_diff
    # two-tailed normal approx
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {
        "significant": p < 0.05 and mu_e > mu_r,
        "p_approx": round(p, 4),
        "z_score": round(z, 3),
        "beats_random_mean": mu_e > mu_r,
    }


def classify_regime(rows: list[dict]) -> str:
    rets = [float(r["features"].get("1h_current_return_pct", 0)) for r in rows]
    med = statistics.median(rets) if rets else 0
    if med >= 1.0:
        return "bull"
    if med <= -0.5:
        return "bear"
    return "sideway"


def failure_analysis(engine: str, samples: list[dict], checks: dict) -> dict:
    if not samples:
        return {"engine": engine, "status": "no_data", "reasons": ["no forward samples"]}
    reasons: list[str] = []
    if not checks.get("beats_random_return"):
        reasons.append("avg_return_2h below random baseline")
    if not checks.get("beats_random_trap"):
        reasons.append("trap_rate not lower than random")
    if not checks.get("beats_a6_return"):
        reasons.append("avg_return_2h below A6")
    if not checks.get("median_positive"):
        reasons.append("median_return_2h not positive")
    if not checks.get("sample_count_ok"):
        reasons.append(f"sample_count < {MIN_CHAMPION_SAMPLES}")

    # regime failures
    by_regime: dict[str, list[float]] = defaultdict(list)
    symbol_fail: dict[str, int] = defaultdict(int)
    for s in samples:
        reg = s.get("regime", "unknown")
        r2h = float(s.get("return_2h", 0))
        by_regime[reg].append(r2h)
        if r2h < 0:
            symbol_fail[s.get("symbol", "?")] += 1
    regime_weak = [reg for reg, rets in by_regime.items() if statistics.mean(rets) < 0]
    if regime_weak:
        reasons.append(f"negative avg in regimes: {', '.join(regime_weak)}")
    top_fail_syms = sorted(symbol_fail.items(), key=lambda x: x[1], reverse=True)[:5]
    return {
        "engine": engine,
        "status": "failed" if reasons else "passed",
        "reasons": reasons,
        "regime_avg_return_2h": {k: round(statistics.mean(v), 4) for k, v in by_regime.items()},
        "top_losing_symbols": [{"symbol": s, "loss_count": c} for s, c in top_fail_syms],
    }


def build_validation_report(
    board: list[dict],
    random_stats: dict,
    a6_row: dict,
    failures: list[dict],
    regime_win: dict[str, dict[str, float]],
    meta: dict,
) -> str:
    lines = [
        "# SCOUT Zero-Base Validation V1 Report",
        "",
        f"Generated: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')} KST",
        f"Train: before {meta.get('train_cutoff')} ({meta.get('train_scans', 0)} scans)",
        f"Blind Validation: June+ ({meta.get('validation_scans', 0)} scans, 5m grid)",
        f"Random: {meta.get('random_draws', 100)} draws, 95% CI [{random_stats.get('ci95_low')}, {random_stats.get('ci95_high')}]",
        "",
        "## ① Engine Rankings",
    ]
    for row in board:
        sig = "✓" if row.get("statistically_significant") else "✗"
        lines.append(
            f"- #{row.get('board_rank')} **{row.get('engine')}** "
            f"avg2h={row.get('avg_return_2h')}% win={row.get('win_rate')}% "
            f"PF={row.get('profit_factor')} sig={sig} [{row.get('tier')}]"
        )

    a6_rank = next((r.get("board_rank") for r in board if r.get("engine") == "A6"), "?")
    lines.extend([
        "",
        "## ② A6 Rank",
        f"- A6 rank: **#{a6_rank}** / {len(board)}",
        f"- A6 avg2h={a6_row.get('avg_return_2h')}% win={a6_row.get('win_rate')}% trap={a6_row.get('trap_rate')}%",
        "",
        "## ③ vs Random",
        f"- Random avg2h={random_stats.get('avg_return_2h')}% (std={random_stats.get('draw_means_std')})",
        f"- Random 95% CI: [{random_stats.get('ci95_low')}, {random_stats.get('ci95_high')}]",
    ])
    for row in board[:8]:
        if row.get("engine") == "RANDOM":
            continue
        lines.append(
            f"- {row.get('engine')}: delta={row.get('avg_return_2h_delta_vs_random', 0)}% "
            f"significant={row.get('statistically_significant')}"
        )

    champs = [r for r in board if r.get("champion_eligible")]
    lines.extend(["", "## ④ Champion Candidates"])
    if champs:
        for c in champs:
            lines.append(f"- **{c.get('engine')}** n={c.get('sample_count')} avg2h={c.get('avg_return_2h')}%")
    else:
        lines.append("- None passed all gates (see section ⑤)")

    lines.extend(["", "## ⑤ Failure Reasons"])
    for f in failures:
        if f.get("status") == "failed":
            lines.append(f"- **{f.get('engine')}**: {'; '.join(f.get('reasons', []))}")

    lines.extend(["", "## ⑥ Regime Win Rates (validation)"])
    for eng, regimes in sorted(regime_win.items()):
        parts = ", ".join(f"{reg}={wr}%" for reg, wr in regimes.items())
        lines.append(f"- {eng}: {parts}")

    promising = [r for r in board if r.get("beats_a6_return") and r.get("beats_random_return")][:3]
    lines.extend(["", "## ⑦ Most Promising Direction"])
    if promising:
        for p in promising:
            lines.append(
                f"- {p.get('engine')}: avg2h={p.get('avg_return_2h')}% "
                f"big_winner={p.get('big_winner_capture_rate')}% (hypothesis tier)"
            )
    else:
        lines.append("- No engine beat both A6 and Random on avg_return_2h")

    lines.extend([
        "",
        "## ⑧ Next Research Priority",
        "1. Obtain May train data for true blind calibration (current bundle: train=0)",
        "2. Re-test champion candidates with trap_rate gate on expanded sample",
        "3. Cross-validate top momentum/breakout engines on July+ live research forward",
        "4. State League replay with position evolution (V1.5) on same validation window",
        "",
        "*Research only. No LIVE changes. Correlation ≠ causation. Probabilistic labels.*",
    ])
    return "\n".join(lines)


class ZeroBaseValidationRunner:
    def __init__(
        self,
        data_dir: Path,
        candidates_path: Path,
        forward_path: Path,
        random_draws: int = 100,
        train_cutoff: str = TRAIN_CUTOFF,
    ) -> None:
        self.data_dir = data_dir
        self.out_dir = data_dir / "zero_base"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path
        self.random_draws = random_draws
        self.train_cutoff = train_cutoff

    def run(self) -> dict:
        print("[ZEROBASE VALIDATION] started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        val_scans = sorted(s for s in by_scan if _is_validation(s, self.train_cutoff))
        train_scans = sorted(s for s in by_scan if not _is_validation(s, self.train_cutoff))

        samples_by_engine: dict[str, list[dict]] = defaultdict(list)
        random_draw_means_all: list[float] = []

        for scan_kst in val_scans:
            rows = by_scan[scan_kst]
            regime = classify_regime(rows)
            symbols = [r["symbol"] for r in rows]

            def metric_fn(sym: str) -> dict | None:
                klines = fwd.get((scan_kst, sym))
                if not klines:
                    return None
                m = compute_forward_metrics(klines)
                if not m:
                    return None
                m["scan_time_kst"] = scan_kst
                m["symbol"] = sym
                m["regime"] = regime
                return m

            # Random CI per scan aggregated
            for i, pick in enumerate(generate_random_draws(symbols, 5, self.random_draws, hash(scan_kst) % 10000)):
                rets = []
                for sym in pick:
                    m = metric_fn(sym)
                    if m:
                        rets.append(float(m["return_2h"]))
                if rets:
                    random_draw_means_all.append(statistics.mean(rets))

            for eng_name in VALIDATION_ENGINES:
                syms = rank_validation_engine(rows, eng_name, top_k=5)
                for sym in syms:
                    m = metric_fn(sym)
                    if m:
                        m = dict(m)
                        m["engine"] = eng_name
                        samples_by_engine[eng_name].append(m)

        # Global random stats
        random_samples = []
        for scan_kst in val_scans:
            rows = by_scan[scan_kst]
            symbols = [r["symbol"] for r in rows]

            def metric_fn(sym: str, sk=scan_kst) -> dict | None:
                klines = fwd.get((sk, sym))
                if not klines:
                    return None
                return compute_forward_metrics(klines)

            for pick in generate_random_draws(symbols, 5, self.random_draws, 42):
                for sym in pick:
                    m = metric_fn(sym)
                    if m:
                        random_samples.append(m)

        random_stats = aggregate_candidate_metrics(random_samples)
        if len(random_draw_means_all) > 1:
            mu = statistics.mean(random_draw_means_all)
            se = statistics.stdev(random_draw_means_all) / math.sqrt(len(random_draw_means_all))
            random_stats["ci95_low"] = round(mu - 1.96 * se, 4)
            random_stats["ci95_high"] = round(mu + 1.96 * se, 4)
            random_stats["draw_means_std"] = round(statistics.stdev(random_draw_means_all), 4)
        random_stats["engine"] = "RANDOM"

        candidate_results: list[dict] = []
        for eng_name in VALIDATION_ENGINES:
            samples = samples_by_engine.get(eng_name, [])
            agg = aggregate_candidate_metrics(samples)
            agg["engine"] = eng_name
            a6_agg = aggregate_candidate_metrics(samples_by_engine.get("A6", []))
            cmp = compare_vs_baseline(agg, random_stats, a6_agg)
            sig = significance_vs_random(
                [float(s["return_2h"]) for s in samples],
                random_draw_means_all,
            )
            agg.update(cmp)
            agg["statistically_significant"] = sig.get("significant", False)
            agg["p_approx"] = sig.get("p_approx")
            candidate_results.append(agg)

        board = build_champion_board(candidate_results, random_stats, samples_by_engine)
        sig_by_eng = {r["engine"]: r.get("statistically_significant", False) for r in candidate_results}
        p_by_eng = {r["engine"]: r.get("p_approx") for r in candidate_results}
        for row in board:
            row["statistically_significant"] = sig_by_eng.get(row.get("engine"), False)
            row["p_approx"] = p_by_eng.get(row.get("engine"))
            if not row.get("statistically_significant"):
                row["champion_eligible"] = False
                if row.get("tier") == "champion_candidate":
                    row["tier"] = "verification_needed"
        champ_by_eng = {c["engine"]: c for c in board}
        for row in candidate_results:
            extra = champ_by_eng.get(row["engine"], {})
            row.update({k: extra[k] for k in ("champion_eligible", "tier", "board_rank", "positive_days") if k in extra})
            row["statistically_significant"] = row.get("statistically_significant", False)

        failures: list[dict] = []
        for eng_name in VALIDATION_ENGINES:
            samples = samples_by_engine.get(eng_name, [])
            row = next((c for c in candidate_results if c["engine"] == eng_name), {})
            checks = champ_by_eng.get(eng_name, {}).get("champion_checks", {})
            fa = failure_analysis(eng_name, samples, checks)
            fa["champion_eligible"] = row.get("champion_eligible", False)
            failures.append(fa)

        regime_win: dict[str, dict[str, float]] = {}
        for eng_name, samples in samples_by_engine.items():
            by_reg: dict[str, list[float]] = defaultdict(list)
            for s in samples:
                by_reg[s.get("regime", "unknown")].append(float(s["return_2h"]))
            regime_win[eng_name] = {
                reg: round(sum(1 for x in rets if x >= 3) / len(rets) * 100, 1)
                for reg, rets in by_reg.items() if rets
            }

        meta = {
            "train_cutoff": self.train_cutoff,
            "train_scans": len(train_scans),
            "validation_scans": len(val_scans),
            "random_draws": self.random_draws,
            "generated_at": datetime.now(KST).isoformat(),
        }

        a6_row = next((c for c in candidate_results if c["engine"] == "A6"), {})
        report = build_validation_report(board, random_stats, a6_row, failures, regime_win, meta)

        report_path = self.out_dir / "validation_v1_report.md"
        report_path.write_text(report, encoding="utf-8")

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "validation_candidate_results.csv", candidate_results)
        (self.out_dir / "validation_failures.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        write_csv(self.out_dir / "validation_champion_board.csv", board)
        (self.out_dir / "validation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print("[ZEROBASE VALIDATION] report generated")
        return {
            "meta": meta,
            "board": board,
            "failures": failures,
            "random_stats": random_stats,
            "report_path": str(report_path),
        }
