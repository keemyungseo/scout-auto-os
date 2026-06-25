"""Directional Zero-Base orchestrator."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.research.directional.engines import (
    LONG_ENGINES,
    SHORT_ENGINES,
    rank_long,
    rank_short,
)
from scout_auto_os.engine.research.directional.evaluation import (
    MAX_LONG_SLOTS,
    MAX_SHORT_SLOTS,
    QUALITY_MIN_AVG_2H,
    QUALITY_MIN_SAMPLES,
    QUALITY_MIN_WIN_RATE,
    aggregate_directional,
    build_champion_board,
    simulate_slots,
    to_long_metrics,
    to_short_metrics,
)
from scout_auto_os.engine.research.directional.patterns import (
    ALL_PATTERNS,
    label_direction_pattern,
    pattern_side,
)
from scout_auto_os.engine.research.directional.report import build_directional_report
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics
from scout_auto_os.engine.research.zero_base.random_baseline import generate_random_draws
from scout_auto_os.engine.research.zero_base.runner import (
    TRAIN_CUTOFF,
    _is_validation,
    load_candidates_jsonl,
    load_forward_klines,
)

KST = timezone(timedelta(hours=9))


class DirectionalZeroBaseRunner:
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

    @research_safe("directional_zerobase")
    def run(self, max_scans: int | None = None) -> dict:
        print("[DIRECTIONAL ZEROBASE] started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        val_scans = sorted(s for s in by_scan if _is_validation(s, self.train_cutoff))
        if max_scans:
            val_scans = val_scans[:max_scans]

        long_by_engine: dict[str, list[dict]] = defaultdict(list)
        short_by_engine: dict[str, list[dict]] = defaultdict(list)
        pattern_long_rets: dict[str, list[float]] = defaultdict(list)
        pattern_short_rets: dict[str, list[float]] = defaultdict(list)
        random_long_samples: list[dict] = []
        random_short_samples: list[dict] = []

        for scan_kst in val_scans:
            rows = by_scan[scan_kst]
            for r in rows:
                r["direction_pattern"] = label_direction_pattern(r["features"])

            symbols = [r["symbol"] for r in rows]

            def metric_fn(sym: str) -> dict | None:
                klines = fwd.get((scan_kst, sym))
                if not klines:
                    return None
                raw = compute_forward_metrics(klines)
                return raw if raw else None

            # Random baselines
            for pick in generate_random_draws(symbols, 5, self.random_draws, hash(scan_kst) % 10000):
                for sym in pick:
                    raw = metric_fn(sym)
                    if raw:
                        random_long_samples.append(to_long_metrics(raw))
                        random_short_samples.append(to_short_metrics(raw))

            for eng in LONG_ENGINES:
                if eng == "RANDOM_LONG":
                    continue
                for sym in rank_long(rows, eng, top_k=5):
                    raw = metric_fn(sym)
                    if raw:
                        m = to_long_metrics(raw)
                        m["engine"] = eng
                        m["scan_time_kst"] = scan_kst
                        m["pattern"] = next(
                            (r["direction_pattern"] for r in rows if r["symbol"] == sym), "UNLABELED",
                        )
                        long_by_engine[eng].append(m)
                        pat = m["pattern"]
                        if pat != "UNLABELED":
                            pattern_long_rets[pat].append(float(m["return_2h"]))

            for eng in SHORT_ENGINES:
                if eng == "RANDOM_SHORT":
                    continue
                for sym in rank_short(rows, eng, top_k=5):
                    raw = metric_fn(sym)
                    if raw:
                        m = to_short_metrics(raw)
                        m["engine"] = eng
                        m["scan_time_kst"] = scan_kst
                        m["pattern"] = next(
                            (r["direction_pattern"] for r in rows if r["symbol"] == sym), "UNLABELED",
                        )
                        short_by_engine[eng].append(m)
                        pat = m["pattern"]
                        if pat != "UNLABELED":
                            pattern_short_rets[pat].append(float(m["short_return_2h"]))

        long_random = aggregate_directional(random_long_samples, "long")
        long_random["engine"] = "RANDOM_LONG"
        short_random = aggregate_directional(random_short_samples, "short")
        short_random["engine"] = "RANDOM_SHORT"

        long_aggs: list[dict] = []
        for eng in LONG_ENGINES:
            agg = aggregate_directional(long_by_engine.get(eng, []), "long")
            agg["engine"] = eng
            long_aggs.append(agg)

        short_aggs: list[dict] = []
        for eng in SHORT_ENGINES:
            agg = aggregate_directional(short_by_engine.get(eng, []), "short")
            agg["engine"] = eng
            short_aggs.append(agg)

        long_board = build_champion_board(long_aggs, long_random, "long", exclude_baseline=("A6_LONG",))
        short_board = build_champion_board(short_aggs, short_random, "short")
        a6_agg = next((a for a in long_aggs if a["engine"] == "A6_LONG"), {})

        pattern_stats: list[dict] = []
        for pat in ALL_PATTERNS:
            if pat == "UNLABELED":
                continue
            lr = pattern_long_rets.get(pat, [])
            sr = pattern_short_rets.get(pat, [])
            pattern_stats.append({
                "pattern": pat,
                "side": pattern_side(pat),
                "sample_count": len(lr) + len(sr),
                "long_avg_return_2h": round(statistics.mean(lr), 4) if lr else None,
                "short_avg_return_2h": round(statistics.mean(sr), 4) if sr else None,
            })

        slot_sim = simulate_slots(long_board, short_board, long_by_engine, short_by_engine)

        meta = {
            "validation_scans": len(val_scans),
            "random_draws": self.random_draws,
            "max_long_slots": MAX_LONG_SLOTS,
            "max_short_slots": MAX_SHORT_SLOTS,
            "quality_min_samples": QUALITY_MIN_SAMPLES,
            "quality_min_win": QUALITY_MIN_WIN_RATE,
            "quality_min_avg2h": QUALITY_MIN_AVG_2H,
            "generated_at": datetime.now(KST).isoformat(),
        }

        report = build_directional_report(
            long_board, short_board, long_random, short_random,
            pattern_stats, slot_sim, a6_agg, meta,
        )
        report_path = self.out_dir / "directional_report.md"
        report_path.write_text(report, encoding="utf-8")

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "long_champion_board.csv", long_board)
        write_csv(self.out_dir / "short_champion_board.csv", short_board)
        write_csv(self.out_dir / "directional_pattern_stats.csv", pattern_stats)
        (self.out_dir / "directional_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print("[DIRECTIONAL ZEROBASE] report generated")
        return {
            "meta": meta,
            "long_board": long_board,
            "short_board": short_board,
            "slot_sim": slot_sim,
            "report_path": str(report_path),
        }
