"""Zero-Base Discovery orchestrator — blind validation on historical bundle or live research data."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.candidates import (
    CANDIDATE_ENGINES,
    EVAL_INTERVALS,
    TRAIN_CUTOFF,
    rank_all_engines,
    rank_engine,
    scan_context,
)
from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics
from scout_auto_os.engine.research.zero_base.random_baseline import random_baseline_for_scan
from scout_auto_os.engine.research.zero_base.ranking import (
    aggregate_candidate_metrics,
    build_champion_board,
    compare_vs_baseline,
    feature_diagnostics,
)
from scout_auto_os.engine.research.zero_base.report import build_zero_base_report
from scout_auto_os.engine.research.zero_base.storage import ZeroBaseStore


def _parse_scan_date(scan_kst: str) -> str:
    return scan_kst[:10]


def _is_validation(scan_kst: str, cutoff: str = TRAIN_CUTOFF) -> bool:
    return _parse_scan_date(scan_kst) >= cutoff


def _interval_match(scan_kst: str, interval: str) -> bool:
    dt = datetime.strptime(scan_kst, "%Y-%m-%d %H:%M:%S")
    if interval == "5m":
        return dt.minute % 5 == 0
    if interval == "15m":
        return dt.minute % 15 == 0
    if interval == "30m":
        return dt.minute % 30 == 0
    if interval == "1h":
        return dt.minute == 0
    return True


def load_candidates_jsonl(path: Path) -> dict[str, list[dict]]:
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_scan[row["scan_kst"]].append({
            "scan_kst": row["scan_kst"],
            "symbol": row["symbol"],
            "features": row["features"],
            "max_up_4h": row.get("max_up_4h"),
        })
    return by_scan


def load_forward_klines(path: Path) -> dict[tuple[str, str], list]:
    out: dict[tuple[str, str], list] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["scan_kst"], row["symbol"])
        out[key] = row.get("forward_klines_15m") or []
    return out


class ZeroBaseRunner:
    """Lab Stream — discover engines that beat A6 baseline on blind validation."""

    def __init__(
        self,
        data_dir: Path,
        candidates_path: Path | None = None,
        forward_klines_path: Path | None = None,
        random_draws: int = 100,
        top_k: int = 5,
        engines: tuple[str, ...] | None = None,
        eval_intervals: tuple[str, ...] = EVAL_INTERVALS,
        train_cutoff: str = TRAIN_CUTOFF,
    ) -> None:
        self.data_dir = data_dir
        self.store = ZeroBaseStore(data_dir)
        root = Path(__file__).resolve().parents[3]
        self.candidates_path = candidates_path or root / "research_bundle" / "seed" / "candidates.jsonl"
        self.forward_path = forward_klines_path or root / "research_bundle" / "forward" / "forward_klines_15m.jsonl"
        self.random_draws = random_draws
        self.top_k = top_k
        self.engines = engines or CANDIDATE_ENGINES
        self.eval_intervals = eval_intervals
        self.train_cutoff = train_cutoff

    def _metric_for(self, fwd: dict, scan_kst: str, symbol: str) -> dict | None:
        klines = fwd.get((scan_kst, symbol))
        if not klines:
            return None
        m = compute_forward_metrics(klines)
        if not m:
            return None
        m["scan_time_kst"] = scan_kst
        m["symbol"] = symbol
        return m

    @research_safe("zerobase")
    def run(self, max_scans: int | None = None, fresh_picks: bool = True) -> dict:
        print("[ZEROBASE] candidate scan started")
        if fresh_picks and self.store.picks_path.exists():
            self.store.picks_path.unlink()
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        scans = sorted(by_scan.keys())
        if max_scans:
            scans = scans[:max_scans]

        val_scans = [s for s in scans if _is_validation(s, self.train_cutoff)]
        train_scans = [s for s in scans if not _is_validation(s, self.train_cutoff)]

        samples_by_engine: dict[str, list[dict]] = defaultdict(list)
        random_rows: list[dict] = []
        all_random_samples: list[dict] = []
        picks_saved = 0

        for scan_kst in scans:
            rows = by_scan[scan_kst]
            split = "validation" if _is_validation(scan_kst, self.train_cutoff) else "train"
            if split != "validation":
                continue

            symbols = [r["symbol"] for r in rows]
            ctx = scan_context(rows)

            def metric_fn(sym: str) -> dict | None:
                return self._metric_for(fwd, scan_kst, sym)

            for interval in self.eval_intervals:
                if not _interval_match(scan_kst, interval):
                    continue

                rand_agg, draw_rows = random_baseline_for_scan(
                    symbols, self.top_k, self.random_draws, metric_fn, seed=hash(scan_kst) % 10000,
                )
                for dr in draw_rows:
                    random_rows.append({
                        "scan_time_kst": scan_kst,
                        "eval_interval": interval,
                        "split": split,
                        **dr,
                    })
                # Collect per-symbol metrics from random draws for fair aggregate
                from scout_auto_os.engine.research.zero_base.random_baseline import generate_random_draws
                for pick in generate_random_draws(symbols, self.top_k, self.random_draws, hash(scan_kst) % 10000):
                    for sym in pick:
                        m = metric_fn(sym)
                        if m:
                            all_random_samples.append(m)

                picks = rank_all_engines(rows, top_k=self.top_k)
                picks["RANDOM_BASELINE"] = rank_engine(rows, "RANDOM_BASELINE", self.top_k, ctx)

                pick_record = {
                    "scan_time_kst": scan_kst,
                    "eval_interval": interval,
                    "split": split,
                    "picks": picks,
                }
                self.store.append_picks(pick_record)
                picks_saved += 1

                for engine, syms in picks.items():
                    if engine not in self.engines:
                        continue
                    for sym in syms:
                        m = metric_fn(sym)
                        if m:
                            m = dict(m)
                            m["engine"] = engine
                            m["eval_interval"] = interval
                            m["split"] = split
                            samples_by_engine[engine].append(m)

        print("[ZEROBASE] forward evaluation updated")
        if random_rows:
            print("[ZEROBASE] random baseline generated")

        random_stats = aggregate_candidate_metrics(all_random_samples)

        candidate_results: list[dict] = []
        for engine in self.engines:
            if engine == "RANDOM_BASELINE":
                row = {
                    "engine": engine,
                    "eval_interval": "all",
                    "split": "validation",
                    "top_k": self.top_k,
                    **random_stats,
                }
                candidate_results.append(row)
                continue
            samples = samples_by_engine.get(engine, [])
            agg = aggregate_candidate_metrics(samples)
            row = {
                "engine": engine,
                "eval_interval": "all",
                "split": "validation",
                "top_k": self.top_k,
                **agg,
            }
            a6_agg = aggregate_candidate_metrics(samples_by_engine.get("A6_CURRENT", []))
            cmp = compare_vs_baseline(agg, random_stats, a6_agg)
            row.update(cmp)
            candidate_results.append(row)

        champion_board = build_champion_board(candidate_results, random_stats, samples_by_engine)
        champ_by_engine = {c["engine"]: c for c in champion_board}
        for row in candidate_results:
            extra = champ_by_engine.get(row.get("engine"), {})
            row.update({
                k: extra[k] for k in (
                    "champion_eligible", "tier", "positive_days",
                    "avg_return_2h_delta_vs_random", "avg_return_2h_delta_vs_a6",
                    "beats_random_return", "beats_a6_return",
                ) if k in extra
            })
        print("[ZEROBASE] champion board updated")

        feat_diag = feature_diagnostics(samples_by_engine)
        meta = {
            "train_cutoff": self.train_cutoff,
            "validation_scans": len(val_scans),
            "train_scans": len(train_scans),
            "random_draws": self.random_draws,
            "top_k": self.top_k,
            "engines": list(self.engines),
            "picks_saved": picks_saved,
            "generated_at": datetime.utcnow().isoformat(),
        }

        self.store.write_candidate_results(candidate_results)
        self.store.write_random_baseline(random_rows)
        self.store.write_champion_board(champion_board)
        self.store.write_feature_diagnostics(feat_diag)
        self.store.write_meta(meta)

        report_text = build_zero_base_report(
            candidate_results, champion_board, random_stats, feat_diag, meta,
        )
        self.store.write_report(report_text)
        print("[ZEROBASE] report generated")

        return {
            "meta": meta,
            "champion_board": champion_board[:10],
            "random_stats": random_stats,
            "a6": next((c for c in candidate_results if c.get("engine") == "A6_CURRENT"), {}),
            "better_than_a6": [c for c in champion_board if c.get("beats_a6_return")],
        }

    @staticmethod
    def aggregate_candidate_metrics(samples: list[dict]) -> dict:
        from scout_auto_os.engine.research.zero_base.ranking import aggregate_candidate_metrics as agg
        return agg(samples)
