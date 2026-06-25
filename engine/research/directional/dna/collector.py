"""Collect pattern-engine samples with scan-time features and forward outcomes."""

from __future__ import annotations

from collections import defaultdict

from scout_auto_os.engine.research.directional.dna.constants import RESEARCH_ENGINES
from scout_auto_os.engine.research.directional.engines import rank_long, rank_short
from scout_auto_os.engine.research.directional.evaluation import to_long_metrics, to_short_metrics
from scout_auto_os.engine.research.directional.patterns import label_direction_pattern
from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines


def _is_success(metrics: dict, direction: str) -> bool:
    r2h = float(metrics.get("return_2h", 0))
    if direction == "short":
        return r2h <= -3.0
    return r2h >= 3.0


def numeric_feature_keys(features: dict) -> list[str]:
    keys: list[str] = []
    for k, v in features.items():
        if k == "price":
            continue
        try:
            float(v)
            keys.append(k)
        except (TypeError, ValueError):
            pass
    return sorted(keys)


def collect_samples(
    by_scan: dict[str, list[dict]],
    fwd: dict,
    val_scans: list[str],
) -> dict[str, list[dict]]:
    """Per engine: list of {features, metrics, success, scan_time_kst, symbol, pattern_label}."""
    by_engine: dict[str, list[dict]] = defaultdict(list)

    for scan_kst in val_scans:
        rows = by_scan[scan_kst]
        for r in rows:
            r["direction_pattern"] = label_direction_pattern(r["features"])

        def metric_fn(sym: str) -> dict | None:
            klines = fwd.get((scan_kst, sym))
            if not klines:
                return None
            return compute_forward_metrics(klines)

        engine_picks: dict[str, list[str]] = {}
        for eng in RESEARCH_ENGINES:
            if eng.startswith("LONG"):
                engine_picks[eng] = rank_long(rows, eng, top_k=5)
            else:
                engine_picks[eng] = rank_short(rows, eng, top_k=5)

        for eng, syms in engine_picks.items():
            direction = "long" if eng.startswith("LONG") else "short"
            for sym in syms:
                raw = metric_fn(sym)
                if not raw:
                    continue
                m = to_long_metrics(raw) if direction == "long" else to_short_metrics(raw)
                row = next(r for r in rows if r["symbol"] == sym)
                by_engine[eng].append({
                    "engine": eng,
                    "direction": direction,
                    "scan_time_kst": scan_kst,
                    "symbol": sym,
                    "pattern_label": row["direction_pattern"],
                    "features": dict(row["features"]),
                    "metrics": m,
                    "success": _is_success(m, direction),
                    "return_2h": float(m.get("return_2h", 0)),
                })
    return dict(by_engine)
