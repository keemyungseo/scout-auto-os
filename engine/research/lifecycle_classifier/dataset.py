"""Build labeled dataset: entry features + post-hoc lifecycle labels."""

from __future__ import annotations

from scout_auto_os.engine.portfolio.rule_loader import PortfolioRules
from scout_auto_os.engine.research.directional.dna.collector import numeric_feature_keys
from scout_auto_os.engine.research.directional.dna.formulas import ClusterFormula
from scout_auto_os.engine.research.directional.dna.validator import split_scans
from scout_auto_os.engine.research.directional.engines import rank_long, rank_short
from scout_auto_os.engine.research.directional.entry_filter.constants import (
    CHAMPION_TOP_K,
    LONG_DIRECTION_CHAMPION,
    SHORT_DIRECTION_CHAMPION,
)
from scout_auto_os.engine.research.lifecycle_classifier.constants import MIN_CLASS_SAMPLES, TRAIN_RATIO
from scout_auto_os.engine.research.lifecycle_classifier.features import build_entry_feature_row
from scout_auto_os.engine.research.signal_lifecycle.shape_classifier import classify_lifecycle_shape
from scout_auto_os.engine.research.signal_lifecycle.timeline import build_signal_timeline


def _signal_id(direction: str, scan_kst: str, symbol: str) -> str:
    safe_scan = scan_kst.replace(" ", "T").replace(":", "")
    return f"{direction}_{safe_scan}_{symbol}"


def lifecycle_label_for_signal(
    klines: list | None,
    direction: str,
    scan_kst: str,
    symbol: str,
) -> tuple[str, dict] | None:
    """Post-hoc label only — never passed as model input."""
    if not klines:
        return None
    sid = _signal_id(direction, scan_kst, symbol)
    _, summary = build_signal_timeline(klines, direction, scan_kst, symbol, sid)
    if not summary:
        return None
    label = classify_lifecycle_shape(summary)
    return label, summary


def collect_lifecycle_dataset(
    by_scan: dict[str, list[dict]],
    fwd: dict,
    scans: list[str],
    rules: PortfolioRules,
    formulas: list[ClusterFormula],
    top_k: int = CHAMPION_TOP_K,
) -> list[dict]:
    latest_scan = scans[-1] if scans else ""
    sample_feats = by_scan[scans[0]][0]["features"] if scans and by_scan.get(scans[0]) else {}
    feature_keys = numeric_feature_keys(sample_feats)

    records: list[dict] = []
    for scan_kst in scans:
        rows = by_scan.get(scan_kst, [])
        if not rows:
            continue
        for direction, engine in (
            ("long", LONG_DIRECTION_CHAMPION),
            ("short", SHORT_DIRECTION_CHAMPION),
        ):
            rank_fn = rank_long if direction == "long" else rank_short
            for sym in rank_fn(rows, engine, top_k):
                row = next(r for r in rows if r["symbol"] == sym)
                labeled = lifecycle_label_for_signal(fwd.get((scan_kst, sym)), direction, scan_kst, sym)
                if not labeled:
                    continue
                label, summary = labeled
                x = build_entry_feature_row(
                    row, direction, engine, rules, formulas, rows, scan_kst, latest_scan, feature_keys,
                )
                sid = _signal_id(direction, scan_kst, sym)
                records.append(
                    {
                        "signal_id": sid,
                        "direction": direction,
                        "engine": engine,
                        "scan_time_kst": scan_kst,
                        "symbol": sym,
                        "lifecycle_label": label,
                        "x": x,
                        "summary": summary,
                    },
                )
    return records


def split_dataset(records: list[dict], train_ratio: float = TRAIN_RATIO) -> tuple[list[dict], list[dict]]:
    scans = sorted({r["scan_time_kst"] for r in records})
    train_scans, val_scans = split_scans(scans, train_ratio)
    train_set = set(train_scans)
    train = [r for r in records if r["scan_time_kst"] in train_set]
    val = [r for r in records if r["scan_time_kst"] not in train_set]
    return train, val


def active_class_names(records: list[dict], min_samples: int = MIN_CLASS_SAMPLES) -> list[str]:
    counts: dict[str, int] = {}
    for r in records:
        lab = r["lifecycle_label"]
        counts[lab] = counts.get(lab, 0) + 1
    return sorted(lab for lab, n in counts.items() if n >= min_samples)
