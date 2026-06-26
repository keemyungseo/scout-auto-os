"""Load blind top-K picks from frozen Long + Short constitutions."""

from __future__ import annotations

from pathlib import Path

from scout_auto_os.engine.portfolio.rule_loader import load_portfolio_rules
from scout_auto_os.engine.research.constitution_validation.long_calendar import load_constitution_dataset
from scout_auto_os.engine.research.constitution_validation.validator import train_frozen_constitution
from scout_auto_os.engine.research.directional.prediction.loader import load_formulas, resolve_formulas_path
from scout_auto_os.engine.research.ranking_engine.metrics import evaluate_strategy_on_blind
from scout_auto_os.engine.research.ranking_engine.models import predict_scores
from scout_auto_os.engine.research.short_constitution.candidate_generator import generate_short_label_candidates
from scout_auto_os.engine.research.short_constitution.dataset import (
    collect_short_dataset,
    prepare_annotated as prepare_short_annotated,
    split_by_scans,
)
from scout_auto_os.engine.research.short_constitution.label_builder import apply_short_label
from scout_auto_os.engine.research.short_constitution.constants import MODEL as SHORT_MODEL
from scout_auto_os.engine.research.short_execution.constants import (
    FROZEN_LONG_LABEL,
    FROZEN_SHORT_LABEL,
    LONG_PICK_TOP,
    PICK_TOP,
    TRAIN_RATIO,
)
from scout_auto_os.engine.research.short_constitution.evaluation import train_short_ranker
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines


def _short_spec():
    return next(s for s in generate_short_label_candidates() if s.label_id == FROZEN_SHORT_LABEL)


def load_frozen_blind_picks(
    candidates_path: Path,
    forward_path: Path,
    data_dir: Path,
    pkg_root: Path,
) -> tuple[list[dict], list[dict], dict]:
    """Return (short_blind_picks, long_blind_picks, meta). Each pick has klines attached."""
    by_scan = load_candidates_jsonl(candidates_path)
    fwd = load_forward_klines(forward_path)
    rules = load_portfolio_rules(data_dir, pkg_root)
    formulas = load_formulas(resolve_formulas_path(data_dir, pkg_root))

    annotated, th, stats = prepare_short_annotated(by_scan)
    short_dataset = collect_short_dataset(annotated, fwd, rules, formulas, th, stats)
    short_train, short_blind_rows = split_by_scans(short_dataset, TRAIN_RATIO)
    feat_names = list(next(iter(short_train))["x"].keys()) if short_train else []

    spec = _short_spec()
    short_bundle = train_short_ranker(short_train, feat_names, spec)
    labeled_blind = apply_short_label(short_blind_rows, spec)

    def short_score(row, peers, b=short_bundle):
        return float(predict_scores(b, [row])[0])

    short_picks, short_metrics, _ = evaluate_strategy_on_blind(labeled_blind, short_score, top_k=PICK_TOP)

    long_dataset, calendar, long_feat = load_constitution_dataset(
        candidates_path, forward_path, data_dir, pkg_root,
    )
    long_scans = sorted({r["scan_kst"] for r in long_dataset})
    split_i = int(len(long_scans) * TRAIN_RATIO)
    train_set = set(long_scans[:split_i])
    long_train = [r for r in long_dataset if r["scan_kst"] in train_set]
    long_blind_rows = [r for r in long_dataset if r["scan_kst"] not in train_set]

    long_bundle = train_frozen_constitution(long_train, long_feat)

    def long_score(row, peers, b=long_bundle):
        return float(predict_scores(b, [row])[0])

    long_picks, long_metrics, _ = evaluate_strategy_on_blind(long_blind_rows, long_score, top_k=LONG_PICK_TOP)

    for p in short_picks:
        p["direction"] = "short"
        p["klines"] = fwd.get((p["scan_kst"], p["symbol"]), [])
        p["return_field"] = "short_return_2h"

    for p in long_picks:
        p["direction"] = "long"
        p["klines"] = fwd.get((p["scan_kst"], p["symbol"]), [])
        p["return_field"] = "return_2h"

    meta = {
        "short_blind_scans": len({p["scan_kst"] for p in short_picks}),
        "long_blind_scans": len({p["scan_kst"] for p in long_picks}),
        "short_pick_count": len(short_picks),
        "long_pick_count": len(long_picks),
        "short_metrics": short_metrics,
        "long_metrics": long_metrics,
        "calendar_days": calendar.get("calendar_days"),
        "frozen_short_label": FROZEN_SHORT_LABEL,
        "frozen_long_label": FROZEN_LONG_LABEL,
    }
    return short_picks, long_picks, meta
