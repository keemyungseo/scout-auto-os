"""Rolling, walk-forward, expanding and sliding window validation."""

from __future__ import annotations

from collections import defaultdict

from scout_auto_os.engine.research.constitution_validation.validator import (
    evaluate_constitution,
    metrics_from_picks,
    train_frozen_constitution,
)
from scout_auto_os.engine.research.ranking_engine.models import RankingModelBundle, predict_scores


def _week_key(s: str) -> str:
    return s[:10]


def _month_key(s: str) -> str:
    return s[:7]


def weekly_validation(
    rows: list[dict],
    feat_names: list[str],
    bundle: RankingModelBundle | None = None,
) -> list[dict]:
    out: list[dict] = []
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[_week_key(r["scan_kst"])].append(r)
    for key, chunk in sorted(buckets.items()):
        if len(chunk) < 15:
            continue
        if bundle:
            _, m, _ = evaluate_constitution(chunk, bundle, split_name=f"weekly_{key}")
        else:
            m = {"split": f"weekly_{key}"}
        m["fold_id"] = key
        m["validation_type"] = "weekly"
        out.append(m)
    return out


def monthly_validation(
    rows: list[dict],
    feat_names: list[str],
    bundle: RankingModelBundle | None = None,
) -> list[dict]:
    out: list[dict] = []
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[_month_key(r["scan_kst"])].append(r)
    for key, chunk in sorted(buckets.items()):
        if len(chunk) < 20:
            continue
        if bundle:
            _, m, _ = evaluate_constitution(chunk, bundle, split_name=f"monthly_{key}")
        else:
            m = {"split": f"monthly_{key}"}
        m["fold_id"] = key
        m["validation_type"] = "monthly"
        out.append(m)
    return out


def walk_forward_validation(
    rows: list[dict],
    feat_names: list[str],
    train_ratio: float = 0.7,
) -> list[dict]:
    scans = sorted({r["scan_kst"] for r in rows})
    cut = max(1, int(len(scans) * train_ratio))
    train_cut = scans[cut - 1]
    train = [r for r in rows if r["scan_kst"] <= train_cut]
    holdout = [r for r in rows if r["scan_kst"] > train_cut]
    if not holdout:
        return []
    bundle = train_frozen_constitution(train, feat_names)
    _, m, _ = evaluate_constitution(holdout, bundle, split_name="walk_forward")
    m["validation_type"] = "walk_forward"
    m["fold_id"] = f"after_{train_cut[:10]}"
    return [m]


def expanding_window_validation(
    rows: list[dict],
    feat_names: list[str],
    min_train_scans: int = 40,
    step_scans: int = 10,
) -> list[dict]:
    scans = sorted({r["scan_kst"] for r in rows})
    out: list[dict] = []
    for i in range(min_train_scans, len(scans) - 5, step_scans):
        train_scans = set(scans[:i])
        test_scans = set(scans[i:i + step_scans])
        if not test_scans:
            break
        train = [r for r in rows if r["scan_kst"] in train_scans]
        test = [r for r in rows if r["scan_kst"] in test_scans]
        if len(test) < 15:
            continue
        bundle = train_frozen_constitution(train, feat_names)
        _, m, _ = evaluate_constitution(test, bundle, split_name=f"expanding_{i}")
        m["validation_type"] = "expanding_window"
        m["fold_id"] = f"train_{i}_test_{i+step_scans}"
        m["train_scans"] = i
        m["test_scans"] = len(test_scans)
        out.append(m)
    return out


def sliding_window_validation(
    rows: list[dict],
    feat_names: list[str],
    train_scans: int = 60,
    test_scans: int = 10,
) -> list[dict]:
    scans = sorted({r["scan_kst"] for r in rows})
    out: list[dict] = []
    for i in range(0, len(scans) - train_scans - test_scans + 1, test_scans):
        train_set = set(scans[i:i + train_scans])
        test_set = set(scans[i + train_scans:i + train_scans + test_scans])
        train = [r for r in rows if r["scan_kst"] in train_set]
        test = [r for r in rows if r["scan_kst"] in test_set]
        if len(test) < 15:
            continue
        bundle = train_frozen_constitution(train, feat_names)
        _, m, _ = evaluate_constitution(test, bundle, split_name=f"sliding_{i}")
        m["validation_type"] = "sliding_window"
        m["fold_id"] = f"win_{i}"
        out.append(m)
    return out


def all_rolling_validations(
    rows: list[dict],
    blind_rows: list[dict],
    feat_names: list[str],
    train_ratio: float,
    frozen_bundle: RankingModelBundle,
) -> list[dict]:
    out: list[dict] = []
    out.extend(weekly_validation(blind_rows, feat_names, frozen_bundle))
    out.extend(monthly_validation(blind_rows, feat_names, frozen_bundle))
    out.extend(walk_forward_validation(rows, feat_names, train_ratio))
    out.extend(expanding_window_validation(rows, feat_names))
    out.extend(sliding_window_validation(rows, feat_names))
    return out
