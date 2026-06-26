"""Annotate candidates with phase20 states for search scoring."""

from __future__ import annotations

import scout_phase20_winner_state_ranking as p20
from scout_auto_os.engine.research.rule_discovery.generator import build_scan_rank_context


def annotate_universe(by_scan: dict[str, list[dict]]) -> tuple[dict, p20.Thresholds, dict]:
    winner_feats: list[dict] = []
    for rows in by_scan.values():
        if len(rows) < 4:
            continue
        ranked = sorted(rows, key=lambda x: -float(x.get("max_up_4h") or 0))
        for r in ranked[:3]:
            winner_feats.append(r["features"])

    th = p20.build_thresholds(winner_feats)
    annotated: dict[str, list[dict]] = {}
    for scan, rows in by_scan.items():
        annotated[scan] = p20.annotate(rows, th)

    ann_by_scan = annotated
    winners, _ = p20.winner_loser_sets(ann_by_scan)
    all_rows = [r for rows in annotated.values() for r in rows]
    profile = p20.build_profile(winners, all_rows)
    return annotated, th, profile


def attach_base_scores(annotated: dict[str, list[dict]], profile: dict) -> None:
    for rows in annotated.values():
        for r in rows:
            r["base_score"] = p20.state_match_score(
                r.get("states", {}),
                r.get("transitions", {}),
                profile,
            )


def label_winner_cohort(rows: list[dict], top_n: int = 3) -> None:
    ranked = sorted(rows, key=lambda r: -float(r.get("max_up_4h") or 0))
    winners = {r["symbol"] for r in ranked[:top_n]}
    for r in rows:
        r["cohort"] = "winner" if r["symbol"] in winners else "loser"
        r["outcome_rank"] = next(
            (i + 1 for i, x in enumerate(ranked) if x["symbol"] == r["symbol"]),
            99,
        )


def attach_scan_rank_context(annotated: dict[str, list[dict]], rank_feats: list[str]) -> None:
    flat = [
        {"scan_time_kst": r["scan_kst"], "symbol": r["symbol"], "features": r["features"]}
        for rows in annotated.values()
        for r in rows
    ]
    build_scan_rank_context(flat, rank_feats)
    ctx_map = {(x["scan_time_kst"], x["symbol"]): x.get("ctx", {}) for x in flat}
    for rows in annotated.values():
        for r in rows:
            r["ctx"] = ctx_map.get((r["scan_kst"], r["symbol"]), {"scan_ranks": {}})
