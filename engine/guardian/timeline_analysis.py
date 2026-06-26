"""Aggregate statistics across Guardian timelines."""

from __future__ import annotations

from collections import Counter, defaultdict

from scout_auto_os.engine.guardian.timeline_engine import TradeTimeline

EXIT_RECS = frozenset({"EXIT", "EMERGENCY_EXIT"})
TRAIL_RECS = frozenset({"TRAIL"})


def _first_rec_minute(points: list[dict], recs: frozenset[str]) -> int | None:
    for p in points:
        if p.get("recommendation") in recs:
            return int(p.get("elapsed_minutes", 0))
    return None


def analyze_timelines(timelines: list[TradeTimeline]) -> dict:
    transition_counts: Counter[str] = Counter()
    transition_rois: dict[str, list[float]] = defaultdict(list)
    hold_until_exit: list[int] = []
    trail_starts: list[int] = []
    exit_starts: list[int] = []
    rec_changes: list[int] = []
    state_freq: Counter[str] = Counter()

    for tl in timelines:
        rec_changes.append(tl.recommendation_changes)
        for p in tl.points:
            state_freq[p.get("guardian_state", "")] += 1
        for tr in tl.transitions:
            key = f"{tr.from_state} → {tr.to_state}"
            transition_counts[key] += 1
            transition_rois[key].append(tr.current_roi)

        exit_m = _first_rec_minute(tl.points, EXIT_RECS)
        if exit_m is not None:
            hold_until_exit.append(exit_m)
            exit_starts.append(exit_m)
        trail_m = _first_rec_minute(tl.points, TRAIL_RECS)
        if trail_m is not None:
            trail_starts.append(trail_m)

    trans_stats = []
    for key, count in transition_counts.most_common():
        rois = transition_rois[key]
        trans_stats.append({
            "transition": key,
            "count": count,
            "avg_roi_at_transition": round(sum(rois) / len(rois), 4) if rois else 0.0,
        })

    def _avg(vals: list[int]) -> float:
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    return {
        "trade_count": len(timelines),
        "total_timeline_points": sum(len(tl.points) for tl in timelines),
        "avg_hold_minutes_until_exit_rec": _avg(hold_until_exit),
        "avg_trail_start_minutes": _avg(trail_starts),
        "avg_exit_rec_minutes": _avg(exit_starts),
        "avg_recommendation_changes": round(sum(rec_changes) / len(rec_changes), 2) if rec_changes else 0,
        "state_point_frequency": dict(state_freq),
        "transition_statistics": trans_stats,
        "transition_frequency": dict(transition_counts),
    }
