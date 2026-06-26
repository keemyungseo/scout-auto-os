"""Execution-layer feature vector (search + first observation bar only)."""

from __future__ import annotations

from scout_auto_os.engine.research.execution_research.observation import compute_observation_features


def _bar_body_range(klines: list) -> tuple[float, float]:
    if not klines:
        return 0.0, 0.0
    bar = klines[0]
    o, h, l, c = float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4])
    mid = (o + c) / 2 or 1.0
    body = abs(c - o) / mid * 100
    rng = (h - l) / (o or 1.0) * 100
    return round(body, 4), round(rng, 4)


def build_candidate_record(
    candidate: dict,
    direction: str,
    scan_kst: str,
    klines: list,
    return_2h: float,
    top5_rank: int,
    top5_size: int,
    gap_to_best: float,
    rank_obs: float,
    rank_entry: float,
) -> dict:
    obs = compute_observation_features(klines, direction, candidate.get("features") or {}) or {}
    body, rng = _bar_body_range(klines)

    feat = {
        "entry_score": float(candidate.get("entry_score", 0)),
        "direction_confidence": float(candidate.get("direction_confidence", 0)),
        "pattern_confidence": float(candidate.get("pattern_confidence", 0)),
        "rule_confidence": float(candidate.get("rule_confidence", 0)),
        "feature_match_ratio": float(candidate.get("feature_match_ratio", 0)),
        "rule_margin": float(candidate.get("rule_margin", 0)),
        "recency": float(candidate.get("recency", 0)),
        "signal_freshness": float(candidate.get("signal_freshness", 0)),
        "top5_rank": float(top5_rank),
        "top5_rank_pct": round(1.0 - (top5_rank - 1) / max(top5_size - 1, 1), 4),
        "gap_to_best_entry": round(gap_to_best, 4),
        "obs_body_pct": body,
        "obs_range_pct": rng,
        "body_range_ratio": round(body / max(rng, 0.01), 4),
        "rank_obs_return_top5": round(rank_obs, 4),
        "rank_entry_score_top5": round(rank_entry, 4),
        **{k: float(obs[k]) for k in obs if isinstance(obs[k], (int, float))},
    }

    return {
        "scan_time_kst": scan_kst,
        "direction": direction,
        "symbol": candidate["symbol"],
        "live_pattern": candidate.get("live_pattern", ""),
        "return_2h": return_2h,
        "features": feat,
        "ctx": {"_symbol": candidate["symbol"], "scan_ranks": {}},
    }


def enrich_top5_ranks(records: list[dict]) -> None:
    """Relative rank inside Top5 cohort (scan-time + obs only)."""
    if not records:
        return
    for key in ("obs_return_pct", "entry_score", "execution_score"):
        ranked = sorted(records, key=lambda r: float(r["features"].get(key, 0)), reverse=True)
        n = len(ranked)
        rank_map: dict[str, float] = {}
        for i, r in enumerate(ranked):
            rank_map[r["symbol"]] = 1.0 - i / max(n - 1, 1)
        for r in records:
            r["ctx"]["scan_ranks"].setdefault(key, {})[r["symbol"]] = rank_map.get(r["symbol"], 0)
            if key == "obs_return_pct":
                r["features"]["rank_obs_return_top5"] = rank_map[r["symbol"]]
            if key == "entry_score":
                r["features"]["rank_entry_score_top5"] = rank_map[r["symbol"]]
