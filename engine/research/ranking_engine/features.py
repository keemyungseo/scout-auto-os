"""Ranking Engine V1 — feature collection."""

from __future__ import annotations

import scout_phase22_search_formula_evolution as p22
import scout_phase23_search_formula_league as p23

from scout_auto_os.engine.portfolio.rule_loader import PortfolioRules
from scout_auto_os.engine.research.directional.dna.collector import numeric_feature_keys
from scout_auto_os.engine.research.directional.dna.formulas import ClusterFormula
from scout_auto_os.engine.research.execution_research.observation import compute_observation_features
from scout_auto_os.engine.research.formula_league_v2.features import enrich_derived_features
from scout_auto_os.engine.research.lifecycle_classifier.features import (
    _cluster_features,
    _portfolio_scan_features,
)
from scout_auto_os.engine.research.zero_base.validation import classify_regime


def _scan_vol_band(rows: list[dict]) -> str:
    import statistics
    ranges = [float(r["features"].get("1h_current_range_pct", 0)) for r in rows]
    if not ranges:
        return "unknown"
    med = statistics.median(ranges)
    if med >= 20.0:
        return "high_volatility"
    if med <= 10.0:
        return "low_volatility"
    return "mid_volatility"


def _one_hot(prefix: str, value: str, choices: tuple[str, ...]) -> dict[str, float]:
    return {f"{prefix}_{c}": 1.0 if value == c else 0.0 for c in choices}


def build_ranking_feature_row(
    row: dict,
    peers: list[dict],
    rules: PortfolioRules,
    formulas: list[ClusterFormula],
    scan_time_kst: str,
    latest_scan: str,
    th,
    stats: dict,
    direction: str = "long",
) -> dict[str, float]:
    """Scan-time features only (+ first observation bar as execution_obs_*)."""
    enrich_derived_features(row)
    features = row.get("features") or {}
    keys = numeric_feature_keys(features)
    vec: dict[str, float] = {f"dna_{k}": float(features.get(k, 0)) for k in keys}
    for k, v in features.items():
        if k.startswith("derived_"):
            vec[f"meta_{k}"] = float(v)

    vec.update(_portfolio_scan_features(row, direction, rules, peers, scan_time_kst, latest_scan))
    vec.update(_cluster_features(features, "A6", direction, formulas))

    base = float(row.get("base_score", 0))
    vec["phase20_base_score"] = base
    for tf in ("5m", "15m", "30m", "1h", "2h"):
        st = row.get("states", {}).get(tf, "unknown")
        vec[f"state_{tf}_{st}"] = 1.0

    vec["a6_formula_score"] = p23.formula_scores_a6(row, peers, base, th, stats)["A6"]
    vec["formula_v2_rank_pct"] = p22.within_scan_pct(row, peers, "1h_current_return_pct")
    vec["search_rank_pct_return"] = p22.within_scan_pct(row, peers, "1h_current_return_pct")
    vec["search_rank_pct_range"] = p22.within_scan_pct(row, peers, "1h_current_range_pct")
    vec["search_rank_pct_volume"] = p22.within_scan_pct(row, peers, "15m_current_volume_ratio")
    vec["search_rank_pct_body"] = p22.within_scan_pct(row, peers, "1h_current_body_pct")

    ctx = row.get("ctx", {}).get("scan_ranks", {})
    sym = row["symbol"]
    for feat, ranks in ctx.items():
        vec[f"ctx_rank_{feat}"] = float(ranks.get(sym, 0.5))

    regime = classify_regime(peers)
    vol = _scan_vol_band(peers)
    vec.update(_one_hot("regime", regime, ("bull", "bear", "sideway", "recovery", "crash", "unknown")))
    vec.update(_one_hot("vol_band", vol, ("high_volatility", "mid_volatility", "low_volatility", "unknown")))

    obs = row.get("obs_features") or {}
    for k, v in obs.items():
        vec[f"exec_obs_{k}"] = float(v)

    vec["direction_long"] = 1.0 if direction == "long" else 0.0
    return vec


def feature_matrix(rows: list[dict], key: str = "x") -> tuple[list[str], list[list[float]]]:
    if not rows:
        return [], []
    names = sorted(rows[0][key].keys())
    mat = [[float(r[key].get(n, 0.0)) for n in names] for r in rows]
    return names, mat
