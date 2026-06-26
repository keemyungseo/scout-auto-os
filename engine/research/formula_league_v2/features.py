"""Search-time feature pool from candidates.jsonl."""

from __future__ import annotations

from scout_auto_os.engine.research.formula_league_v2.constants import SEARCH_FEATURE_ALIASES


def numeric_feature_keys(features: dict) -> list[str]:
    return sorted(k for k, v in features.items() if isinstance(v, (int, float)))


def build_feature_pool(sample_features: dict) -> list[str]:
    keys = set(numeric_feature_keys(sample_features))
    keys.update(SEARCH_FEATURE_ALIASES.values())
    keys.update(SEARCH_FEATURE_ALIASES.keys())
    priority = [k for k in keys if any(
        t in k for t in ("body", "range", "return", "volume", "momentum", "compression", "ma20", "seq")
    )]
    rest = sorted(keys - set(priority))
    return list(dict.fromkeys(priority + rest))


def enrich_derived_features(row: dict) -> None:
    """Add derived search features in-place (scan-time computable)."""
    f = row.setdefault("features", {})
    body = float(f.get("1h_current_body_pct", 0))
    rng = float(f.get("1h_current_range_pct", 0) or 1e-9)
    f.setdefault("derived_body_share", round(body / max(rng, 1e-9), 6))
    f.setdefault("derived_upper_wick", float(f.get("1h_current_close_position", 0.5)))
    f.setdefault("derived_lower_wick", round(1.0 - float(f.get("1h_current_close_position", 0.5)), 6))
    f.setdefault("derived_atr_proxy", float(f.get("1h_current_range_pct", 0)))
    f.setdefault("derived_atr_expansion", float(f.get("5m_release", 0)))
    f.setdefault("derived_atr_compression", float(f.get("5m_compression", 0)))
    f.setdefault("derived_volume_ratio", float(f.get("15m_current_volume_ratio", 0)))
    f.setdefault("derived_vwap_distance", float(f.get("1h_current_ma20_distance_pct", 0)))
    f.setdefault("derived_momentum", float(f.get("5m_momentum", 0)))
    f.setdefault("derived_range_expansion", float(f.get("1h_current_range_pct", 0)) - float(f.get("1h_previous_range_pct", 0)))
    f.setdefault("derived_breakout_flag", 1.0 if float(f.get("5m_release", 0)) > 0 else 0.0)
    f.setdefault("derived_false_breakout", float(f.get("5m_seq_return_sum_6", 0)) if float(f.get("5m_release", 0)) > 0 else 0.0)
    f.setdefault("derived_clv", float(f.get("1h_current_close_position", 0.5)))
    f.setdefault("derived_obv_proxy", float(f.get("5m_seq_volume_energy_6", 0)))
    f.setdefault("derived_mtf_alignment", (
        float(f.get("15m_current_return_pct", 0) > 0)
        + float(f.get("1h_current_return_pct", 0) > 0)
        + float(f.get("2h_current_return_pct", 0) > 0)
    ) / 3.0)
