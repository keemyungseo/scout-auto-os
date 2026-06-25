"""Feature league — single-feature and state condition performance."""

from __future__ import annotations

import statistics

FEATURE_SPECS: list[tuple[str, str, callable]] = []  # built dynamically in _build_specs


def _fval(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def _bool(row: dict, key: str) -> bool:
    return str(row.get(key, "")).lower() in ("1", "true", "yes")


def _build_specs() -> list[tuple[str, str, callable]]:
    return [
        ("volume_ratio", ">=1.5", lambda r: _fval(r, "volume_ratio") >= 1.5),
        ("volume_ratio", ">=2.0", lambda r: _fval(r, "volume_ratio") >= 2.0),
        ("momentum_15m", ">0", lambda r: _fval(r, "momentum_15m") > 0),
        ("momentum_1h", ">0", lambda r: _fval(r, "momentum_1h") > 0),
        ("atr_pct", ">=2.0", lambda r: _fval(r, "atr_pct") >= 2.0),
        ("compression_score", "<=5", lambda r: _fval(r, "compression_score") <= 5),
        ("breakout_score", ">=1.0", lambda r: _fval(r, "breakout_score") >= 1.0),
        ("range_pct", ">=2.0", lambda r: _fval(r, "range_pct") >= 2.0),
        ("btc_context", "bullish", lambda r: str(r.get("btc_context", "")).startswith("bull")),
        ("reason_1h", "Acceleration", lambda r: "Acceleration" in str(r.get("reason_1h", ""))),
        ("reason_2h", "TrendAlive", lambda r: "TrendAlive" in str(r.get("reason_2h", ""))),
        ("a6_score", ">=5", lambda r: _fval(r, "a6_score") >= 5),
        ("expected_ev", ">=3", lambda r: _fval(r, "expected_ev") >= 3),
    ]


def compute_feature_league(candidate_rows: list[dict], forward_rows: list[dict]) -> list[dict]:
    fwd = {(r["scan_time_kst"], r["symbol"]): r for r in forward_rows if r.get("return_2h") not in ("", None)}
    merged: list[dict] = []
    for c in candidate_rows:
        key = (c.get("scan_time_kst"), c.get("symbol"))
        f = fwd.get(key)
        if not f:
            continue
        merged.append({**c, **f})

    out: list[dict] = []
    for name, condition, pred in _build_specs():
        subset = [r for r in merged if pred(r)]
        if len(subset) < 3:
            out.append({
                "feature_name": name,
                "condition": condition,
                "sample_count": len(subset),
                "win_rate_2h": 0,
                "avg_return_2h": 0,
                "median_return_2h": 0,
                "big_winner_rate": 0,
                "trap_rate": 0,
                "comment": "insufficient sample",
            })
            continue
        r2h = [_fval(s, "return_2h") for s in subset]
        big = sum(1 for s in subset if _bool(s, "label_big_winner"))
        trap = sum(1 for s in subset if _bool(s, "label_trap"))
        win = sum(1 for x in r2h if x >= 3.0)
        comment = "hypothesis"
        if win / len(subset) >= 0.5:
            comment = "verification_candidate"
        if trap / len(subset) >= 0.4:
            comment = "apply_forbidden_trap_risk"
        out.append({
            "feature_name": name,
            "condition": condition,
            "sample_count": len(subset),
            "win_rate_2h": round(win / len(subset) * 100, 2),
            "avg_return_2h": round(statistics.mean(r2h), 4),
            "median_return_2h": round(statistics.median(r2h), 4),
            "big_winner_rate": round(big / len(subset) * 100, 2),
            "trap_rate": round(trap / len(subset) * 100, 2),
            "comment": comment,
        })
    out.sort(key=lambda x: (x["win_rate_2h"], x["avg_return_2h"]), reverse=True)
    return out
