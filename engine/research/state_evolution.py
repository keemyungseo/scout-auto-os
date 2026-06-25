"""State Evolution analysis — component contribution & improvement proposals."""

from __future__ import annotations

import statistics
from collections import defaultdict


def _fval(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def _success(row: dict) -> bool:
    pnl = _fval(row, "realized_pnl_pct")
    if pnl != 0:
        return pnl >= 3.0
    # replay: use return at 2h checkpoint proxy
    return _fval(row, "alive_delta") > 0 and _fval(row, "alive_score") >= 70


def analyze_evolution(rows: list[dict], max_samples: int = 500) -> dict:
    data = rows[-max_samples:] if rows else []
    if len(data) < 10:
        return {
            "sample_count": len(data),
            "status": "insufficient_sample",
            "proposals": [],
            "component_contribution": [],
            "high_score_failures": [],
            "low_score_successes": [],
        }

    components = ("trend_alive", "momentum_alive", "volume_alive", "expansion_alive")
    contrib: list[dict] = []
    for comp in components:
        success_vals = [_fval(r, comp) for r in data if _success(r)]
        fail_vals = [_fval(r, comp) for r in data if not _success(r)]
        if not success_vals or not fail_vals:
            continue
        delta = statistics.mean(success_vals) - statistics.mean(fail_vals)
        contrib.append({
            "component": comp,
            "success_mean": round(statistics.mean(success_vals), 2),
            "fail_mean": round(statistics.mean(fail_vals), 2),
            "delta": round(delta, 2),
            "correlation_proxy": round(delta / (statistics.pstdev(success_vals + fail_vals) or 1), 3),
        })
    contrib.sort(key=lambda x: abs(x["correlation_proxy"]), reverse=True)

    high_fail = [
        r for r in data
        if _fval(r, "alive_score") >= 70 and not _success(r)
    ][:10]
    low_win = [
        r for r in data
        if _fval(r, "alive_score") < 45 and _success(r)
    ][:10]

    proposals = _build_proposals(contrib, len(data), high_fail, low_win)

    return {
        "sample_count": len(data),
        "status": "ok",
        "component_contribution": contrib[:4],
        "high_score_failures": [
            {"symbol": r.get("symbol"), "alive": r.get("alive_score"), "pnl": r.get("realized_pnl_pct")}
            for r in high_fail
        ],
        "low_score_successes": [
            {"symbol": r.get("symbol"), "alive": r.get("alive_score"), "pnl": r.get("realized_pnl_pct")}
            for r in low_win
        ],
        "proposals": proposals,
        "recent_100_win_rate": _recent_win_rate(data, 100),
    }


def _recent_win_rate(data: list[dict], n: int) -> float:
    chunk = data[-n:]
    if not chunk:
        return 0.0
    return round(sum(1 for r in chunk if _success(r)) / len(chunk) * 100, 2)


def _build_proposals(contrib, n, high_fail, low_win) -> list[dict]:
    proposals: list[dict] = []
    if n < 100:
        proposals.append({
            "tier": "hypothesis",
            "title": "Insufficient sample for LIVE change",
            "detail": f"Only {n} evolution records — need 100+ before State Candidate review",
            "action": "continue_collection",
        })
        return proposals

    if contrib:
        best = contrib[0]
        proposals.append({
            "tier": "verification_needed",
            "title": f"Component emphasis: {best['component']}",
            "detail": f"Success-fail delta={best['delta']:+.1f} on {n} samples — test in State League only",
            "action": "state_league_grid_search",
        })

    if len(high_fail) >= 5:
        proposals.append({
            "tier": "hypothesis",
            "title": "High Alive Score failures detected",
            "detail": f"{len(high_fail)} cases score>=70 but failed — review exhaustion weight in League",
            "action": "analyze_exhaustion_formula",
        })

    if len(low_win) >= 3:
        proposals.append({
            "tier": "hypothesis",
            "title": "Low score successes exist",
            "detail": f"{len(low_win)} cases score<45 but succeeded — momentum/volume formula variant?",
            "action": "state_formula_B_C_test",
        })

    top_comp = contrib[0]["component"] if contrib else "trend_alive"
    proposals.append({
        "tier": "state_candidate",
        "title": "Promotion gate reminder",
        "detail": (
            "LIVE unchanged until: blind pass + league rank top3 + n>=100 + user approval. "
            f"Current lead component: {top_comp}"
        ),
        "action": "no_live_auto_apply",
    })
    return proposals
