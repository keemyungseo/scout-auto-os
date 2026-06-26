"""Rank and select discovered rules by LIVE objective."""

from __future__ import annotations

from scout_auto_os.engine.research.rule_discovery.constants import MIN_BLIND_PASS, PRECISION_TOLERANCE, TOP_OUTPUT


def rank_candidates(
    blind_metrics: list[dict],
    v2_precision: float,
) -> list[dict]:
    floor = v2_precision - PRECISION_TOLERANCE
    eligible = [
        m for m in blind_metrics
        if m.get("pass_count", 0) >= MIN_BLIND_PASS and float(m.get("precision", 0)) >= floor
    ]
    pool = eligible if eligible else blind_metrics

    def _sort_key(m: dict) -> tuple:
        return (
            float(m.get("pass_per_day", 0)),
            float(m.get("avg_return_2h", 0)),
            float(m.get("precision", 0)),
            float(m.get("lift", 0)),
        )

    ranked = sorted(pool, key=_sort_key, reverse=True)
    for i, row in enumerate(ranked[:TOP_OUTPUT], start=1):
        row["rank"] = i
    return ranked[:TOP_OUTPUT]


def decide_adoption(
    top_long: list[dict],
    top_short: list[dict],
    v2_long: dict,
    v2_short: dict,
    hybrid_rows: list[dict],
) -> dict:
    def _one(
        direction: str,
        top: list[dict],
        v2_row: dict,
    ) -> dict:
        if not top:
            return {
                "decision": "Reject",
                "reason": f"No {direction} candidate met precision floor on blind validation",
                "recommended_rule_id": None,
                "recommended_rule_expr": None,
                "deployment_mode": "keep_v2",
            }
        best = top[0]
        v2_ppd = float(v2_row.get("pass_per_day", 0))
        v2_prec = float(v2_row.get("precision", 0))
        best_ppd = float(best.get("pass_per_day", 0))
        best_prec = float(best.get("precision", 0))

        hybrid = next(
            (h for h in hybrid_rows if h.get("direction") == direction and h.get("scenario") == "hybrid_or"),
            None,
        )
        hybrid_ppd = float(hybrid.get("pass_per_day", 0)) if hybrid else 0
        hybrid_prec = float(hybrid.get("precision", 0)) if hybrid else 0

        if best_ppd > v2_ppd * 1.15 and best_prec >= v2_prec - PRECISION_TOLERANCE and float(best.get("lift", 0)) >= 1.1:
            return {
                "decision": "Adopt",
                "reason": f"Candidate beats V2 pass/day ({best_ppd} vs {v2_ppd}) with precision {best_prec}",
                "recommended_rule_id": best.get("rule_id"),
                "recommended_rule_expr": best.get("rule_expr"),
                "deployment_mode": "replace_v2",
            }
        if hybrid and hybrid_ppd > v2_ppd and hybrid_prec >= v2_prec - PRECISION_TOLERANCE:
            return {
                "decision": "Needs further validation",
                "reason": f"Hybrid OR improves {direction} coverage ({hybrid_ppd} vs {v2_ppd} pass/day) — extend blind window",
                "recommended_rule_id": best.get("rule_id"),
                "recommended_rule_expr": hybrid.get("rule_expr"),
                "deployment_mode": "hybrid_or_v2",
            }
        if best_prec >= v2_prec - PRECISION_TOLERANCE and best_ppd > v2_ppd:
            return {
                "decision": "Needs further validation",
                "reason": f"Marginal {direction} coverage gain ({best_ppd} vs {v2_ppd} pass/day)",
                "recommended_rule_id": best.get("rule_id"),
                "recommended_rule_expr": best.get("rule_expr"),
                "deployment_mode": "candidate_only",
            }
        return {
            "decision": "Reject",
            "reason": f"{direction} candidates do not improve trades/day under precision constraint",
            "recommended_rule_id": best.get("rule_id"),
            "recommended_rule_expr": best.get("rule_expr"),
            "deployment_mode": "keep_v2",
        }

    long_dec = _one("long", top_long, v2_long)
    short_dec = _one("short", top_short, v2_short)

    priority = {"Adopt": 3, "Needs further validation": 2, "Reject": 1}
    overall = long_dec if priority.get(long_dec["decision"], 0) >= priority.get(short_dec["decision"], 0) else short_dec

    return {
        "decision": overall["decision"],
        "reason": f"Long: {long_dec['decision']}. Short: {short_dec['decision']}. {overall['reason']}",
        "recommended_rule_id": overall.get("recommended_rule_id"),
        "recommended_rule_expr": overall.get("recommended_rule_expr"),
        "direction": "long" if overall is long_dec else "short",
        "deployment_mode": overall.get("deployment_mode"),
        "long": long_dec,
        "short": short_dec,
    }
