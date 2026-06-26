"""Policy A/B replay evaluation."""

from __future__ import annotations

from scout_auto_os.engine.predator.policies import POLICIES, evaluate_policy, score_band
from scout_auto_os.engine.predator.value_gate import GateAction
from scout_auto_os.engine.research.ranking_engine.metrics import equity_mdd, sharpe

WIN_THRESHOLD = 3.0
BIG_WIN_THRESHOLD = 10.0


def gated_block(gate: dict, actual_roi: float) -> dict:
    action = gate["action"]
    size = float(gate["recommended_size"])
    if action in (GateAction.SKIP.value, GateAction.NO_ACTION.value):
        return {"action": action, "size": 0.0, "weighted_roi": 0.0, "taken": False, "reason": gate["reason"]}
    if action == GateAction.SHADOW_ONLY.value:
        return {
            "action": action, "size": size,
            "weighted_roi": actual_roi * size, "taken": False,
            "shadow_simulated": True, "reason": gate["reason"],
        }
    return {
        "action": action, "size": size,
        "weighted_roi": actual_roi * size, "taken": True, "reason": gate["reason"],
    }


def apply_policy_to_rows(rows: list[dict], policy_key: str) -> list[dict]:
    out = []
    for row in rows:
        gate = evaluate_policy(policy_key, row)
        g = gated_block(gate, float(row["actual_roi"]))
        out.append({**row, "policy": policy_key, "gate": gate, "gated": g})
    return out


def compute_metrics(trades: list[dict]) -> dict:
    rets: list[float] = []
    taken_rois: list[float] = []
    skipped_rois: list[float] = []
    for t in trades:
        g = t["gated"]
        roi = float(t["actual_roi"])
        if g.get("taken"):
            rets.append(float(g["weighted_roi"]))
            taken_rois.append(roi)
        elif g.get("action") == GateAction.SKIP.value or g.get("shadow_simulated"):
            if g.get("action") == GateAction.SKIP.value or not g.get("taken"):
                skipped_rois.append(roi)
            if g.get("shadow_simulated"):
                rets.append(float(g["weighted_roi"]))
    active = [r for r in rets if r != 0] or rets
    wins = sum(1 for r in taken_rois if r >= WIN_THRESHOLD)
    mdd = equity_mdd(active) if active else 0.0
    total = round(sum(rets), 4)
    tc = len(taken_rois)
    abs_mdd = max(abs(mdd), 0.01)
    return {
        "total_roi": total,
        "avg_roi": round(sum(taken_rois) / tc, 4) if tc else 0.0,
        "win_rate": round(wins / tc * 100, 2) if tc else 0.0,
        "sharpe": sharpe(active) if active else 0.0,
        "mdd": mdd,
        "trade_count": tc,
        "skipped_count": len(skipped_rois),
        "skipped_avg_roi": round(sum(skipped_rois) / len(skipped_rois), 4) if skipped_rois else 0.0,
        "accepted_avg_roi": round(sum(taken_rois) / tc, 4) if tc else 0.0,
        "return_per_risk": round(total / abs_mdd, 4),
        "return_per_trade": round(total / tc, 4) if tc else 0.0,
        "mdd_adjusted_return": round(total / abs_mdd, 4),
    }


def false_skips(trades: list[dict]) -> list[dict]:
    out = []
    for t in trades:
        g = t["gated"]
        if g.get("taken"):
            continue
        if g["action"] not in (GateAction.SKIP.value, GateAction.SHADOW_ONLY.value):
            continue
        roi = float(t["actual_roi"])
        if roi < BIG_WIN_THRESHOLD:
            continue
        out.append(_case_row(t, "false_skip"))
    return out


def false_accepts(trades: list[dict]) -> list[dict]:
    out = []
    for t in trades:
        g = t["gated"]
        if not g.get("taken"):
            continue
        roi = float(t["actual_roi"])
        if roi >= 0:
            continue
        out.append(_case_row(t, "false_accept"))
    return out


def _case_row(t: dict, kind: str) -> dict:
    g = t["gated"]
    return {
        "policy": t["policy"],
        "kind": kind,
        "trade_key": t["trade_key"],
        "symbol": t["symbol"],
        "direction": t["direction"],
        "value_score": t["value_score"],
        "score_band": score_band(float(t["value_score"])),
        "gate_action": g["action"],
        "gate_reason": g.get("reason", t["gate"].get("reason", "")),
        "recommended_size": g["size"],
        "actual_roi": float(t["actual_roi"]),
        "actual_dna_type": t["actual_dna_type"],
        "predicted_dna_type": t["predicted_dna_type"],
        "predicted_roi": t["predicted_roi"],
        "predicted_drawdown": t["predicted_drawdown"],
        "predicted_win_prob": t["predicted_win_prob"],
        "runner_probability": t["runner_probability"],
        "type1_false_pass": int(
            t["predicted_dna_type"] == "TYPE_1" or t["actual_dna_type"] == "TYPE_1"
        ),
        "type0_pred_actual_type1": int(
            t["predicted_dna_type"] == "TYPE_0" and t["actual_dna_type"] == "TYPE_1"
        ),
        "live_pattern": t.get("live_pattern", ""),
    }


def best_missed(trades: list[dict]) -> dict | None:
    skipped = [t for t in trades if not t["gated"].get("taken")]
    return max(skipped, key=lambda t: float(t["actual_roi"])) if skipped else None


def worst_accepted(trades: list[dict]) -> dict | None:
    taken = [t for t in trades if t["gated"].get("taken")]
    return min(taken, key=lambda t: float(t["actual_roi"])) if taken else None


def band_analysis(trades: list[dict], policy_key: str) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for t in trades:
        b = score_band(float(t["value_score"]))
        buckets.setdefault(b, []).append(t)
    order = ["0-49", "50-59", "60-69", "70-79", "80+"]
    rows = []
    for band in order:
        items = buckets.get(band, [])
        if not items:
            rows.append({
                "policy": policy_key, "score_band": band,
                "trade_count": 0, "taken_count": 0, "skip_count": 0,
                "avg_actual_roi": 0, "avg_weighted_roi": 0, "win_rate_pct": 0,
            })
            continue
        taken = [x for x in items if x["gated"].get("taken")]
        rois = [float(x["actual_roi"]) for x in items]
        wrois = [float(x["gated"]["weighted_roi"]) for x in items if x["gated"].get("taken")]
        wins = sum(1 for x in taken if float(x["actual_roi"]) >= WIN_THRESHOLD)
        rows.append({
            "policy": policy_key,
            "score_band": band,
            "trade_count": len(items),
            "taken_count": len(taken),
            "skip_count": len(items) - len(taken),
            "avg_actual_roi": round(sum(rois) / len(rois), 4),
            "avg_weighted_roi": round(sum(wrois) / len(wrois), 4) if wrois else 0,
            "win_rate_pct": round(wins / len(taken) * 100, 2) if taken else 0,
        })
    return rows


def split_by_side(trades: list[dict], policy_key: str) -> list[dict]:
    rows = []
    for side in ("long", "short"):
        subset = [t for t in trades if t["direction"] == side]
        m = compute_metrics(subset)
        fs = len(false_skips(subset))
        fa = len(false_accepts(subset))
        m.update({
            "policy": policy_key,
            "side": side.upper(),
            "false_skip_count": fs,
            "false_accept_count": fa,
        })
        rows.append(m)
    return rows


def policy_score_for_selection(m: dict, v1: dict, *, long_row: dict, short_row: dict) -> float:
    """Higher is better. Composite against V1 baseline policy metrics."""
    score = 0.0
    if m["mdd"] >= -10:
        score += 3
    elif m["mdd"] >= v1["mdd"]:
        score += 1
    if m["win_rate"] >= 75:
        score += 3
    elif m["win_rate"] >= v1["win_rate"]:
        score += 1
    if m.get("false_accept_count", 99) <= v1.get("false_accept_count", 0) + 1:
        score += 2
    if m.get("false_skip_count", 99) < v1.get("false_skip_count", 99):
        score += 2
    if m["trade_count"] >= 35:
        score += 1
    if m["trade_count"] >= 25:
        score += 1
    lc, sc = long_row["trade_count"], short_row["trade_count"]
    total = lc + sc or 1
    if max(lc, sc) / total <= 0.85:
        score += 1
    if m["sharpe"] >= v1["sharpe"]:
        score += 2
    return score
