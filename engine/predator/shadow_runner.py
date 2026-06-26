"""Shadow mode — baseline vs Value Gate side-by-side (no live orders)."""

from __future__ import annotations

from scout_auto_os.engine.predator.predator_output import enrich_predator_candidate
from scout_auto_os.engine.predator.value_gate import GateAction, evaluate_gate
from scout_auto_os.engine.research.ranking_engine.metrics import equity_mdd, sharpe


WIN_THRESHOLD = 3.0
BIG_WIN_THRESHOLD = 10.0


def process_trade_shadow(row: dict) -> dict:
    """Record baseline (A) and value-gated (B) decisions for one replay trade."""
    candidate = {
        "symbol": row["symbol"],
        "side": row["side"],
        "entry_score": row["entry_score"],
        "direction_confidence": row["direction_confidence"],
    }
    predictions = {
        "symbol": row["symbol"],
        "direction": row["direction"],
        "entry_score": row["entry_score"],
        "value_score": row["value_score"],
        "predicted_roi": row["predicted_roi"],
        "predicted_peak_roi": row["predicted_peak_roi"],
        "predicted_drawdown": row["predicted_drawdown"],
        "predicted_win_prob": row["predicted_win_prob"],
        "predicted_sharpe": row["predicted_sharpe"],
        "predicted_dna_type": row["predicted_dna_type"],
        "runner_probability": row["runner_probability"],
        "dna_type_probability": row["runner_probability"],
    }
    enriched = enrich_predator_candidate(candidate, predictions)
    actual_roi = float(row["actual_roi"])

    baseline = {
        "strategy": "baseline_predator",
        "action": "ENTER",
        "size": 1.0,
        "weighted_roi": actual_roi,
        "taken": True,
    }
    gate_action = enriched["gate_action"]
    size = float(enriched["recommended_size"])
    if gate_action in (GateAction.SKIP.value, GateAction.NO_ACTION.value):
        gated = {
            "strategy": "value_gate",
            "action": gate_action,
            "size": 0.0,
            "weighted_roi": 0.0,
            "taken": False,
        }
    elif gate_action == GateAction.SHADOW_ONLY.value:
        gated = {
            "strategy": "value_gate",
            "action": gate_action,
            "size": size,
            "weighted_roi": actual_roi * size,
            "taken": False,
            "shadow_simulated": True,
        }
    else:
        gated = {
            "strategy": "value_gate",
            "action": gate_action,
            "size": size,
            "weighted_roi": actual_roi * size,
            "taken": True,
        }

    return {
        "trade_key": row["trade_key"],
        "scan_kst": row["scan_kst"],
        "symbol": row["symbol"],
        "direction": row["direction"],
        "actual_roi": actual_roi,
        "actual_dna_type": row["actual_dna_type"],
        "predicted_dna_type": row["predicted_dna_type"],
        "live_pattern": row.get("live_pattern", ""),
        "value_score": enriched["value_score"],
        "gate_action": gate_action,
        "gate_reason": enriched["gate_reason"],
        "recommended_size": size,
        "baseline": baseline,
        "gated": gated,
        "enriched": enriched,
        "trade_contract": enriched["trade_contract"],
    }


def portfolio_metrics(trades: list[dict], key: str) -> dict:
    """key = 'baseline' or 'gated'"""
    rets: list[float] = []
    taken_rois: list[float] = []
    skipped_rois: list[float] = []
    for t in trades:
        block = t[key]
        roi = float(t["actual_roi"])
        if block.get("taken"):
            rets.append(float(block["weighted_roi"]))
            taken_rois.append(roi)
        elif block.get("action") == GateAction.SKIP.value:
            skipped_rois.append(roi)
        elif block.get("shadow_simulated"):
            rets.append(float(block["weighted_roi"]))
            skipped_rois.append(roi)

    active = [r for r in rets if r != 0] or rets
    wins = sum(1 for r in taken_rois if r >= WIN_THRESHOLD)
    return {
        "total_roi": round(sum(rets), 4),
        "avg_roi": round(sum(taken_rois) / len(taken_rois), 4) if taken_rois else 0.0,
        "win_rate": round(wins / len(taken_rois) * 100, 2) if taken_rois else 0.0,
        "sharpe": sharpe(active) if active else 0.0,
        "mdd": equity_mdd(active) if active else 0.0,
        "trade_count": len(taken_rois),
        "skipped_count": len(skipped_rois),
        "skipped_avg_roi": round(sum(skipped_rois) / len(skipped_rois), 4) if skipped_rois else 0.0,
        "accepted_avg_roi": round(sum(taken_rois) / len(taken_rois), 4) if taken_rois else 0.0,
    }


def false_skip_cases(trades: list[dict]) -> list[dict]:
    """Skipped by gate but actual ROI was strong."""
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
        out.append({
            "trade_key": t["trade_key"],
            "symbol": t["symbol"],
            "direction": t["direction"],
            "value_score": t["value_score"],
            "gate_action": g["action"],
            "gate_reason": t["gate_reason"],
            "actual_roi": roi,
            "actual_dna_type": t["actual_dna_type"],
            "predicted_dna_type": t["predicted_dna_type"],
            "predicted_roi": t["enriched"]["predicted_roi"],
            "predicted_win_prob": t["enriched"]["predicted_win_prob"],
            "runner_probability": t["enriched"]["dna_type_probability"],
            "live_pattern": t.get("live_pattern", ""),
        })
    return out


def false_accept_cases(trades: list[dict]) -> list[dict]:
    """Accepted (ENTER, size>0) but lost money."""
    out = []
    for t in trades:
        g = t["gated"]
        if not g.get("taken"):
            continue
        roi = float(t["actual_roi"])
        if roi >= 0:
            continue
        e = t["enriched"]
        out.append({
            "trade_key": t["trade_key"],
            "symbol": t["symbol"],
            "direction": t["direction"],
            "value_score": t["value_score"],
            "recommended_size": g["size"],
            "actual_roi": roi,
            "actual_dna_type": t["actual_dna_type"],
            "predicted_dna_type": t["predicted_dna_type"],
            "predicted_roi": e["predicted_roi"],
            "predicted_drawdown": e["predicted_drawdown"],
            "predicted_win_prob": e["predicted_win_prob"],
            "dna_type_probability": e["dna_type_probability"],
            "gate_reason": t["gate_reason"],
            "drawdown_prediction_error": round(
                abs(float(e["predicted_drawdown"])) - abs(float(t.get("actual_drawdown", 0))), 4
            ),
            "type1_false_pass": int(t["predicted_dna_type"] == "TYPE_1" or t["actual_dna_type"] == "TYPE_1"),
        })
    return out


def best_missed_trade(trades: list[dict]) -> dict | None:
    skipped = [t for t in trades if not t["gated"].get("taken")]
    if not skipped:
        return None
    return max(skipped, key=lambda t: float(t["actual_roi"]))


def worst_accepted_trade(trades: list[dict]) -> dict | None:
    taken = [t for t in trades if t["gated"].get("taken")]
    if not taken:
        return None
    return min(taken, key=lambda t: float(t["actual_roi"]))
