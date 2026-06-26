"""Empirical lifecycle shape labels from measured trajectory features."""

from __future__ import annotations

from scout_auto_os.engine.research.signal_lifecycle.constants import SUCCESS_RETURN_PCT


def classify_lifecycle_shape(summary: dict) -> str:
    """
    Assign a descriptive lifecycle label from post-entry dynamics.
    Probabilistic taxonomy — not a trading rule.
    """
    peak_time = float(summary.get("peak_time_min", 0))
    peak_ret = float(summary.get("peak_return_pct", 0))
    mfe = float(summary.get("mfe_full", 0))
    ret_2h = float(summary.get("return_2h", 0))
    ret_6h = float(summary.get("return_6h", 0))
    ret_end = float(summary.get("return_at_end", 0))
    ret_90m = float(summary.get("return_90m", 0))
    early_mae = float(summary.get("early_mae_1h", 0))
    peak_frac = float(summary.get("peak_fraction", 0))
    end_time = float(summary.get("end_time_min", 1)) or 1.0

    if mfe < 1.0 and abs(ret_6h) < 0.5 and abs(ret_2h) < 0.5:
        return "Dead Signal"

    if mfe >= 3.0 and ret_end < 1.0 and ret_end <= peak_ret * 0.35:
        return "Fake Breakout"

    if early_mae <= -1.5 and ret_6h >= 2.0 and ret_90m <= 0:
        return "V-Reversal"

    if peak_time <= 45 and peak_ret >= 3.0:
        return "Immediate Explosion"

    if abs(ret_90m) < 1.0 and peak_time >= 120 and mfe >= 2.0:
        return "Delayed Breakout"

    if peak_frac >= 0.75 and peak_ret >= 2.0:
        return "Late Runner"

    if ret_6h >= 2.0 and peak_frac >= 0.55 and ret_end >= ret_6h * 0.8:
        return "Continuous Trend"

    if peak_time >= 120 and ret_end >= 1.0 and mfe >= 2.0:
        return "Slow Trend"

    if ret_6h >= SUCCESS_RETURN_PCT and ret_2h < SUCCESS_RETURN_PCT:
        return "Delayed Breakout"

    return "Unclassified"


def evaluation_flags(summary: dict) -> dict:
    ret_2h = float(summary.get("return_2h", 0))
    ret_6h = float(summary.get("return_6h", 0))
    success_2h = ret_2h >= SUCCESS_RETURN_PCT
    success_6h = ret_6h >= SUCCESS_RETURN_PCT
    shift = round(ret_6h - ret_2h, 4)

    if success_2h and not success_6h:
        eval_shift = "winner_2h_fade_6h"
    elif not success_2h and success_6h:
        eval_shift = "loser_2h_runner_6h"
    elif success_2h and success_6h:
        eval_shift = "sustained_winner"
    else:
        eval_shift = "underperformer"

    return {
        "success_2h": success_2h,
        "success_6h": success_6h,
        "return_shift_2h_to_6h": shift,
        "evaluation_shift": eval_shift,
    }


def shape_feature_row(summary: dict, label: str, engine: str) -> dict:
    flags = evaluation_flags(summary)
    return {
        "signal_id": summary["signal_id"],
        "direction": summary["direction"],
        "scan_time_kst": summary["scan_time_kst"],
        "symbol": summary["symbol"],
        "engine": engine,
        "lifecycle_label": label,
        "peak_time_min": summary["peak_time_min"],
        "peak_return_pct": summary["peak_return_pct"],
        "mfe_full": summary["mfe_full"],
        "mae_full": summary["mae_full"],
        "return_2h": summary["return_2h"],
        "return_6h": summary["return_6h"],
        "return_12h": summary.get("return_12h"),
        "return_at_end": summary["return_at_end"],
        "peak_fraction": summary["peak_fraction"],
        "track_hours": summary["track_hours"],
        **flags,
    }
