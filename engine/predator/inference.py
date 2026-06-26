"""Load replay predictions from Trade DNA + Value Estimator outputs."""

from __future__ import annotations

import csv
from pathlib import Path


def load_replay_bundle(trade_dna_dir: Path) -> list[dict]:
    """Merge value_prediction.csv + dna_prediction_model.csv + trade_cluster labels."""
    value_path = trade_dna_dir / "value_prediction.csv"
    dna_path = trade_dna_dir / "dna_prediction_model.csv"
    cluster_path = trade_dna_dir / "trade_cluster.csv"
    if not value_path.exists():
        raise FileNotFoundError(f"Missing {value_path} — run trade_value_estimator_v1 first")

    values = {r["trade_key"]: r for r in _read_csv(value_path)}
    dna = {r["trade_key"]: r for r in _read_csv(dna_path)} if dna_path.exists() else {}
    cluster = {r["trade_key"]: r for r in _read_csv(cluster_path)} if cluster_path.exists() else {}

    rows: list[dict] = []
    for trade_key, val in values.items():
        d = dna.get(trade_key, {})
        c = cluster.get(trade_key, {})
        actual_roi = float(val.get("actual_expected_roi", c.get("final_roi_2h", 0)))
        predicted_dna = d.get("predicted_type", c.get("trade_type_id", "UNKNOWN"))
        rows.append({
            "trade_key": trade_key,
            "scan_kst": trade_key.split("|")[0] if "|" in trade_key else "",
            "symbol": val["symbol"],
            "direction": val["direction"],
            "side": val["direction"],
            "entry_score": float(d.get("entry_score", c.get("entry_score", 0))),
            "direction_confidence": float(d.get("confidence", 0)),
            "runner_probability": float(d.get("runner_probability", 0)),
            "failed_probability": float(d.get("failed_probability", 0)),
            "predicted_dna_type": predicted_dna,
            "actual_dna_type": c.get("trade_type_id", ""),
            "predicted_roi": float(val.get("pred_expected_roi", 0)),
            "predicted_peak_roi": float(val.get("pred_expected_peak_roi", 0)),
            "predicted_drawdown": float(val.get("pred_expected_drawdown", 0)),
            "predicted_win_prob": float(val.get("pred_expected_win_prob", 0)),
            "predicted_sharpe": float(val.get("pred_expected_sharpe_contrib", 0)),
            "value_score": float(val.get("value_score", 0)),
            "actual_roi": actual_roi,
            "actual_peak_roi": float(val.get("actual_expected_peak_roi", c.get("peak_roi", 0))),
            "actual_drawdown": float(val.get("actual_expected_drawdown", c.get("max_drawdown", 0))),
            "is_winner": int(float(val.get("actual_expected_win_prob", c.get("is_winner", 0)))),
            "live_pattern": c.get("live_pattern", ""),
        })
    return rows


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))
