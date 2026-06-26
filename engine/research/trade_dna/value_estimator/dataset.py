"""Value estimator dataset — entry features + replay labels."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from scout_auto_os.engine.research.trade_dna.predictor.dataset import build_entry_dataset


def _sharpe_contrib(roi: float, drawdown: float) -> float:
    risk = max(abs(drawdown), 0.5)
    return round(roi / risk, 4)


def build_value_dataset(
    cluster_path: Path,
    data_dir: Path,
    pkg_root: Path,
    candidates_path: Path,
) -> list[dict]:
    label_map = {r["trade_key"]: r for r in _load_cluster(cluster_path)}
    rows = build_entry_dataset(cluster_path, data_dir, pkg_root, candidates_path)

    out: list[dict] = []
    for row in rows:
        lab = label_map.get(row["trade_key"])
        if not lab:
            continue
        roi = float(lab["final_roi_2h"])
        peak = float(lab["peak_roi"])
        dd = abs(float(lab["max_drawdown"]))
        peak_time = int(float(lab["peak_timing_min"]))
        win = int(lab.get("is_winner", 0))
        hold_efficiency = round(roi / peak, 4) if abs(peak) > 0.01 else 0.0

        row["y"] = {
            "expected_roi": round(roi, 4),
            "expected_peak_roi": round(peak, 4),
            "expected_hold_time": peak_time,
            "expected_drawdown": round(dd, 4),
            "expected_win_prob": float(win),
            "expected_sharpe_contrib": _sharpe_contrib(roi, dd),
            "hold_efficiency": hold_efficiency,
        }
        out.append(row)
    return out


def _load_cluster(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))
