"""Build entry-only dataset from Trade DNA cluster labels."""

from __future__ import annotations

import csv
from pathlib import Path

from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.research.directional.dna.formulas import ClusterFormula
from scout_auto_os.engine.research.lifecycle_classifier.features import build_entry_feature_row
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl
from scout_auto_os.engine.research.zero_base.validation import classify_regime

# Post-entry columns — must never be used as features
LEAK_COLUMNS = frozenset({
    "peak_timing_min", "peak_roi", "final_roi_2h", "final_roi_4h", "max_drawdown",
    "alive_delta_proxy", "exit_pressure_proxy", "is_winner",
    "roi_5m", "roi_15m", "roi_30m", "roi_60m", "roi_120m", "roi_240m",
    "vol_5m", "vol_15m", "vol_30m", "vol_60m", "vol_120m", "vol_240m",
    "dd_5m", "dd_15m", "dd_30m", "dd_60m", "dd_120m", "dd_240m",
})


def load_cluster_labels(cluster_path: Path) -> list[dict]:
    with cluster_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_entry_dataset(
    cluster_path: Path,
    data_dir: Path,
    pkg_root: Path,
    candidates_path: Path,
) -> list[dict]:
    labels = load_cluster_labels(cluster_path)
    by_scan = load_candidates_jsonl(candidates_path)
    engine = PortfolioEngine.from_paths(data_dir, pkg_root)
    formulas: list[ClusterFormula] = []
    latest_scan = max(by_scan.keys()) if by_scan else ""

    rows: list[dict] = []
    for lab in labels:
        scan = lab["scan_kst"]
        sym = lab["symbol"]
        direction = lab["direction"]
        peers = by_scan.get(scan, [])
        cand = next((r for r in peers if r["symbol"] == sym), None)
        if not cand:
            continue

        features = cand.get("features") or {}
        x = build_entry_feature_row(
            cand, direction, "A6", engine.rules, formulas,
            peers, scan, latest_scan,
        )
        h4 = float(features.get("h4_score", 0))
        x["search_h4_score"] = h4
        x["search_a6_proxy"] = h4 if h4 > 0 else float(x.get("entry_score", 0))
        x["constitution_entry_score"] = float(lab.get("entry_score", x.get("entry_score", 0)))
        regime = classify_regime(peers)
        for r in ("bull", "bear", "sideway", "recovery", "crash", "unknown"):
            x[f"regime_{r}"] = 1.0 if regime == r else 0.0

        y = int(lab["cluster_id"])
        rows.append({
            "trade_key": lab["trade_key"],
            "scan_kst": scan,
            "symbol": sym,
            "direction": direction,
            "trade_type_id": lab["trade_type_id"],
            "label_runner": 1 if y == 0 else 0,
            "label_failed": 1 if y == 1 else 0,
            "cluster_id": y,
            "live_pattern": lab.get("live_pattern", ""),
            "x": x,
        })
    return rows
