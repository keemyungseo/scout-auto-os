"""Load cluster formulas and validation statistics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scout_auto_os.engine.research.directional.dna.formulas import ClusterFormula


def load_formulas(path: Path) -> list[ClusterFormula]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        ClusterFormula(
            name=row["name"],
            engine=row["engine"],
            cluster_id=row["cluster_id"],
            direction=row["direction"],
            weights=row.get("weights", {}),
            train_samples=int(row.get("train_samples", 0)),
        )
        for row in raw
    ]


def resolve_formulas_path(data_dir: Path, pkg_root: Path) -> Path:
    local = data_dir / "zero_base" / "directional_dna_formulas.json"
    if local.exists():
        return local
    bundle = pkg_root / "research_bundle" / "reports" / "directional_dna_v1_formulas.json"
    if bundle.exists():
        return bundle
    raise FileNotFoundError("directional_dna_formulas.json not found in data/zero_base or research_bundle")


def load_expected_returns(clusters_csv: Path, split: str = "train") -> dict[str, dict]:
    """Per-formula expected return from cluster train stats (no blind leakage)."""
    if not clusters_csv.exists():
        return {}
    out: dict[str, dict] = {}
    with clusters_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") != split:
                continue
            name = row["formula_name"]
            out[name] = {
                "avg_return_2h": float(row.get("avg_return_2h", 0)),
                "win_rate": float(row.get("win_rate", 0)),
                "trap_rate": float(row.get("trap_rate", 0)),
                "sample_count": int(row.get("sample_count", 0)),
            }
    return out


def load_blind_validation(validation_csv: Path) -> dict[str, dict]:
    """Blind validation stats for reporting only."""
    if not validation_csv.exists():
        return {}
    out: dict[str, dict] = {}
    with validation_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") != "blind":
                continue
            name = row["formula_name"]
            if name in ("RANDOM", "PATTERN_CHAMPION"):
                continue
            out[name] = {
                "engine": row.get("engine", ""),
                "avg_return_2h": float(row.get("avg_return_2h", 0)),
                "win_rate": float(row.get("win_rate", 0)),
                "delta_vs_random": float(row.get("delta_vs_random", 0)),
                "delta_vs_champion": float(row.get("delta_vs_champion", 0)),
            }
    return out


def best_cluster_formula(blind_stats: dict[str, dict], direction: str) -> str | None:
    prefix = "LONG_" if direction == "long" else "SHORT_"
    candidates = {
        k: v for k, v in blind_stats.items()
        if k.startswith(prefix) and k.count("_") >= 2
    }
    if not candidates:
        return None
    return max(candidates, key=lambda k: candidates[k]["avg_return_2h"])
