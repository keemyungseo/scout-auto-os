"""Auto-generate cluster formulas from DNA importance."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClusterFormula:
    name: str
    engine: str
    cluster_id: str
    direction: str
    weights: dict[str, float] = field(default_factory=dict)
    train_samples: int = 0

    def score(self, features: dict) -> float:
        total = 0.0
        for k, w in self.weights.items():
            total += w * float(features.get(k, 0))
        return total

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "engine": self.engine,
            "cluster_id": self.cluster_id,
            "direction": self.direction,
            "weights": self.weights,
            "train_samples": self.train_samples,
        }


def build_cluster_formula(
    engine: str,
    cluster_letter: str,
    direction: str,
    cluster_samples: list[dict],
    importance: list[dict],
    top_n: int = 12,
) -> ClusterFormula:
    """Weights from success-biased feature deltas within cluster."""
    weights: dict[str, float] = {}
    for row in importance[:top_n]:
        feat = row["feature"]
        delta = float(row["delta"])
        if direction == "short" and feat.endswith("_return_pct"):
            # for short, lower returns in features may be positive signal — keep signed delta
            pass
        weights[feat] = round(delta, 6)
    # normalize max abs weight to 1
    max_w = max((abs(v) for v in weights.values()), default=1.0) or 1.0
    weights = {k: round(v / max_w, 6) for k, v in weights.items()}
    return ClusterFormula(
        name=f"{engine}_{cluster_letter}",
        engine=engine,
        cluster_id=cluster_letter,
        direction=direction,
        weights=weights,
        train_samples=len(cluster_samples),
    )


def rank_by_formula(rows: list[dict], formula: ClusterFormula, top_k: int = 5) -> list[str]:
    ranked = sorted(rows, key=lambda r: formula.score(r["features"]), reverse=True)
    return [r["symbol"] for r in ranked[:top_k]]
