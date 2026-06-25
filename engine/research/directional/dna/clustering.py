"""Simple k-means clustering without external deps."""

from __future__ import annotations

import random
import statistics
from typing import Sequence


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def kmeans(
    data: list[list[float]],
    k: int,
    seed: int = 42,
    max_iter: int = 40,
) -> tuple[list[int], list[list[float]]]:
    if not data or k <= 0:
        return [], []
    k = min(k, len(data))
    rng = random.Random(seed)
    centroids = [list(data[i]) for i in rng.sample(range(len(data)), k)]
    labels = [0] * len(data)

    for _ in range(max_iter):
        changed = False
        for i, point in enumerate(data):
            best_j = min(range(k), key=lambda j: _dist(point, centroids[j]))
            if labels[i] != best_j:
                labels[i] = best_j
                changed = True
        new_centroids: list[list[float]] = []
        for j in range(k):
            cluster_pts = [data[i] for i, lbl in enumerate(labels) if lbl == j]
            if not cluster_pts:
                new_centroids.append(list(centroids[j]))
                continue
            dim = len(cluster_pts[0])
            new_centroids.append([
                statistics.mean(p[d] for p in cluster_pts) for d in range(dim)
            ])
        centroids = new_centroids
        if not changed:
            break
    return labels, centroids


def choose_k(n: int) -> int:
    if n < 40:
        return 1
    if n < 120:
        return 2
    if n < 300:
        return 3
    return min(4, max(2, n // 150))


def build_feature_matrix(samples: list[dict], feature_keys: list[str]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for s in samples:
        matrix.append([float(s["features"].get(k, 0)) for k in feature_keys])
    return matrix


def normalize_matrix(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        return []
    dims = len(matrix[0])
    cols = [[row[d] for row in matrix] for d in range(dims)]
    norms: list[tuple[float, float]] = []
    for col in cols:
        mu = statistics.mean(col)
        sd = statistics.pstdev(col) if len(col) > 1 else 1.0
        norms.append((mu, sd or 1.0))
    out: list[list[float]] = []
    for row in matrix:
        out.append([(row[d] - norms[d][0]) / norms[d][1] for d in range(dims)])
    return out
