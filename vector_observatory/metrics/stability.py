from __future__ import annotations

import numpy as np


def cluster_jaccard_matrix(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Pairwise Jaccard overlap between clusters in two label arrays.

    Noise points (-1) are excluded.
    Returns (matrix[n_a, n_b], cluster_ids_a, cluster_ids_b).
    """
    ids_a = sorted(set(labels_a.tolist()) - {-1})
    ids_b = sorted(set(labels_b.tolist()) - {-1})

    if not ids_a or not ids_b:
        return np.zeros((len(ids_a), len(ids_b)), dtype=np.float32), ids_a, ids_b

    sets_a = [set(np.where(labels_a == c)[0].tolist()) for c in ids_a]
    sets_b = [set(np.where(labels_b == c)[0].tolist()) for c in ids_b]

    matrix = np.zeros((len(ids_a), len(ids_b)), dtype=np.float32)
    for i, sa in enumerate(sets_a):
        for j, sb in enumerate(sets_b):
            inter = len(sa & sb)
            union = len(sa | sb)
            matrix[i, j] = inter / union if union > 0 else 0.0

    return matrix, ids_a, ids_b


def stability_score(matrix: np.ndarray) -> float:
    """Mean best-match Jaccard across all clusters in run A (0–1, higher = more stable)."""
    if matrix.size == 0:
        return 0.0
    return float(matrix.max(axis=1).mean())
