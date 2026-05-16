from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClusterMetrics:
    n_clusters: int
    noise_fraction: float
    cluster_sizes: dict[int, int]  # cluster_id → count
    largest_cluster_fraction: float
    silhouette_score: float | None  # None if not computed (expensive)


def compute_cluster_metrics(
    labels: np.ndarray,
    embeddings: np.ndarray | None = None,
    compute_silhouette: bool = False,
) -> ClusterMetrics:
    """Compute cluster quality metrics from label array.

    Args:
        labels: (N,) int array of cluster assignments. -1 = noise.
        embeddings: Required only if compute_silhouette=True.
        compute_silhouette: Silhouette score is expensive — opt-in.
    """
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(k): int(v) for k, v in zip(unique, counts, strict=False) if k >= 0}

    n_clusters = len(cluster_sizes)
    noise_count = int((labels == -1).sum())
    noise_fraction = noise_count / len(labels) if len(labels) > 0 else 0.0
    largest = max(cluster_sizes.values()) / len(labels) if cluster_sizes else 0.0

    silhouette = None
    if compute_silhouette and embeddings is not None and n_clusters >= 2:
        from sklearn.metrics import silhouette_score

        valid_mask = labels >= 0
        if valid_mask.sum() > n_clusters:
            silhouette = float(silhouette_score(embeddings[valid_mask], labels[valid_mask]))

    return ClusterMetrics(
        n_clusters=n_clusters,
        noise_fraction=noise_fraction,
        cluster_sizes=cluster_sizes,
        largest_cluster_fraction=largest,
        silhouette_score=silhouette,
    )
