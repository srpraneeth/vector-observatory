from __future__ import annotations

import numpy as np


def compute_outlier_scores(embeddings: np.ndarray, k: int = 5) -> np.ndarray:
    """Score each point by mean cosine distance to its k nearest neighbors.

    Higher score = more isolated = stronger outlier signal.
    Returns float32 array of shape (N,).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    # column 0 is distance to self (always 0) — skip it
    return distances[:, 1:].mean(axis=1).astype(np.float32)
