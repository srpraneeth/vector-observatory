from __future__ import annotations

import numpy as np
import pandas as pd


def find_near_duplicates(
    embeddings: np.ndarray,
    ids: np.ndarray,
    threshold: float = 0.95,
    k: int = 10,
) -> pd.DataFrame:
    """Find near-duplicate pairs by cosine similarity.

    threshold: minimum similarity (1 - cosine_distance) to report.
    Returns DataFrame[id_a, id_b, similarity] sorted by similarity desc.
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(embeddings)
    k_actual = min(k + 1, n)

    nn = NearestNeighbors(n_neighbors=k_actual, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    rows = []
    seen: set[tuple[int, int]] = set()
    for i in range(n):
        for pos in range(1, k_actual):
            sim = 1.0 - float(distances[i, pos])
            if sim < threshold:
                break  # distances are sorted ascending — safe to stop
            j = int(indices[i, pos])
            pair = (min(i, j), max(i, j))
            if pair not in seen:
                seen.add(pair)
                rows.append({"id_a": ids[i], "id_b": ids[j], "similarity": round(sim, 4)})

    if not rows:
        return pd.DataFrame(columns=["id_a", "id_b", "similarity"])
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)
