from __future__ import annotations

import numpy as np
import pandas as pd


def find_hard_negatives(
    embeddings: np.ndarray,
    ids: np.ndarray,
    metadata: pd.DataFrame,
    label_col: str,
    k: int = 10,
) -> pd.DataFrame:
    """Surface pairs close in embedding space but with different labels.

    Returns DataFrame[id_a, <label_col>_a, id_b, <label_col>_b, similarity]
    sorted by similarity desc (highest similarity = hardest negative first).
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(embeddings)
    k_actual = min(k + 1, n)

    nn = NearestNeighbors(n_neighbors=k_actual, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    label_values = metadata[label_col].values
    col_a, col_b = f"{label_col}_a", f"{label_col}_b"

    rows = []
    seen: set[tuple[int, int]] = set()
    for i in range(n):
        for pos in range(1, k_actual):
            j = int(indices[i, pos])
            if label_values[i] == label_values[j]:
                continue
            pair = (min(i, j), max(i, j))
            if pair not in seen:
                seen.add(pair)
                rows.append(
                    {
                        "id_a": ids[i],
                        col_a: label_values[i],
                        "id_b": ids[j],
                        col_b: label_values[j],
                        "similarity": round(1.0 - float(distances[i, pos]), 4),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["id_a", col_a, "id_b", col_b, "similarity"])
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)
