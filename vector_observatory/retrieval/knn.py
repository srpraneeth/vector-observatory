from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NeighborResult:
    query_id: str | int
    neighbor_ids: list[str | int]
    distances: list[float]
    metadata: pd.DataFrame   # rows aligned with neighbor_ids


class KNNIndex:
    """Brute-force k-nearest-neighbor index backed by scikit-learn.

    Operates in the original embedding space (not reduced coordinates).
    Sufficient for datasets up to ~100K points. Replace with FAISS in v2
    for larger scales.
    """

    def __init__(self, metric: str = "cosine") -> None:
        self._metric = metric
        self._index = None
        self._ids: np.ndarray | None = None
        self._metadata: pd.DataFrame | None = None

    def build(self, dataset) -> None:
        """Build the index from an EmbeddingDataset."""
        from sklearn.neighbors import NearestNeighbors
        from ..dataset import EmbeddingDataset  # avoid circular at module level

        self._ids = dataset.ids
        self._metadata = dataset.metadata
        self._index = NearestNeighbors(metric=self._metric, algorithm="brute")
        self._index.fit(dataset.embeddings)

    def search(self, query: np.ndarray, k: int = 10) -> NeighborResult:
        """Find k nearest neighbors for a single query vector."""
        if self._index is None:
            raise RuntimeError("Call build() before search().")

        query_2d = query.reshape(1, -1)
        distances, indices = self._index.kneighbors(query_2d, n_neighbors=k)
        indices = indices[0]
        distances = distances[0]

        return NeighborResult(
            query_id="",   # caller sets this
            neighbor_ids=list(self._ids[indices]),
            distances=list(distances.astype(float)),
            metadata=self._metadata.iloc[indices].reset_index(drop=True),
        )

    def search_by_id(self, id_value, dataset, k: int = 10) -> NeighborResult:
        """Find k nearest neighbors for a point already in the index."""
        from .dataset import EmbeddingDataset
        pos = dataset.index_of(id_value)
        result = self.search(dataset.embeddings[pos], k=k + 1)
        # Exclude the query point itself if it appears in results
        filtered_ids = []
        filtered_dist = []
        filtered_rows = []
        for i, nid in enumerate(result.neighbor_ids):
            if nid != id_value:
                filtered_ids.append(nid)
                filtered_dist.append(result.distances[i])
                filtered_rows.append(result.metadata.iloc[i])
        meta_df = pd.DataFrame(filtered_rows).reset_index(drop=True)
        return NeighborResult(
            query_id=id_value,
            neighbor_ids=filtered_ids[:k],
            distances=filtered_dist[:k],
            metadata=meta_df.iloc[:k],
        )
