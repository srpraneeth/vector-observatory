from __future__ import annotations

import numpy as np

from ..schema import KMeansConfig


class KMeansClusterer:
    """K-Means clusterer.

    Fixed number of clusters, no noise detection.
    Use when the number of clusters is known in advance.
    """

    def __init__(self, n_clusters: int = 8, random_state: int = 42) -> None:
        self._config = KMeansConfig(n_clusters=n_clusters, random_state=random_state)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import KMeans
        clusterer = KMeans(
            n_clusters=self._config.n_clusters,
            random_state=self._config.random_state,
            n_init="auto",
        )
        return clusterer.fit_predict(X).astype(np.int32)

    @property
    def config(self) -> dict:
        return {"type": "kmeans", **vars(self._config)}
