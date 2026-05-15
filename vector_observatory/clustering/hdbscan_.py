from __future__ import annotations

import numpy as np

from ..schema import HDBSCANConfig


class HDBSCANClusterer:
    """HDBSCAN clusterer — recommended default.

    Variable-density clusters, robust noise detection.
    Assigns -1 to noise points.
    """

    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: int = 5,
        metric: str = "euclidean",
    ) -> None:
        self._config = HDBSCANConfig(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
        )

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self._config.min_cluster_size,
            min_samples=self._config.min_samples,
            metric=self._config.metric,
        )
        return clusterer.fit_predict(X).astype(np.int32)

    @property
    def config(self) -> dict:
        return {"type": "hdbscan", **vars(self._config)}
