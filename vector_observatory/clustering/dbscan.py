from __future__ import annotations

import numpy as np

from ..schema import DBSCANConfig


class DBSCANClusterer:
    """DBSCAN clusterer.

    Fixed-density clusters. Requires tuning eps for the dataset.
    Assigns -1 to noise points.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ) -> None:
        self._config = DBSCANConfig(eps=eps, min_samples=min_samples, metric=metric)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(
            eps=self._config.eps,
            min_samples=self._config.min_samples,
            metric=self._config.metric,
        )
        return clusterer.fit_predict(X).astype(np.int32)

    @property
    def config(self) -> dict:
        return {"type": "dbscan", **vars(self._config)}
