from __future__ import annotations

import numpy as np

from ..schema import UMAPConfig


class UMAPReducer:
    """UMAP dimensionality reducer.

    Primary reducer for visual exploration. Preserves both local and
    global structure better than t-SNE at larger scales.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42,
    ) -> None:
        self._config = UMAPConfig(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        self._reducer = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        import umap
        self._reducer = umap.UMAP(
            n_components=self._config.n_components,
            n_neighbors=self._config.n_neighbors,
            min_dist=self._config.min_dist,
            metric=self._config.metric,
            random_state=self._config.random_state,
        )
        return self._reducer.fit_transform(X).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._reducer is None:
            raise RuntimeError("Call fit_transform before transform.")
        return self._reducer.transform(X).astype(np.float32)

    @property
    def config(self) -> dict:
        return {"type": "umap", **vars(self._config)}
