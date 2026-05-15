from __future__ import annotations

import numpy as np

from ..schema import PCAConfig


class PCAReducer:
    """PCA dimensionality reducer.

    Fast linear baseline. Use for quick previews or as a pre-processing
    step before UMAP on very high-dimensional embeddings.
    """

    def __init__(self, n_components: int = 2) -> None:
        self._config = PCAConfig(n_components=n_components)
        self._reducer = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA
        self._reducer = PCA(n_components=self._config.n_components)
        return self._reducer.fit_transform(X).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._reducer is None:
            raise RuntimeError("Call fit_transform before transform.")
        return self._reducer.transform(X).astype(np.float32)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        if self._reducer is None:
            raise RuntimeError("Call fit_transform first.")
        return self._reducer.explained_variance_ratio_

    @property
    def config(self) -> dict:
        return {"type": "pca", **vars(self._config)}
