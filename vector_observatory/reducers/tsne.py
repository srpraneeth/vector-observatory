from __future__ import annotations

import numpy as np

from ..schema import TSNEConfig


class TSNEReducer:
    """t-SNE dimensionality reducer.

    Good for revealing fine-grained cluster structure. Does not support
    transform() on new points — refit required for new data.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        metric: str = "cosine",
        random_state: int = 42,
        max_iter: int = 1000,
    ) -> None:
        self._config = TSNEConfig(
            n_components=n_components,
            perplexity=perplexity,
            metric=metric,
            random_state=random_state,
            max_iter=max_iter,
        )

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=self._config.n_components,
            perplexity=self._config.perplexity,
            metric=self._config.metric,
            random_state=self._config.random_state,
            max_iter=self._config.max_iter,
        )
        return reducer.fit_transform(X).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "t-SNE does not support projecting new points. "
            "Use UMAPReducer if you need transform()."
        )

    @property
    def config(self) -> dict:
        return {"type": "tsne", **vars(self._config)}
