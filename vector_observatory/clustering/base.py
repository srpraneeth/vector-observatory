from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Clusterer(Protocol):
    """Structural protocol for clustering algorithms.

    Clustering operates on original high-dimensional embeddings,
    not on reduced coordinates.
    """

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels. Shape: (N,). -1 indicates noise."""
        ...

    @property
    def config(self) -> dict:
        """Serializable config dict for reproducibility."""
        ...
