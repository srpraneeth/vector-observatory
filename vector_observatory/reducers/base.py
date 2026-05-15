from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Reducer(Protocol):
    """Structural protocol for dimensionality reducers.

    All reducers must satisfy this interface. No inheritance required.
    GPU backends (cuML) and custom reducers are drop-in replacements.
    """

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return reduced coordinates. Shape: (N, n_components)."""
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new points using a previously fitted reducer."""
        ...

    @property
    def config(self) -> dict:
        """Serializable config dict for reproducibility."""
        ...
