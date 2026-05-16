from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dataset import EmbeddingDataset
from ..reducers.base import Reducer


@dataclass
class DriftResult:
    """Result of comparing two embedding datasets in a shared coordinate space."""

    dataset_a: EmbeddingDataset  # with reduced_coords in shared space
    dataset_b: EmbeddingDataset  # with reduced_coords in shared space
    mmd_score: float  # Maximum Mean Discrepancy (0 = identical distributions)
    reducer_config: dict  # reducer was fit on A ∪ B
    cluster_overlap: dict[int, dict]  # {cluster_id: {"a_count": int, "b_count": int}}


class DriftComparison:
    """Compare two embedding datasets by projecting them into a shared UMAP space.

    The reducer is fit on the combined (A ∪ B) embeddings so both datasets
    share the same coordinate space. This is the only valid way to visually
    compare two embedding distributions.
    """

    def __init__(
        self,
        dataset_a: EmbeddingDataset,
        dataset_b: EmbeddingDataset,
        reducer: Reducer | None = None,
    ) -> None:
        from ..reducers.umap import UMAPReducer

        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.reducer = reducer or UMAPReducer()

    def run(self) -> DriftResult:
        combined = np.vstack([self.dataset_a.embeddings, self.dataset_b.embeddings])
        all_coords = self.reducer.fit_transform(combined)

        n_a = len(self.dataset_a.embeddings)
        coords_a = all_coords[:n_a]
        coords_b = all_coords[n_a:]

        ds_a = self.dataset_a.with_reduction(coords_a)
        ds_b = self.dataset_b.with_reduction(coords_b)

        mmd = _compute_mmd(self.dataset_a.embeddings, self.dataset_b.embeddings)
        overlap = _compute_cluster_overlap(ds_a, ds_b)

        return DriftResult(
            dataset_a=ds_a,
            dataset_b=ds_b,
            mmd_score=mmd,
            reducer_config=self.reducer.config,
            cluster_overlap=overlap,
        )


def _compute_mmd(X: np.ndarray, Y: np.ndarray, n_samples: int = 500) -> float:
    """Unbiased Maximum Mean Discrepancy with RBF kernel."""
    rng = np.random.default_rng(42)
    nx, ny = len(X), len(Y)
    ix = rng.choice(nx, size=min(n_samples, nx), replace=False)
    iy = rng.choice(ny, size=min(n_samples, ny), replace=False)
    X, Y = X[ix], Y[iy]

    # Median heuristic for bandwidth
    all_pts = np.vstack([X, Y])
    pairwise = np.sum((all_pts[:, None] - all_pts[None, :]) ** 2, axis=-1)
    sigma2 = float(np.median(pairwise[pairwise > 0]))
    if sigma2 == 0:
        return 0.0

    def rbf(A, B):
        diff = A[:, None] - B[None, :]
        return np.exp(-np.sum(diff**2, axis=-1) / (2 * sigma2))

    kxx = rbf(X, X)
    kyy = rbf(Y, Y)
    kxy = rbf(X, Y)

    n, m = len(X), len(Y)
    mmd = kxx.sum() / (n * n) + kyy.sum() / (m * m) - 2 * kxy.sum() / (n * m)
    return float(max(mmd, 0.0))


def _compute_cluster_overlap(
    ds_a: EmbeddingDataset,
    ds_b: EmbeddingDataset,
) -> dict[int, dict]:
    if ds_a.cluster_labels is None or ds_b.cluster_labels is None:
        return {}

    all_ids = set(np.unique(ds_a.cluster_labels)) | set(np.unique(ds_b.cluster_labels))
    overlap = {}
    for cid in sorted(all_ids):
        overlap[int(cid)] = {
            "a_count": int((ds_a.cluster_labels == cid).sum()),
            "b_count": int((ds_b.cluster_labels == cid).sum()),
        }
    return overlap
