"""Synthetic dataset generators with known geometric properties.

Used in unit tests to validate metrics and transformations against
ground truth we control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vector_observatory.dataset import EmbeddingDataset


def isotropic_gaussian(n: int = 500, dim: int = 128, seed: int = 42) -> EmbeddingDataset:
    """Uniform Gaussian embeddings — expected: low anisotropy, high isotropy."""
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.array([f"id_{i}" for i in range(n)])
    metadata = pd.DataFrame({"index": np.arange(n)})
    return EmbeddingDataset(ids=ids, embeddings=embeddings, metadata=metadata, name="isotropic_gaussian")


def tight_clusters(n: int = 500, n_clusters: int = 5, dim: int = 64, seed: int = 42) -> EmbeddingDataset:
    """Well-separated Gaussian clusters — expected: clear cluster structure."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)) * 10
    base = n // n_clusters
    # distribute remainder so total is exactly n
    sizes = [base + (1 if i < n % n_clusters else 0) for i in range(n_clusters)]
    embeddings, labels = [], []
    for i, (center, size) in enumerate(zip(centers, sizes)):
        pts = center + rng.standard_normal((size, dim)) * 0.5
        embeddings.append(pts)
        labels.extend([i] * size)
    embeddings = np.vstack(embeddings).astype(np.float32)
    ids = np.array([f"id_{i}" for i in range(len(embeddings))])
    metadata = pd.DataFrame({"label": labels})
    return EmbeddingDataset(ids=ids, embeddings=embeddings, metadata=metadata, name="tight_clusters")


def collapsed_embeddings(n: int = 500, dim: int = 128, seed: int = 42) -> EmbeddingDataset:
    """All embeddings in a narrow cone — expected: anisotropy near 1.0."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(dim)
    base /= np.linalg.norm(base)
    noise = rng.standard_normal((n, dim)) * 0.01
    embeddings = (base + noise).astype(np.float32)
    ids = np.array([f"id_{i}" for i in range(n)])
    metadata = pd.DataFrame({"index": np.arange(n)})
    return EmbeddingDataset(ids=ids, embeddings=embeddings, metadata=metadata, name="collapsed")


def two_dataset_pair(
    n: int = 300,
    dim: int = 64,
    shift: float = 5.0,
    seed: int = 42,
) -> tuple[EmbeddingDataset, EmbeddingDataset]:
    """Two related datasets with a controlled distribution shift for drift testing."""
    rng = np.random.default_rng(seed)
    emb_a = rng.standard_normal((n, dim)).astype(np.float32)
    emb_b = (rng.standard_normal((n, dim)) + shift).astype(np.float32)
    ids_a = np.array([f"a_{i}" for i in range(n)])
    ids_b = np.array([f"b_{i}" for i in range(n)])
    meta = pd.DataFrame({"index": np.arange(n)})
    ds_a = EmbeddingDataset(ids=ids_a, embeddings=emb_a, metadata=meta.copy(), name="dataset_a")
    ds_b = EmbeddingDataset(ids=ids_b, embeddings=emb_b, metadata=meta.copy(), name="dataset_b")
    return ds_a, ds_b
