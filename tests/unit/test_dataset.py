import numpy as np
import pytest

from tests.fixtures.synthetic import isotropic_gaussian, tight_clusters
from vector_observatory.dataset import EmbeddingDataset


def test_basic_properties():
    ds = isotropic_gaussian(n=100, dim=32)
    assert ds.n_samples == 100
    assert ds.dim == 32
    assert ds.reduced_coords is None
    assert ds.cluster_labels is None


def test_with_reduction_immutable():
    ds = isotropic_gaussian(n=50, dim=16)
    coords = np.random.rand(50, 2).astype(np.float32)
    ds2 = ds.with_reduction(coords)
    assert ds.reduced_coords is None        # original unchanged
    assert ds2.reduced_coords is not None


def test_with_clusters_immutable():
    ds = isotropic_gaussian(n=50, dim=16)
    labels = np.zeros(50, dtype=np.int32)
    ds2 = ds.with_clusters(labels)
    assert ds.cluster_labels is None
    assert ds2.cluster_labels is not None


def test_filter_by_metadata():
    ds = tight_clusters(n=100, n_clusters=2)
    filtered = ds.filter_by_metadata("label", [0])
    assert all(filtered.metadata["label"] == 0)
    assert filtered.n_samples < ds.n_samples


def test_filter_by_cluster():
    ds = tight_clusters(n=100, n_clusters=3)
    labels = ds.metadata["label"].to_numpy()
    ds = ds.with_clusters(labels)
    cluster_ds = ds.filter_by_cluster(0)
    assert (cluster_ds.cluster_labels == 0).all()


def test_filter_preserves_alignment():
    ds = isotropic_gaussian(n=100, dim=32)
    coords = np.random.rand(100, 2).astype(np.float32)
    labels = np.random.randint(0, 3, 100).astype(np.int32)
    ds = ds.with_reduction(coords).with_clusters(labels)

    mask = np.zeros(100, dtype=bool)
    mask[:30] = True
    filtered = ds.filter(mask)

    assert filtered.n_samples == 30
    assert filtered.embeddings.shape == (30, 32)
    assert filtered.reduced_coords.shape == (30, 2)
    assert filtered.cluster_labels.shape == (30,)
    assert len(filtered.metadata) == 30


def test_search_metadata():
    import pandas as pd
    ids = np.array(["a", "b", "c"])
    embs = np.random.rand(3, 8).astype(np.float32)
    meta = pd.DataFrame({"text": ["hello world", "foo bar", "hello there"]})
    ds = EmbeddingDataset(ids=ids, embeddings=embs, metadata=meta)
    result = ds.search_metadata("hello")
    assert result.n_samples == 2


def test_n_clusters_excludes_noise():
    ds = isotropic_gaussian(n=10, dim=4)
    labels = np.array([-1, -1, 0, 0, 1, 1, 2, 2, -1, -1], dtype=np.int32)
    ds = ds.with_clusters(labels)
    assert ds.n_clusters == 3
    assert ds.noise_fraction == 0.4
