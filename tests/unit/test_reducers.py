import numpy as np
import pytest

from tests.fixtures.synthetic import isotropic_gaussian
from vector_observatory.reducers import PCAReducer, TSNEReducer, UMAPReducer


@pytest.fixture
def small_ds():
    return isotropic_gaussian(n=100, dim=32)


def test_pca_output_shape(small_ds):
    reducer = PCAReducer(n_components=2)
    coords = reducer.fit_transform(small_ds.embeddings)
    assert coords.shape == (100, 2)
    assert coords.dtype == np.float32


def test_pca_transform(small_ds):
    reducer = PCAReducer(n_components=2)
    reducer.fit_transform(small_ds.embeddings)
    new_pts = np.random.rand(5, 32).astype(np.float32)
    out = reducer.transform(new_pts)
    assert out.shape == (5, 2)


def test_pca_config(small_ds):
    reducer = PCAReducer(n_components=3)
    reducer.fit_transform(small_ds.embeddings)
    assert reducer.config["type"] == "pca"
    assert reducer.config["n_components"] == 3


def test_tsne_output_shape(small_ds):
    reducer = TSNEReducer(n_components=2, max_iter=250)
    coords = reducer.fit_transform(small_ds.embeddings)
    assert coords.shape == (100, 2)


def test_tsne_transform_raises(small_ds):
    reducer = TSNEReducer()
    reducer.fit_transform(small_ds.embeddings)
    with pytest.raises(NotImplementedError):
        reducer.transform(small_ds.embeddings[:5])


def test_umap_output_shape(small_ds):
    reducer = UMAPReducer(n_components=2, n_neighbors=10)
    coords = reducer.fit_transform(small_ds.embeddings)
    assert coords.shape == (100, 2)
    assert coords.dtype == np.float32


def test_umap_config(small_ds):
    reducer = UMAPReducer(n_neighbors=20, min_dist=0.2)
    assert reducer.config["n_neighbors"] == 20
    assert reducer.config["min_dist"] == 0.2
