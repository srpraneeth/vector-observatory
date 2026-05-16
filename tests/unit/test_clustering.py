import numpy as np
import pytest

from tests.fixtures.synthetic import tight_clusters
from vector_observatory.clustering import DBSCANClusterer, HDBSCANClusterer, KMeansClusterer


@pytest.fixture
def clusterable():
    return tight_clusters(n=200, n_clusters=3, dim=16)


def test_hdbscan_output_shape(clusterable):
    labels = HDBSCANClusterer(min_cluster_size=5).fit_predict(clusterable.embeddings)
    assert labels.shape == (200,)
    assert labels.dtype == np.int32


def test_hdbscan_finds_clusters(clusterable):
    labels = HDBSCANClusterer(min_cluster_size=5).fit_predict(clusterable.embeddings)
    n_clusters = len(set(labels) - {-1})
    assert n_clusters >= 2


def test_dbscan_output_shape(clusterable):
    labels = DBSCANClusterer(eps=2.0, min_samples=3).fit_predict(clusterable.embeddings)
    assert labels.shape == (200,)


def test_kmeans_output_shape(clusterable):
    labels = KMeansClusterer(n_clusters=3).fit_predict(clusterable.embeddings)
    assert labels.shape == (200,)
    assert set(labels) == {0, 1, 2}  # no noise for kmeans


def test_kmeans_no_noise(clusterable):
    labels = KMeansClusterer(n_clusters=4).fit_predict(clusterable.embeddings)
    assert -1 not in labels


@pytest.mark.parametrize(
    "ClustererClass,kwargs",
    [
        (HDBSCANClusterer, {"min_cluster_size": 5}),
        (DBSCANClusterer, {"eps": 2.0}),
        (KMeansClusterer, {"n_clusters": 3}),
    ],
)
def test_config_has_type(ClustererClass, kwargs, clusterable):
    c = ClustererClass(**kwargs)
    assert "type" in c.config
