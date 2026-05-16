import numpy as np
import pytest

from tests.fixtures.synthetic import collapsed_embeddings, isotropic_gaussian
from vector_observatory.metrics.cluster import compute_cluster_metrics
from vector_observatory.metrics.geometry import compute_geometry_metrics


class TestGeometryMetrics:
    def test_isotropic_low_anisotropy(self):
        ds = isotropic_gaussian(n=300, dim=64)
        metrics = compute_geometry_metrics(ds.embeddings)
        assert metrics.anisotropy < 0.3, "Isotropic Gaussian should have low anisotropy"

    def test_collapsed_high_anisotropy(self):
        ds = collapsed_embeddings(n=300, dim=64)
        metrics = compute_geometry_metrics(ds.embeddings)
        assert metrics.anisotropy > 0.8, "Collapsed embeddings should have high anisotropy"

    def test_variance_per_dim_shape(self):
        ds = isotropic_gaussian(n=100, dim=32)
        metrics = compute_geometry_metrics(ds.embeddings)
        assert metrics.variance_per_dim.shape == (32,)

    def test_intrinsic_dim_positive(self):
        ds = isotropic_gaussian(n=200, dim=64)
        metrics = compute_geometry_metrics(ds.embeddings)
        assert metrics.intrinsic_dim > 0

    def test_isotropy_score_positive(self):
        ds = isotropic_gaussian(n=200, dim=32)
        metrics = compute_geometry_metrics(ds.embeddings)
        assert metrics.isotropy_score >= 0


class TestClusterMetrics:
    def test_basic_counts(self):
        labels = np.array([0, 0, 1, 1, 2, -1, -1])
        m = compute_cluster_metrics(labels)
        assert m.n_clusters == 3
        assert m.noise_fraction == pytest.approx(2 / 7)

    def test_all_noise(self):
        labels = np.full(10, -1, dtype=np.int32)
        m = compute_cluster_metrics(labels)
        assert m.n_clusters == 0
        assert m.noise_fraction == 1.0

    def test_no_noise(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        m = compute_cluster_metrics(labels)
        assert m.noise_fraction == 0.0

    def test_cluster_sizes(self):
        labels = np.array([0, 0, 0, 1, 1])
        m = compute_cluster_metrics(labels)
        assert m.cluster_sizes[0] == 3
        assert m.cluster_sizes[1] == 2
