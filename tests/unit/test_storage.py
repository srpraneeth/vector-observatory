import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.synthetic import isotropic_gaussian, tight_clusters
from vector_observatory.clustering import KMeansClusterer
from vector_observatory.reducers import PCAReducer
from vector_observatory.storage.experiment import Experiment
from vector_observatory.storage.store import DuckDBStore


@pytest.fixture
def tmp_experiment(tmp_path):
    return Experiment.load_or_create("test", experiments_dir=tmp_path)


def test_save_and_load_dataset(tmp_experiment):
    ds = isotropic_gaussian(n=50, dim=16)
    tmp_experiment.store.save_dataset(ds)
    loaded = tmp_experiment.store.load_dataset(ds.name)
    assert loaded.n_samples == ds.n_samples
    assert loaded.dim == ds.dim
    np.testing.assert_allclose(loaded.embeddings, ds.embeddings, atol=1e-5)


def test_save_and_load_run(tmp_experiment):
    ds = isotropic_gaussian(n=50, dim=16)
    coords = PCAReducer().fit_transform(ds.embeddings)
    ds = ds.with_reduction(coords)
    labels = KMeansClusterer(n_clusters=3).fit_predict(ds.embeddings)
    ds = ds.with_clusters(labels)

    config = {"run_id": "abc123", "reducer": {"type": "pca"}, "clusterer": {"type": "kmeans"}}
    tmp_experiment.store.save_run(ds, config)

    loaded = tmp_experiment.store.load_run(ds.name, "abc123")
    assert loaded.reduced_coords is not None
    assert loaded.cluster_labels is not None
    assert loaded.n_samples == ds.n_samples


def test_list_datasets(tmp_experiment):
    ds = isotropic_gaussian(n=20, dim=8)
    tmp_experiment.store.save_dataset(ds)
    names = tmp_experiment.store.list_datasets()
    assert ds.name in names


def test_list_runs(tmp_experiment):
    ds = isotropic_gaussian(n=20, dim=8)
    config = {"run_id": "run1", "reducer": {"type": "pca"}, "clusterer": {"type": "kmeans"}}
    tmp_experiment.store.save_run(ds, config)
    runs = tmp_experiment.store.list_runs(ds.name)
    assert any(r["run_id"] == "run1" for r in runs)


def test_project_list_all(tmp_path):
    Experiment.load_or_create("alpha", experiments_dir=tmp_path)
    Experiment.load_or_create("beta", experiments_dir=tmp_path)
    names = Experiment.list_all(experiments_dir=tmp_path)
    assert "alpha" in names and "beta" in names


def test_project_delete(tmp_path):
    Experiment.load_or_create("to_delete", experiments_dir=tmp_path)
    Experiment.delete("to_delete", experiments_dir=tmp_path)
    names = Experiment.list_all(experiments_dir=tmp_path)
    assert "to_delete" not in names
