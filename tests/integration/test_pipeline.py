"""End-to-end pipeline integration test — parquet → reduce → cluster → store → reload."""

import pandas as pd
import pytest

from tests.fixtures.synthetic import tight_clusters
from vector_observatory.clustering import KMeansClusterer
from vector_observatory.pipeline.pipeline import EmbeddingPipeline
from vector_observatory.reducers import PCAReducer
from vector_observatory.storage.experiment import Experiment


@pytest.fixture
def parquet_file(tmp_path):
    ds = tight_clusters(n=100, n_clusters=3, dim=16)
    df = pd.DataFrame(
        {
            "id": ds.ids,
            "embedding": [row.tolist() for row in ds.embeddings],
            "label": ds.metadata["label"].tolist(),
        }
    )
    path = tmp_path / "test.parquet"
    df.to_parquet(path, index=False)
    return path


def test_full_pipeline(parquet_file, tmp_path):
    result = (
        EmbeddingPipeline()
        .from_parquet(parquet_file, id_col="id", embedding_col="embedding", metadata_cols=["label"])
        .reduce(PCAReducer(n_components=2))
        .cluster(KMeansClusterer(n_clusters=3))
        .compute_metrics()
        .store(Experiment.load_or_create("integration_test", experiments_dir=tmp_path))
        .run()
    )

    assert result.dataset.n_samples == 100
    assert result.dataset.reduced_coords is not None
    assert result.dataset.cluster_labels is not None
    assert result.geometry_metrics is not None
    assert result.cluster_metrics is not None
    assert result.cluster_metrics.n_clusters == 3


def test_pipeline_without_store(parquet_file):
    result = (
        EmbeddingPipeline()
        .from_parquet(parquet_file, id_col="id", embedding_col="embedding")
        .reduce(PCAReducer())
        .cluster(KMeansClusterer(n_clusters=2))
        .run()
    )
    assert result.dataset.reduced_coords is not None
    assert result.run_id is not None
