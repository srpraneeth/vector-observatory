from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..dataset import EmbeddingDataset
from ..reducers.umap import UMAPReducer
from ..clustering.hdbscan_ import HDBSCANClusterer
from .result import PipelineResult

if TYPE_CHECKING:
    from ..reducers.base import Reducer
    from ..clustering.base import Clusterer
    from ..storage.experiment import Experiment


class EmbeddingPipeline:
    """Fluent builder for the full embedding analysis pipeline.

    Example::

        result = (
            EmbeddingPipeline()
            .from_parquet("embeddings.parquet", id_col="id", embedding_col="vector")
            .reduce(UMAPReducer(n_neighbors=15))
            .cluster(HDBSCANClusterer(min_cluster_size=10))
            .compute_metrics()
            .store(project="my-project")
            .run()
        )
    """

    def __init__(self) -> None:
        self._dataset: EmbeddingDataset | None = None
        self._reducer: Reducer = UMAPReducer()
        self._clusterer: Clusterer = HDBSCANClusterer()
        self._should_compute_metrics: bool = True
        self._project_name: str | None = None

    # ------------------------------------------------------------------
    # Data source
    # ------------------------------------------------------------------

    def from_parquet(
        self,
        path: str | Path,
        id_col: str,
        embedding_col: str,
        metadata_cols: list[str] | None = None,
        name: str = "",
    ) -> EmbeddingPipeline:
        self._dataset = EmbeddingDataset.from_parquet(path, id_col, embedding_col, metadata_cols, name)
        return self

    def from_csv(
        self,
        path: str | Path,
        id_col: str,
        embedding_col: str,
        metadata_cols: list[str] | None = None,
        name: str = "",
    ) -> EmbeddingPipeline:
        self._dataset = EmbeddingDataset.from_csv(path, id_col, embedding_col, metadata_cols, name)
        return self

    def from_dataset(self, dataset: EmbeddingDataset) -> EmbeddingPipeline:
        self._dataset = dataset
        return self

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def reduce(self, reducer: Reducer) -> EmbeddingPipeline:
        self._reducer = reducer
        return self

    def cluster(self, clusterer: Clusterer) -> EmbeddingPipeline:
        self._clusterer = clusterer
        return self

    def compute_metrics(self, enabled: bool = True) -> EmbeddingPipeline:
        self._should_compute_metrics = enabled
        return self

    def store(self, experiment: str | Experiment) -> EmbeddingPipeline:
        self._project_name = experiment if isinstance(experiment, str) else experiment.name
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        if self._dataset is None:
            raise RuntimeError("No data source set. Call from_parquet(), from_csv(), or from_dataset() first.")

        run_id = str(uuid.uuid4())[:8]
        ds = self._dataset

        coords = self._reducer.fit_transform(ds.embeddings)
        ds = ds.with_reduction(coords)

        labels = self._clusterer.fit_predict(ds.embeddings)
        ds = ds.with_clusters(labels)

        geo_metrics = None
        clust_metrics = None
        if self._should_compute_metrics:
            from ..metrics.geometry import compute_geometry_metrics
            from ..metrics.cluster import compute_cluster_metrics
            geo_metrics = compute_geometry_metrics(ds.embeddings)
            clust_metrics = compute_cluster_metrics(ds.cluster_labels, ds.embeddings)

        config = {
            "run_id": run_id,
            "reducer": self._reducer.config,
            "clusterer": self._clusterer.config,
        }

        if self._project_name is not None:
            from ..storage.experiment import Experiment
            experiment = Experiment.load_or_create(self._project_name)
            experiment.store.save_run(ds, config, geo_metrics, clust_metrics)

        return PipelineResult(
            dataset=ds,
            geometry_metrics=geo_metrics,
            cluster_metrics=clust_metrics,
            run_id=run_id,
            config=config,
        )
