from __future__ import annotations

from dataclasses import dataclass

from ..dataset import EmbeddingDataset
from ..metrics.cluster import ClusterMetrics
from ..metrics.geometry import GeometryMetrics


@dataclass
class PipelineResult:
    dataset: EmbeddingDataset
    geometry_metrics: GeometryMetrics | None
    cluster_metrics: ClusterMetrics | None
    run_id: str
    config: dict
