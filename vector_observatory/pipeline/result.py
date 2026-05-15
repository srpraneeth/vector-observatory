from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..dataset import EmbeddingDataset
from ..metrics.geometry import GeometryMetrics
from ..metrics.cluster import ClusterMetrics


@dataclass
class PipelineResult:
    dataset: EmbeddingDataset
    geometry_metrics: GeometryMetrics | None
    cluster_metrics: ClusterMetrics | None
    run_id: str
    config: dict
