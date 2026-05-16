from __future__ import annotations

from dataclasses import dataclass, field

# ------------------------------------------------------------------
# Reducer configs
# ------------------------------------------------------------------


@dataclass
class UMAPConfig:
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 42


@dataclass
class TSNEConfig:
    n_components: int = 2
    perplexity: float = 30.0
    metric: str = "cosine"
    random_state: int = 42
    max_iter: int = 1000


@dataclass
class PCAConfig:
    n_components: int = 2


# ------------------------------------------------------------------
# Clusterer configs
# ------------------------------------------------------------------


@dataclass
class HDBSCANConfig:
    min_cluster_size: int = 10
    min_samples: int = 5
    metric: str = "euclidean"


@dataclass
class DBSCANConfig:
    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"


@dataclass
class KMeansConfig:
    n_clusters: int = 8
    random_state: int = 42


# ------------------------------------------------------------------
# Pipeline config
# ------------------------------------------------------------------


@dataclass
class PipelineConfig:
    reducer: UMAPConfig | TSNEConfig | PCAConfig = field(default_factory=UMAPConfig)
    clusterer: HDBSCANConfig | DBSCANConfig | KMeansConfig = field(default_factory=HDBSCANConfig)
    compute_metrics: bool = True
    build_index: bool = True
