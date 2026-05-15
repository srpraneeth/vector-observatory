"""Vector Observatory — observability and debugging for embedding spaces."""

from .dataset import EmbeddingDataset
from .pipeline.pipeline import EmbeddingPipeline
from .reducers import UMAPReducer, TSNEReducer, PCAReducer
from .clustering import HDBSCANClusterer, DBSCANClusterer, KMeansClusterer
from .retrieval.knn import KNNIndex
from .drift.comparison import DriftComparison
from .storage.experiment import Experiment

__version__ = "0.1.0"

__all__ = [
    "EmbeddingDataset",
    "EmbeddingPipeline",
    "UMAPReducer",
    "TSNEReducer",
    "PCAReducer",
    "HDBSCANClusterer",
    "DBSCANClusterer",
    "KMeansClusterer",
    "KNNIndex",
    "DriftComparison",
    "Experiment",
]
