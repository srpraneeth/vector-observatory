"""Vector Observatory — observability and debugging for embedding spaces."""

from .clustering import DBSCANClusterer, HDBSCANClusterer, KMeansClusterer
from .dataset import EmbeddingDataset
from .drift.comparison import DriftComparison
from .pipeline.pipeline import EmbeddingPipeline
from .reducers import PCAReducer, TSNEReducer, UMAPReducer
from .retrieval.knn import KNNIndex
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
