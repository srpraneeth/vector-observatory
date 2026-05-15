from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmbeddingDataset:
    """Immutable container for embeddings, IDs, and metadata.

    All transforms (filter, with_reduction, with_clusters) return new instances.
    """

    ids: np.ndarray           # (N,) primary keys
    embeddings: np.ndarray    # (N, D) float32
    metadata: pd.DataFrame    # (N, M) user-defined columns
    name: str = ""
    reduced_coords: np.ndarray | None = None   # (N, 2) after reduction
    cluster_labels: np.ndarray | None = None   # (N,) int; -1 = noise

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1]

    @property
    def n_clusters(self) -> int:
        if self.cluster_labels is None:
            return 0
        valid = self.cluster_labels[self.cluster_labels >= 0]
        return int(valid.max()) + 1 if len(valid) > 0 else 0

    @property
    def noise_fraction(self) -> float:
        if self.cluster_labels is None:
            return 0.0
        return float((self.cluster_labels == -1).sum() / self.n_samples)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        id_col: str,
        embedding_col: str,
        metadata_cols: list[str] | None = None,
        name: str = "",
    ) -> EmbeddingDataset:
        from .ingestion.parquet import load_parquet
        return load_parquet(path, id_col, embedding_col, metadata_cols, name)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        id_col: str,
        embedding_col: str,
        metadata_cols: list[str] | None = None,
        name: str = "",
    ) -> EmbeddingDataset:
        from .ingestion.csv import load_csv
        return load_csv(path, id_col, embedding_col, metadata_cols, name)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        id_col: str,
        embedding_col: str,
        metadata_cols: list[str] | None = None,
        name: str = "",
    ) -> EmbeddingDataset:
        from .ingestion.json_ import load_json
        return load_json(path, id_col, embedding_col, metadata_cols, name)

    # ------------------------------------------------------------------
    # Transforms (return new instances)
    # ------------------------------------------------------------------

    def with_reduction(self, coords: np.ndarray) -> EmbeddingDataset:
        return EmbeddingDataset(
            ids=self.ids,
            embeddings=self.embeddings,
            metadata=self.metadata,
            name=self.name,
            reduced_coords=coords,
            cluster_labels=self.cluster_labels,
        )

    def with_clusters(self, labels: np.ndarray) -> EmbeddingDataset:
        return EmbeddingDataset(
            ids=self.ids,
            embeddings=self.embeddings,
            metadata=self.metadata,
            name=self.name,
            reduced_coords=self.reduced_coords,
            cluster_labels=labels,
        )

    def rename(self, name: str) -> EmbeddingDataset:
        return EmbeddingDataset(
            ids=self.ids,
            embeddings=self.embeddings,
            metadata=self.metadata,
            name=name,
            reduced_coords=self.reduced_coords,
            cluster_labels=self.cluster_labels,
        )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(self, mask: np.ndarray) -> EmbeddingDataset:
        """Return a new dataset containing only rows where mask is True."""
        mask = mask.astype(bool)
        return EmbeddingDataset(
            ids=self.ids[mask],
            embeddings=self.embeddings[mask],
            metadata=self.metadata.iloc[mask].reset_index(drop=True),
            name=self.name,
            reduced_coords=self.reduced_coords[mask] if self.reduced_coords is not None else None,
            cluster_labels=self.cluster_labels[mask] if self.cluster_labels is not None else None,
        )

    def filter_by_metadata(self, column: str, values: list) -> EmbeddingDataset:
        mask = self.metadata[column].isin(values).to_numpy()
        return self.filter(mask)

    def filter_by_cluster(self, cluster_id: int) -> EmbeddingDataset:
        if self.cluster_labels is None:
            raise ValueError("Dataset has no cluster labels. Run clustering first.")
        return self.filter(self.cluster_labels == cluster_id)

    def filter_by_range(self, column: str, min_val: float, max_val: float) -> EmbeddingDataset:
        col = self.metadata[column]
        mask = ((col >= min_val) & (col <= max_val)).to_numpy()
        return self.filter(mask)

    def search_metadata(self, query: str, columns: list[str] | None = None) -> EmbeddingDataset:
        """Substring search across text metadata columns."""
        cols = columns or list(self.metadata.select_dtypes(include="object").columns)
        mask = pd.Series([False] * len(self.metadata))
        for col in cols:
            mask |= self.metadata[col].astype(str).str.contains(query, case=False, na=False)
        return self.filter(mask.to_numpy())

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def index_of(self, id_value) -> int:
        """Return the integer position of a given ID."""
        matches = np.where(self.ids == id_value)[0]
        if len(matches) == 0:
            raise KeyError(f"ID {id_value!r} not found in dataset")
        return int(matches[0])

    def __repr__(self) -> str:
        reduced = self.reduced_coords is not None
        clustered = self.cluster_labels is not None
        return (
            f"EmbeddingDataset(name={self.name!r}, n={self.n_samples}, "
            f"dim={self.dim}, reduced={reduced}, clustered={clustered})"
        )
