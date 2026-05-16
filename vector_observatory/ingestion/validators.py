from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd

from ..dataset import EmbeddingDataset


def _parse_embedding_column(series: pd.Series) -> np.ndarray:
    """Parse an embedding column that may be stored as a string, list, or array."""
    first = series.iloc[0]
    if isinstance(first, (list, np.ndarray)):
        return np.array(series.tolist(), dtype=np.float32)
    if isinstance(first, str):
        parsed = series.apply(lambda x: json.loads(x) if x.startswith("[") else ast.literal_eval(x))
        return np.array(parsed.tolist(), dtype=np.float32)
    raise ValueError(
        f"Cannot parse embedding column with dtype {series.dtype}. "
        "Expected list, numpy array, or JSON string."
    )


def _build_dataset(
    df: pd.DataFrame,
    id_col: str,
    embedding_col: str,
    metadata_cols: list[str] | None,
    name: str,
) -> EmbeddingDataset:
    if id_col not in df.columns:
        raise ValueError(f"ID column {id_col!r} not found. Available: {list(df.columns)}")
    if embedding_col not in df.columns:
        raise ValueError(
            f"Embedding column {embedding_col!r} not found. Available: {list(df.columns)}"
        )

    ids = df[id_col].to_numpy().astype(str)
    embeddings = _parse_embedding_column(df[embedding_col])

    if metadata_cols is not None:
        missing = [c for c in metadata_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Metadata columns not found: {missing}")
        metadata = df[metadata_cols].reset_index(drop=True)
    else:
        exclude = {id_col, embedding_col}
        metadata = df[[c for c in df.columns if c not in exclude]].reset_index(drop=True)

    return EmbeddingDataset(ids=ids, embeddings=embeddings, metadata=metadata, name=name)
