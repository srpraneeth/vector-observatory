from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..dataset import EmbeddingDataset
from .validators import _build_dataset


def load_parquet(
    path: str | Path,
    id_col: str,
    embedding_col: str,
    metadata_cols: list[str] | None = None,
    name: str = "",
) -> EmbeddingDataset:
    df = pd.read_parquet(path)
    return _build_dataset(df, id_col, embedding_col, metadata_cols, name or Path(path).stem)
