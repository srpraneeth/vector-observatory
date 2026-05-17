from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ..dataset import EmbeddingDataset

_SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS datasets (
    id           VARCHAR,
    dataset_name VARCHAR,
    embedding    DOUBLE[],
    metadata     JSON,
    PRIMARY KEY (id, dataset_name)
);

CREATE TABLE IF NOT EXISTS reductions (
    id           VARCHAR,
    dataset_name VARCHAR,
    run_id       VARCHAR,
    x            DOUBLE,
    y            DOUBLE,
    PRIMARY KEY (id, dataset_name, run_id)
);

CREATE TABLE IF NOT EXISTS clusters (
    id           VARCHAR,
    dataset_name VARCHAR,
    run_id       VARCHAR,
    cluster_id   INTEGER,
    is_noise     BOOLEAN,
    PRIMARY KEY (id, dataset_name, run_id)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id           VARCHAR PRIMARY KEY,
    dataset_name     VARCHAR,
    reducer_type     VARCHAR,
    reducer_config   JSON,
    clusterer_type   VARCHAR,
    clusterer_config JSON,
    geometry_metrics JSON,
    cluster_metrics  JSON,
    created_at       TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS drift_comparisons (
    comparison_id  VARCHAR PRIMARY KEY,
    dataset_a_name VARCHAR,
    dataset_b_name VARCHAR,
    reducer_config JSON,
    mmd_score      DOUBLE,
    created_at     TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS drift_reductions (
    id             VARCHAR,
    dataset_name   VARCHAR,
    comparison_id  VARCHAR,
    x              DOUBLE,
    y              DOUBLE,
    PRIMARY KEY (id, dataset_name, comparison_id)
);

CREATE TABLE IF NOT EXISTS cluster_labels (
    run_id     VARCHAR,
    cluster_id INTEGER,
    label      VARCHAR,
    PRIMARY KEY (run_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT now()
);
"""


def _to_json(obj: Any) -> str:
    """json.dumps with numpy scalar/array support."""

    def _default(o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return json.dumps(obj, default=_default)


def _metrics_to_dict(metrics) -> dict:
    """Serialize a metrics dataclass to a JSON-safe dict (converts ndarrays to lists)."""
    if metrics is None:
        return {}
    result = {}
    for k, v in vars(metrics).items():
        result[k] = v.tolist() if hasattr(v, "tolist") else v
    return result


class DuckDBStore:
    """DuckDB-backed storage for embedding datasets, runs, and drift comparisons."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._bootstrap()

    def _bootstrap(self) -> None:
        with self._cx() as conn:
            conn.execute(_DDL)
            version_rows = conn.execute("SELECT version FROM schema_version").fetchall()
            if not version_rows:
                conn.execute(f"INSERT INTO schema_version (version) VALUES ({_SCHEMA_VERSION})")

    @contextmanager
    def _cx(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        conn = duckdb.connect(str(self.path))
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Dataset CRUD
    # ------------------------------------------------------------------

    def save_dataset(self, dataset: EmbeddingDataset) -> None:
        rows = []
        for i, (id_val, emb) in enumerate(zip(dataset.ids, dataset.embeddings, strict=False)):
            meta = dataset.metadata.iloc[i].to_dict()
            rows.append((str(id_val), dataset.name, emb.tolist(), _to_json(meta)))
        with self._cx() as conn:
            conn.executemany("INSERT OR REPLACE INTO datasets VALUES (?, ?, ?, ?)", rows)

    def load_dataset(self, dataset_name: str) -> EmbeddingDataset:
        with self._cx() as conn:
            rows = conn.execute(
                "SELECT id, embedding, metadata FROM datasets WHERE dataset_name = ?",
                [dataset_name],
            ).fetchall()
        if not rows:
            raise KeyError(f"Dataset {dataset_name!r} not found.")

        ids = np.array([r[0] for r in rows])
        embeddings = np.array([r[1] for r in rows], dtype=np.float32)
        meta_records = [json.loads(r[2]) for r in rows]
        metadata = pd.DataFrame(meta_records)
        return EmbeddingDataset(
            ids=ids, embeddings=embeddings, metadata=metadata, name=dataset_name
        )

    def list_datasets(self) -> list[str]:
        with self._cx() as conn:
            rows = conn.execute(
                "SELECT DISTINCT dataset_name FROM datasets ORDER BY dataset_name"
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Run CRUD
    # ------------------------------------------------------------------

    def save_run(
        self,
        dataset: EmbeddingDataset,
        config: dict,
        geo_metrics=None,
        clust_metrics=None,
    ) -> None:
        run_id = config["run_id"]

        rows: list[tuple] = []
        for i, (id_val, emb) in enumerate(zip(dataset.ids, dataset.embeddings, strict=False)):
            meta = dataset.metadata.iloc[i].to_dict()
            rows.append((str(id_val), dataset.name, emb.tolist(), _to_json(meta)))

        reduction_rows: list[tuple] = []
        if dataset.reduced_coords is not None:
            reduction_rows = [
                (str(id_val), dataset.name, run_id, float(x), float(y))
                for id_val, (x, y) in zip(dataset.ids, dataset.reduced_coords, strict=False)
            ]

        cluster_rows: list[tuple] = []
        if dataset.cluster_labels is not None:
            cluster_rows = [
                (str(id_val), dataset.name, run_id, int(label), bool(label == -1))
                for id_val, label in zip(dataset.ids, dataset.cluster_labels, strict=False)
            ]

        reducer_cfg = config.get("reducer", {})
        clusterer_cfg = config.get("clusterer", {})

        with self._cx() as conn:
            conn.executemany("INSERT OR REPLACE INTO datasets VALUES (?, ?, ?, ?)", rows)
            if reduction_rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO reductions VALUES (?, ?, ?, ?, ?)", reduction_rows
                )
            if cluster_rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO clusters VALUES (?, ?, ?, ?, ?)", cluster_rows
                )
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, now())",
                [
                    run_id,
                    dataset.name,
                    reducer_cfg.get("type", ""),
                    _to_json(reducer_cfg),
                    clusterer_cfg.get("type", ""),
                    _to_json(clusterer_cfg),
                    _to_json(_metrics_to_dict(geo_metrics)),
                    _to_json(_metrics_to_dict(clust_metrics)),
                ],
            )

    def load_run(self, dataset_name: str, run_id: str) -> EmbeddingDataset:
        ds = self.load_dataset(dataset_name)

        with self._cx() as conn:
            coords_rows = conn.execute(
                "SELECT id, x, y FROM reductions WHERE dataset_name=? AND run_id=? ORDER BY id",
                [dataset_name, run_id],
            ).fetchall()
            label_rows = conn.execute(
                "SELECT id, cluster_id FROM clusters WHERE dataset_name=? AND run_id=? ORDER BY id",
                [dataset_name, run_id],
            ).fetchall()

        if coords_rows:
            id_to_coord = {r[0]: (r[1], r[2]) for r in coords_rows}
            coords = np.array([id_to_coord.get(str(i), (0, 0)) for i in ds.ids], dtype=np.float32)
            ds = ds.with_reduction(coords)

        if label_rows:
            id_to_label = {r[0]: r[1] for r in label_rows}
            labels = np.array([id_to_label.get(str(i), -1) for i in ds.ids], dtype=np.int32)
            ds = ds.with_clusters(labels)

        return ds

    def list_runs(self, dataset_name: str) -> list[dict]:
        with self._cx() as conn:
            rows = conn.execute(
                "SELECT run_id, reducer_type, clusterer_type, created_at FROM runs "
                "WHERE dataset_name=? ORDER BY created_at DESC",
                [dataset_name],
            ).fetchall()
        return [
            {"run_id": r[0], "reducer": r[1], "clusterer": r[2], "created_at": str(r[3])}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Cluster labels
    # ------------------------------------------------------------------

    def save_cluster_label(self, run_id: str, cluster_id: int, label: str) -> None:
        with self._cx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cluster_labels VALUES (?, ?, ?)",
                [run_id, cluster_id, label],
            )

    def delete_cluster_label(self, run_id: str, cluster_id: int) -> None:
        with self._cx() as conn:
            conn.execute(
                "DELETE FROM cluster_labels WHERE run_id=? AND cluster_id=?",
                [run_id, cluster_id],
            )

    def load_cluster_labels(self, run_id: str) -> dict[int, str]:
        with self._cx() as conn:
            rows = conn.execute(
                "SELECT cluster_id, label FROM cluster_labels WHERE run_id=?",
                [run_id],
            ).fetchall()
        return {int(r[0]): r[1] for r in rows}
