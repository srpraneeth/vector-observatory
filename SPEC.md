# Vector Observatory — Technical Specification

> Architecture and implementation reference. For project overview, goals, and usage see [README.md](README.md).

---

## Table of Contents

1. [Core Concepts](#1-core-concepts)
2. [Architecture](#2-architecture)
3. [Package Structure](#3-package-structure)
4. [Data Model](#4-data-model)
5. [Core API Design](#5-core-api-design)
6. [Storage Design](#6-storage-design)
7. [Streamlit App Design](#7-streamlit-app-design)
8. [Scale & Backend Strategy](#8-scale--backend-strategy)
9. [Testing Strategy](#9-testing-strategy)
10. [Packaging & Distribution](#10-packaging--distribution)

---

## 1. Core Concepts

### EmbeddingDataset
The central unit of work. An immutable container of IDs, embedding vectors, and metadata. Everything in the library operates on or produces an `EmbeddingDataset`.

### Experiment
A named, persistent workspace backed by a single DuckDB file on disk. A project stores one or more datasets, their reduction/cluster results, run configs, and computed metrics.

### Run
A single execution of the analysis pipeline (reduce + cluster + metrics) against a dataset. Runs are stored with their full config so results are reproducible.

### Reducer
A dimensionality reduction algorithm that maps `(N, D)` embeddings to `(N, 2)` coordinates for visualization. Interchangeable via a shared Protocol interface.

### Clusterer
An algorithm that assigns cluster labels to embeddings. Operates in the original high-dimensional space, not on reduced coordinates.

### DriftComparison
A side-by-side analysis of two datasets. A single reducer is fit on the combined embeddings, then both datasets are transformed through it, placing them in a shared coordinate space for visual comparison.

### KNNIndex
A nearest-neighbor index built over a dataset's embeddings. Used for point inspection and neighbor panels.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit App                      │
│  (upload → call library → render → interact)        │
└────────────────────────┬────────────────────────────┘
                         │ calls
┌────────────────────────▼────────────────────────────┐
│              vector_observatory (library)            │
│                                                     │
│  ingestion → EmbeddingDataset → pipeline            │
│                                                     │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐ │
│  │ reducers │  │ clustering│  │     metrics      │ │
│  └──────────┘  └───────────┘  └──────────────────┘ │
│                                                     │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐ │
│  │retrieval │  │   drift   │  │   visualization  │ │
│  └──────────┘  └───────────┘  └──────────────────┘ │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │             storage (DuckDB)                 │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**Key discipline:** The visualization module returns `plotly.graph_objects.Figure` objects. It never calls `st.*`. The Streamlit layer only passes data in and renders figures out.

---

## 3. Package Structure

```
vector-observatory/
│
├── vector_observatory/
│   ├── __init__.py               # public API surface
│   ├── dataset.py                # EmbeddingDataset
│   ├── schema.py                 # config dataclasses
│   │
│   ├── ingestion/
│   │   ├── parquet.py
│   │   ├── csv.py
│   │   ├── json_.py
│   │   └── validators.py         # column detection, embedding validation
│   │
│   ├── reducers/
│   │   ├── base.py               # Reducer protocol
│   │   ├── umap.py
│   │   ├── tsne.py
│   │   └── pca.py
│   │
│   ├── clustering/
│   │   ├── base.py               # Clusterer protocol
│   │   ├── hdbscan_.py
│   │   ├── dbscan.py
│   │   └── kmeans.py
│   │
│   ├── metrics/
│   │   ├── geometry.py           # anisotropy, isotropy, intrinsic_dim
│   │   └── cluster.py            # cluster stats, noise fraction
│   │
│   ├── retrieval/
│   │   └── knn.py                # KNNIndex (brute-force sklearn)
│   │
│   ├── drift/
│   │   └── comparison.py         # DriftComparison, DriftResult
│   │
│   ├── storage/
│   │   ├── experiment.py            # Experiment (create/load/list/delete)
│   │   └── store.py              # DuckDBStore (short-lived connections)
│   │
│   ├── visualization/
│   │   ├── scatter.py            # 2D scatter figure builder
│   │   ├── cluster_map.py        # cluster overview charts
│   │   ├── drift_overlay.py      # two-dataset overlay figure
│   │   └── metrics_chart.py      # geometry metric charts
│   │
│   └── pipeline/
│       ├── pipeline.py           # EmbeddingPipeline (fluent builder)
│       └── result.py             # PipelineResult
│
├── app/
│   ├── app.py                    # entry point, sidebar, navigation
│   ├── state.py                  # typed session state wrapper (AppState)
│   ├── views/
│   │   ├── experiment_list.py
│   │   ├── new_experiment.py
│   │   ├── detail.py
│   │   ├── tab_overview.py
│   │   ├── tab_explore.py
│   │   ├── tab_clusters.py
│   │   ├── tab_metrics.py
│   │   └── tab_drift.py
│   └── components/
│       ├── filter_bar.py
│       └── neighbor_panel.py
│
├── data/
│   ├── movies_demo.parquet       # demo source file
│   └── projects/                 # DuckDB experiment files (gitignored)
│
├── scripts/
│   └── generate_demo.py          # regenerate movies demo experiment
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│       └── synthetic.py          # known-property dataset generators
│
└── examples/
    ├── basic_analysis.py
    ├── drift_comparison.py
    └── neighbor_inspection.py
```

---

## 4. Data Model

### Input Contract

Users provide a file (parquet, CSV, JSON) with:

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| ID column | any scalar | Yes | User specifies column name on upload |
| Embedding column | list/array of floats | Yes | User specifies column name on upload |
| Metadata columns | any | No | User selects which columns to include |

Supported embedding dimensions: any. Tested targets: 384, 768, 1536, 3072.

### EmbeddingDataset

```python
@dataclass(frozen=True)
class EmbeddingDataset:
    ids: np.ndarray           # shape (N,), dtype str
    embeddings: np.ndarray    # shape (N, D), dtype float32
    metadata: pd.DataFrame    # shape (N, M), user-defined columns
    name: str

    reduced_coords: np.ndarray | None  # shape (N, 2), populated after reduction
    cluster_labels: np.ndarray | None  # shape (N,), int; -1 = noise

    # Computed properties
    n_samples: int
    dim: int
    n_clusters: int            # excludes noise label -1
    noise_fraction: float

    # Returns new instance, never mutates
    def with_reduction(self, coords: np.ndarray) -> EmbeddingDataset: ...
    def with_clusters(self, labels: np.ndarray) -> EmbeddingDataset: ...
    def filter(self, mask: np.ndarray) -> EmbeddingDataset: ...
    def filter_by_metadata(self, column, values) -> EmbeddingDataset: ...
    def filter_by_cluster(self, cluster_id: int) -> EmbeddingDataset: ...
    def filter_by_range(self, column, min_val, max_val) -> EmbeddingDataset: ...
    def search_metadata(self, query: str, columns=None) -> EmbeddingDataset: ...
```

### Config Dataclasses

```python
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
    max_iter: int = 1000       # renamed from n_iter in sklearn 1.5
    metric: str = "cosine"
    random_state: int = 42

@dataclass
class PCAConfig:
    n_components: int = 2

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
```

### DriftResult

```python
@dataclass(frozen=True)
class DriftResult:
    dataset_a: EmbeddingDataset   # with shared reduced coords
    dataset_b: EmbeddingDataset   # with shared reduced coords
    mmd_score: float              # Maximum Mean Discrepancy
    reducer_config: dict
    cluster_overlap: dict         # {cluster_id: {"a_count": int, "b_count": int}}
```

---

## 5. Core API Design

### Protocols

All reducers and clusterers satisfy structural protocols — no forced inheritance. Enables drop-in GPU replacements without a rewrite.

```python
class Reducer(Protocol):
    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    @property
    def config(self) -> dict: ...

class Clusterer(Protocol):
    def fit_predict(self, X: np.ndarray) -> np.ndarray: ...
    @property
    def config(self) -> dict: ...
```

### Pipeline (fluent builder)

```python
result = (
    EmbeddingPipeline()
    .from_parquet("embeddings.parquet", id_col="id", embedding_col="vector",
                  metadata_cols=["title", "category"])
    .reduce(UMAPReducer(n_components=2, n_neighbors=15))
    .cluster(HDBSCANClusterer(min_cluster_size=10))
    .compute_metrics()
    .store(Experiment.load_or_create("my-project"))
    .run()
)
# result.dataset   — EmbeddingDataset with coords + labels populated
# result.run_id    — str, use to reload this exact run
# result.geometry_metrics, result.cluster_metrics
```

### Metrics

```python
geo = compute_geometry_metrics(embeddings)
# geo.anisotropy        — avg cosine sim between random pairs; >0.5 = collapse risk
# geo.isotropy_score    — min/max partition function ratio (Mu & Viswanath 2018); 0–1
# geo.intrinsic_dim     — Two-NN estimate of effective dimensionality (Facco et al. 2017)
# geo.variance_per_dim  — per-dimension variance array

clust = compute_cluster_metrics(cluster_labels)
# clust.n_clusters, clust.noise_fraction, clust.largest_cluster_fraction
```

---

## 6. Storage Design

### Layout

Each experiment is a single DuckDB file in `data/experiments/`. Files are self-contained and portable — copy or share them freely.

```
data/experiments/
├── movies-demo.duckdb
├── my-experiment.duckdb
└── ...
```

### DuckDB Schema

```sql
CREATE TABLE datasets (
    id           VARCHAR,
    dataset_name VARCHAR,
    embedding    DOUBLE[],
    metadata     JSON,
    PRIMARY KEY (id, dataset_name)
);

CREATE TABLE reductions (
    id           VARCHAR,
    dataset_name VARCHAR,
    run_id       VARCHAR,
    x            DOUBLE,
    y            DOUBLE,
    PRIMARY KEY (id, dataset_name, run_id)
);

CREATE TABLE clusters (
    id           VARCHAR,
    dataset_name VARCHAR,
    run_id       VARCHAR,
    cluster_id   INTEGER,
    is_noise     BOOLEAN,
    PRIMARY KEY (id, dataset_name, run_id)
);

CREATE TABLE runs (
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

CREATE TABLE drift_comparisons (
    comparison_id  VARCHAR PRIMARY KEY,
    dataset_a_name VARCHAR,
    dataset_b_name VARCHAR,
    reducer_config JSON,
    mmd_score      DOUBLE,
    created_at     TIMESTAMP DEFAULT now()
);

CREATE TABLE drift_reductions (
    id             VARCHAR,
    dataset_name   VARCHAR,
    comparison_id  VARCHAR,
    x              DOUBLE,
    y              DOUBLE,
    PRIMARY KEY (id, dataset_name, comparison_id)
);

CREATE TABLE schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT now()
);
```

### Connection Strategy

`DuckDBStore` uses short-lived connections — each method opens a connection, executes, and closes immediately. This avoids file lock conflicts with Streamlit's hot-reload process spawning multiple Python workers.

### Why DuckDB

- Single-file, zero-server — works on a laptop, no infra required
- Native array columns for embeddings
- SQL filtering over metadata JSON via `json_extract`
- Analytical aggregations for cluster stats and metric history
- Trivially portable: copy one file to share an experiment

DuckDB is **not** used for kNN search. The `KNNIndex` is in-memory sklearn in v0.x, FAISS in v1.0.

---

## 7. Streamlit App Design

### Session State

All session state lives in a typed `AppState` wrapper (`app/state.py`). Views never access `st.session_state` directly.

```python
@dataclass
class AppState:
    view: str                          # "list" | "new" | "detail"
    active_experiment: Experiment | None
    active_dataset: EmbeddingDataset | None
    active_run_id: str | None
    knn_index: KNNIndex | None
    filter_mask: np.ndarray | None
    selected_point_id: Any
```

### Navigation Flow

```
experiment_list  →  new_experiment  →  detail
                                         ├── tab_overview
                                         ├── tab_explore
                                         ├── tab_clusters
                                         ├── tab_metrics
                                         └── tab_drift
```

### Filter Spec

Filters are composable — all active filters are AND'd. Applied to `state.visible_dataset`, which is recomputed on every rerun.

| Column type | Widget |
|-------------|--------|
| Categorical (≤20 unique values) | `st.multiselect` |
| Numeric | `st.slider` with min/max from data |
| Free text | `st.text_input` with substring match |

---

## 8. Scale & Backend Strategy

### v0.x — CPU single node

Default backend uses scikit-learn, umap-learn, hdbscan. Practical limits:

- UMAP: ~200K embeddings in reasonable time on a modern laptop
- HDBSCAN: ~500K embeddings
- kNN brute-force: ~100K before it becomes slow

For larger datasets: subsample on load, warn the user.

### v1.0 — GPU single node (CUDA)

RAPIDS/cuML provides drop-in GPU versions of UMAP, DBSCAN, K-Means. The Protocol interface means this is a backend swap:

```python
# CPU
from vector_observatory.reducers import UMAPReducer
# GPU (CUDA only, Linux, conda install)
from vector_observatory.reducers.gpu import RAPIDSUMAPReducer
```

### v2.0 — Multi-node

Dask or Ray for distributed processing. FAISS GPU index for large-scale retrieval.

---

## 9. Testing Strategy

### Principles

- Unit tests use synthetic datasets with known geometric properties
- Parametrize across all Reducer and Clusterer implementations
- Integration tests run a full pipeline against a real DuckDB file
- No mocking of the database — integration tests hit real DuckDB

### Synthetic Dataset Generators

All in `tests/fixtures/synthetic.py`:

| Generator | Known properties |
|-----------|-----------------|
| `isotropic_gaussian(n, dim)` | Low anisotropy, isotropy score ~1.0 |
| `tight_clusters(n, n_clusters, dim)` | Clear cluster structure, HDBSCAN recovers clusters |
| `collapsed_embeddings(n, dim)` | Anisotropy near 1.0 |
| `two_dataset_pair(n, dim, shift)` | Controlled distribution shift for drift testing |

### CI

- GitHub Actions: Python 3.12
- `ruff check` + `ruff format --check`
- `pytest --cov=vector_observatory`
- Docker build smoke test (planned)

---

## 10. Packaging & Distribution

```toml
[project]
name = "vector-observatory"
requires-python = ">=3.12"

dependencies = [
    "numpy>=1.26", "pandas>=2.1", "pyarrow>=14.0",
    "scikit-learn>=1.4", "umap-learn>=0.5", "hdbscan>=0.8",
    "duckdb>=0.10", "plotly>=5.18",
]

[project.optional-dependencies]
app = ["streamlit>=1.31"]
# gpu: conda install -c rapidsai cuml  (CUDA + Linux only)
```

### Public import surface

```python
from vector_observatory import (
    EmbeddingDataset,
    EmbeddingPipeline,
    UMAPReducer, TSNEReducer, PCAReducer,
    HDBSCANClusterer, DBSCANClusterer, KMeansClusterer,
    KNNIndex,
    DriftComparison,
    Experiment,
)
```

---

*Spec version: 0.2 — May 2026*
