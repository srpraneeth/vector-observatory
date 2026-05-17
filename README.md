# Vector Observatory

> Observability and debugging for embedding spaces.

Modern ML systems — RAG pipelines, recommendation engines, semantic search — depend on embeddings as a core primitive. But embeddings are invisible. They live in high-dimensional vector spaces that resist direct inspection.

When retrieval quality degrades, when recommendations become stale, when a model retraining changes embedding geometry — engineers have no tools to answer the question: *what actually changed, and why?*

Vector Observatory gives ML engineers direct visibility into their embedding spaces. It treats embeddings as a first-class observability signal, not just a component inside a black box.

## Who it's for

ML engineers and data scientists working with embedding-based systems, specifically:

- Debugging why a RAG pipeline returns poor results
- Inspecting how a fine-tuned model changed embedding geometry
- Understanding cluster structure in a recommendation embedding space
- Comparing embeddings before and after retraining
- Detecting embedding collapse or dimensional degeneration
- Building intuition about what your model has actually learned

## Goals

- Reusable Python library for embedding analysis as the core value — no Streamlit required
- Thin Streamlit frontend that calls the library, no analysis logic in page files
- Dimensionality reduction, clustering, and visualization out of the box
- Rich metadata filtering and slicing on the scatter plot
- Drift comparison between two embedding datasets
- Lightweight nearest-neighbor inspection
- Embedding health metrics — geometry, collapse detection, intrinsic dimensionality
- Persistent experiments backed by DuckDB — one file, no servers
- Usable headlessly in notebooks and scripts
- Deployable via Docker

## What it does

You bring an embeddings file. Vector Observatory runs dimensionality reduction, clustering, and geometry analysis, then gives you a browser-based UI to explore the results:

- **Visualise** your embedding space with UMAP, t-SNE, or PCA
- **Cluster** with HDBSCAN, DBSCAN, or K-Means and drill into each cluster
- **Measure** embedding health: anisotropy, isotropy, intrinsic dimensionality, per-dimension variance
- **Detect drift** between two embedding sets using Maximum Mean Discrepancy (MMD)
- **Inspect neighbors** — click any point to see its nearest neighbors in the original embedding space
- **Filter** the scatter plot by any metadata column

All results are stored locally in DuckDB — one file per experiment, no external services needed.

## How it looks

**Experiment list**
![Experiment list](images/Screenshot%20Experiments.png)

**Explore — UMAP scatter with metadata filters and neighbor inspection**
![Explore tab](images/Screenshot%20Explore.png)

**Overview — dataset stats and metadata distributions**
![Overview tab 1](images/Screenshot%20Overview1.png)
![Overview tab 2](images/Screenshot%20Overview2.png)

**Clusters — breakdown table with per-cluster metrics**
![Clusters tab](images/Screenshot%20Clusters.png)

**Metrics — embedding health dashboard**
![Metrics tab](images/Screenshot%20Metrics.png)

**Outliers — most isolated points by kNN distance**
![Outliers tab](images/Screenshot%20Outliers.png)

## Quickstart

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://https://github.com/srpraneeth/vector-observatory
cd vector-observatory

# Install dependencies
uv sync --extra app

# Load the movies demo experiment
uv run python scripts/generate_demo.py

# Start the app
uv run streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) and you'll see the `movies-demo` experiment pre-loaded.

## Using your own data

Your embeddings file needs three things:
- An **ID column** (any unique string or integer per row)
- An **embedding column** (a list/array of floats per row)
- Any number of **metadata columns** (title, label, score, etc.)

Supported formats: **Parquet**, **CSV**, **JSON**

In the app, click **New Experiment**, upload your file, assign the columns, pick a reducer and clusterer, and hit **Create Experiment**.

### Using the library without the app

The core library has no Streamlit dependency and works in scripts, notebooks, and CI pipelines. See the [`examples/`](examples/) directory:

| Example | What it shows |
|---|---|
| [`basic_analysis.py`](examples/basic_analysis.py) | Run the full pipeline headlessly and save as an experiment |
| [`neighbor_inspection.py`](examples/neighbor_inspection.py) | Build a KNN index and find nearest neighbors by ID |
| [`drift_comparison.py`](examples/drift_comparison.py) | Compare two embedding datasets and get an MMD drift score |

### Example: creating an experiment from Python

```python
import pandas as pd
from vector_observatory.ingestion.validators import _build_dataset
from vector_observatory.reducers import UMAPReducer
from vector_observatory.clustering import HDBSCANClusterer
from vector_observatory.metrics.geometry import compute_geometry_metrics
from vector_observatory.storage.experiment import Experiment
import uuid

df = pd.read_parquet("my_embeddings.parquet")
ds = _build_dataset(df, id_col="id", embedding_col="embedding",
                    metadata_cols=["text", "label"], name="my-experiment")

reducer = UMAPReducer(n_neighbors=15, min_dist=0.1)
ds = ds.with_reduction(reducer.fit_transform(ds.embeddings))

clusterer = HDBSCANClusterer(min_cluster_size=10)
ds = ds.with_clusters(clusterer.fit_predict(ds.embeddings))

geo = compute_geometry_metrics(ds.embeddings)
run_id = str(uuid.uuid4())[:8]

experiment = Experiment.load_or_create("my-experiment")
experiment.store.save_run(ds, {"run_id": run_id, "reducer": reducer.config,
                             "clusterer": clusterer.config}, geo)
```

## Project layout

```
vector_observatory/   # core library (installable, no Streamlit dependency)
  dataset.py          # EmbeddingDataset — immutable data container
  reducers/           # UMAP, t-SNE, PCA
  clustering/         # HDBSCAN, DBSCAN, K-Means
  metrics/            # geometry (anisotropy, isotropy, intrinsic dim) + cluster metrics
  drift/              # MMD-based drift comparison
  retrieval/          # KNN index for neighbor lookup
  storage/            # DuckDB-backed experiment storage
  ingestion/          # parquet / CSV / JSON readers + validation
  visualization/      # Plotly chart builders

app/                  # Streamlit frontend
  app.py
  views/              # experiment list, new experiment, detail tabs
  components/         # filter bar, neighbor panel

data/
  movies_demo.parquet       # demo source file
  experiments/              # DuckDB experiment files (gitignored)

scripts/
  generate_demo.py    # regenerate demo experiment
```

## Running tests

```bash
uv run pytest
```

## Docker

**Build the image:**

```bash
docker build -t vector-observatory .
```

**Run the app:**

```bash
docker run -p 8501:8501 \
  -v $(pwd)/data/experiments:/app/data/experiments \
  vector-observatory
```

Open [http://localhost:8501](http://localhost:8501).

The `-v` mount persists your experiments to `data/experiments/` on the host so they survive container restarts. The demo experiment is baked into the image and will appear automatically on first launch.

## Experiment storage

Each experiment is a single `.duckdb` file in `data/experiments/`. You can copy, share, or delete them freely. The app stores:

- Raw embeddings + metadata
- Reduced 2D coordinates (one set per run)
- Cluster labels (one set per run)
- Geometry and cluster metrics snapshots
- Drift comparison results

---

## Roadmap

### v0.1 — Current
- [x] Experiment-centric UI — create, browse, and delete experiments
- [x] File ingestion — Parquet, CSV, JSON with column assignment
- [x] Dimensionality reduction — UMAP, t-SNE, PCA
- [x] Clustering — HDBSCAN, DBSCAN, K-Means
- [x] Geometry metrics — anisotropy, isotropy, intrinsic dimensionality, per-dimension variance
- [x] Cluster metrics — count, noise fraction, largest cluster fraction
- [x] Interactive scatter — filter by metadata, color by any column, click to inspect
- [x] Neighbor panel — nearest neighbors in original embedding space
- [x] Drift detection — MMD score + overlay plot, compare saved experiment or upload inline
- [x] Local DuckDB storage — one file per experiment, no external services
- [x] Movies demo — pre-built experiment ready on first launch

### v0.2 — Quick wins
- [x] Metadata distributions — histograms and value counts per column in Overview
- [x] Export — download the current filtered view as CSV
- [x] Outlier detection — surface the most isolated points by distance to kNN
- [x] Per-cluster metrics — anisotropy and intrinsic dimensionality broken down per cluster

### v0.3 — Deeper experiment management
- [x] Re-run — apply a different reducer or clusterer to an existing experiment without re-uploading
- [x] Cluster labeling — name clusters (e.g. "Cluster 0 → Action/Thriller") and persist the labels
- [x] Semantic deduplication — find near-duplicate embeddings above a cosine similarity threshold
- [x] Cluster stability — measure how much cluster assignments change across runs (Jaccard overlap)
- [x] Hard negative inspection — surface pairs that are close in embedding space but semantically distant

### v1.0 — Full observability
- [ ] Multi-run comparison — compare UMAP vs t-SNE or different cluster settings side-by-side
- [ ] Embedding diff — given two runs on the same IDs, show which points moved the most
- [ ] Temporal analysis — track how embedding geometry evolves across model versions or time windows
- [ ] Retrieval quality evaluation — measure precision@k / recall@k against a ground-truth query set
- [ ] Annotation mode — mark individual points as good/bad/interesting and export those labels
- [ ] Real encoder integration — paste raw text, encode it live, find nearest neighbors

---

MIT License
