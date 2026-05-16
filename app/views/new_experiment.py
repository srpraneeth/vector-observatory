"""New experiment form — name + upload + column config + run."""

import io
import uuid

import streamlit as st
import pandas as pd

from vector_observatory.ingestion.validators import _build_dataset
from vector_observatory.reducers import UMAPReducer, TSNEReducer, PCAReducer
from vector_observatory.clustering import HDBSCANClusterer, DBSCANClusterer, KMeansClusterer
from vector_observatory.retrieval.knn import KNNIndex
from vector_observatory.metrics.geometry import compute_geometry_metrics
from vector_observatory.metrics.cluster import compute_cluster_metrics
from vector_observatory.storage.experiment import Experiment


def render(state) -> None:
    st.title("New Experiment")

    if st.button("← Back", type="secondary"):
        state.view = "list"
        state.write_to_session(st.session_state)
        st.rerun()

    st.divider()

    # ------------------------------------------------------------------
    # Step 1 — Name + file
    # ------------------------------------------------------------------
    st.subheader("1. Name & File")

    col1, col2 = st.columns([1, 2])
    with col1:
        name = st.text_input("Experiment name", placeholder="my-embeddings")
    with col2:
        uploaded = st.file_uploader("Embeddings file", type=["parquet", "csv", "json"])

    if not uploaded:
        st.stop()

    @st.cache_data
    def _load_preview(data: bytes, filename: str) -> pd.DataFrame:
        buf = io.BytesIO(data)
        if filename.endswith(".parquet"):
            return pd.read_parquet(buf)
        if filename.endswith(".csv"):
            return pd.read_csv(buf)
        return pd.read_json(buf)

    df = _load_preview(uploaded.read(), uploaded.name)
    st.dataframe(df.head(5), use_container_width=True)
    st.caption(f"{len(df):,} rows · {len(df.columns)} columns")

    # ------------------------------------------------------------------
    # Step 2 — Column assignment
    # ------------------------------------------------------------------
    st.subheader("2. Column Assignment")
    cols = list(df.columns)

    col1, col2 = st.columns(2)
    with col1:
        id_col = st.selectbox("ID column", cols)
    with col2:
        remaining = [c for c in cols if c != id_col]
        embedding_col = st.selectbox("Embedding column", remaining, index=len(remaining) - 1)

    metadata_cols = st.multiselect(
        "Metadata columns",
        [c for c in cols if c not in (id_col, embedding_col)],
        default=[c for c in cols if c not in (id_col, embedding_col)],
    )

    # ------------------------------------------------------------------
    # Step 3 — Analysis config
    # ------------------------------------------------------------------
    st.subheader("3. Analysis Settings")

    col1, col2 = st.columns(2)
    with col1:
        reducer_choice = st.selectbox("Reducer", ["UMAP", "PCA", "t-SNE"])
        if reducer_choice == "UMAP":
            n_neighbors = st.slider("n_neighbors", 5, 100, 15)
            min_dist = st.slider("min_dist", 0.0, 1.0, 0.1)
            reducer = UMAPReducer(n_neighbors=int(n_neighbors), min_dist=float(min_dist))
        elif reducer_choice == "t-SNE":
            perplexity = st.slider("perplexity", 5, 100, 30)
            reducer = TSNEReducer(perplexity=float(perplexity))
        else:
            reducer = PCAReducer()

    with col2:
        clusterer_choice = st.selectbox("Clusterer", ["HDBSCAN", "DBSCAN", "K-Means"])
        if clusterer_choice == "HDBSCAN":
            min_cluster_size = st.slider("min_cluster_size", 2, 100, 10)
            clusterer = HDBSCANClusterer(min_cluster_size=int(min_cluster_size))
        elif clusterer_choice == "DBSCAN":
            eps = st.slider("eps", 0.01, 2.0, 0.5)
            clusterer = DBSCANClusterer(eps=float(eps))
        else:
            n_clusters = st.slider("n_clusters", 2, 50, 8)
            clusterer = KMeansClusterer(n_clusters=int(n_clusters))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    st.divider()
    exp_name = name.strip() or uploaded.name.rsplit(".", 1)[0]

    if st.button("Create Experiment", type="primary", use_container_width=True):
        if not exp_name:
            st.error("Enter an experiment name.")
            st.stop()

        if Experiment.list_all().__contains__(exp_name):
            st.error(f"Experiment '{exp_name}' already exists.")
            st.stop()

        buf = io.BytesIO(uploaded.getvalue())

        with st.spinner("Loading data…"):
            if uploaded.name.endswith(".parquet"):
                df_full = pd.read_parquet(buf)
            elif uploaded.name.endswith(".csv"):
                df_full = pd.read_csv(buf)
            else:
                df_full = pd.read_json(buf)
            ds = _build_dataset(df_full, id_col, embedding_col, metadata_cols or None, exp_name)

        with st.spinner(f"Running {reducer_choice}…"):
            coords = reducer.fit_transform(ds.embeddings)
            ds = ds.with_reduction(coords)

        with st.spinner(f"Running {clusterer_choice}…"):
            labels = clusterer.fit_predict(ds.embeddings)
            ds = ds.with_clusters(labels)

        with st.spinner("Computing metrics & saving…"):
            run_id = str(uuid.uuid4())[:8]
            config = {"run_id": run_id, "reducer": reducer.config, "clusterer": clusterer.config}
            geo = compute_geometry_metrics(ds.embeddings)
            clust = compute_cluster_metrics(ds.cluster_labels)

            project = Experiment.create(exp_name)
            project.store.save_run(ds, config, geo, clust)

            index = KNNIndex()
            index.build(ds)

        state.active_experiment = project
        state.active_dataset = ds
        state.active_run_id = run_id
        state.knn_index = index
        state.filter_mask = None
        state.selected_point_id = None
        state.view = "detail"
        state.write_to_session(st.session_state)
        st.rerun()
