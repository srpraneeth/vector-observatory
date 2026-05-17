"""Runs tab — browse existing runs, switch active run, create new runs."""

from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st
from state import AppState

from vector_observatory.clustering import DBSCANClusterer, HDBSCANClusterer, KMeansClusterer
from vector_observatory.metrics.cluster import compute_cluster_metrics
from vector_observatory.metrics.geometry import compute_geometry_metrics
from vector_observatory.reducers import PCAReducer, TSNEReducer, UMAPReducer
from vector_observatory.retrieval.knn import KNNIndex


def render(state: AppState) -> None:
    exp = state.active_experiment
    ds = state.active_dataset

    # ------------------------------------------------------------------
    # Existing runs
    # ------------------------------------------------------------------
    st.subheader("Analysis Runs")

    runs = exp.store.list_runs(ds.name)
    if runs:
        runs_df = pd.DataFrame(
            [
                {
                    "Run ID": r["run_id"],
                    "Reducer": r["reducer"] or "—",
                    "Clusterer": r["clusterer"] or "—",
                    "Created": r["created_at"],
                    "Active": "✓" if r["run_id"] == state.active_run_id else "",
                }
                for r in runs
            ]
        )
        selected = st.dataframe(
            runs_df,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="runs_table",
            hide_index=True,
        )

        if selected and selected.get("selection", {}).get("rows"):
            row_idx = selected["selection"]["rows"][0]
            chosen_run_id = runs[row_idx]["run_id"]
            if chosen_run_id != state.active_run_id and st.button(
                "Load selected run", type="primary"
            ):
                with st.spinner("Loading run…"):
                    loaded_ds = exp.store.load_run(ds.name, chosen_run_id)
                    index = KNNIndex()
                    index.build(loaded_ds)
                state.active_dataset = loaded_ds
                state.active_run_id = chosen_run_id
                state.knn_index = index
                state.filter_mask = None
                state.selected_point_id = None
                state.write_to_session(st.session_state)
                st.rerun()

    st.divider()

    # ------------------------------------------------------------------
    # New run
    # ------------------------------------------------------------------
    with st.expander("New Run", expanded=not runs):
        col1, col2 = st.columns(2)
        with col1:
            reducer_choice = st.selectbox("Reducer", ["UMAP", "PCA", "t-SNE"], key="rerun_reducer")
            if reducer_choice == "UMAP":
                n_neighbors = st.slider("n_neighbors", 5, 100, 15, key="rerun_umap_nn")
                min_dist = st.slider("min_dist", 0.0, 1.0, 0.1, key="rerun_umap_md")
                reducer = UMAPReducer(n_neighbors=int(n_neighbors), min_dist=float(min_dist))
            elif reducer_choice == "t-SNE":
                perplexity = st.slider("perplexity", 5, 100, 30, key="rerun_tsne_perp")
                reducer = TSNEReducer(perplexity=float(perplexity))
            else:
                reducer = PCAReducer()

        with col2:
            clusterer_choice = st.selectbox(
                "Clusterer", ["HDBSCAN", "DBSCAN", "K-Means"], key="rerun_clusterer"
            )
            if clusterer_choice == "HDBSCAN":
                min_cluster_size = st.slider(
                    "min_cluster_size", 2, 100, 10, key="rerun_hdbscan_mcs"
                )
                clusterer = HDBSCANClusterer(min_cluster_size=int(min_cluster_size))
            elif clusterer_choice == "DBSCAN":
                eps = st.slider("eps", 0.01, 2.0, 0.5, key="rerun_dbscan_eps")
                clusterer = DBSCANClusterer(eps=float(eps))
            else:
                n_clusters = st.slider("n_clusters", 2, 50, 8, key="rerun_kmeans_k")
                clusterer = KMeansClusterer(n_clusters=int(n_clusters))

        if st.button("Create Run", type="primary", use_container_width=True):
            with st.spinner("Loading embeddings…"):
                raw_ds = exp.store.load_dataset(ds.name)

            with st.spinner(f"Running {reducer_choice}…"):
                coords = reducer.fit_transform(raw_ds.embeddings)
                raw_ds = raw_ds.with_reduction(coords)

            with st.spinner(f"Running {clusterer_choice}…"):
                labels = clusterer.fit_predict(raw_ds.embeddings)
                raw_ds = raw_ds.with_clusters(labels)

            with st.spinner("Saving…"):
                run_id = str(uuid.uuid4())[:8]
                config = {
                    "run_id": run_id,
                    "reducer": reducer.config,
                    "clusterer": clusterer.config,
                }
                geo = compute_geometry_metrics(raw_ds.embeddings)
                clust = compute_cluster_metrics(raw_ds.cluster_labels)
                exp.store.save_run(raw_ds, config, geo, clust)

                index = KNNIndex()
                index.build(raw_ds)

            state.active_dataset = raw_ds
            state.active_run_id = run_id
            state.knn_index = index
            state.filter_mask = None
            state.selected_point_id = None
            state.write_to_session(st.session_state)
            st.rerun()
