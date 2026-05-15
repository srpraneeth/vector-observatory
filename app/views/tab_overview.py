"""Overview tab — dataset info + health snapshot."""

import pandas as pd
import streamlit as st

from state import AppState
from vector_observatory.metrics.geometry import compute_geometry_metrics
from vector_observatory.metrics.cluster import compute_cluster_metrics
from vector_observatory.visualization.metrics_chart import (
    build_geometry_metrics_chart,
    build_variance_per_dim_chart,
)


def render(state: AppState) -> None:
    ds = state.active_dataset

    # ------------------------------------------------------------------
    # Dataset info
    # ------------------------------------------------------------------
    st.subheader("Dataset")

    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", f"{ds.n_samples:,}")
    c2.metric("Embedding dim", ds.dim)
    c3.metric("Metadata columns", len(ds.metadata.columns))

    # Column schema
    schema_rows = [
        {"Column": col, "Type": str(ds.metadata[col].dtype), "Non-null": int(ds.metadata[col].notna().sum())}
        for col in ds.metadata.columns
    ]
    with st.expander("Schema", expanded=True):
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

    st.subheader("Sample rows")
    st.dataframe(ds.metadata.head(10).reset_index(drop=True), use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # Embedding health
    # ------------------------------------------------------------------
    st.subheader("Embedding Health")

    with st.spinner("Computing metrics…"):
        geo = compute_geometry_metrics(ds.embeddings)
        clust = compute_cluster_metrics(ds.cluster_labels)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Anisotropy", f"{geo.anisotropy:.3f}",
              help="Measures embedding collapse. Think of throwing darts at a globe — 0 means they're spread all over (healthy), 1 means they all land in one spot (collapsed). Real sentence-transformer models score 0.02–0.1.")
    c2.metric("Isotropy", f"{geo.isotropy_score:.3f}",
              help="Does the embedding cloud look like a sphere (1.0) or a pancake/needle (→0)? Measured by checking if the cloud looks the same from 200 random angles. Higher is better.")
    c3.metric("Intrinsic Dim", f"{geo.intrinsic_dim:.1f}",
              help=f"Your embeddings are {ds.dim}-dimensional, but how many dimensions do they actually use? This estimates the true number of independent axes of variation. Real RAG pipelines typically score 20–60.")
    c4.metric("Noise Fraction", f"{clust.noise_fraction:.1%}",
              help="Points that didn't fit into any cluster (HDBSCAN only). 0% with K-Means since every point is assigned. With HDBSCAN, 5–20% noise is normal — those are your edge cases and outliers.")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(build_geometry_metrics_chart(geo), use_container_width=True, key="overview_geo_chart")
    with col2:
        st.plotly_chart(build_variance_per_dim_chart(geo), use_container_width=True, key="overview_variance_chart")
