"""Metrics tab — embedding health dashboard."""

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

    with st.spinner("Computing metrics…"):
        geo = compute_geometry_metrics(ds.embeddings)
        clust = compute_cluster_metrics(ds.cluster_labels) if ds.cluster_labels is not None else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Anisotropy", f"{geo.anisotropy:.4f}",
              help="Measures embedding collapse. Think of throwing darts at a globe — 0 means they're spread all over (healthy), 1 means they all land in one spot (collapsed). Real sentence-transformer models score 0.02–0.1.")
    c2.metric("Isotropy Score", f"{geo.isotropy_score:.4f}",
              help="Does the embedding cloud look like a sphere (1.0) or a pancake/needle (→0)? Measured by checking if the cloud looks the same from 200 random angles. Range 0–1, higher is better.")
    c3.metric("Intrinsic Dim", f"{geo.intrinsic_dim:.1f}",
              help=f"Your embeddings are {geo.variance_per_dim.shape[0]}-dimensional, but how many dimensions do they actually use? Estimates the true number of independent axes of variation. Real RAG pipelines typically score 20–60.")

    if clust:
        c4, c5, c6 = st.columns(3)
        c4.metric("Clusters", clust.n_clusters,
                  help="Number of clusters found. With K-Means this is what you set; with HDBSCAN it's discovered automatically from the data's density structure.")
        c5.metric("Noise Fraction", f"{clust.noise_fraction:.1%}",
                  help="Points that didn't fit into any cluster (HDBSCAN only). 0% with K-Means. With HDBSCAN, 5–20% is normal — those are your edge cases and outliers.")
        c6.metric("Largest Cluster", f"{clust.largest_cluster_fraction:.1%}",
                  help="Fraction of all points in the biggest cluster. Very high values (>80%) suggest the clustering isn't separating the data well.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(build_geometry_metrics_chart(geo), use_container_width=True, key="metrics_geo_chart")
    with col2:
        st.plotly_chart(build_variance_per_dim_chart(geo), use_container_width=True, key="metrics_variance_chart")

    with st.expander("What do these metrics mean?"):
        st.markdown("""
**Anisotropy** — measures how much embeddings cluster in a narrow cone vs filling the space.
High values (>0.5) indicate embedding collapse, where all vectors point in similar directions.
Healthy sentence-transformer models typically score 0.02–0.1.

**Isotropy** — min/max partition function ratio (Mu & Viswanath 2018). Measures how uniformly
the hypersphere is occupied. Range 0–1: 1 = perfectly uniform, 0 = fully collapsed into a cone.

**Intrinsic Dimensionality** — Two-NN estimator (Facco et al. 2017). Tells you how many
dimensions your data actually uses. Significantly lower than the embedding dimension means
the model is compressing information aggressively.

**Per-dimension variance** — Dead dimensions (near-zero variance) represent wasted capacity.
A healthy model uses most of its dimensions.
        """)
