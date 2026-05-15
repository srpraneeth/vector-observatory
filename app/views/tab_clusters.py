"""Clusters tab — size breakdown + per-cluster drill-down."""

import numpy as np
import pandas as pd
import streamlit as st

from state import AppState
from vector_observatory.visualization.cluster_map import build_cluster_overview


def render(state: AppState) -> None:
    ds = state.active_dataset

    if ds.cluster_labels is None:
        st.info("No cluster labels available.")
        return

    st.plotly_chart(build_cluster_overview(ds), use_container_width=True, key="clusters_overview_chart")

    st.subheader("Cluster Breakdown")

    unique_labels, counts = np.unique(ds.cluster_labels, return_counts=True)
    rows = []
    for label, count in zip(unique_labels, counts):
        row = {
            "Cluster": "Noise" if label == -1 else int(label),
            "Points": int(count),
            "% of total": f"{count / ds.n_samples:.1%}",
        }
        mask = ds.cluster_labels == label
        cluster_meta = ds.metadata[mask]
        for col in list(ds.metadata.columns)[:3]:
            top = cluster_meta[col].mode()
            row[f"Top {col}"] = str(top.iloc[0]) if len(top) > 0 else "—"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    selected = st.dataframe(
        summary_df,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="cluster_table",
    )

    if selected and selected.get("selection", {}).get("rows"):
        row_idx = selected["selection"]["rows"][0]
        cluster_id = rows[row_idx]["Cluster"]

        if cluster_id == "Noise":
            cluster_ds = ds.filter(ds.cluster_labels == -1)
        else:
            cluster_ds = ds.filter_by_cluster(int(cluster_id))

        st.subheader(f"Cluster {cluster_id} — {cluster_ds.n_samples:,} points")
        st.dataframe(cluster_ds.metadata.reset_index(drop=True), use_container_width=True)
