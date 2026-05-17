"""Clusters tab — size breakdown, cluster labeling, per-cluster drill-down, stability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from state import AppState

from vector_observatory.metrics.geometry import compute_geometry_metrics
from vector_observatory.metrics.stability import cluster_jaccard_matrix, stability_score
from vector_observatory.visualization.cluster_map import build_cluster_overview


def render(state: AppState) -> None:
    ds = state.active_dataset
    exp = state.active_experiment

    if ds.cluster_labels is None:
        st.info("No cluster labels available.")
        return

    st.plotly_chart(
        build_cluster_overview(ds), use_container_width=True, key="clusters_overview_chart"
    )

    st.subheader("Cluster Breakdown")

    unique_labels, counts = np.unique(ds.cluster_labels, return_counts=True)

    saved_labels: dict[int, str] = {}
    if exp and state.active_run_id:
        saved_labels = exp.store.load_cluster_labels(state.active_run_id)

    rows = []
    for label, count in zip(unique_labels, counts, strict=False):
        row = {
            "Cluster": "Noise" if label == -1 else int(label),
            "Name": saved_labels.get(int(label), ""),
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

    # ------------------------------------------------------------------
    # Cluster drill-down + labeling
    # ------------------------------------------------------------------
    if selected and selected.get("selection", {}).get("rows"):
        row_idx = selected["selection"]["rows"][0]
        cluster_id = rows[row_idx]["Cluster"]
        cluster_id_int = -1 if cluster_id == "Noise" else int(cluster_id)

        if cluster_id == "Noise":
            cluster_ds = ds.filter(ds.cluster_labels == -1)
        else:
            cluster_ds = ds.filter_by_cluster(cluster_id_int)

        st.subheader(f"Cluster {cluster_id} — {cluster_ds.n_samples:,} points")

        # Cluster label editor
        if exp and state.active_run_id and cluster_id != "Noise":
            current_name = saved_labels.get(cluster_id_int, "")
            lc1, lc2, lc3 = st.columns([3, 1, 1])
            with lc1:
                new_name = st.text_input(
                    "Cluster name",
                    value=current_name,
                    placeholder="e.g. Action / Thriller",
                    key=f"label_input_{cluster_id_int}",
                )
            with lc2:
                st.write("")
                st.write("")
                if st.button("Save", key=f"label_save_{cluster_id_int}"):
                    if new_name.strip():
                        exp.store.save_cluster_label(
                            state.active_run_id, cluster_id_int, new_name.strip()
                        )
                    else:
                        exp.store.delete_cluster_label(state.active_run_id, cluster_id_int)
                    st.rerun()
            with lc3:
                st.write("")
                st.write("")
                if current_name and st.button("Clear", key=f"label_clear_{cluster_id_int}"):
                    exp.store.delete_cluster_label(state.active_run_id, cluster_id_int)
                    st.rerun()

        # Per-cluster geometry metrics
        with st.spinner("Computing cluster metrics…"):
            geo = compute_geometry_metrics(cluster_ds.embeddings)

        mc1, mc2 = st.columns(2)
        mc1.metric(
            "Anisotropy",
            f"{geo.anisotropy:.3f}",
            help="Embedding collapse within this cluster. Lower = more spread out.",
        )
        mc2.metric(
            "Intrinsic Dim",
            f"{geo.intrinsic_dim:.1f}",
            help="Effective dimensionality of this cluster's embeddings.",
        )

        st.dataframe(cluster_ds.metadata.reset_index(drop=True), use_container_width=True)

    # ------------------------------------------------------------------
    # Cluster stability
    # ------------------------------------------------------------------
    if exp and state.active_run_id:
        other_runs = [r for r in exp.store.list_runs(ds.name) if r["run_id"] != state.active_run_id]
        if other_runs:
            st.divider()
            st.subheader("Cluster Stability")
            st.caption(
                "How stable are the cluster assignments compared to another run? "
                "Jaccard overlap of 1.0 = identical; 0.0 = no overlap."
            )

            compare_options = {
                f"{r['run_id']} ({r['reducer']} + {r['clusterer']})": r["run_id"]
                for r in other_runs
            }
            compare_label = st.selectbox(
                "Compare against run", list(compare_options.keys()), key="stability_run_select"
            )
            compare_run_id = compare_options[compare_label]

            with st.spinner("Computing stability…"):
                compare_ds = exp.store.load_run(ds.name, compare_run_id)

            if compare_ds.cluster_labels is not None:
                matrix, ids_a, ids_b = cluster_jaccard_matrix(
                    ds.cluster_labels, compare_ds.cluster_labels
                )
                score = stability_score(matrix)

                st.metric(
                    "Stability Score",
                    f"{score:.3f}",
                    help="Mean best-match Jaccard across clusters in the active run.",
                )

                if matrix.size > 0:
                    fig = go.Figure(
                        go.Heatmap(
                            z=matrix.tolist(),
                            x=[f"Run B — {c}" for c in ids_b],
                            y=[f"Run A — {c}" for c in ids_a],
                            colorscale="Blues",
                            zmin=0,
                            zmax=1,
                            text=[[f"{v:.2f}" for v in row] for row in matrix.tolist()],
                            texttemplate="%{text}",
                        )
                    )
                    fig.update_layout(
                        title="Jaccard Overlap Matrix",
                        xaxis_title="Compare run clusters",
                        yaxis_title="Active run clusters",
                        template="plotly_dark",
                        height=400,
                        margin=dict(t=50, b=20, l=20, r=20),
                    )
                    st.plotly_chart(fig, use_container_width=True, key="stability_heatmap")
            else:
                st.warning("The selected run has no cluster labels.")
