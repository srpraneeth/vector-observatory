from __future__ import annotations

import streamlit as st

from vector_observatory.dataset import EmbeddingDataset
from vector_observatory.retrieval.knn import KNNIndex


def render_neighbor_panel(selected_id, index: KNNIndex, dataset: EmbeddingDataset, k: int = 10) -> None:
    """Render nearest neighbors for a selected point."""
    st.subheader(f"Nearest Neighbors — ID: {selected_id}")

    try:
        result = index.search_by_id(selected_id, dataset, k=k)
    except KeyError:
        st.warning(f"ID {selected_id!r} not found in index.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.caption("Selected point metadata")
        pos = dataset.index_of(selected_id)
        st.dataframe(dataset.metadata.iloc[[pos]], use_container_width=True)

    with col2:
        st.caption(f"Top {k} nearest neighbors")
        import pandas as pd
        neighbor_df = result.metadata.copy()
        neighbor_df.insert(0, "id", result.neighbor_ids)
        neighbor_df.insert(1, "distance", [f"{d:.4f}" for d in result.distances])
        st.dataframe(neighbor_df, use_container_width=True)
