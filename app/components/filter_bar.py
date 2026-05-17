from __future__ import annotations

import numpy as np
import streamlit as st

from vector_observatory.dataset import EmbeddingDataset


def render_filter_bar(ds: EmbeddingDataset) -> np.ndarray | None:
    """Render sidebar filter widgets and return a combined boolean mask.

    Returns None if no filters are active (show all points).
    """
    import pandas as pd

    active_filters = []

    # Cluster filter
    if ds.cluster_labels is not None:
        unique_clusters = sorted(set(ds.cluster_labels.tolist()))
        cluster_labels_display = ["Noise" if c == -1 else f"Cluster {c}" for c in unique_clusters]
        selected = st.multiselect(
            "Clusters", cluster_labels_display, default=cluster_labels_display
        )
        selected_ids = [unique_clusters[cluster_labels_display.index(s)] for s in selected]
        if len(selected_ids) < len(unique_clusters):
            active_filters.append(np.isin(ds.cluster_labels, selected_ids))

    # Per-column filters
    for col in ds.metadata.columns:
        series = ds.metadata[col]

        if pd.api.types.is_numeric_dtype(series):
            min_val, max_val = float(series.min()), float(series.max())
            if min_val < max_val:
                selected_range = st.slider(col, min_val, max_val, (min_val, max_val))
                if selected_range != (min_val, max_val):
                    mask = (series >= selected_range[0]) & (series <= selected_range[1])
                    active_filters.append(mask.to_numpy())

        elif series.nunique() <= 20:
            options = sorted(series.dropna().unique().tolist())
            selected_vals = st.multiselect(col, options, default=options)
            if len(selected_vals) < len(options):
                active_filters.append(series.isin(selected_vals).to_numpy())

        else:
            query = st.text_input(f"Search {col}", "")
            if query:
                mask = series.astype(str).str.contains(query, case=False, na=False)
                active_filters.append(mask.to_numpy())

    if not active_filters:
        return None

    combined = active_filters[0]
    for m in active_filters[1:]:
        combined = combined & m
    return combined
