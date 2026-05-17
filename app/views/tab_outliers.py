"""Outliers tab — most isolated points by kNN distance."""

import numpy as np
import pandas as pd
import streamlit as st
from state import AppState

from vector_observatory.metrics.outliers import compute_outlier_scores


def render(state: AppState) -> None:
    ds = state.active_dataset

    col1, col2 = st.columns([3, 1])
    with col1:
        top_n = st.slider("Show top N outliers", min_value=5, max_value=100, value=20, step=5)
    with col2:
        k = st.number_input(
            "kNN k",
            min_value=2,
            max_value=20,
            value=5,
            help="Number of neighbors used to compute isolation score.",
        )

    with st.spinner("Computing outlier scores…"):
        scores = compute_outlier_scores(ds.embeddings, k=int(k))

    top_idx = np.argsort(scores)[::-1][:top_n]

    rows = []
    for rank, idx in enumerate(top_idx, start=1):
        row = {"Rank": rank, "ID": ds.ids[idx], "Outlier Score": round(float(scores[idx]), 4)}
        row.update(ds.metadata.iloc[idx].to_dict())
        rows.append(row)

    result_df = pd.DataFrame(rows)

    st.caption(
        f"Ranked by mean cosine distance to {k} nearest neighbors. "
        "Higher score = more isolated from its neighbours."
    )
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download outliers CSV",
        data=result_df.to_csv(index=False).encode(),
        file_name=f"{ds.name}_outliers.csv",
        mime="text/csv",
    )
