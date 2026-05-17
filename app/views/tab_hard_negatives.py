"""Hard negatives tab — pairs close in embedding space but with different labels."""

import streamlit as st
from state import AppState

from vector_observatory.retrieval.hard_negatives import find_hard_negatives


def render(state: AppState) -> None:
    ds = state.active_dataset

    if ds.metadata.empty or len(ds.metadata.columns) == 0:
        st.info("No metadata columns available to define labels.")
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        label_col = st.selectbox(
            "Label column",
            list(ds.metadata.columns),
            help="Pairs where this column disagrees are hard negatives.",
        )
    with col2:
        k = st.number_input(
            "kNN k",
            min_value=2,
            max_value=50,
            value=10,
            help="Neighbors checked per point.",
        )
    with col3:
        st.write("")
        st.write("")
        run = st.button("Find hard negatives", type="primary", use_container_width=True)

    if not run:
        st.caption(
            f"Will find pairs with different **{label_col}** values that are "
            "nonetheless close in embedding space. Click **Find hard negatives** to search."
        )
        return

    with st.spinner("Scanning for hard negatives…"):
        results = find_hard_negatives(
            ds.embeddings, ds.ids, ds.metadata, label_col=label_col, k=int(k)
        )

    if results.empty:
        st.success("No hard negatives found — all close neighbors share the same label.")
        return

    st.caption(
        f"**{len(results):,}** hard negative pair(s) — sorted by similarity "
        "(highest = most surprising mismatch)."
    )
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.download_button(
        "Download hard negatives CSV",
        data=results.to_csv(index=False).encode(),
        file_name=f"{ds.name}_hard_negatives.csv",
        mime="text/csv",
    )
