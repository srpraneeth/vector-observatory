"""Deduplication tab — surface near-duplicate embeddings by cosine similarity."""

import streamlit as st
from state import AppState

from vector_observatory.deduplication import find_near_duplicates


def render(state: AppState) -> None:
    ds = state.active_dataset

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        threshold = st.slider(
            "Similarity threshold",
            min_value=0.80,
            max_value=0.999,
            value=0.95,
            step=0.005,
            format="%.3f",
            help="Pairs with cosine similarity ≥ this value are flagged as near-duplicates.",
        )
    with col2:
        k = st.number_input(
            "kNN k",
            min_value=2,
            max_value=50,
            value=10,
            help="How many neighbors to check per point.",
        )
    with col3:
        st.write("")
        st.write("")
        run = st.button("Find duplicates", type="primary", use_container_width=True)

    if not run:
        st.caption("Adjust the threshold and click **Find duplicates** to search.")
        return

    with st.spinner("Scanning for near-duplicates…"):
        pairs = find_near_duplicates(ds.embeddings, ds.ids, threshold=threshold, k=int(k))

    if pairs.empty:
        st.success(f"No pairs found with similarity ≥ {threshold:.3f}.")
        return

    st.caption(
        f"**{len(pairs):,}** near-duplicate pair(s) found with similarity ≥ {threshold:.3f}."
    )

    # Enrich with metadata for both sides
    id_to_idx = {str(id_val): i for i, id_val in enumerate(ds.ids)}
    meta_cols = list(ds.metadata.columns)[:4]  # show up to 4 metadata cols

    enriched_rows = []
    for _, row in pairs.iterrows():
        r = {"id_a": row["id_a"], "id_b": row["id_b"], "similarity": row["similarity"]}
        idx_a = id_to_idx.get(str(row["id_a"]))
        idx_b = id_to_idx.get(str(row["id_b"]))
        for col in meta_cols:
            r[f"{col}_a"] = ds.metadata.iloc[idx_a][col] if idx_a is not None else "—"
            r[f"{col}_b"] = ds.metadata.iloc[idx_b][col] if idx_b is not None else "—"
        enriched_rows.append(r)

    import pandas as pd

    result_df = pd.DataFrame(enriched_rows)
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download duplicates CSV",
        data=result_df.to_csv(index=False).encode(),
        file_name=f"{ds.name}_duplicates.csv",
        mime="text/csv",
    )
