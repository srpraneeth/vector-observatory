"""Explore tab — scatter + filters + neighbor inspection."""

import streamlit as st
from components.filter_bar import render_filter_bar
from components.neighbor_panel import render_neighbor_panel
from state import AppState

from vector_observatory.visualization.scatter import build_scatter_2d


def render(state: AppState) -> None:
    if not state.has_reduction:
        st.info("No reduced coordinates found.")
        return

    ds = state.active_dataset

    col_main, col_sidebar = st.columns([3, 1])

    with col_sidebar:
        st.subheader("Filters")
        filter_mask = render_filter_bar(ds)
        if filter_mask is not None:
            state.filter_mask = filter_mask
        else:
            state.filter_mask = None
        state.write_to_session(st.session_state)

        st.divider()
        color_options = ["cluster"] + list(ds.metadata.columns)
        color_by = st.selectbox("Color by", color_options)

        st.divider()
        st.subheader("Export")
        visible_ds = state.visible_dataset
        export_df = visible_ds.metadata.copy()
        export_df.insert(0, "id", visible_ds.ids)
        st.download_button(
            "Download visible points CSV",
            data=export_df.to_csv(index=False).encode(),
            file_name=f"{ds.name}_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    visible_ds = state.visible_dataset
    st.caption(f"{visible_ds.n_samples:,} / {ds.n_samples:,} points visible")

    with col_main:
        fig = build_scatter_2d(
            visible_ds,
            color_by=color_by,
            hover_cols=list(visible_ds.metadata.columns),
            highlight_ids=[state.selected_point_id] if state.selected_point_id else None,
        )
        event = st.plotly_chart(
            fig, use_container_width=True, on_select="rerun", key="explore_scatter"
        )

    if event and event.get("selection", {}).get("points"):
        clicked_id = event["selection"]["points"][0].get("text")
        if clicked_id and clicked_id != state.selected_point_id:
            state.selected_point_id = clicked_id
            state.write_to_session(st.session_state)
            st.rerun()

    if state.selected_point_id and state.knn_index:
        st.divider()
        render_neighbor_panel(state.selected_point_id, state.knn_index, ds)
