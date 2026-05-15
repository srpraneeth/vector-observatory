"""Experiment detail — tabbed view over a loaded experiment."""

import streamlit as st

from state import AppState
from vector_observatory.storage.experiment import Experiment


def render(state: AppState) -> None:
    ds = state.active_dataset
    exp = state.active_experiment

    # Header
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        st.title(exp.name)
    with col2:
        if st.button("← All Experiments", use_container_width=True):
            state.view = "list"
            state.write_to_session(st.session_state)
            st.rerun()
    with col3:
        if st.button("Delete", type="secondary", use_container_width=True):
            st.session_state["_confirm_delete"] = True

    if st.session_state.get("_confirm_delete"):
        st.warning(
            f"Delete **{exp.name}**? This cannot be undone.",
            icon="⚠️",
        )
        c_yes, c_no = st.columns(2)
        if c_yes.button("Yes, delete", type="primary", use_container_width=True):
            Experiment.delete(exp.name)
            st.session_state.pop("_confirm_delete", None)
            state.view = "list"
            state.active_experiment = None
            state.active_dataset = None
            state.active_run_id = None
            state.knn_index = None
            state.write_to_session(st.session_state)
            st.rerun()
        if c_no.button("Cancel", use_container_width=True):
            st.session_state.pop("_confirm_delete", None)
            st.rerun()
        st.stop()

    if ds is None:
        st.warning("No analysis runs found in this experiment.")
        st.stop()

    st.divider()

    # Tabs
    tab_overview, tab_explore, tab_clusters, tab_metrics, tab_drift = st.tabs([
        "Overview", "Explore", "Clusters", "Metrics", "Drift"
    ])

    with tab_overview:
        from views.tab_overview import render as render_tab
        render_tab(state)

    with tab_explore:
        from views.tab_explore import render as render_tab
        render_tab(state)

    with tab_clusters:
        from views.tab_clusters import render as render_tab
        render_tab(state)

    with tab_metrics:
        from views.tab_metrics import render as render_tab
        render_tab(state)

    with tab_drift:
        from views.tab_drift import render as render_tab
        render_tab(state)
