"""Vector Observatory — single-page app with experiment-centric navigation."""

import streamlit as st
from state import AppState, load_experiment

from vector_observatory.storage.experiment import Experiment

st.set_page_config(
    page_title="Vector Observatory",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

state = AppState.from_session(st.session_state)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.title("🔭 Vector Observatory")
    st.caption("Observability for embedding spaces")
    st.divider()

    if st.button("＋ New Experiment", use_container_width=True, type="primary"):
        state.view = "new"
        state.active_experiment = None
        state.active_dataset = None
        state.write_to_session(st.session_state)
        st.rerun()

    experiments = Experiment.list_all()
    if experiments:
        st.subheader("Experiments")
        for name in experiments:
            is_active = state.active_experiment is not None and state.active_experiment.name == name
            label = f"**{name}**" if is_active else name
            if st.button(label, key=f"exp_{name}", use_container_width=True):
                with st.spinner(f"Loading {name}…"):
                    state = load_experiment(name, state)
                    state.view = "detail"
                    state.write_to_session(st.session_state)
                st.rerun()

# ------------------------------------------------------------------
# Main area routing
# ------------------------------------------------------------------

if state.view == "new":
    from views.new_experiment import render

    render(state)

elif state.view == "detail" and state.active_experiment is not None:
    from views.detail import render

    render(state)

else:
    from views.experiment_list import render

    render(state)
