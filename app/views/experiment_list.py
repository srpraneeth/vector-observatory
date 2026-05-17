"""Landing page — list of all experiments as cards."""

import streamlit as st

from vector_observatory.storage.experiment import Experiment


def render(state) -> None:
    st.title("Vector Observatory")
    st.caption("Observability and debugging for embedding spaces.")
    st.divider()

    experiments = Experiment.list_all()
    if not experiments:
        st.info("No experiments yet. Click **＋ New Experiment** in the sidebar to get started.")
        return

    st.subheader(f"{len(experiments)} experiment{'s' if len(experiments) != 1 else ''}")

    cols = st.columns(3)
    for i, name in enumerate(experiments):
        with cols[i % 3], st.container(border=True):
            st.markdown(f"### {name}")

            # Load lightweight stats from DuckDB
            try:
                project = Experiment.load(name)
                datasets = project.store.list_datasets()
                runs = project.store.list_runs(datasets[0]) if datasets else []

                if runs:
                    run = runs[0]
                    st.caption(f"Reducer: {run['reducer']} · Clusterer: {run['clusterer']}")
                    st.caption(f"Last run: {run['created_at'][:19]}")
                else:
                    st.caption("No runs yet")
            except Exception:
                st.caption("—")

            if st.button("Open", key=f"open_{name}", use_container_width=True, type="primary"):
                from state import load_experiment

                with st.spinner(f"Loading {name}…"):
                    load_experiment(name, state)
                    state.view = "detail"
                    state.write_to_session(st.session_state)
                st.rerun()
