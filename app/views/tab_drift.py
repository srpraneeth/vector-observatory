"""Drift tab — compare this experiment against another."""

import io

import pandas as pd
import streamlit as st
from state import AppState

from vector_observatory.drift.comparison import DriftComparison
from vector_observatory.ingestion.validators import _build_dataset
from vector_observatory.reducers import UMAPReducer
from vector_observatory.storage.experiment import Experiment
from vector_observatory.visualization.drift_overlay import build_drift_overlay


def render(state: AppState) -> None:
    ds_a = state.active_dataset
    st.info(f"**Base:** {state.active_experiment.name} — {ds_a.n_samples:,} points, dim={ds_a.dim}")

    with st.expander("How does drift comparison work?", expanded=False):
        st.markdown("""
**Comparing versions of the same data** — e.g. embeddings before and after fine-tuning:
- Create a second experiment (e.g. `my-model-v2`) and then come back here to compare, **or**
- Upload the new embeddings file directly below — no need to save it as an experiment first.

**Comparing different datasets** — e.g. train vs. eval split:
- Same options apply: save as separate experiments, or upload one inline.

The drift score (MMD) and overlay plot tell you how much the two distributions have shifted.
        """)

    st.subheader("Compare Against")

    all_experiments = [e for e in Experiment.list_all() if e != state.active_experiment.name]

    compare_mode = st.radio(
        "Source",
        ["Saved experiment", "Upload embeddings file"],
        horizontal=True,
        help="Pick a saved experiment or upload a second embeddings file (parquet/csv/json).",
    )

    ds_b = None

    if compare_mode == "Saved experiment":
        if not all_experiments:
            st.warning(
                "No other experiments saved yet. Create one first, or upload a file directly."
            )
            return
        selected = st.selectbox("Experiment", all_experiments)
        if st.button("Load experiment", use_container_width=True):
            project_b = Experiment.load(selected)
            datasets_b = project_b.store.list_datasets()
            if datasets_b:
                runs_b = project_b.store.list_runs(datasets_b[0])
                if runs_b:
                    ds_b = project_b.store.load_run(datasets_b[0], runs_b[0]["run_id"])
                    st.session_state["drift_ds_b"] = ds_b
        ds_b = st.session_state.get("drift_ds_b")

    else:
        st.caption(
            "Upload a second embeddings file — e.g. a newer model version or a different split."
        )
        uploaded = st.file_uploader("Embeddings file", type=["parquet", "csv", "json"])
        if uploaded:
            buf = io.BytesIO(uploaded.read())
            if uploaded.name.endswith(".parquet"):
                df_b = pd.read_parquet(buf)
            elif uploaded.name.endswith(".csv"):
                df_b = pd.read_csv(buf)
            else:
                df_b = pd.read_json(buf)

            cols_b = list(df_b.columns)
            c1, c2 = st.columns(2)
            with c1:
                id_col_b = st.selectbox("ID column", cols_b, key="drift_id")
            with c2:
                emb_col_b = st.selectbox(
                    "Embedding column", cols_b, index=len(cols_b) - 1, key="drift_emb"
                )
            meta_b = [c for c in cols_b if c not in (id_col_b, emb_col_b)]
            name_b = uploaded.name.rsplit(".", 1)[0]
            ds_b = _build_dataset(df_b, id_col_b, emb_col_b, meta_b or None, name_b)

    if ds_b is None:
        return

    st.divider()

    if ds_a.dim != ds_b.dim:
        st.error(
            f"Embedding dimensions don't match: dataset A is {ds_a.dim}-dim, "
            f"dataset B is {ds_b.dim}-dim. Drift comparison requires both datasets "
            "to use the same embedding model."
        )
        return

    if st.button("Run Comparison", type="primary", use_container_width=True):
        with st.spinner("Fitting combined UMAP on A ∪ B…"):
            comparison = DriftComparison(ds_a, ds_b, reducer=UMAPReducer())
            result = comparison.run()
            st.session_state["drift_result"] = result

    result = st.session_state.get("drift_result")
    if result is None:
        return

    st.metric(
        "MMD Score",
        f"{result.mmd_score:.4f}",
        help="Maximum Mean Discrepancy. 0 = identical distributions.",
    )

    st.plotly_chart(
        build_drift_overlay(result), use_container_width=True, key="drift_overlay_chart"
    )

    if result.cluster_overlap:
        st.subheader("Cluster Overlap")
        rows = [
            {
                "Cluster": "Noise" if cid == -1 else cid,
                f"{state.active_experiment.name}": v["a_count"],
                f"{ds_b.name}": v["b_count"],
                "Shift": v["b_count"] - v["a_count"],
            }
            for cid, v in result.cluster_overlap.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
