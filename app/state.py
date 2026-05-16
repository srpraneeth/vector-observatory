from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vector_observatory.dataset import EmbeddingDataset
from vector_observatory.retrieval.knn import KNNIndex
from vector_observatory.storage.experiment import Experiment


@dataclass
class AppState:
    """Typed wrapper around st.session_state."""

    view: str = "list"                         # "list" | "new" | "detail"
    active_experiment: Experiment | None = None
    active_dataset: EmbeddingDataset | None = None
    active_run_id: str | None = None
    knn_index: KNNIndex | None = None
    filter_mask: np.ndarray | None = None
    selected_point_id: Any = None

    @classmethod
    def from_session(cls, ss) -> AppState:
        return cls(
            view=ss.get("view", "list"),
            active_experiment=ss.get("active_experiment"),
            active_dataset=ss.get("active_dataset"),
            active_run_id=ss.get("active_run_id"),
            knn_index=ss.get("knn_index"),
            filter_mask=ss.get("filter_mask"),
            selected_point_id=ss.get("selected_point_id"),
        )

    def write_to_session(self, ss) -> None:
        ss["view"] = self.view
        ss["active_experiment"] = self.active_experiment
        ss["active_dataset"] = self.active_dataset
        ss["active_run_id"] = self.active_run_id
        ss["knn_index"] = self.knn_index
        ss["filter_mask"] = self.filter_mask
        ss["selected_point_id"] = self.selected_point_id

    @property
    def has_dataset(self) -> bool:
        return self.active_dataset is not None

    @property
    def has_reduction(self) -> bool:
        return self.has_dataset and self.active_dataset.reduced_coords is not None

    @property
    def visible_dataset(self) -> EmbeddingDataset | None:
        if self.active_dataset is None:
            return None
        if self.filter_mask is not None:
            return self.active_dataset.filter(self.filter_mask)
        return self.active_dataset


def load_experiment(name: str, state: AppState) -> AppState:
    """Load the most recent run from an experiment into AppState."""
    project = Experiment.load(name)
    state.active_experiment = project
    state.active_dataset = None
    state.active_run_id = None
    state.knn_index = None
    state.filter_mask = None
    state.selected_point_id = None

    datasets = project.store.list_datasets()
    if not datasets:
        return state

    runs = project.store.list_runs(datasets[0])
    if not runs:
        return state

    ds = project.store.load_run(datasets[0], runs[0]["run_id"])
    index = KNNIndex()
    index.build(ds)

    state.active_dataset = ds
    state.active_run_id = runs[0]["run_id"]
    state.knn_index = index
    return state
