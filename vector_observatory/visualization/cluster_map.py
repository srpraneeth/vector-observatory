from __future__ import annotations

import plotly.graph_objects as go

from ..dataset import EmbeddingDataset


def build_cluster_overview(dataset: EmbeddingDataset) -> go.Figure:
    """Bar chart of cluster sizes, noise shown separately."""
    if dataset.cluster_labels is None:
        raise ValueError("Dataset has no cluster labels.")

    import numpy as np

    unique, counts = np.unique(dataset.cluster_labels, return_counts=True)
    labels = ["Noise" if c == -1 else f"Cluster {c}" for c in unique]
    palette = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]
    colors = ["#cccccc" if c == -1 else palette[int(c) % len(palette)] for c in unique]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Cluster Size Distribution",
        xaxis_title="Cluster",
        yaxis_title="Points",
        template="plotly_dark",
        height=400,
    )
    return fig
