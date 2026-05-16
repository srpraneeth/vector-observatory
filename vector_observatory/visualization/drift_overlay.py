from __future__ import annotations

import plotly.graph_objects as go

from ..drift.comparison import DriftResult

_COLOR_A = "#636EFA"
_COLOR_B = "#EF553B"


def build_drift_overlay(
    result: DriftResult,
    hover_cols_a: list[str] | None = None,
    hover_cols_b: list[str] | None = None,
) -> go.Figure:
    """Scatter plot with dataset A and B overlaid in shared coordinate space."""
    ds_a = result.dataset_a
    ds_b = result.dataset_b

    if ds_a.reduced_coords is None or ds_b.reduced_coords is None:
        raise ValueError("DriftResult datasets must have reduced_coords populated.")

    traces = [
        go.Scattergl(
            x=ds_a.reduced_coords[:, 0],
            y=ds_a.reduced_coords[:, 1],
            mode="markers",
            marker=dict(color=_COLOR_A, size=5, opacity=0.6),
            name=ds_a.name or "Dataset A",
            text=ds_a.ids,
            hovertemplate="<b>A</b> ID: %{text}<extra></extra>",
        ),
        go.Scattergl(
            x=ds_b.reduced_coords[:, 0],
            y=ds_b.reduced_coords[:, 1],
            mode="markers",
            marker=dict(color=_COLOR_B, size=5, opacity=0.6),
            name=ds_b.name or "Dataset B",
            text=ds_b.ids,
            hovertemplate="<b>B</b> ID: %{text}<extra></extra>",
        ),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Drift Comparison — MMD: {result.mmd_score:.4f}",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        template="plotly_dark",
        height=650,
        hovermode="closest",
    )
    return fig
