from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..dataset import EmbeddingDataset

_NOISE_COLOR = "#cccccc"
_PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def build_scatter_2d(
    dataset: EmbeddingDataset,
    color_by: str = "cluster",
    hover_cols: list[str] | None = None,
    highlight_ids: list | None = None,
    title: str = "",
) -> go.Figure:
    """Build the main 2D scatter plot figure.

    Args:
        dataset: Must have reduced_coords populated.
        color_by: "cluster" to color by cluster_id, or a metadata column name.
        hover_cols: Metadata columns to include in hover tooltip.
        highlight_ids: IDs to render with a highlight marker.
        title: Figure title.
    """
    if dataset.reduced_coords is None:
        raise ValueError("Dataset has no reduced coordinates. Run a reducer first.")

    coords = dataset.reduced_coords
    hover_cols = hover_cols or list(dataset.metadata.columns)

    if color_by == "cluster" and dataset.cluster_labels is not None:
        colors, legend_entries = _colors_from_clusters(dataset.cluster_labels)
    elif color_by in dataset.metadata.columns:
        colors, legend_entries = _colors_from_column(dataset.metadata[color_by])
    else:
        colors = ["#636EFA"] * dataset.n_samples
        legend_entries = None

    customdata = dataset.metadata[hover_cols].values if hover_cols else None
    hover_template = _build_hover_template(dataset.ids, hover_cols)

    traces = _build_traces(
        coords=coords,
        ids=dataset.ids,
        colors=colors,
        labels=dataset.cluster_labels,
        customdata=customdata,
        hover_template=hover_template,
        highlight_ids=set(highlight_ids) if highlight_ids else None,
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title or dataset.name,
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        legend_title=color_by.capitalize(),
        hovermode="closest",
        height=650,
        template="plotly_dark",
    )
    return fig


def _colors_from_clusters(labels: np.ndarray) -> tuple[list[str], None]:
    colors = []
    for label in labels:
        if label == -1:
            colors.append(_NOISE_COLOR)
        else:
            colors.append(_PALETTE[int(label) % len(_PALETTE)])
    return colors, None


def _colors_from_column(series) -> tuple[list[str], None]:
    import pandas as pd
    if pd.api.types.is_numeric_dtype(series):
        return list(series.astype(str)), None
    unique_vals = series.dropna().unique()
    color_map = {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(unique_vals)}
    return [color_map.get(v, _NOISE_COLOR) for v in series], None


def _build_hover_template(ids: np.ndarray, hover_cols: list[str]) -> str:
    lines = ["<b>ID:</b> %{text}"]
    for i, col in enumerate(hover_cols):
        lines.append(f"<b>{col}:</b> %{{customdata[{i}]}}")
    lines.append("<extra></extra>")
    return "<br>".join(lines)


def _build_traces(
    coords,
    ids,
    colors,
    labels,
    customdata,
    hover_template,
    highlight_ids,
) -> list[go.Scattergl]:
    """Use Scattergl (WebGL) for performance on large datasets."""
    normal_mask = np.ones(len(ids), dtype=bool)
    if highlight_ids:
        highlight_mask = np.array([i in highlight_ids for i in ids])
        normal_mask = ~highlight_mask
    else:
        highlight_mask = np.zeros(len(ids), dtype=bool)

    traces = []

    if normal_mask.any():
        traces.append(go.Scattergl(
            x=coords[normal_mask, 0],
            y=coords[normal_mask, 1],
            mode="markers",
            marker=dict(
                color=[colors[i] for i in np.where(normal_mask)[0]],
                size=5,
                opacity=0.8,
            ),
            text=ids[normal_mask],
            customdata=customdata[normal_mask] if customdata is not None else None,
            hovertemplate=hover_template,
            name="Points",
            showlegend=False,
        ))

    if highlight_mask.any():
        traces.append(go.Scattergl(
            x=coords[highlight_mask, 0],
            y=coords[highlight_mask, 1],
            mode="markers",
            marker=dict(
                color="yellow",
                size=10,
                symbol="star",
                line=dict(color="white", width=1),
            ),
            text=ids[highlight_mask],
            customdata=customdata[highlight_mask] if customdata is not None else None,
            hovertemplate=hover_template,
            name="Highlighted",
        ))

    return traces
