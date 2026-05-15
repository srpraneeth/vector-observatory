from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..metrics.geometry import GeometryMetrics
from ..metrics.cluster import ClusterMetrics


def build_geometry_metrics_chart(metrics: GeometryMetrics) -> go.Figure:
    """Bar chart for scalar geometry metrics."""
    names = ["Anisotropy", "Isotropy Score", "Intrinsic Dim"]
    values = [metrics.anisotropy, metrics.isotropy_score, metrics.intrinsic_dim]

    fig = go.Figure(go.Bar(x=names, y=values, text=[f"{v:.3f}" for v in values], textposition="outside"))
    fig.update_layout(title="Embedding Geometry Metrics", template="plotly_dark", height=350)
    return fig


def build_variance_per_dim_chart(metrics: GeometryMetrics, top_n: int = 50) -> go.Figure:
    """Show per-dimension variance — dead dimensions appear near zero."""
    var = np.sort(metrics.variance_per_dim)[::-1][:top_n]
    fig = go.Figure(go.Bar(x=list(range(len(var))), y=var))
    fig.update_layout(
        title=f"Per-Dimension Variance (top {top_n})",
        xaxis_title="Dimension (sorted by variance)",
        yaxis_title="Variance",
        template="plotly_dark",
        height=350,
    )
    return fig
