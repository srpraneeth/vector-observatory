from __future__ import annotations

import streamlit as st


def render_metrics_card(label: str, value: str, help_text: str = "") -> None:
    st.metric(label=label, value=value, help=help_text)
