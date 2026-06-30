"""Inventory-trajectory line-chart factory.

A stateless Plotly factory: it takes a materialized
Ledger frame (``Ledger.to_dataframe()`` / :func:`experiments.single_run.run`) and returns a
``go.Figure``. No Streamlit, no state — a pure function, trivially unit-testable. The page
layer (a later bullet) renders the returned figure with ``st.plotly_chart``.

The chart shows the inventory state over time: on-hand, inventory position, and on-order as
lines on the primary y-axis, with per-period demand as bars on a secondary y-axis (demand
~10 vs inventory 0–100 are different scales; a shared axis would flatten demand).

An optional ``highlight_period`` draws a dashed vertical marker at one period — the visual
half of the time-scrubber: the page's slider drives it so dragging the timeline moves
the marker across the trajectory. Default ``None`` leaves the figure marker-free (byte-
identical to the un-highlighted chart). The marker colour is a light neutral with opacity so
it reads against the dark theme (Plotly's default shape line is a dark grey — low contrast
here); it spans the full plot height regardless of the dual y-axes (``add_vline`` is paper-
referenced in y).
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import plotly.graph_objects as go

# Ledger columns read by the factory. Validated up front so a malformed frame fails loud
# (project style — cf. analytics.kpis / analytics.bullwhip) instead of a cryptic KeyError.
_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"period", "on_hand", "on_order", "inventory_position", "demand"}
)


def inventory_timeseries(ledger: pd.DataFrame, *, highlight_period: int | None = None) -> go.Figure:
    """Build the inventory-trajectory figure for one run's Ledger frame.

    Lines (primary y-axis, "Units"): ``on_hand``, ``inventory_position``, ``on_order``.
    Bars (secondary y-axis): ``demand``. ``x`` is the ``period`` column. When
    ``highlight_period`` is set, a dashed vertical marker is drawn at that period (the
    time-scrubber's visual cue); ``None`` leaves the figure marker-free. Raises
    :class:`ValueError` if any required column is missing.
    """
    missing = _REQUIRED_COLUMNS - set(ledger.columns)
    if missing:
        raise ValueError(f"ledger is missing required columns: {sorted(missing)}")

    period = ledger["period"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=period, y=ledger["on_hand"], mode="lines", name="On hand"))
    fig.add_trace(
        go.Scatter(
            x=period, y=ledger["inventory_position"], mode="lines", name="Inventory position"
        )
    )
    fig.add_trace(go.Scatter(x=period, y=ledger["on_order"], mode="lines", name="On order"))
    fig.add_trace(go.Bar(x=period, y=ledger["demand"], name="Demand", yaxis="y2", opacity=0.3))
    fig.update_layout(
        title="Inventory trajectory",
        xaxis_title="Period",
        yaxis_title="Units",
        yaxis2={"title": "Demand", "overlaying": "y", "side": "right", "showgrid": False},
        hovermode="x unified",
        legend={"orientation": "h"},
    )
    if highlight_period is not None:
        fig.add_vline(
            x=highlight_period,
            line_dash="dash",
            line_color="rgba(180,180,180,0.5)",
            annotation_text=f"t={highlight_period}",
        )
    return fig
