"""Pareto-frontier scatter factory — the service-vs-cost cloud.

A stateless Plotly factory: it takes the
:func:`analytics.pareto.pareto_sweep` output frame (one row per policy-grid point: the
swept policy-param columns + the 12 KPI means + a boolean ``on_frontier``) and returns a
``go.Figure`` — the service-vs-cost cloud with the efficient-frontier points highlighted
and connected. No Streamlit, no state — a pure function, unit-testable. The page layer
 renders it and wires click-to-load of a point's parameters.

Axes default to the same KPIs ``pareto_sweep`` ranks on (service = ``fill_rate`` on x,
cost = ``total_cost`` on y); pass ``cost_kpi`` / ``service_kpi`` to match a sweep run on
other axes. The ideal corner is bottom-right (high service, low cost); the frontier is
that lower-right envelope, drawn as a line through its points **sorted by service** — a
visual guide through the discrete efficient set, not a claim of continuous dominance.

Colours are left to the active Plotly theme; the dominated cloud recedes via low marker
opacity and the frontier stands out via its connecting line + larger markers (added last,
so it sits on top) — no hard-coded, theme-coupled colours. Each point's swept parameters
are shown on hover (the columns that are neither KPIs nor ``on_frontier``) so a user can
read which params produced it. Each point also carries its source-frame row index as
``customdata`` (``[[idx], …]``): a Streamlit point-selection event surfaces that index, so
the page layer maps a clicked point straight back to ``sweep_df.loc[idx]`` to load its
parameters — robust to the cloud/frontier split and the frontier's by-service re-sort.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Final

import pandas as pd
import plotly.graph_objects as go

from analytics.kpis import KPIs

# KPI-mean column names in a sweep frame; every other column (minus ``on_frontier``) is a
# swept policy parameter, surfaced on hover.
_KPI_FIELDS: Final[frozenset[str]] = frozenset(f.name for f in fields(KPIs))

# Friendly axis labels for the likely Pareto axes; any other column falls back to its raw
# name (cf. visualization.comparison._METRIC_LABELS).
_AXIS_LABELS: Final[dict[str, str]] = {
    "total_cost": "Total cost ($)",
    "fill_rate": "Fill rate (Type 2)",
    "cycle_service_level": "Cycle service level (Type 1)",
    "ready_rate": "Ready rate (Type 3)",
}


def _axis_label(kpi: str) -> str:
    return _AXIS_LABELS.get(kpi, kpi)


def _hover(frame: pd.DataFrame, param_cols: list[str]) -> list[str]:
    """One ``"name=value<br>…"`` string per row from the swept-param columns."""
    return ["<br>".join(f"{col}={row[col]:g}" for col in param_cols) for _, row in frame.iterrows()]


def _customdata(frame: pd.DataFrame) -> list[list[Any]]:
    """Per-point ``[[idx], …]`` of each row's source-frame index (the click-to-load key)."""
    return [[idx] for idx in frame.index]


def pareto_scatter(
    sweep_df: pd.DataFrame, *, cost_kpi: str = "total_cost", service_kpi: str = "fill_rate"
) -> go.Figure:
    """Build the service-vs-cost Pareto scatter from a :func:`pareto_sweep` frame.

    ``sweep_df`` has one row per grid point: the swept policy-param columns, the 12 KPI
    means, and a boolean ``on_frontier``. Dominated points are drawn as a faded cloud; the
    non-dominated frontier points are highlighted and connected (sorted by service). Hover
    shows each point's swept parameters. ``x`` = ``service_kpi``, ``y`` = ``cost_kpi``.
    Raises :class:`ValueError` if the frame is empty or any required column is missing.
    """
    if sweep_df.empty:
        raise ValueError("sweep_df is empty: need at least one grid point to plot")
    missing = {cost_kpi, service_kpi, "on_frontier"} - set(sweep_df.columns)
    if missing:
        raise ValueError(f"sweep_df is missing required columns: {sorted(missing)}")

    param_cols = [c for c in sweep_df.columns if c not in _KPI_FIELDS and c != "on_frontier"]
    cloud = sweep_df[~sweep_df["on_frontier"]]
    frontier = sweep_df[sweep_df["on_frontier"]].sort_values(service_kpi)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cloud[service_kpi],
            y=cloud[cost_kpi],
            mode="markers",
            name="Dominated",
            opacity=0.4,
            hovertext=_hover(cloud, param_cols),
            hoverinfo="text",
            customdata=_customdata(cloud),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frontier[service_kpi],
            y=frontier[cost_kpi],
            mode="lines+markers",
            name="Efficient frontier",
            marker={"size": 10},
            hovertext=_hover(frontier, param_cols),
            hoverinfo="text",
            customdata=_customdata(frontier),
        )
    )
    fig.update_layout(
        title="Pareto frontier — service vs. cost",
        xaxis_title=_axis_label(service_kpi),
        yaxis_title=_axis_label(cost_kpi),
        hovermode="closest",
        legend={"orientation": "h"},
    )
    return fig
