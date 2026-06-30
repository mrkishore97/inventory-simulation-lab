"""Tornado chart — one-at-a-time sensitivity ranking.

A stateless Plotly factory. It takes the
:func:`analytics.sensitivity.sensitivity_oat` frame — one row per perturbed parameter, already
sorted by ``abs_swing`` descending (the tornado order) — and draws the classic tornado:
horizontal bars, one per parameter, each spanning ``[kpi_low, kpi_high]`` anchored at the shared
unperturbed ``kpi_baseline``, widest swing on top.

Both bar traces set ``base=kpi_baseline`` and a signed length ``x = kpi_{low,high} - kpi_baseline``,
so each bar *ends at the true KPI value* on the x-axis — the axis reads in real KPI units, with a
dashed reference line at the baseline. ``autorange="reversed"`` puts row 0 (the widest
``abs_swing``) at the top.

Like the other factories here it visualizes only: the OAT compute + ranking live in
``analytics.sensitivity``; this renders the frame it is handed (it does **not** re-sort — the
analytic owns the ranking). No Streamlit, no state; trace colours are left to the active Plotly
theme. ``output_kpi`` labels the axis/title (the frame's columns are KPI-agnostic).
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import plotly.graph_objects as go

# Columns read off the sensitivity_oat frame. ``abs_swing`` is required too — not read for geometry,
# but its presence proves the input is a genuine tornado frame (the one sorted by it), not just any
# frame that happens to carry low/high KPI columns.
_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"parameter", "low", "high", "kpi_low", "kpi_baseline", "kpi_high", "abs_swing"}
)


def _side_bar(
    params: list[object],
    settings: list[object],
    kpi: list[float],
    baseline: float,
    side: str,
    output_kpi: str,
) -> go.Bar:
    """One side of the tornado (``side`` = "low" or "high"): a horizontal bar per parameter.

    ``base=baseline`` with a signed length ``kpi - baseline`` makes each bar end exactly at its
    ``kpi`` value on the x-axis; the hover carries the parameter's setting and the resulting KPI.
    """
    return go.Bar(
        y=params,
        x=[k - baseline for k in kpi],
        base=baseline,
        orientation="h",
        name=f"Parameter {side}",
        customdata=[[s, k] for s, k in zip(settings, kpi, strict=True)],
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"{side} = %{{customdata[0]}}<br>"
            f"{output_kpi} = %{{customdata[1]:.2f}}"
            "<extra></extra>"
        ),
    )


def tornado_chart(sensitivity: pd.DataFrame, *, output_kpi: str = "total_cost") -> go.Figure:
    """Build the tornado figure from a :func:`analytics.sensitivity.sensitivity_oat` frame.

    ``sensitivity`` is one row per parameter — ``parameter`` / ``low`` / ``high`` / ``kpi_low`` /
    ``kpi_baseline`` / ``kpi_high`` / ``abs_swing`` — **already sorted by ``abs_swing`` descending**
    (the tornado order; this renders that order, it does not re-sort). Two horizontal bars per
    parameter, anchored at the shared ``kpi_baseline`` so each ends at its ``kpi_low`` /
    ``kpi_high`` on the x-axis; a dashed reference line marks the baseline; widest swing on top.
    ``output_kpi`` labels the axis/title only. Raises :class:`ValueError` if the frame is empty or
    missing a required column.
    """
    if sensitivity.empty:
        raise ValueError("sensitivity frame is empty: need at least one parameter to plot")
    missing = _REQUIRED_COLUMNS - set(sensitivity.columns)
    if missing:
        raise ValueError(f"sensitivity frame is missing required columns: {sorted(missing)}")

    params = sensitivity["parameter"].tolist()
    baseline = float(sensitivity["kpi_baseline"].iloc[0])  # constant across rows (one baseline run)

    fig = go.Figure()
    fig.add_trace(
        _side_bar(
            params,
            sensitivity["low"].tolist(),
            sensitivity["kpi_low"].tolist(),
            baseline,
            "low",
            output_kpi,
        )
    )
    fig.add_trace(
        _side_bar(
            params,
            sensitivity["high"].tolist(),
            sensitivity["kpi_high"].tolist(),
            baseline,
            "high",
            output_kpi,
        )
    )
    fig.add_vline(x=baseline, line_dash="dash", annotation_text="baseline")
    fig.update_layout(
        title=f"Tornado — sensitivity of {output_kpi}",
        xaxis_title=output_kpi,
        yaxis_title="Parameter",
        barmode="overlay",
        legend={"orientation": "h"},
    )
    fig.update_yaxes(autorange="reversed")  # widest abs_swing (row 0) on top
    return fig
