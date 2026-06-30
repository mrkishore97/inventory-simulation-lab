"""Integration: the Pareto scatter factory — service-vs-cost frontier."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import pytest

from analytics.pareto import pareto_sweep
from core.config import RunConfig
from visualization.pareto import pareto_scatter


def _sweep() -> pd.DataFrame:
    """A small synthetic sweep frame: 2 swept params + the two axes + a known frontier mask.

    on_frontier is True at rows 0, 1, 3 (cost 500/450/700, fill 0.80/0.90/0.95); row 2
    (cost 600, fill 0.92) is the lone dominated point.
    """
    return pd.DataFrame(
        {
            "reorder_point": [10, 20, 30, 40],
            "order_quantity": [30, 30, 30, 30],
            "total_cost": [500.0, 450.0, 600.0, 700.0],
            "fill_rate": [0.80, 0.90, 0.92, 0.95],
            "on_frontier": [True, True, False, True],
        }
    )


def _trace(fig: go.Figure, name: str) -> go.Scatter:
    return next(t for t in fig.data if t.name == name)


def test_returns_figure() -> None:
    assert isinstance(pareto_scatter(_sweep()), go.Figure)


def test_two_traces_cloud_then_frontier() -> None:
    fig = pareto_scatter(_sweep())
    assert [t.name for t in fig.data] == ["Dominated", "Efficient frontier"]


def test_frontier_trace_is_only_frontier_points_sorted_by_service() -> None:
    frontier = _trace(pareto_scatter(_sweep()), "Efficient frontier")
    assert list(frontier.x) == [0.80, 0.90, 0.95]  # sorted by fill_rate
    assert list(frontier.y) == [500.0, 450.0, 700.0]  # cost follows the sorted rows


def test_cloud_trace_is_the_dominated_points() -> None:
    cloud = _trace(pareto_scatter(_sweep()), "Dominated")
    assert list(cloud.x) == [0.92]
    assert list(cloud.y) == [600.0]


def test_default_axes_are_service_x_cost_y() -> None:
    fig = pareto_scatter(_sweep())
    assert fig.layout.xaxis.title.text == "Fill rate (Type 2)"
    assert fig.layout.yaxis.title.text == "Total cost ($)"


def test_axis_override_and_label_fallback() -> None:
    df = _sweep().assign(total_holding_cost=[100.0, 90.0, 120.0, 140.0])
    fig = pareto_scatter(df, cost_kpi="total_holding_cost")
    # total_holding_cost is a real KPI column but not in _AXIS_LABELS -> raw-name fallback.
    assert fig.layout.yaxis.title.text == "total_holding_cost"
    frontier = _trace(fig, "Efficient frontier")
    assert list(frontier.y) == [100.0, 90.0, 140.0]  # holding cost, sorted by fill_rate


def test_hover_shows_swept_params() -> None:
    frontier = _trace(pareto_scatter(_sweep()), "Efficient frontier")
    assert "reorder_point=10" in frontier.hovertext[0]  # lowest-service frontier point
    assert "order_quantity=30" in frontier.hovertext[0]


def test_empty_frame_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        pareto_scatter(pd.DataFrame())


def test_missing_on_frontier_raises() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        pareto_scatter(_sweep().drop(columns=["on_frontier"]))


def test_missing_axis_column_raises() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        pareto_scatter(_sweep(), service_kpi="nonexistent_kpi")


def test_all_points_on_frontier_empty_cloud() -> None:
    fig = pareto_scatter(_sweep().assign(on_frontier=[True, True, True, True]))
    cloud = _trace(fig, "Dominated")
    frontier = _trace(fig, "Efficient frontier")
    assert cloud.x is None or len(cloud.x) == 0
    assert len(frontier.x) == 4


def test_no_frontier_points_empty_frontier() -> None:
    # Theoretical (a correct on_frontier always keeps >=1 point) but the factory stays robust.
    fig = pareto_scatter(_sweep().assign(on_frontier=[False, False, False, False]))
    cloud = _trace(fig, "Dominated")
    frontier = _trace(fig, "Efficient frontier")
    assert frontier.x is None or len(frontier.x) == 0
    assert len(cloud.x) == 4


def test_smoke_on_real_sweep(make_config: Callable[..., RunConfig]) -> None:
    sweep = pareto_sweep(
        make_config(),
        {"reorder_point": [10, 20], "order_quantity": [20, 30]},
        n_replications=5,
        n_jobs=1,
    )
    fig = pareto_scatter(sweep)
    assert [t.name for t in fig.data] == ["Dominated", "Efficient frontier"]
    total_points = sum(len(t.x) for t in fig.data if t.x is not None)
    assert total_points == 4  # the full 2x2 grid, split across cloud + frontier


def test_frontier_customdata_is_source_row_index() -> None:
    # Frontier rows 0/1/3 (already ascending in fill_rate, so the by-service sort is a no-op);
    # customdata carries the original sweep_df index — the click-to-load lookup key.
    frontier = _trace(pareto_scatter(_sweep()), "Efficient frontier")
    assert [int(c[0]) for c in frontier.customdata] == [0, 1, 3]


def test_cloud_customdata_is_source_row_index() -> None:
    cloud = _trace(pareto_scatter(_sweep()), "Dominated")
    assert [int(c[0]) for c in cloud.customdata] == [2]  # the lone dominated row


def test_customdata_empty_when_subset_empty() -> None:
    fig = pareto_scatter(_sweep().assign(on_frontier=[True, True, True, True]))
    cloud = _trace(fig, "Dominated")
    assert cloud.customdata is None or len(cloud.customdata) == 0
