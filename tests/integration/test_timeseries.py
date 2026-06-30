"""Integration: inventory-trajectory chart factory."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import pytest

from core.config import RunConfig
from experiments import single_run
from visualization.timeseries import inventory_timeseries


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": [0, 1, 2, 3],
            "on_hand": [100.0, 90.0, 80.0, 110.0],
            "on_order": [0.0, 30.0, 30.0, 0.0],
            "inventory_position": [100.0, 120.0, 110.0, 110.0],
            "demand": [10.0, 11.0, 9.0, 10.0],
        }
    )


def test_returns_figure() -> None:
    assert isinstance(inventory_timeseries(_frame()), go.Figure)


def test_trace_names_in_order() -> None:
    fig = inventory_timeseries(_frame())
    assert [t.name for t in fig.data] == ["On hand", "Inventory position", "On order", "Demand"]


def test_line_trace_encodes_its_column() -> None:
    df = _frame()
    fig = inventory_timeseries(df)
    on_hand = next(t for t in fig.data if t.name == "On hand")
    assert list(on_hand.y) == df["on_hand"].tolist()
    assert list(on_hand.x) == df["period"].tolist()


def test_demand_is_bars_on_secondary_axis() -> None:
    fig = inventory_timeseries(_frame())
    demand = next(t for t in fig.data if t.name == "Demand")
    assert isinstance(demand, go.Bar)
    assert demand.yaxis == "y2"


def test_layout_axis_titles() -> None:
    fig = inventory_timeseries(_frame())
    assert fig.layout.xaxis.title.text == "Period"
    assert fig.layout.yaxis.title.text == "Units"


@pytest.mark.parametrize("drop", ["period", "on_hand", "on_order", "inventory_position", "demand"])
def test_missing_column_raises(drop: str) -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        inventory_timeseries(_frame().drop(columns=[drop]))


def test_smoke_on_real_ledger(make_config: Callable[..., RunConfig]) -> None:
    df = single_run.run(make_config())
    fig = inventory_timeseries(df)
    assert len(fig.data) == 4
    on_hand = next(t for t in fig.data if t.name == "On hand")
    assert len(on_hand.x) == len(df)


def test_no_marker_by_default() -> None:
    fig = inventory_timeseries(_frame())
    assert len(fig.layout.shapes) == 0


def test_explicit_none_is_marker_free() -> None:
    assert len(inventory_timeseries(_frame(), highlight_period=None).layout.shapes) == 0


def test_highlight_period_adds_one_marker() -> None:
    fig = inventory_timeseries(_frame(), highlight_period=2)
    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0].x0 == 2


@pytest.mark.parametrize("period", [0, 3])
def test_highlight_period_at_boundaries(period: int) -> None:
    fig = inventory_timeseries(_frame(), highlight_period=period)
    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0].x0 == period
