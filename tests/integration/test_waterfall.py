"""Integration: cost-decomposition waterfall factory."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import pytest

from analytics.kpis import compute_kpis
from core.config import RunConfig
from experiments import single_run
from visualization.waterfall import cost_waterfall


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "holding_cost": [1.0, 2.0, 3.0],  # sum 6
            "ordering_cost": [0.0, 20.0, 0.0],  # sum 20
            "stockout_cost": [0.0, 0.0, 4.0],  # sum 4
            "purchase_cost": [50.0, 50.0, 50.0],  # sum 150
        }
    )


def test_returns_figure() -> None:
    assert isinstance(cost_waterfall(_frame()), go.Figure)


def test_default_excludes_purchase() -> None:
    fig = cost_waterfall(_frame())
    wf = fig.data[0]
    assert list(wf.x) == ["Holding", "Ordering", "Stockout", "Total controllable"]
    assert list(wf.measure) == ["relative", "relative", "relative", "total"]
    assert list(wf.y)[:3] == [6.0, 20.0, 4.0]
    assert fig.layout.title.text == "Controllable cost decomposition"


def test_include_purchase_adds_bar_and_relabels() -> None:
    fig = cost_waterfall(_frame(), include_purchase=True)
    wf = fig.data[0]
    assert list(wf.x) == ["Holding", "Ordering", "Stockout", "Purchase", "Total"]
    assert list(wf.y)[:4] == [6.0, 20.0, 4.0, 150.0]
    assert fig.layout.title.text == "Cost decomposition"


def test_default_missing_controllable_column_raises() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        cost_waterfall(_frame().drop(columns=["holding_cost"]))


def test_purchase_column_required_only_in_full_mode() -> None:
    df = _frame().drop(columns=["purchase_cost"])
    cost_waterfall(df)  # default mode never reads purchase -> no raise
    with pytest.raises(ValueError, match="missing required columns"):
        cost_waterfall(df, include_purchase=True)


def test_full_total_reconciles_with_kpis_total_cost(
    make_config: Callable[..., RunConfig],
) -> None:
    df = single_run.run(make_config())
    fig = cost_waterfall(df, include_purchase=True)
    component_sum = sum(fig.data[0].y[:4])  # holding + ordering + stockout + purchase
    assert component_sum == pytest.approx(compute_kpis(df).total_cost)


def test_smoke_default(make_config: Callable[..., RunConfig]) -> None:
    assert isinstance(cost_waterfall(single_run.run(make_config())), go.Figure)
