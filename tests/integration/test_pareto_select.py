"""Unit: the Pareto click-to-load mapping — selection → swept params."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ui.components.pareto_select import clicked_index, clicked_params


def _sweep() -> pd.DataFrame:
    """A synthetic sweep frame: 2 swept params + 2 KPI means + on_frontier (RangeIndex 0..2)."""
    return pd.DataFrame(
        {
            "reorder_point": [10.0, 20.0, 30.0],
            "order_quantity": [20, 30, 40],
            "total_cost": [500.0, 450.0, 600.0],
            "fill_rate": [0.80, 0.90, 0.92],
            "on_frontier": [True, True, False],
        }
    )


def _selection(idx: int) -> dict[str, Any]:
    """A selection event as Streamlit delivers it: one point carrying its row index customdata."""
    return {"points": [{"customdata": [idx]}]}


def test_clicked_index_returns_customdata_row() -> None:
    assert clicked_index(_selection(2)) == 2


def test_clicked_index_none_when_nothing_selected() -> None:
    assert clicked_index(None) is None
    assert clicked_index({"points": []}) is None


def test_clicked_params_returns_only_swept_columns() -> None:
    params = clicked_params(_sweep(), _selection(1))
    assert params == {"reorder_point": 20.0, "order_quantity": 30}  # KPIs/on_frontier dropped


def test_clicked_params_casts_int_and_float_fields() -> None:
    params = clicked_params(_sweep(), _selection(1))
    assert params is not None
    assert isinstance(params["order_quantity"], int)  # int-field widget type
    assert isinstance(params["reorder_point"], float)


def test_clicked_params_none_when_nothing_selected() -> None:
    assert clicked_params(_sweep(), None) is None


def test_clicked_params_none_when_index_is_stale() -> None:
    # A click that predates a re-sweep to a smaller grid: the index no longer exists.
    assert clicked_params(_sweep(), _selection(99)) is None
