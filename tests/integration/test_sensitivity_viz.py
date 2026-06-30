"""Integration: the tornado factory — one-at-a-time sensitivity ranking."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import pytest

from analytics.sensitivity import sensitivity_oat
from core.config import RunConfig
from visualization.sensitivity import tornado_chart

_BASELINE = 100.0

_COLUMNS = [
    "parameter",
    "low",
    "high",
    "kpi_low",
    "kpi_baseline",
    "kpi_high",
    "swing",
    "abs_swing",
]


def _frame() -> pd.DataFrame:
    """A 3-row tornado frame mirroring sensitivity_oat's columns, sorted abs_swing descending."""
    rows = [
        # parameter, low, high, kpi_low, kpi_baseline, kpi_high, swing, abs_swing
        ("costs.ordering_fixed", 10.0, 50.0, 80.0, _BASELINE, 160.0, 80.0, 80.0),
        ("demand.std", 1.0, 5.0, 130.0, _BASELINE, 90.0, -40.0, 40.0),
        ("policy.reorder_point", 10.0, 30.0, 95.0, _BASELINE, 108.0, 13.0, 13.0),
    ]
    return pd.DataFrame(rows, columns=_COLUMNS)


def test_returns_figure() -> None:
    assert isinstance(tornado_chart(_frame()), go.Figure)


def test_two_horizontal_bar_traces() -> None:
    fig = tornado_chart(_frame())
    assert [t.type for t in fig.data] == ["bar", "bar"]
    assert [t.orientation for t in fig.data] == ["h", "h"]
    assert [t.name for t in fig.data] == ["Parameter low", "Parameter high"]


def test_bars_anchored_at_baseline() -> None:
    fig = tornado_chart(_frame())
    assert fig.data[0].base == _BASELINE
    assert fig.data[1].base == _BASELINE


def test_bar_endpoints_are_kpi_values() -> None:
    # base + signed length == the true kpi_low / kpi_high per row (the geometry proof).
    df = _frame()
    fig = tornado_chart(df)
    low_ends = [fig.data[0].base + d for d in fig.data[0].x]
    high_ends = [fig.data[1].base + d for d in fig.data[1].x]
    assert low_ends == df["kpi_low"].tolist()
    assert high_ends == df["kpi_high"].tolist()


def test_parameters_in_frame_order() -> None:
    fig = tornado_chart(_frame())
    assert list(fig.data[0].y) == ["costs.ordering_fixed", "demand.std", "policy.reorder_point"]


def test_widest_swing_on_top() -> None:
    # autorange reversed => row 0 (widest abs_swing) renders at the top (the tornado convention).
    assert tornado_chart(_frame()).layout.yaxis.autorange == "reversed"


def test_baseline_reference_line() -> None:
    fig = tornado_chart(_frame())
    assert len(fig.layout.shapes) == 1  # the dashed baseline line
    texts = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "baseline" in texts


def test_axis_title_uses_output_kpi() -> None:
    fig = tornado_chart(_frame(), output_kpi="fill_rate")
    assert fig.layout.xaxis.title.text == "fill_rate"
    assert "fill_rate" in fig.layout.title.text


def test_hover_carries_settings() -> None:
    fig = tornado_chart(_frame())
    # low trace customdata per row = [low setting, kpi_low]; high trace = [high setting, kpi_high].
    assert list(fig.data[0].customdata[0]) == [10.0, 80.0]
    assert list(fig.data[1].customdata[0]) == [50.0, 160.0]


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        tornado_chart(pd.DataFrame())


def test_missing_columns_raises() -> None:
    df = _frame().drop(columns=["abs_swing"])  # drop the tornado-order proof column
    with pytest.raises(ValueError, match="missing required columns"):
        tornado_chart(df)


def test_smoke_real_sensitivity(make_config: Callable[..., RunConfig]) -> None:
    """Faithfully renders a real sensitivity_oat frame (analytics owns the numbers)."""
    df = sensitivity_oat(
        make_config(),
        {"costs.ordering_fixed": (10.0, 40.0), "demand.std": (1.0, 5.0)},
        n_replications=6,
        n_jobs=1,
    )
    fig = tornado_chart(df)
    assert [t.type for t in fig.data] == ["bar", "bar"]
    assert list(fig.data[0].y) == df["parameter"].tolist()
