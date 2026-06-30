"""Integration: the bullwhip bar factory — order-vs-demand variance amplification."""

from __future__ import annotations

import math
from collections.abc import Callable

import plotly.graph_objects as go
import pytest

from analytics.bullwhip import BullwhipMetrics, bullwhip_metrics
from core.config import BaseStockPolicyConfig, RunConfig
from experiments import single_run
from visualization.bullwhip import bullwhip_bar


def _metrics() -> dict[str, BullwhipMetrics]:
    """Two synthetic bundles: a smoothing run (ratio < 1) and an amplifying run (ratio > 1)."""
    return {
        "base_stock(50)": BullwhipMetrics(
            demand_mean=10.0,
            demand_variance=4.0,
            order_mean=10.0,
            order_variance=2.0,
            bullwhip_ratio=0.5,
        ),
        "sQ(20,30)": BullwhipMetrics(
            demand_mean=10.0,
            demand_variance=4.0,
            order_mean=20.0,
            order_variance=400.0,
            bullwhip_ratio=100.0,
        ),
    }


def test_returns_figure() -> None:
    assert isinstance(bullwhip_bar(_metrics()), go.Figure)


def test_bar_heights_are_ratios() -> None:
    fig = bullwhip_bar(_metrics())
    assert fig.data[0].type == "bar"
    assert list(fig.data[0].y) == [0.5, 100.0]


def test_labels_preserved_in_order() -> None:
    assert list(bullwhip_bar(_metrics()).data[0].x) == ["base_stock(50)", "sQ(20,30)"]


def test_pass_through_reference_line() -> None:
    fig = bullwhip_bar(_metrics())
    assert len(fig.layout.shapes) == 1  # the dashed y=1.0 line
    texts = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "pass-through" in texts


def test_hover_carries_both_variances() -> None:
    fig = bullwhip_bar(_metrics())
    # customdata per bar = [Var(demand), Var(orders)]
    assert list(fig.data[0].customdata[0]) == [4.0, 2.0]
    assert list(fig.data[0].customdata[1]) == [4.0, 400.0]


def test_nan_ratio_does_not_crash() -> None:
    metrics = {
        "constant_demand": BullwhipMetrics(
            demand_mean=10.0,
            demand_variance=0.0,
            order_mean=30.0,
            order_variance=900.0,
            bullwhip_ratio=float("nan"),
        )
    }
    fig = bullwhip_bar(metrics)  # nan ratio is a documented valid output, not an error
    assert math.isnan(fig.data[0].y[0])


def test_log_y_sets_axis_type() -> None:
    assert bullwhip_bar(_metrics(), log_y=True).layout.yaxis.type == "log"


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        bullwhip_bar({})


def test_smoke_real_runs(make_config: Callable[..., RunConfig]) -> None:
    """Faithfully renders real computed metrics (no flaky inequality — analytics owns that)."""
    sq = single_run.run(make_config())  # the fixture is (s,Q)
    bs = single_run.run(
        make_config().model_copy(update={"policy": BaseStockPolicyConfig(target_level=50.0)})
    )
    m_sq = bullwhip_metrics(sq)
    m_bs = bullwhip_metrics(bs)
    fig = bullwhip_bar({"sQ": m_sq, "base_stock": m_bs})
    assert fig.data[0].type == "bar"
    assert list(fig.data[0].y) == pytest.approx([m_sq.bullwhip_ratio, m_bs.bullwhip_ratio])
