"""Integration: the KPI-distribution histogram factory — MC spread + risk markers."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import pytest

from analytics.monte_carlo import monte_carlo
from analytics.risk import RiskMetrics, risk_metrics
from core.config import RunConfig
from visualization.distribution import kpi_distribution_histogram


def _dist() -> pd.DataFrame:
    """A small synthetic monte_carlo frame: meta cols + two KPI columns."""
    return pd.DataFrame(
        {
            "replication": [0, 1, 2, 3, 4],
            "seed": [10, 11, 12, 13, 14],
            "total_cost": [500.0, 520.0, 480.0, 600.0, 510.0],
            "fill_rate": [0.90, 0.92, 0.88, 0.95, 0.91],
        }
    )


def _risk() -> RiskMetrics:
    return RiskMetrics(
        column="total_cost",
        mean=522.0,
        value_at_risk=580.0,
        conditional_value_at_risk=600.0,
        confidence_level=0.95,
        tail_probability=0.05,
        tail="upper",
    )


def test_returns_figure() -> None:
    assert isinstance(kpi_distribution_histogram(_dist()), go.Figure)


def test_histogram_trace_on_column() -> None:
    fig = kpi_distribution_histogram(_dist())
    assert fig.data[0].type == "histogram"
    assert list(fig.data[0].x) == [500.0, 520.0, 480.0, 600.0, 510.0]


def test_default_column_is_total_cost() -> None:
    assert kpi_distribution_histogram(_dist()).layout.xaxis.title.text == "Total cost ($)"


def test_column_override_and_label_fallback() -> None:
    df = _dist().assign(custom_kpi=[1.0, 2.0, 3.0, 4.0, 5.0])
    fig = kpi_distribution_histogram(df, column="custom_kpi")
    assert fig.layout.xaxis.title.text == "custom_kpi"  # not in _LABELS -> raw-name fallback
    assert list(fig.data[0].x) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_no_risk_has_no_vlines() -> None:
    assert len(kpi_distribution_histogram(_dist()).layout.shapes) == 0


def test_risk_adds_mean_var_cvar_markers() -> None:
    fig = kpi_distribution_histogram(_dist(), risk=_risk())
    assert len(fig.layout.shapes) == 3  # mean + VaR + CVaR vertical lines
    texts = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "VaR" in texts and "CVaR" in texts and "Mean" in texts


def test_empty_frame_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        kpi_distribution_histogram(pd.DataFrame())


def test_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="no column"):
        kpi_distribution_histogram(_dist(), column="nonexistent_kpi")


def test_smoke_on_real_monte_carlo(make_config: Callable[..., RunConfig]) -> None:
    dist = monte_carlo(make_config(), 20, n_jobs=1)
    fig = kpi_distribution_histogram(dist, risk=risk_metrics(dist))
    assert fig.data[0].type == "histogram"
    assert len(fig.data[0].x) == 20
    assert len(fig.layout.shapes) == 3
