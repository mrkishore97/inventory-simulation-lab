"""Integration: policy-comparison factories — the race."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import pytest

from analytics.kpis import KPIs, compute_kpis
from core.config import RunConfig
from experiments import single_run
from visualization.comparison import comparison_table, policy_race_timeseries


def _ledgers() -> dict[str, pd.DataFrame]:
    period = [0, 1, 2, 3]
    return {
        "sQ(20,30)": pd.DataFrame(
            {
                "period": period,
                "on_hand": [100.0, 90.0, 80.0, 110.0],
                "inventory_position": [100.0, 120.0, 110.0, 110.0],
                "on_order": [0.0, 30.0, 30.0, 0.0],
            }
        ),
        "base_stock(60)": pd.DataFrame(
            {
                "period": period,
                "on_hand": [60.0, 52.0, 48.0, 55.0],
                "inventory_position": [60.0, 60.0, 60.0, 60.0],
                "on_order": [0.0, 8.0, 12.0, 5.0],
            }
        ),
    }


def _kpis(total_cost: float) -> KPIs:
    return KPIs(
        cycle_service_level=0.9,
        fill_rate=0.95,
        ready_rate=0.8,
        total_holding_cost=100.0,
        total_ordering_cost=200.0,
        total_purchase_cost=300.0,
        total_stockout_cost=50.0,
        total_cost=total_cost,
        order_count=12.0,
        average_on_hand=53.0,
        inventory_turns=2.0,
        days_of_supply=5.3,
    )


# --- policy_race_timeseries ------------------------------------------------------------


def test_returns_figure() -> None:
    assert isinstance(policy_race_timeseries(_ledgers()), go.Figure)


def test_one_trace_per_policy_named_by_label() -> None:
    fig = policy_race_timeseries(_ledgers())
    assert [t.name for t in fig.data] == ["sQ(20,30)", "base_stock(60)"]


def test_default_metric_is_on_hand() -> None:
    ledgers = _ledgers()
    fig = policy_race_timeseries(ledgers)
    sq = next(t for t in fig.data if t.name == "sQ(20,30)")
    assert list(sq.y) == ledgers["sQ(20,30)"]["on_hand"].tolist()


def test_metric_override() -> None:
    ledgers = _ledgers()
    fig = policy_race_timeseries(ledgers, metric="inventory_position")
    sq = next(t for t in fig.data if t.name == "sQ(20,30)")
    assert list(sq.y) == ledgers["sQ(20,30)"]["inventory_position"].tolist()
    assert fig.layout.yaxis.title.text == "Inventory position (units)"


def test_empty_ledgers_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        policy_race_timeseries({})


def test_missing_metric_column_raises() -> None:
    ledgers = _ledgers()
    ledgers["sQ(20,30)"] = ledgers["sQ(20,30)"].drop(columns=["on_hand"])
    with pytest.raises(ValueError, match="missing required columns"):
        policy_race_timeseries(ledgers)


def test_missing_period_column_raises() -> None:
    ledgers = _ledgers()
    ledgers["base_stock(60)"] = ledgers["base_stock(60)"].drop(columns=["period"])
    with pytest.raises(ValueError, match="missing required columns"):
        policy_race_timeseries(ledgers)


def test_smoke_on_real_multi_policy_run(make_config: Callable[..., RunConfig]) -> None:
    base = make_config()
    policies = {
        "sQ": {"kind": "sQ", "reorder_point": 20.0, "order_quantity": 30},
        "base_stock": {"kind": "base_stock", "target_level": 60.0},
    }
    ledgers = {
        label: single_run.run(RunConfig.model_validate({**base.model_dump(), "policy": p}))
        for label, p in policies.items()
    }
    # Same master_seed, different policy -> identical demand stream (the race premise).
    assert ledgers["sQ"]["demand"].equals(ledgers["base_stock"]["demand"])
    fig = policy_race_timeseries(ledgers)
    assert len(fig.data) == 2


# --- policy_race_timeseries: disruption-window shading -----------------------------

_WINDOWS = [(30, 60, "LT ×2"), (80, 90, "LT ×3")]


def test_shade_windows_adds_a_rect_per_window() -> None:
    fig = policy_race_timeseries(_ledgers(), shade_windows=_WINDOWS)
    rects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(rects) == 2
    assert [(r.x0, r.x1) for r in rects] == [(30, 60), (80, 90)]


def test_shade_windows_labels_each_band() -> None:
    # The label carries the severity (the multiplier), not just the timing.
    fig = policy_race_timeseries(_ledgers(), shade_windows=_WINDOWS)
    assert [a.text for a in fig.layout.annotations] == ["LT ×2", "LT ×3"]


def test_no_shade_windows_is_unshaded() -> None:
    # Default (and an empty list) leave the figure band-free — byte-identical to the race.
    assert len(policy_race_timeseries(_ledgers()).layout.shapes) == 0
    assert len(policy_race_timeseries(_ledgers(), shade_windows=[]).layout.shapes) == 0


def test_shade_windows_keeps_one_trace_per_policy() -> None:
    fig = policy_race_timeseries(_ledgers(), shade_windows=[(30, 60, "LT ×2")])
    assert [t.name for t in fig.data] == ["sQ(20,30)", "base_stock(60)"]


# --- comparison_table ------------------------------------------------------------------


def test_table_one_row_per_policy() -> None:
    table = comparison_table({"a": _kpis(1000.0), "b": _kpis(2000.0)})
    assert list(table.index) == ["a", "b"]


def test_table_columns_are_the_kpi_fields() -> None:
    table = comparison_table({"a": _kpis(1000.0)})
    assert list(table.columns) == [f.name for f in dataclasses.fields(KPIs)]


def test_table_values_match_asdict() -> None:
    table = comparison_table({"a": _kpis(1234.0)})
    assert table.loc["a", "total_cost"] == 1234.0
    assert table.loc["a", "fill_rate"] == 0.95


def test_table_order_preserved() -> None:
    table = comparison_table({"z": _kpis(1.0), "a": _kpis(2.0), "m": _kpis(3.0)})
    assert list(table.index) == ["z", "a", "m"]


def test_table_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        comparison_table({})


def test_table_from_real_kpis(make_config: Callable[..., RunConfig]) -> None:
    df = single_run.run(make_config())
    table = comparison_table({"baseline": compute_kpis(df)})
    assert table.shape == (1, len(dataclasses.fields(KPIs)))
