"""Integration: Pareto sweep — policy-grid service-vs-cost frontier."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from analytics.pareto import _pareto_mask, pareto_sweep
from core.config import (
    CostConfig,
    DeterministicLeadTimeConfig,
    NormalDemandConfig,
    RunConfig,
    SimulationConfig,
    SSPolicyConfig,
)

_KPI_COLUMNS = (
    "cycle_service_level",
    "fill_rate",
    "ready_rate",
    "total_holding_cost",
    "total_ordering_cost",
    "total_purchase_cost",
    "total_stockout_cost",
    "total_cost",
    "order_count",
    "average_on_hand",
    "inventory_turns",
    "days_of_supply",
)


def _ss_config() -> RunConfig:
    """An (s,S) config (the make_config fixture is (s,Q)) for the cross-field test."""
    return RunConfig(
        simulation=SimulationConfig(horizon=30, initial_on_hand=50),
        demand=NormalDemandConfig(mean=10.0, std=2.0),
        lead_time=DeterministicLeadTimeConfig(lead_time=2),
        policy=SSPolicyConfig(reorder_point=20.0, order_up_to=50.0),
        costs=CostConfig(
            unit_cost=5.0,
            holding_per_unit_per_period=0.5,
            ordering_fixed=20.0,
            backorder_per_unit_per_period=2.0,
        ),
        master_seed=42,
    )


def test_shape_and_columns(make_config: Callable[..., RunConfig]) -> None:
    df = pareto_sweep(
        make_config(),
        {"reorder_point": [15.0, 25.0], "order_quantity": [20, 40]},
        n_replications=5,
        n_jobs=1,
    )
    assert len(df) == 4  # 2x2 grid
    for col in ("reorder_point", "order_quantity", *_KPI_COLUMNS, "on_frontier"):
        assert col in df.columns


def test_pareto_mask_known_case() -> None:
    # minimize cost, maximize service. C is dominated by A (cheaper, equal service).
    cost = np.array([10.0, 20.0, 20.0, 30.0])
    service = np.array([0.90, 0.95, 0.90, 0.99])
    assert list(_pareto_mask(cost, service)) == [True, True, False, True]


def test_pareto_mask_ties_both_kept() -> None:
    cost = np.array([10.0, 10.0])
    service = np.array([0.9, 0.9])
    assert list(_pareto_mask(cost, service)) == [True, True]


def test_frontier_nonempty_and_monotone(make_config: Callable[..., RunConfig]) -> None:
    df = pareto_sweep(
        make_config(),
        {"reorder_point": [10.0, 20.0, 30.0], "order_quantity": [20, 30, 40]},
        n_replications=8,
        n_jobs=1,
    )
    frontier = df[df["on_frontier"]].sort_values("total_cost")
    assert len(frontier) >= 1
    # Efficient frontier: sorted by cost ascending, service is non-decreasing.
    assert frontier["fill_rate"].is_monotonic_increasing


def test_determinism(make_config: Callable[..., RunConfig]) -> None:
    grid: dict[str, list[float | int]] = {"reorder_point": [15.0, 25.0], "order_quantity": [25, 35]}
    a = pareto_sweep(make_config(), grid, n_replications=6, n_jobs=1)
    b = pareto_sweep(make_config(), grid, n_replications=6, n_jobs=1)
    pd.testing.assert_frame_equal(a, b)


def test_empty_grid_raises(make_config: Callable[..., RunConfig]) -> None:
    with pytest.raises(ValueError, match="param_grid must be non-empty"):
        pareto_sweep(make_config(), {}, n_replications=3, n_jobs=1)


def test_unknown_param_raises(make_config: Callable[..., RunConfig]) -> None:
    # pydantic ValidationError (extra="forbid") is a subclass of ValueError.
    with pytest.raises(ValueError):
        pareto_sweep(make_config(), {"nope": [1, 2]}, n_replications=3, n_jobs=1)


def test_bad_kpi_column_raises(make_config: Callable[..., RunConfig]) -> None:
    with pytest.raises(ValueError, match="not a KPI column"):
        pareto_sweep(
            make_config(),
            {"reorder_point": [15.0, 25.0]},
            n_replications=3,
            n_jobs=1,
            cost_kpi="bogus",
        )


def test_ss_cross_field_invalid_combo_raises() -> None:
    # (s,S) requires reorder_point < order_up_to; a grid point with s >= S must fail loud,
    # proving model_validate enforces the cross-field invariant per grid point.
    with pytest.raises(ValueError):
        pareto_sweep(
            _ss_config(),
            {"reorder_point": [10.0, 60.0]},  # 60 >= order_up_to=50 -> invalid
            n_replications=3,
            n_jobs=1,
        )
