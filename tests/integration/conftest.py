"""Shared fixtures for engine integration tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from core.config import (
    CostConfig,
    DeterministicLeadTimeConfig,
    NormalDemandConfig,
    RunConfig,
    SimulationConfig,
    SQPolicyConfig,
)


def _make_config(
    *,
    horizon: int = 50,
    initial_on_hand: int = 100,
    initial_on_order: int = 0,
    backorder_policy: str = "backorder",
    demand_mean: float = 10.0,
    demand_std: float = 2.0,
    lead_time: int = 3,
    master_seed: int = 42,
) -> RunConfig:
    return RunConfig(
        simulation=SimulationConfig(
            horizon=horizon,
            initial_on_hand=initial_on_hand,
            initial_on_order=initial_on_order,
            backorder_policy=backorder_policy,  # type: ignore[arg-type]
        ),
        demand=NormalDemandConfig(mean=demand_mean, std=demand_std),
        lead_time=DeterministicLeadTimeConfig(lead_time=lead_time),
        policy=SQPolicyConfig(reorder_point=20.0, order_quantity=30),
        costs=CostConfig(
            unit_cost=5.0,
            holding_per_unit_per_period=0.5,
            ordering_fixed=20.0,
            backorder_per_unit_per_period=2.0,
            lost_sale_per_unit=10.0,
        ),
        master_seed=master_seed,
    )


@pytest.fixture
def basic_config() -> RunConfig:
    return _make_config()


@pytest.fixture
def make_config() -> Callable[..., RunConfig]:
    return _make_config
