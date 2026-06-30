"""Integration: Monte Carlo runner — seeded replications + KPI distributions."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from analytics.kpis import compute_kpis
from analytics.monte_carlo import monte_carlo
from core.config import RunConfig
from experiments import single_run

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


def test_shape_and_metadata_columns(make_config: Callable[..., RunConfig]) -> None:
    frame = monte_carlo(make_config(), 50, n_jobs=1)
    assert len(frame) == 50
    assert list(frame.columns) == ["replication", "seed", *_KPI_COLUMNS]
    assert list(frame["replication"]) == list(range(50))


def test_determinism(make_config: Callable[..., RunConfig]) -> None:
    config = make_config()
    pd.testing.assert_frame_equal(
        monte_carlo(config, 20, n_jobs=1), monte_carlo(config, 20, n_jobs=1)
    )


def test_seed_column_reproduces_replication(make_config: Callable[..., RunConfig]) -> None:
    config = make_config()
    frame = monte_carlo(config, 10, n_jobs=1)

    seed = int(frame["seed"].iloc[7])
    reproduced = compute_kpis(single_run.run(config.model_copy(update={"master_seed": seed})))
    assert frame["total_cost"].iloc[7] == reproduced.total_cost
    assert frame["fill_rate"].iloc[7] == reproduced.fill_rate


def test_seed_is_float64_safe(make_config: Callable[..., RunConfig]) -> None:
    # 53-bit seeds stay exactly representable as float64, so even a row Series
    # (df.iloc[i], which upcasts the int seed to float) round-trips to the same seed.
    frame = monte_carlo(make_config(), 20, n_jobs=1)
    assert frame["seed"].max() <= 2**53
    assert int(frame.iloc[7]["seed"]) == int(frame["seed"].iloc[7])


def test_parallel_matches_serial(make_config: Callable[..., RunConfig]) -> None:
    config = make_config()
    pd.testing.assert_frame_equal(
        monte_carlo(config, 16, n_jobs=1), monte_carlo(config, 16, n_jobs=2)
    )


def test_replications_vary(make_config: Callable[..., RunConfig]) -> None:
    # std>0 demand -> distinct per-rep streams -> the distribution is not degenerate.
    frame = monte_carlo(make_config(demand_std=3.0), 30, n_jobs=1)
    assert frame["total_cost"].nunique() > 1


def test_different_master_seed_changes_distribution(make_config: Callable[..., RunConfig]) -> None:
    a = monte_carlo(make_config(master_seed=1), 20, n_jobs=1)
    b = monte_carlo(make_config(master_seed=2), 20, n_jobs=1)
    assert not a["total_cost"].equals(b["total_cost"])


def test_single_replication(make_config: Callable[..., RunConfig]) -> None:
    assert len(monte_carlo(make_config(), 1, n_jobs=1)) == 1


def test_zero_replications_raises(make_config: Callable[..., RunConfig]) -> None:
    with pytest.raises(ValueError, match="n_replications must be >= 1"):
        monte_carlo(make_config(), 0)
