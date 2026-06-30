"""Integration: bullwhip metric — order-vs-demand variance amplification."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from analytics.bullwhip import BullwhipMetrics, bullwhip_metrics
from core.config import BaseStockPolicyConfig, RunConfig
from experiments import single_run


def test_exact_hand_computed() -> None:
    # demand var = 4.0 (10 ± 2); orders mean 20, var 400.0; ratio = 400 / 4 = 100.
    df = pd.DataFrame({"demand": [8.0, 12.0, 8.0, 12.0], "order_placed": [0.0, 40.0, 0.0, 40.0]})
    m = bullwhip_metrics(df)
    assert m.demand_mean == pytest.approx(10.0)
    assert m.demand_variance == pytest.approx(4.0)
    assert m.order_mean == pytest.approx(20.0)
    assert m.order_variance == pytest.approx(400.0)
    assert m.bullwhip_ratio == pytest.approx(100.0)


def test_sq_clearly_amplifies(make_config: Callable[..., RunConfig]) -> None:
    # The concept being taught: an (s,Q) batching policy amplifies demand variance.
    m = bullwhip_metrics(single_run.run(make_config()))
    assert m.bullwhip_ratio > 1.0


def test_base_stock_smooths_relative_to_sq(make_config: Callable[..., RunConfig]) -> None:
    # Seed-averaged (de-flaked): an order-up-to policy reorders ~what was consumed, so it
    # tracks demand far more closely than (s,Q) batching. Averaged over seeds to remove
    # single-seed wobble (the margin is ~16x, so this is robust).
    def mean_ratio(policy: object | None) -> float:
        ratios = []
        for seed in range(5):
            cfg = make_config(master_seed=seed)
            if policy is not None:
                cfg = cfg.model_copy(update={"policy": policy})
            ratios.append(bullwhip_metrics(single_run.run(cfg)).bullwhip_ratio)
        return float(np.mean(ratios))

    sq_mean = mean_ratio(None)  # the fixture is (s,Q)
    base_stock_mean = mean_ratio(BaseStockPolicyConfig(target_level=50.0))
    assert base_stock_mean < sq_mean


def test_zero_demand_variance_gives_nan_ratio() -> None:
    # Constant demand -> Var(demand) = 0 -> ratio undefined (nan), not a crash.
    df = pd.DataFrame({"demand": [10.0, 10.0, 10.0], "order_placed": [0.0, 30.0, 0.0]})
    m = bullwhip_metrics(df)
    assert m.demand_variance == 0.0
    assert np.isnan(m.bullwhip_ratio)


def test_determinism_pure_reduction(make_config: Callable[..., RunConfig]) -> None:
    df = single_run.run(make_config())
    assert bullwhip_metrics(df) == bullwhip_metrics(df)  # frozen dataclass equality


def test_returns_bullwhip_metrics_instance() -> None:
    df = pd.DataFrame({"demand": [1.0, 2.0, 3.0], "order_placed": [0.0, 5.0, 0.0]})
    assert isinstance(bullwhip_metrics(df), BullwhipMetrics)


def test_missing_demand_column_raises() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        bullwhip_metrics(pd.DataFrame({"order_placed": [0.0, 5.0]}))


def test_missing_order_placed_column_raises() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        bullwhip_metrics(pd.DataFrame({"demand": [1.0, 2.0]}))
