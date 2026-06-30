"""Validate the simulator against the safety-stock law.

Two layers:

* **Layer A** pins the closed-form helpers ``analytics.classical.safety_stock``
  (``z·σ·√L``) and ``z_for_service_level`` (``Φ⁻¹``) and their input guards.
* **Layer B** is a single-seed-class simulator sanity check: provisioning safety
  stock for a *rising* target service level must raise the *achieved* service
  level. This validates the **direction** of the law — safety stock buys service
  — not the magnitude. The rigorous ``|achieved − target|`` claim needs many
  Monte Carlo replications and is deferred to M3; demand here is stochastic, so
  the measurement is averaged over a handful of seeds only to stabilize it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import pairwise

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from analytics.classical import safety_stock, z_for_service_level
from core.config import (
    CostConfig,
    DeterministicLeadTimeConfig,
    NormalDemandConfig,
    RunConfig,
    SimulationConfig,
    SQPolicyConfig,
)

# --------------------------------------------------------------------------- #
# Layer A — the closed-form helpers                                           #
# --------------------------------------------------------------------------- #


def test_safety_stock_matches_textbook_formula() -> None:
    # z·σ·√L = 2·3·√4 = 12.
    assert safety_stock(2.0, 3.0, 4.0) == pytest.approx(2.0 * 3.0 * math.sqrt(4.0))
    assert safety_stock(2.0, 3.0, 4.0) == pytest.approx(12.0)


def test_safety_stock_scales_with_sqrt_lead_time() -> None:
    # Safety stock scales with √L: quadrupling the lead time doubles it.
    base = safety_stock(1.645, 3.0, 2.0)
    assert safety_stock(1.645, 3.0, 8.0) == pytest.approx(2.0 * base)


def test_safety_stock_zero_variability_or_lead_time_is_zero() -> None:
    assert safety_stock(1.645, 0.0, 4.0) == 0.0  # no demand variability
    assert safety_stock(1.645, 3.0, 0.0) == 0.0  # no lead time to cover


def test_safety_stock_negative_z_gives_negative_stock() -> None:
    # A sub-50% service level (z < 0) means holding less than mean lead-time demand.
    assert safety_stock(-1.0, 3.0, 4.0) < 0.0


@pytest.mark.parametrize("demand_std", [-0.1, -1.0])
def test_safety_stock_negative_std_raises(demand_std: float) -> None:
    with pytest.raises(ValueError, match="demand_std must be >= 0"):
        safety_stock(1.645, demand_std, 4.0)


def test_safety_stock_negative_lead_time_raises() -> None:
    with pytest.raises(ValueError, match="lead_time must be >= 0"):
        safety_stock(1.645, 3.0, -1.0)


def test_z_for_service_level_known_quantiles() -> None:
    assert z_for_service_level(0.5) == pytest.approx(0.0, abs=1e-9)
    assert z_for_service_level(0.95) == pytest.approx(1.6448536, abs=1e-6)
    assert z_for_service_level(0.975) == pytest.approx(1.9599640, abs=1e-6)
    assert z_for_service_level(0.99) == pytest.approx(2.3263479, abs=1e-6)


def test_z_for_service_level_is_monotone_increasing() -> None:
    levels = [0.6, 0.8, 0.9, 0.95, 0.99]
    zs = [z_for_service_level(sl) for sl in levels]
    assert all(b > a for a, b in pairwise(zs))


@pytest.mark.parametrize("service_level", [0.0, 1.0, -0.5, 1.5])
def test_z_for_service_level_out_of_range_raises(service_level: float) -> None:
    with pytest.raises(ValueError, match=r"service_level must be in \(0, 1\)"):
        z_for_service_level(service_level)


@given(
    z=st.floats(min_value=-5.0, max_value=5.0),
    demand_std=st.floats(min_value=0.0, max_value=1e4),
    lead_time=st.floats(min_value=0.0, max_value=1e4),
)
def test_safety_stock_finite_and_sign_tracks_z(
    z: float, demand_std: float, lead_time: float
) -> None:
    ss = safety_stock(z, demand_std, lead_time)
    assert math.isfinite(ss)
    # demand_std, lead_time >= 0 by construction, so sign(ss) follows sign(z)
    # (an all-tiny product may underflow to exactly 0.0, which satisfies both bounds).
    if z >= 0:
        assert ss >= 0.0
    else:
        assert ss <= 0.0


# --------------------------------------------------------------------------- #
# Layer B — simulator sanity: safety stock buys service level                 #
# --------------------------------------------------------------------------- #

_MEAN = 10.0
_STD = 3.0
_LEAD_TIME = 4
_ORDER_QUANTITY = 40
_HORIZON = 730
_SEEDS = range(10)  # averaged only to stabilize the stochastic measurement
_SERVICE_LEVELS = [0.80, 0.90, 0.95, 0.99]


def _safety_stock_scenario(service_level: float, seed: int) -> RunConfig:
    """Normal-demand ``(s,Q)`` whose reorder point carries safety stock for ``service_level``.

    ``s = μL + z·σ·√L`` — cycle stock plus the safety stock under test. Raising
    the target service level raises ``z``, hence ``s``. Lost-sales mode so unmet
    demand is a clean per-period shortfall the cycle-service measure can read.
    """
    z = z_for_service_level(service_level)
    s = _MEAN * _LEAD_TIME + safety_stock(z, _STD, float(_LEAD_TIME))
    return RunConfig(
        simulation=SimulationConfig(
            horizon=_HORIZON, initial_on_hand=80, backorder_policy="lost_sales"
        ),
        demand=NormalDemandConfig(mean=_MEAN, std=_STD),
        lead_time=DeterministicLeadTimeConfig(lead_time=_LEAD_TIME),
        policy=SQPolicyConfig(reorder_point=s, order_quantity=_ORDER_QUANTITY),
        costs=CostConfig(
            unit_cost=0.0,
            holding_per_unit_per_period=1.0,
            ordering_fixed=20.0,
            lost_sale_per_unit=10.0,
        ),
        master_seed=seed,
    )


def _cycle_service_level(df: pd.DataFrame) -> float:
    """Fraction of replenishment cycles with no stockout — the measure ``z`` targets.

    A cycle runs between consecutive order arrivals (``order_received > 0``); a
    cycle "stocks out" if any of its periods had a lost sale. This is the
    cycle-service-level form of ``z = Φ⁻¹(service level)``.
    """
    cycle = (df["order_received"] > 0).cumsum()
    had_stockout = df.groupby(cycle)["lost_sales"].max() > 0
    return 1.0 - float(had_stockout.mean())


def _mean_cycle_service_level(
    run_to_ledger: Callable[[RunConfig], pd.DataFrame], service_level: float
) -> float:
    return sum(
        _cycle_service_level(run_to_ledger(_safety_stock_scenario(service_level, seed)))
        for seed in _SEEDS
    ) / len(_SEEDS)


def test_higher_service_level_raises_achieved_service(
    run_to_ledger: Callable[[RunConfig], pd.DataFrame],
) -> None:
    achieved = [_mean_cycle_service_level(run_to_ledger, sl) for sl in _SERVICE_LEVELS]

    # (1) Achieved service rises monotonically with the target service level.
    assert achieved == sorted(achieved), f"achieved service not monotone in target: {achieved}"

    # (2) The lift is real — provisioning for 99% clearly beats 80% (direction,
    #     not magnitude; |achieved - target| rigor is deferred to M3).
    assert achieved[-1] - achieved[0] > 0.02, f"service-level lift too small: {achieved}"
