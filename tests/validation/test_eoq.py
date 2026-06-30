"""Validate the simulator against the EOQ law.

Two layers:

* **Layer A** pins the closed-form helper ``analytics.classical.eoq`` to the
  textbook ``sqrt(2 D K / h)`` and its input guards.
* **Layer B** is the digital-twin check: under deterministic demand, sweep the
  ``(s, Q)`` order quantity through the real engine and confirm the
  cost-minimizing ``Q`` matches the analytical EOQ. Only holding + ordering
  cost is compared — purchase cost is ``Q``-independent over the horizon, and
  stockout cost is held at zero by provisioning ``s`` above lead-time demand.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from analytics.classical import eoq
from core.config import (
    CostConfig,
    DeterministicLeadTimeConfig,
    NormalDemandConfig,
    RunConfig,
    SimulationConfig,
    SQPolicyConfig,
)

# --------------------------------------------------------------------------- #
# Layer A — the closed-form helper                                            #
# --------------------------------------------------------------------------- #


def test_eoq_matches_textbook_formula() -> None:
    assert eoq(10.0, 20.0, 1.0) == math.sqrt(2 * 10.0 * 20.0 / 1.0)
    assert eoq(10.0, 20.0, 1.0) == pytest.approx(20.0)


def test_eoq_quadrupling_order_cost_doubles_quantity() -> None:
    # Q* scales with sqrt(K): 4x the ordering cost -> 2x the quantity.
    base = eoq(10.0, 20.0, 1.0)
    assert eoq(10.0, 80.0, 1.0) == pytest.approx(2 * base)


@pytest.mark.parametrize("holding", [0.0, -1.0])
def test_eoq_nonpositive_holding_raises(holding: float) -> None:
    with pytest.raises(ValueError, match="holding_cost must be > 0"):
        eoq(10.0, 20.0, holding)


def test_eoq_negative_demand_raises() -> None:
    with pytest.raises(ValueError, match="demand_rate must be >= 0"):
        eoq(-1.0, 20.0, 1.0)


def test_eoq_negative_ordering_raises() -> None:
    with pytest.raises(ValueError, match="ordering_cost must be >= 0"):
        eoq(10.0, -1.0, 1.0)


@given(
    demand_rate=st.floats(min_value=0.0, max_value=1e6),
    ordering_cost=st.floats(min_value=0.0, max_value=1e6),
    holding_cost=st.floats(min_value=1e-6, max_value=1e6),
)
def test_eoq_is_nonnegative_and_finite(
    demand_rate: float, ordering_cost: float, holding_cost: float
) -> None:
    q = eoq(demand_rate, ordering_cost, holding_cost)
    assert q >= 0.0
    assert math.isfinite(q)


# --------------------------------------------------------------------------- #
# Layer B — simulator twin-match                                              #
# --------------------------------------------------------------------------- #

# Deterministic-demand scenario with analytical EOQ ~= 20:
#   EOQ = sqrt(2 * D * K / h) = sqrt(2 * 10.3 * 20 / 1) = 20.30  ->  round(EOQ) = 20.
# Demand is deliberately *non-integer* (10.3, not 10): with exactly-integer
# demand the deterministic sawtooth resonates with the integer period grid at
# every Q that is a whole multiple of the demand rate (inter-order time Q/d
# becomes an integer), spiking the cost at round Q values -- including the EOQ
# itself. A non-integer rate de-phases the sawtooth, giving the smooth unimodal
# U-shaped cost curve EOQ theory predicts.
_DEMAND_RATE = 10.3
_ORDERING_COST = 20.0
_HOLDING_COST = 1.0
_LEAD_TIME = 1
_REORDER_POINT = 50.0  # ~40 safety stock over lead-time demand 10.3 -> no stockouts
_Q_LO, _Q_HI = 12, 40  # sweep bounds; both > demand rate so every cycle is stable


def _eoq_scenario(order_quantity: int) -> RunConfig:
    """A deterministic-demand ``(s, Q)`` config with the given order quantity.

    Demand is exactly ``_DEMAND_RATE`` every period (Normal with std=0), so the
    only thing that varies across the sweep is ``Q``. ``s`` sits well above
    lead-time demand, so no stockout occurs and the holding/ordering trade-off
    that EOQ optimizes is isolated. Unit cost is 0 (purchase cost is excluded
    from the comparison regardless).
    """
    return RunConfig(
        simulation=SimulationConfig(horizon=365, initial_on_hand=60),
        demand=NormalDemandConfig(mean=_DEMAND_RATE, std=0.0),
        lead_time=DeterministicLeadTimeConfig(lead_time=_LEAD_TIME),
        policy=SQPolicyConfig(reorder_point=_REORDER_POINT, order_quantity=order_quantity),
        costs=CostConfig(
            unit_cost=0.0,
            holding_per_unit_per_period=_HOLDING_COST,
            ordering_fixed=_ORDERING_COST,
            backorder_per_unit_per_period=100.0,
        ),
        master_seed=42,
    )


def test_simulated_cost_minimizing_quantity_matches_eoq(
    run_to_ledger: Callable[[RunConfig], pd.DataFrame],
) -> None:
    analytical = eoq(_DEMAND_RATE, _ORDERING_COST, _HOLDING_COST)  # 20.30

    relevant_cost: dict[int, float] = {}
    for q in range(_Q_LO, _Q_HI + 1):
        df = run_to_ledger(_eoq_scenario(q))
        assert float(df["stockout_cost"].sum()) == 0.0, f"unexpected stockout at Q={q}"
        relevant_cost[q] = float((df["holding_cost"] + df["ordering_cost"]).sum())

    best_q = min(relevant_cost, key=lambda q: relevant_cost[q])

    # (1) The simulated cost-minimizing quantity matches the analytical EOQ,
    #     to within the integer rounding of a continuous optimum.
    assert abs(best_q - round(analytical)) <= 1, (
        f"simulated argmin Q={best_q} not within 1 of EOQ={analytical:.2f}; costs={relevant_cost}"
    )

    # (2) The curve is a genuine U: the optimum is strictly cheaper than both
    #     ends of the sweep (rules out a flat or monotone cost curve).
    assert relevant_cost[best_q] < relevant_cost[_Q_LO]
    assert relevant_cost[best_q] < relevant_cost[_Q_HI]

    # (3) The analytical EOQ itself realizes near-minimal cost -- the EOQ total-
    #     cost curve is famously flat near the optimum.
    assert relevant_cost[round(analytical)] <= relevant_cost[best_q] * 1.02
