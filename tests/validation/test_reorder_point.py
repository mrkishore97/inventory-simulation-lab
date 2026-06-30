"""Validate the simulator against the reorder-point law.

The reorder point is the ``(s, Q)`` trigger level: cycle stock ``μL`` + safety
stock ``z·σ·√L``. Two layers:

* **Layer A** pins the closed-form ``analytics.classical.reorder_point`` and its
  composition with the ``safety_stock`` helper.
* **Layer B** is a deterministic twin-match of the **cycle-stock** term: under
  σ=0 the reorder point is exactly ``μL``, and the simulator's on-hand floor
  sits at ``s − μL`` (so ``μL`` units are consumed during the lead time) with
  stockouts beginning right below ``μL``. The safety-stock term was validated
  stochastically elsewhere; here demand is deterministic, so the ``μL`` term is
  checked noise-free (the EOQ technique).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from analytics.classical import reorder_point, safety_stock
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


def test_reorder_point_matches_textbook_formula() -> None:
    # μL + z·σ·√L = 10·4 + 2·3·√4 = 40 + 12 = 52.
    assert reorder_point(10.0, 3.0, 4.0, 2.0) == pytest.approx(52.0)
    assert reorder_point(10.0, 3.0, 4.0, 2.0) == pytest.approx(40.0 + 2.0 * 3.0 * math.sqrt(4.0))


def test_reorder_point_is_cycle_stock_plus_safety_stock() -> None:
    # Composition identity: reorder point = cycle stock μL + the safety stock.
    assert reorder_point(10.0, 3.0, 4.0, 1.645) == pytest.approx(
        10.0 * 4.0 + safety_stock(1.645, 3.0, 4.0)
    )


def test_reorder_point_zero_safety_factor_is_cycle_stock() -> None:
    assert reorder_point(10.0, 3.0, 4.0, 0.0) == pytest.approx(40.0)  # z=0 → μL


def test_reorder_point_zero_variability_is_cycle_stock() -> None:
    assert reorder_point(10.0, 0.0, 4.0, 1.645) == pytest.approx(40.0)  # σ=0 → μL


def test_reorder_point_increases_with_z() -> None:
    assert reorder_point(10.0, 3.0, 4.0, 1.645) > reorder_point(10.0, 3.0, 4.0, 0.0)


def test_reorder_point_negative_demand_rate_raises() -> None:
    with pytest.raises(ValueError, match="demand_rate must be >= 0"):
        reorder_point(-1.0, 3.0, 4.0, 1.645)


@pytest.mark.parametrize("demand_std", [-0.1, -1.0])
def test_reorder_point_propagates_negative_std_raise(demand_std: float) -> None:
    # demand_std / lead_time guards are delegated to safety_stock().
    with pytest.raises(ValueError, match="demand_std must be >= 0"):
        reorder_point(10.0, demand_std, 4.0, 1.645)


def test_reorder_point_negative_lead_time_raises() -> None:
    with pytest.raises(ValueError, match="lead_time must be >= 0"):
        reorder_point(10.0, 3.0, -1.0, 1.645)


@given(
    demand_rate=st.floats(min_value=0.0, max_value=1e4),
    demand_std=st.floats(min_value=0.0, max_value=1e4),
    lead_time=st.floats(min_value=0.0, max_value=1e4),
    z=st.floats(min_value=-5.0, max_value=5.0),
)
def test_reorder_point_equals_cycle_plus_safety(
    demand_rate: float, demand_std: float, lead_time: float, z: float
) -> None:
    rp = reorder_point(demand_rate, demand_std, lead_time, z)
    assert math.isfinite(rp)
    assert rp == demand_rate * lead_time + safety_stock(z, demand_std, lead_time)


# --------------------------------------------------------------------------- #
# Layer B — simulator twin-match: μL is the deterministic cycle-stock level    #
# --------------------------------------------------------------------------- #

# Non-integer demand (10.3) de-phases the integer period grid (the lesson).
# σ=0, z=0  ⇒  reorder_point = μL = 10.3 · 4 = 41.2.
_D = 10.3
_LEAD_TIME = 4
_ORDER_QUANTITY = 60  # cycle (Q/d ≈ 5.8) > L so at most one order is in flight


def _deterministic_scenario(reorder: float) -> RunConfig:
    """σ=0 ``(s,Q)`` scenario at the given reorder point; demand is exactly ``_D`` each period."""
    return RunConfig(
        simulation=SimulationConfig(
            horizon=365, initial_on_hand=120, backorder_policy="lost_sales"
        ),
        demand=NormalDemandConfig(mean=_D, std=0.0),
        lead_time=DeterministicLeadTimeConfig(lead_time=_LEAD_TIME),
        policy=SQPolicyConfig(reorder_point=reorder, order_quantity=_ORDER_QUANTITY),
        costs=CostConfig(
            unit_cost=0.0,
            holding_per_unit_per_period=1.0,
            ordering_fixed=20.0,
            lost_sale_per_unit=10.0,
        ),
        master_seed=42,
    )


def test_deterministic_reorder_point_is_cycle_stock(
    run_to_ledger: Callable[[RunConfig], pd.DataFrame],
) -> None:
    mu_l = reorder_point(_D, 0.0, float(_LEAD_TIME), 0.0)
    assert mu_l == pytest.approx(_D * _LEAD_TIME)  # 41.2 — pure cycle stock (z=0, σ=0)

    # Above μL: no stockout, and the on-hand floor is exactly s − μL — i.e. μL
    # units are consumed during the lead time (the cycle-stock meaning of μL).
    for s in (44.0, 48.0, 52.0):
        df = run_to_ledger(_deterministic_scenario(s))
        assert float(df["lost_sales"].sum()) == 0.0, f"unexpected stockout at s={s}"
        min_on_hand = float(df["on_hand"].min())
        assert s - min_on_hand == pytest.approx(mu_l, abs=0.5), f"floor off at s={s}"

    # Below μL: the reorder fires too late — stock runs out before replenishment.
    for s in (34.0, 38.0):
        df = run_to_ledger(_deterministic_scenario(s))
        assert float(df["lost_sales"].sum()) > 0.0, f"expected stockout at s={s}"
