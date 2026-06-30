"""Validate the simulator against the newsvendor law.

The newsvendor order-up-to level ``S* = μ + Φ⁻¹(Cu/(Cu+Co))·σ`` is the
cost-optimal single-period base stock; the **critical ratio** ``Cu/(Cu+Co)`` is
the optimal in-stock probability. Two layers:

* **Layer A** pins the closed-form ``analytics.classical.newsvendor_level`` and
  its composition with the ``z_for_service_level`` helper.
* **Layer B** validates the critical ratio *as a probability*: the newsvendor is
  a single-period model, but the engine forbids lead-time 0, so a multi-period
  base-stock would validate the protection-interval version (``μτ + zσ√τ``), not
  this one. Instead we run **1-period episodes** (stock ``S*``, face one demand
  draw) across many seeds and confirm the no-stockout fraction ≈ ``Cu/(Cu+Co)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from analytics.classical import newsvendor_level, z_for_service_level
from core.config import (
    BaseStockPolicyConfig,
    CostConfig,
    DeterministicLeadTimeConfig,
    NormalDemandConfig,
    RunConfig,
    SimulationConfig,
)

# --------------------------------------------------------------------------- #
# Layer A — the closed-form helper                                            #
# --------------------------------------------------------------------------- #


def test_newsvendor_level_matches_textbook_formula() -> None:
    # S* = μ + Φ⁻¹(Cu/(Cu+Co))·σ. Cu=19, Co=1 → CR=0.95 → z ≈ 1.645.
    assert newsvendor_level(100.0, 20.0, 19.0, 1.0) == pytest.approx(
        100.0 + z_for_service_level(0.95) * 20.0
    )


def test_newsvendor_balanced_costs_give_mean() -> None:
    # Cu = Co → critical ratio 0.5 → z = 0 → S* = μ (no bias).
    assert newsvendor_level(100.0, 20.0, 5.0, 5.0) == pytest.approx(100.0)


def test_newsvendor_higher_underage_raises_level() -> None:
    # Stockouts hurting more than leftovers → stock more.
    assert newsvendor_level(100.0, 20.0, 9.0, 1.0) > newsvendor_level(100.0, 20.0, 1.0, 1.0)


def test_newsvendor_higher_overage_lowers_level() -> None:
    # Leftovers hurting more than stockouts → stock less.
    assert newsvendor_level(100.0, 20.0, 1.0, 9.0) < newsvendor_level(100.0, 20.0, 1.0, 1.0)


def test_newsvendor_negative_demand_rate_raises() -> None:
    with pytest.raises(ValueError, match="demand_rate must be >= 0"):
        newsvendor_level(-1.0, 20.0, 5.0, 5.0)


def test_newsvendor_negative_demand_std_raises() -> None:
    with pytest.raises(ValueError, match="demand_std must be >= 0"):
        newsvendor_level(100.0, -1.0, 5.0, 5.0)


@pytest.mark.parametrize("underage", [0.0, -1.0])
def test_newsvendor_nonpositive_underage_raises(underage: float) -> None:
    with pytest.raises(ValueError, match="underage_cost must be > 0"):
        newsvendor_level(100.0, 20.0, underage, 5.0)


@pytest.mark.parametrize("overage", [0.0, -1.0])
def test_newsvendor_nonpositive_overage_raises(overage: float) -> None:
    with pytest.raises(ValueError, match="overage_cost must be > 0"):
        newsvendor_level(100.0, 20.0, 5.0, overage)


@given(
    demand_rate=st.floats(min_value=0.0, max_value=1e4),
    demand_std=st.floats(min_value=0.0, max_value=1e4),
    underage_cost=st.floats(min_value=1e-3, max_value=1e4),
    overage_cost=st.floats(min_value=1e-3, max_value=1e4),
)
def test_newsvendor_equals_mean_plus_z_sigma(
    demand_rate: float, demand_std: float, underage_cost: float, overage_cost: float
) -> None:
    s = newsvendor_level(demand_rate, demand_std, underage_cost, overage_cost)
    assert math.isfinite(s)
    cr = underage_cost / (underage_cost + overage_cost)
    assert s == demand_rate + z_for_service_level(cr) * demand_std


# --------------------------------------------------------------------------- #
# Layer B — single-period achieved critical ratio                             #
# --------------------------------------------------------------------------- #

_MU = 100.0
_SIGMA = 20.0  # large σ keeps the integer-initial-stock rounding negligible
_SEEDS = range(1000)
# (underage_cost, overage_cost) → critical ratio 0.5 / 0.8 / 0.95
_COST_PAIRS = [(1.0, 1.0), (4.0, 1.0), (19.0, 1.0)]


def _newsvendor_episode(target: int, seed: int) -> RunConfig:
    """A single-period (horizon=1) base-stock episode stocked at ``target``.

    With ``initial_on_hand == target`` the base-stock policy orders nothing, so
    the lone period faces demand with exactly ``target`` units on hand — a
    stockout iff that period's demand exceeds ``target``.
    """
    return RunConfig(
        simulation=SimulationConfig(
            horizon=1, initial_on_hand=target, backorder_policy="lost_sales"
        ),
        demand=NormalDemandConfig(mean=_MU, std=_SIGMA),
        lead_time=DeterministicLeadTimeConfig(lead_time=1),
        policy=BaseStockPolicyConfig(target_level=float(target)),
        costs=CostConfig(
            unit_cost=0.0,
            holding_per_unit_per_period=1.0,
            ordering_fixed=20.0,
            lost_sale_per_unit=1.0,
        ),
        master_seed=seed,
    )


def _achieved_in_stock_fraction(
    run_to_ledger: Callable[[RunConfig], pd.DataFrame], underage: float, overage: float
) -> float:
    """Fraction of 1-period episodes (stocked at S*) that avoid a stockout."""
    target = round(newsvendor_level(_MU, _SIGMA, underage, overage))
    no_stockout = sum(
        float(run_to_ledger(_newsvendor_episode(target, seed))["lost_sales"].sum()) == 0.0
        for seed in _SEEDS
    )
    return no_stockout / len(_SEEDS)


@pytest.mark.parametrize(("underage", "overage"), _COST_PAIRS)
def test_newsvendor_level_achieves_critical_ratio(
    run_to_ledger: Callable[[RunConfig], pd.DataFrame], underage: float, overage: float
) -> None:
    critical_ratio = underage / (underage + overage)
    achieved = _achieved_in_stock_fraction(run_to_ledger, underage, overage)
    # The critical ratio IS the optimal in-stock probability: at S*, the chance
    # of meeting demand equals Cu/(Cu+Co). (Direction + magnitude, deterministic
    # under fixed seeds; tolerance absorbs integer-stock rounding + sampling.)
    assert achieved == pytest.approx(critical_ratio, abs=0.05), (
        f"in-stock {achieved:.4f} != CR {critical_ratio:.4f} (Cu={underage}, Co={overage})"
    )
