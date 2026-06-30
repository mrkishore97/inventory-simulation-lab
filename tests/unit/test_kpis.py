"""Tests for analytics.kpis — service-level trinity."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from analytics.kpis import compute_kpis


def _ledger(
    *,
    demand: list[float],
    sales: list[float] | None = None,
    on_hand: list[float] | None = None,
    order_received: list[float] | None = None,
    holding_cost: list[float] | None = None,
    ordering_cost: list[float] | None = None,
    purchase_cost: list[float] | None = None,
    stockout_cost: list[float] | None = None,
    order_placed: list[float] | None = None,
) -> pd.DataFrame:
    """Minimal Ledger-shaped frame with the columns compute_kpis reads.

    Defaults model a no-stockout, always-stocked, no-receipt, zero-cost run, so
    each test overrides only the columns its KPI depends on. Cast to float64 to
    mirror the real Ledger dtype.
    """
    n = len(demand)
    zeros: list[float] = [0.0] * n
    frame = pd.DataFrame(
        {
            "demand": demand,
            "sales": demand if sales is None else sales,
            "on_hand": [1.0] * n if on_hand is None else on_hand,
            "order_received": zeros if order_received is None else order_received,
            "holding_cost": zeros if holding_cost is None else holding_cost,
            "ordering_cost": zeros if ordering_cost is None else ordering_cost,
            "purchase_cost": zeros if purchase_cost is None else purchase_cost,
            "stockout_cost": zeros if stockout_cost is None else stockout_cost,
            "order_placed": zeros if order_placed is None else order_placed,
        }
    )
    return frame.astype("float64")


# --------------------------------------------------------------------------- #
# Type 2 — fill rate                                                          #
# --------------------------------------------------------------------------- #


def test_fill_rate_is_immediate_fill_fraction() -> None:
    kpis = compute_kpis(_ledger(demand=[10, 10], sales=[10, 5]))
    assert kpis.fill_rate == pytest.approx(0.75)  # 15 sold / 20 demanded


def test_fill_rate_zero_demand_is_one() -> None:
    kpis = compute_kpis(_ledger(demand=[0, 0], sales=[0, 0]))
    assert kpis.fill_rate == 1.0


# --------------------------------------------------------------------------- #
# Type 3 — ready rate                                                         #
# --------------------------------------------------------------------------- #


def test_ready_rate_is_fraction_of_positive_on_hand_periods() -> None:
    kpis = compute_kpis(_ledger(demand=[0, 0, 0, 0], on_hand=[5, 0, 3, 0]))
    assert kpis.ready_rate == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Type 1 — cycle service level (receipt-based cycles)                         #
# --------------------------------------------------------------------------- #


def test_cycle_service_two_cycles_one_stockout() -> None:
    # Receipts at t0 and t2 -> cycles {t0,t1} and {t2,t3}; stockout only in cycle 1.
    kpis = compute_kpis(
        _ledger(
            demand=[5, 5, 5, 5],
            sales=[5, 3, 5, 5],
            order_received=[1, 0, 1, 0],
        )
    )
    assert kpis.cycle_service_level == pytest.approx(0.5)


def test_cycle_service_no_receipts_is_single_cycle_stockout() -> None:
    # No receipts -> one cycle spanning the run; it stocks out -> CSL 0.
    kpis = compute_kpis(_ledger(demand=[5, 5], sales=[5, 3]))
    assert kpis.cycle_service_level == 0.0


def test_cycle_service_no_receipts_clean_is_one() -> None:
    kpis = compute_kpis(_ledger(demand=[5, 5], sales=[5, 5]))
    assert kpis.cycle_service_level == 1.0


def test_cycle_service_all_cycles_clean_is_one() -> None:
    kpis = compute_kpis(
        _ledger(demand=[5, 5, 5, 5], sales=[5, 5, 5, 5], order_received=[1, 0, 1, 0])
    )
    assert kpis.cycle_service_level == 1.0


# --------------------------------------------------------------------------- #
# Input contract                                                             #
# --------------------------------------------------------------------------- #


def test_missing_required_column_raises() -> None:
    frame = pd.DataFrame({"demand": [1.0], "sales": [1.0]})  # no on_hand / order_received
    with pytest.raises(ValueError, match="missing required columns"):
        compute_kpis(frame)


# --------------------------------------------------------------------------- #
# Property — service levels are valid probabilities        #
# --------------------------------------------------------------------------- #


@given(data=st.data())
def test_service_levels_are_valid_probabilities(data: st.DataObject) -> None:
    n = data.draw(st.integers(min_value=1, max_value=30))
    demand = data.draw(st.lists(st.floats(0.0, 1e4), min_size=n, max_size=n))
    fracs = data.draw(st.lists(st.floats(0.0, 1.0), min_size=n, max_size=n))
    on_hand = data.draw(st.lists(st.floats(0.0, 1e4), min_size=n, max_size=n))
    received = data.draw(st.lists(st.floats(0.0, 1e2), min_size=n, max_size=n))
    sales = [f * d for f, d in zip(fracs, demand, strict=True)]

    kpis = compute_kpis(
        _ledger(demand=demand, sales=sales, on_hand=on_hand, order_received=received)
    )

    for value in (kpis.cycle_service_level, kpis.fill_rate, kpis.ready_rate):
        assert -1e-9 <= value <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# Cost decomposition                                                    #
# --------------------------------------------------------------------------- #


def test_cost_decomposition_sums_columns() -> None:
    kpis = compute_kpis(
        _ledger(
            demand=[0, 0],
            holding_cost=[1, 2],
            ordering_cost=[3, 0],
            purchase_cost=[5, 5],
            stockout_cost=[0, 4],
        )
    )
    assert kpis.total_holding_cost == pytest.approx(3.0)
    assert kpis.total_ordering_cost == pytest.approx(3.0)
    assert kpis.total_purchase_cost == pytest.approx(10.0)
    assert kpis.total_stockout_cost == pytest.approx(4.0)
    assert kpis.total_cost == pytest.approx(20.0)  # 3 + 3 + 10 + 4


def test_order_count_counts_order_periods_not_units() -> None:
    kpis = compute_kpis(_ledger(demand=[0, 0, 0, 0], order_placed=[30, 0, 30, 0]))
    assert kpis.order_count == 2.0  # two ordering events, not 60 units


@given(data=st.data())
def test_cost_kpis_are_nonnegative_and_total_is_their_sum(data: st.DataObject) -> None:
    n = data.draw(st.integers(min_value=1, max_value=30))
    cost_cols = {
        name: data.draw(st.lists(st.floats(0.0, 1e6), min_size=n, max_size=n))
        for name in ("holding_cost", "ordering_cost", "purchase_cost", "stockout_cost")
    }
    kpis = compute_kpis(_ledger(demand=[0.0] * n, **cost_cols))

    components = (
        kpis.total_holding_cost,
        kpis.total_ordering_cost,
        kpis.total_purchase_cost,
        kpis.total_stockout_cost,
    )
    for value in components:
        assert value >= 0.0
    assert kpis.total_cost == pytest.approx(sum(components))


# --------------------------------------------------------------------------- #
# Efficiency KPIs — closes the suite                                    #
# --------------------------------------------------------------------------- #


def test_average_on_hand_is_mean_on_hand() -> None:
    kpis = compute_kpis(_ledger(demand=[0, 0, 0], on_hand=[2, 4, 6]))
    assert kpis.average_on_hand == pytest.approx(4.0)


def test_inventory_turns_is_demand_over_average_on_hand() -> None:
    kpis = compute_kpis(_ledger(demand=[10, 10], on_hand=[5, 5]))
    assert kpis.inventory_turns == pytest.approx(4.0)  # 20 demand / 5 avg on-hand


def test_inventory_turns_is_nan_when_no_on_hand() -> None:
    kpis = compute_kpis(_ledger(demand=[5, 5], on_hand=[0, 0]))
    assert math.isnan(kpis.inventory_turns)


def test_days_of_supply_is_on_hand_over_mean_demand() -> None:
    kpis = compute_kpis(_ledger(demand=[5, 5], on_hand=[10, 10]))
    assert kpis.days_of_supply == pytest.approx(2.0)  # 10 avg on-hand / 5 mean demand


def test_days_of_supply_is_nan_when_no_demand() -> None:
    kpis = compute_kpis(_ledger(demand=[0, 0], on_hand=[10, 10]))
    assert math.isnan(kpis.days_of_supply)


@given(data=st.data())
def test_efficiency_kpis_are_nan_or_nonnegative(data: st.DataObject) -> None:
    n = data.draw(st.integers(min_value=1, max_value=30))
    demand = data.draw(st.lists(st.floats(0.0, 1e4), min_size=n, max_size=n))
    on_hand = data.draw(st.lists(st.floats(0.0, 1e4), min_size=n, max_size=n))
    kpis = compute_kpis(_ledger(demand=demand, on_hand=on_hand))

    assert kpis.average_on_hand >= 0.0
    for ratio in (kpis.inventory_turns, kpis.days_of_supply):
        assert math.isnan(ratio) or ratio >= 0.0
