"""Integration: KPIs over a real engine run."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from analytics.kpis import compute_kpis
from core.config import RunConfig
from experiments import single_run


def test_kpis_on_real_sq_run(make_config: Callable[..., RunConfig]) -> None:
    frame = single_run.run(make_config())
    kpis = compute_kpis(frame)

    # All three service levels are valid probabilities.
    for value in (kpis.cycle_service_level, kpis.fill_rate, kpis.ready_rate):
        assert 0.0 <= value <= 1.0

    # Fill rate matches the immediate-fill definition over the real frame.
    expected_fill = float(frame["sales"].sum()) / float(frame["demand"].sum())
    assert kpis.fill_rate == pytest.approx(expected_fill)

    # Cost KPIs match the Ledger column sums, and total_cost is their sum.
    assert kpis.total_holding_cost == pytest.approx(float(frame["holding_cost"].sum()))
    assert kpis.total_stockout_cost == pytest.approx(float(frame["stockout_cost"].sum()))
    assert kpis.total_cost == pytest.approx(
        kpis.total_holding_cost
        + kpis.total_ordering_cost
        + kpis.total_purchase_cost
        + kpis.total_stockout_cost
    )
    assert kpis.order_count == float((frame["order_placed"] > 0).sum())

    # Efficiency KPIs match the frame reductions (real run has positive demand + on-hand).
    assert kpis.average_on_hand == pytest.approx(float(frame["on_hand"].mean()))
    assert kpis.inventory_turns == pytest.approx(
        float(frame["demand"].sum()) / float(frame["on_hand"].mean())
    )
    assert kpis.days_of_supply == pytest.approx(
        float(frame["on_hand"].mean()) / float(frame["demand"].mean())
    )
