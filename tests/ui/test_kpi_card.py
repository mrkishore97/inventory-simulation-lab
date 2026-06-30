"""Integration: the KPI summary card component."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _host() -> None:
    # AppTest.from_function execs this body as a standalone script, so imports live inside.
    from analytics.kpis import KPIs
    from ui.components.kpi_card import render_kpi_card

    kpis = KPIs(
        cycle_service_level=0.9,
        fill_rate=0.95,
        ready_rate=0.8,
        total_holding_cost=478.0,
        total_ordering_cost=560.0,
        total_purchase_cost=4200.0,
        total_stockout_cost=324.0,
        total_cost=5562.0,
        order_count=14.0,
        average_on_hand=53.0,
        inventory_turns=float("nan"),  # exercises the nan -> "—" path
        days_of_supply=5.3,
    )
    render_kpi_card(kpis)


def test_renders_twelve_metrics() -> None:
    at = AppTest.from_function(_host, default_timeout=30).run()
    assert not at.exception
    assert len(at.metric) == 12


def test_service_level_formatted_as_percent_with_help() -> None:
    at = AppTest.from_function(_host, default_timeout=30).run()
    fill = next(m for m in at.metric if m.label == "Fill rate (Type 2)")
    assert fill.value == "95.0%"
    assert fill.help  # a formula tooltip is attached


def test_nan_metric_renders_as_dash() -> None:
    at = AppTest.from_function(_host, default_timeout=30).run()
    turns = next(m for m in at.metric if m.label == "Inventory turns")
    assert turns.value == "—"


def test_total_cost_formatted_as_money() -> None:
    at = AppTest.from_function(_host, default_timeout=30).run()
    total = next(m for m in at.metric if m.label == "Total cost")
    assert total.value == "$5,562"
