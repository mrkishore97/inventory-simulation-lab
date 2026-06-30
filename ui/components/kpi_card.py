"""KPI summary card — render a :class:`KPIs` bundle as ``st.metric`` tiles.

A reusable Streamlit component: it takes the frozen ``KPIs``
dataclass (the output of :func:`analytics.kpis.compute_kpis`) — **not** a Ledger — and
renders it as grouped ``st.metric`` tiles, each carrying a ``help=`` formula tooltip. The
page computes ``compute_kpis(df)`` once and passes the result here, so this component stays
decoupled from the engine and analytics.
"""

from __future__ import annotations

import math

import streamlit as st

from analytics.kpis import KPIs
from ui.components.glossary import tip


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _num(x: float, places: int) -> str:
    return "—" if math.isnan(x) else f"{x:.{places}f}"


def render_kpi_card(kpis: KPIs) -> None:
    """Render the 12-field KPIs bundle as grouped ``st.metric`` tiles with formula tooltips."""
    st.subheader("Service levels")
    s1, s2, s3 = st.columns(3)
    s1.metric(
        "Cycle service (Type 1)",
        _pct(kpis.cycle_service_level),
        help=tip("cycle_service"),
    )
    s2.metric(
        "Fill rate (Type 2)",
        _pct(kpis.fill_rate),
        help=tip("fill_rate"),
    )
    s3.metric(
        "Ready rate (Type 3)",
        _pct(kpis.ready_rate),
        help=tip("ready_rate"),
    )

    st.subheader("Costs")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Total cost",
        _money(kpis.total_cost),
        help=tip("total_cost"),
    )
    c2.metric("Holding", _money(kpis.total_holding_cost), help=tip("holding_cost"))
    c3.metric("Ordering", _money(kpis.total_ordering_cost), help=tip("ordering_cost"))
    c4.metric("Purchase", _money(kpis.total_purchase_cost), help=tip("purchase_cost"))
    c5.metric("Stockout", _money(kpis.total_stockout_cost), help=tip("stockout_cost"))

    st.subheader("Efficiency")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric(
        "Inventory turns",
        _num(kpis.inventory_turns, 2),
        help=tip("inventory_turns"),
    )
    e2.metric(
        "Days of supply",
        _num(kpis.days_of_supply, 1),
        help=tip("days_of_supply"),
    )
    e3.metric("Avg on-hand", _num(kpis.average_on_hand, 1), help=tip("avg_on_hand"))
    e4.metric("Orders", f"{int(kpis.order_count)}", help=tip("order_count"))
