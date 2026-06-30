"""Time-scrubber replay — freeze a period and inspect that moment.

a draggable timeline lets users freeze any moment of a run and inspect
inventory position, the on-order pipeline, and recent demand — the tool for understanding
*why* a stockout happened. Two small Streamlit functions over the last-run Ledger frame:

- :func:`render_time_scrubber` renders the period slider and returns the selected period. The
  page passes that period to :func:`visualization.timeseries.inventory_timeseries` as
  ``highlight_period`` (the dashed marker) and then to :func:`render_period_detail`.
- :func:`render_period_detail` renders the frozen moment's state/flow as ``st.metric`` tiles.

The split is dictated by the page layout: the slider sits *above* the trajectory chart (so
dragging it moves the marker on the chart below) while the detail tiles sit *below* it.

This is pure presentation over the cached frame — moving the slider re-reads the existing
Ledger, never re-simulates, and (not being part of the config) never counts as config drift.

Next scheduled event deferred until the engine exposes a pending-arrivals/event view: the
Ledger records only aggregate ``on_order`` and realized ``order_received``, not the pending-
arrival calendar, so a true "next scheduled event" cannot be read from the frame.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components.glossary import tip

_RECENT_WINDOW = 5  # periods, inclusive trailing window for "recent demand"


def _units(x: float) -> str:
    return f"{x:,.0f}"


def render_time_scrubber(ledger: pd.DataFrame) -> int:
    """Render the period slider and return the selected period (defaults to the last period).

    Single-period runs (horizon=1, e.g. newsvendor) have no range to slide over —
    ``st.slider`` requires ``min < max`` — so the slider is skipped and the lone period is
    returned with a caption.
    """
    periods = ledger["period"]
    pmin, pmax = int(periods.min()), int(periods.max())
    if pmin == pmax:
        st.caption(f"Single-period run — showing period {pmin}.")
        return pmin
    return int(
        st.slider("Freeze period", min_value=pmin, max_value=pmax, value=pmax, key="sr_scrub_t")
    )


def render_period_detail(ledger: pd.DataFrame, period: int) -> None:
    """Render the frozen moment at ``period`` as ``st.metric`` tiles (state + flows)."""
    row = ledger.loc[ledger["period"] == period].iloc[0]
    lo = max(0, period - (_RECENT_WINDOW - 1))
    recent = ledger.loc[(ledger["period"] >= lo) & (ledger["period"] <= period), "demand"].mean()

    st.subheader(f"State at period {period}")
    a1, a2, a3 = st.columns(3)
    a1.metric(
        "Inventory position",
        _units(row["inventory_position"]),
        help=tip("inventory_position"),
    )
    a2.metric("On hand", _units(row["on_hand"]), help=tip("on_hand"))
    a3.metric(
        "On order (pipeline)",
        _units(row["on_order"]),
        help=tip("on_order"),
    )

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Demand", _units(row["demand"]), help=tip("demand"))
    b2.metric(
        f"Recent demand (avg {_RECENT_WINDOW})",
        f"{recent:,.1f}",
        help=tip("recent_demand").format(n=_RECENT_WINDOW),
    )
    b3.metric("Order placed", _units(row["order_placed"]), help=tip("order_placed"))
    b4.metric(
        "Order received",
        _units(row["order_received"]),
        help=tip("order_received"),
    )
