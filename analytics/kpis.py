"""Run-level service-level KPIs computed from a materialized Ledger frame.

:func:`compute_kpis` is a **pure reduction** over one ``Ledger.to_dataframe()``
frame — no config, no RNG, no engine state — mirroring the pure-function
philosophy of :mod:`analytics.classical`. The three service levels are reported
*together* because most tools show only one and conflate them.

This module lands the :class:`KPIs` bundle and the
service-level trinity (Type 1/2/3). Cost KPIs and efficiency KPIs
 widen the *same* :class:`KPIs` dataclass and the *same*
:func:`compute_kpis` additively — new fields are appended at the end (à la the
Ledger ``_SCHEMA`` extend-at-the-end rule), never split into separate types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

# A period has an (immediate) stockout when current-period demand exceeds what was
# sold from stock. demand/sales are float columns, so compare against a small
# tolerance rather than relying on exact float equality.
_UNMET_TOL: Final[float] = 1e-9

# Ledger columns that compute_kpis reads today. Bullets 28/29 extend this set as
# they consume more columns. Validated up front so a malformed frame fails loud
# (project style — cf. Ledger.record's KeyError) instead of raising a cryptic
# pandas KeyError deep inside the reduction.
_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        # service-level trinity
        "demand",
        "sales",
        "on_hand",
        "order_received",
        # cost decomposition
        "holding_cost",
        "ordering_cost",
        "purchase_cost",
        "stockout_cost",
        "order_placed",
    }
)


@dataclass(frozen=True)
class KPIs:
    """Run-level KPIs for a single simulation run.

    Service-level trinity, each in ``[0, 1]``:

    - ``cycle_service_level`` (Type 1): fraction of replenishment cycles with no
      stockout.
    - ``fill_rate`` (Type 2): fraction of demand satisfied immediately from stock.
    - ``ready_rate`` (Type 3): fraction of periods with positive on-hand.

    Cost decomposition, non-negative totals over the run:

    - ``total_holding_cost`` / ``total_ordering_cost`` / ``total_purchase_cost`` /
      ``total_stockout_cost``: sums of the engine's pre-accrued cost columns.
    - ``total_cost``: their sum. **Excludes expediting cost** — the Ledger has no
      ``expediting_cost`` column until M5.
    - ``order_count``: number of orders placed (periods with ``order_placed > 0``),
      not units ordered.

    Efficiency KPIs:

    - ``average_on_hand``: mean end-of-period on-hand inventory.
    - ``inventory_turns``: total simulated demand ÷ average on-hand — **over the
      simulated horizon, not annualized** (equals annual turns only at a one-year
      horizon); ``nan`` when average on-hand is 0.
    - ``days_of_supply``: average on-hand ÷ mean demand per *period* ("periods of
      supply"); ``nan`` when there is no demand.

    The bundle has 3 service + 6 cost + 3 efficiency KPIs.
    Any future field is appended at the end so existing access stays stable.
    """

    # service-level trinity
    cycle_service_level: float
    fill_rate: float
    ready_rate: float
    # cost decomposition
    total_holding_cost: float
    total_ordering_cost: float
    total_purchase_cost: float
    total_stockout_cost: float
    total_cost: float
    order_count: float
    # efficiency KPIs
    average_on_hand: float
    inventory_turns: float
    days_of_supply: float


def compute_kpis(ledger: pd.DataFrame) -> KPIs:
    """Compute run-level KPIs from a materialized Ledger frame.

    ``ledger`` is the frame returned by ``Ledger.to_dataframe()`` — one row per
    period, carrying the columns declared in ``Ledger._SCHEMA``. The subset
    required for the service-level trinity is :data:`_REQUIRED_COLUMNS`; a frame
    missing any of them raises :class:`ValueError`.

    **Type 2 — fill rate** is ``sales.sum() / demand.sum()``: the fraction of
    demand met *immediately* from on-hand stock. ``sales[t]`` is current-period
    demand filled now; it deliberately excludes ``backorders_cleared`` (past
    demand satisfied late). With zero total demand the fill rate is defined as
    ``1.0`` — no demand went unfilled.

    **Type 3 — ready rate** is ``(on_hand > 0).mean()``: the fraction of periods
    that end with positive on-hand inventory.

    **Type 1 — cycle service level** is the fraction of *replenishment cycles*
    with no stockout. This is the **receipt-based** convention: cycle boundaries
    fall at order *receipts* (``order_received > 0``), so a new cycle begins each
    time stock arrives. (An alternative convention delimits cycles by order
    *placement*; the two agree in count under constant lead time and differ only
    at the horizon boundaries / under variable lead time.) A cycle counts as
    stocked-out if *any* period within it has unmet current-period demand
    (``demand − sales > tol``) — a signal that holds in **both** backorder and
    lost-sales modes, since the ``sales ≤ demand`` invariant
    routes every shortfall to either lost sales or new backorders. A run with no
    receipts is treated as a single cycle spanning the whole horizon.

    **Cost KPIs** sum the engine's pre-accrued cost columns —
    ``total_holding_cost`` / ``total_ordering_cost`` / ``total_purchase_cost`` /
    ``total_stockout_cost`` and their ``total_cost`` (expediting excluded — no
    column until M5). ``order_count`` is the number of ordering events
    (``order_placed > 0``), so ``total_ordering_cost == K * order_count``.

    **Efficiency KPIs**: ``average_on_hand`` (mean on-hand),
    ``inventory_turns`` (total simulated demand ÷ average on-hand — horizon-based,
    not annualized), and ``days_of_supply`` (average on-hand ÷ mean demand per
    period). Each ratio is ``nan`` when its denominator is 0.
    """
    missing = _REQUIRED_COLUMNS - set(ledger.columns)
    if missing:
        raise ValueError(f"ledger frame missing required columns: {sorted(missing)}")

    demand = ledger["demand"]
    sales = ledger["sales"]
    on_hand = ledger["on_hand"]
    received = ledger["order_received"]

    # Type 2 — fill rate.
    total_demand = float(demand.sum())
    fill_rate = 1.0 if total_demand == 0.0 else float(sales.sum()) / total_demand

    # Type 3 — ready rate.
    ready_rate = float((on_hand > 0).mean())

    # Type 1 — cycle service level (receipt-based cycles).
    stockout_period = (demand - sales) > _UNMET_TOL
    cycle_id = (received > 0).cumsum()
    stocked_out_by_cycle = stockout_period.groupby(cycle_id).any()
    cycle_service_level = float((~stocked_out_by_cycle).mean())

    # Cost decomposition — sums of the engine's pre-accrued cost columns.
    total_holding_cost = float(ledger["holding_cost"].sum())
    total_ordering_cost = float(ledger["ordering_cost"].sum())
    total_purchase_cost = float(ledger["purchase_cost"].sum())
    total_stockout_cost = float(ledger["stockout_cost"].sum())
    total_cost = (
        total_holding_cost + total_ordering_cost + total_purchase_cost + total_stockout_cost
    )
    order_count = float((ledger["order_placed"] > 0).sum())

    # Efficiency KPIs. Turns are horizon-based (not annualized); undefined ratios -> nan.
    average_on_hand = float(on_hand.mean())
    mean_demand = total_demand / len(demand)
    inventory_turns = total_demand / average_on_hand if average_on_hand > 0.0 else float("nan")
    days_of_supply = average_on_hand / mean_demand if mean_demand > 0.0 else float("nan")

    return KPIs(
        cycle_service_level=cycle_service_level,
        fill_rate=fill_rate,
        ready_rate=ready_rate,
        total_holding_cost=total_holding_cost,
        total_ordering_cost=total_ordering_cost,
        total_purchase_cost=total_purchase_cost,
        total_stockout_cost=total_stockout_cost,
        total_cost=total_cost,
        order_count=order_count,
        average_on_hand=average_on_hand,
        inventory_turns=inventory_turns,
        days_of_supply=days_of_supply,
    )
