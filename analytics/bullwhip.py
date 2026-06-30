"""Bullwhip metric — order-vs-demand variance amplification over one run's Ledger.

The bullwhip effect is demand-variance **amplification**: how much more variable are
the orders a stage places than the demand it faces. On a single echelon the classic
measure (Chen / Lee et al.) is the ratio ``Var(orders) / Var(demand)`` — ``> 1`` means
amplification, ``= 1`` pass-through, ``< 1`` smoothing.

:func:`bullwhip_metrics` is a **pure reduction over one run's Ledger frame** — the
``analytics.kpis.compute_kpis(ledger)`` role, not the Monte-Carlo-distribution input
of :func:`analytics.risk.risk_metrics` nor the config-in input of
:mod:`analytics.pareto` / :mod:`analytics.sensitivity`.

Why ``order_placed`` (not, say, a smoothed order rate): it is the per-period order
*quantity* — ``0`` in periods with no order, the order size when one fires — so its
variance **includes the on/off batching lumpiness, which is exactly why bullwhip
exists**. The canonical teaching example: perfectly smooth demand

    demand = [10, 10, 10, 10, 10, 10]   # Var(demand) = 0  (or tiny under noise)

drives a lumpy ``(s, Q)`` order stream

    orders = [0,  0, 50,  0,  0, 50]    # Var(orders) large

so ``Var(orders) / Var(demand)`` blows up — batching manufactures variability the
customer never had.

Non-invasive: composes a Ledger frame; no engine or config change. ``numpy``-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

# Ledger columns bullwhip_metrics reads. Validated up front so a malformed frame fails
# loud (project style — cf. compute_kpis / Ledger.record) instead of a cryptic KeyError.
_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset({"demand", "order_placed"})


@dataclass(frozen=True)
class BullwhipMetrics:
    """Order-vs-demand variance amplification for a single simulation run.

    - ``demand_mean`` / ``order_mean``: per-period means (flow context — in a
      stationary single echelon the order rate tracks the demand rate).
    - ``demand_variance`` / ``order_variance``: population variances (``ddof=0``) of
      the per-period ``demand`` and ``order_placed`` series — the two quantities the
      "order variance vs. demand variance" view plots.
    - ``bullwhip_ratio``: ``order_variance / demand_variance`` — the headline. ``> 1``
      amplification, ``= 1`` pass-through, ``< 1`` smoothing. ``nan`` when demand has
      zero variance (constant demand — the ratio is undefined). The ratio itself is
      ``ddof``-invariant (the ``N/(N-1)`` factors cancel); only the two ``*_variance``
      fields depend on the ``ddof`` choice.
    """

    demand_mean: float
    demand_variance: float
    order_mean: float
    order_variance: float
    bullwhip_ratio: float


def bullwhip_metrics(ledger: pd.DataFrame) -> BullwhipMetrics:
    """Compute the bullwhip (order-vs-demand variance amplification) for one run.

    ``ledger`` is a materialized Ledger frame (``Ledger.to_dataframe()`` /
    :func:`experiments.single_run.run`); only the ``demand`` and ``order_placed``
    columns are read. Returns a :class:`BullwhipMetrics`; a pure reduction, so two
    calls on the same frame return identical results.

    ``bullwhip_ratio`` is ``Var(order_placed) / Var(demand)`` (population variance),
    or ``nan`` when ``Var(demand) == 0`` (constant demand — undefined, not an error,
    since a constant-demand run is valid).
    """
    missing = _REQUIRED_COLUMNS - set(ledger.columns)
    if missing:
        raise ValueError(f"ledger is missing required columns: {sorted(missing)}")

    demand = ledger["demand"].to_numpy(dtype=float)
    orders = ledger["order_placed"].to_numpy(dtype=float)
    demand_variance = float(demand.var())  # population variance (ddof=0)
    order_variance = float(orders.var())

    bullwhip_ratio = order_variance / demand_variance if demand_variance > 0.0 else float("nan")
    return BullwhipMetrics(
        demand_mean=float(demand.mean()),
        demand_variance=demand_variance,
        order_mean=float(orders.mean()),
        order_variance=order_variance,
        bullwhip_ratio=bullwhip_ratio,
    )
