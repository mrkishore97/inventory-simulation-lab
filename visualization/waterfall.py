"""Cost-decomposition waterfall factory.

A stateless Plotly factory: a ``go.Waterfall``
bridging the cost components of one run to their total.

By default it shows only the **controllable** (policy-sensitive) costs — holding, ordering,
stockout — and **excludes purchase cost**. Purchase (≈ ``unit_cost × units sold``) is largely
policy-invariant and on a typical run several times the controllable total, so including it
would visually swamp the costs that actually differ between policies. Pass
``include_purchase=True`` for the full-accounting view, whose total then reconciles with
:attr:`analytics.kpis.KPIs.total_cost` (the only missing component, expediting, has no Ledger
column until M5). The default's total bar is labelled "Total controllable" and the title
"Controllable cost decomposition" so it is never confused with the KPI ``total_cost``.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import plotly.graph_objects as go

# The policy-sensitive cost columns, always shown.
_CONTROLLABLE_COLUMNS: Final[tuple[str, ...]] = ("holding_cost", "ordering_cost", "stockout_cost")


def cost_waterfall(ledger: pd.DataFrame, *, include_purchase: bool = False) -> go.Figure:
    """Build the cost-decomposition waterfall for one run's Ledger frame.

    With ``include_purchase=False`` (default) the bars are Holding, Ordering, Stockout and a
    "Total controllable" total. With ``include_purchase=True`` a Purchase bar is added and the
    total ("Total") reconciles with :attr:`analytics.kpis.KPIs.total_cost`. Raises
    :class:`ValueError` if a required column is missing (``purchase_cost`` is required only in
    the ``include_purchase=True`` mode).
    """
    required = set(_CONTROLLABLE_COLUMNS)
    if include_purchase:
        required.add("purchase_cost")
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f"ledger is missing required columns: {sorted(missing)}")

    labels = ["Holding", "Ordering", "Stockout"]
    values = [float(ledger[c].sum()) for c in _CONTROLLABLE_COLUMNS]
    if include_purchase:
        labels.append("Purchase")
        values.append(float(ledger["purchase_cost"].sum()))

    total_label = "Total" if include_purchase else "Total controllable"
    title = "Cost decomposition" if include_purchase else "Controllable cost decomposition"

    fig = go.Figure(
        go.Waterfall(
            x=[*labels, total_label],
            measure=["relative"] * len(labels) + ["total"],
            # The "total" entry's y is ignored by Plotly (it shows the running sum); 0.0 is a
            # placeholder so the array lengths line up.
            y=[*values, 0.0],
            text=[f"${v:,.0f}" for v in values] + [f"${sum(values):,.0f}"],
            textposition="outside",
        )
    )
    fig.update_layout(title=title, yaxis_title="Cost ($)", showlegend=False)
    return fig
