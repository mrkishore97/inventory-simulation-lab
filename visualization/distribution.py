"""KPI-distribution histogram — the Monte Carlo spread + VaR/CVaR markers.

A stateless Plotly factory: it takes the
:func:`analytics.monte_carlo.monte_carlo` output frame (one row per replication:
``replication`` / ``seed`` + the 12 KPI values) and draws a histogram of one KPI column across
replications — the spread a single mean hides. Pass a :class:`analytics.risk.RiskMetrics` bundle
to overlay the **mean**, **VaR**, and **CVaR** as vertical markers, so "good on average" and
"catastrophic in the tail" read on the same axis. No Streamlit, no state — a pure
function, unit-testable; the page layer wires the replication / KPI / confidence controls.

Colours are left to the active Plotly theme. Friendly axis labels fall back to the raw column
name (cf. ``visualization.pareto._AXIS_LABELS``).
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import plotly.graph_objects as go

from analytics.risk import RiskMetrics

# Friendly labels for the KPI columns a user is likely to analyze; any other column falls back
# to its raw name.
_LABELS: Final[dict[str, str]] = {
    "total_cost": "Total cost ($)",
    "total_holding_cost": "Holding cost ($)",
    "total_ordering_cost": "Ordering cost ($)",
    "total_stockout_cost": "Stockout cost ($)",
    "total_purchase_cost": "Purchase cost ($)",
    "fill_rate": "Fill rate (Type 2)",
    "cycle_service_level": "Cycle service level (Type 1)",
    "ready_rate": "Ready rate (Type 3)",
    "inventory_turns": "Inventory turns",
    "days_of_supply": "Days of supply",
    "average_on_hand": "Average on-hand (units)",
    "order_count": "Order count",
}


def _label(column: str) -> str:
    return _LABELS.get(column, column)


def kpi_distribution_histogram(
    distribution: pd.DataFrame,
    *,
    column: str = "total_cost",
    risk: RiskMetrics | None = None,
) -> go.Figure:
    """Histogram of ``distribution[column]`` across Monte Carlo replications.

    ``distribution`` is the :func:`analytics.monte_carlo.monte_carlo` frame (one row per
    replication; the 12 KPI columns plus ``replication`` / ``seed``). When ``risk`` is given, the
    mean, VaR, and CVaR are drawn as vertical markers (annotated with the confidence level), so
    the tail risk a mean hides is visible on the same axis. ``x`` = ``column``. Raises
    :class:`ValueError` if the frame is empty or ``column`` is missing.
    """
    if distribution.empty:
        raise ValueError("distribution is empty: need at least one replication to plot")
    if column not in distribution.columns:
        raise ValueError(
            f"distribution has no column {column!r}; columns: {list(distribution.columns)}"
        )

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=distribution[column], name="Replications"))
    if risk is not None:
        conf = f"{risk.confidence_level:.0%}"
        fig.add_vline(x=risk.mean, line_dash="dot", annotation_text="Mean")
        fig.add_vline(x=risk.value_at_risk, line_dash="dash", annotation_text=f"VaR {conf}")
        fig.add_vline(
            x=risk.conditional_value_at_risk, line_dash="solid", annotation_text=f"CVaR {conf}"
        )
    fig.update_layout(
        title=f"{_label(column)} across {len(distribution)} replications",
        xaxis_title=_label(column),
        yaxis_title="Replications",
        bargap=0.05,
    )
    return fig
