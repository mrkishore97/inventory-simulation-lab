"""Bullwhip bar — order-vs-demand variance amplification across policies.

A stateless Plotly factory. It takes a mapping of
label -> :class:`analytics.bullwhip.BullwhipMetrics` (one bundle per run, already reduced by
:func:`analytics.bullwhip.bullwhip_metrics`) and draws one bar per label whose height is the
``bullwhip_ratio`` = Var(orders) / Var(demand). A dashed reference line at 1.0 marks pass-through:
bars above it amplify demand variance (e.g. an (s, Q) batching policy), bars below it smooth it
(e.g. an order-up-to base-stock policy). The two raw variances ride the hover — the literal
"order variance vs demand variance" the view is about.

Like the other factories here it visualizes only: the compute (``bullwhip_metrics``) lives in
``analytics``, and the page layer computes the metrics once and reuses them for both this
chart and its ``st.metric`` tiles — mirroring ``comparison_table({policy: KPIs})``, not raw
ledgers. No Streamlit, no state; colours are left to the active Plotly theme.

``log_y`` switches the y-axis to log scale, useful because the ratio can span ~1 (smoothing) to
100+ (heavy batching) and a linear axis squashes the small bars. Caveat: a ``nan`` ratio
(constant demand — a documented ``bullwhip_metrics`` output) renders no bar, and on a log axis a
``0`` ratio (constant orders) is likewise omitted by Plotly; both are acceptable, not errors.
"""

from __future__ import annotations

from collections.abc import Mapping

import plotly.graph_objects as go

from analytics.bullwhip import BullwhipMetrics


def bullwhip_bar(metrics: Mapping[str, BullwhipMetrics], *, log_y: bool = False) -> go.Figure:
    """Bar of ``bullwhip_ratio`` per run label, with a pass-through reference line at 1.0.

    ``metrics`` maps a run label (e.g. a policy name) to its :class:`BullwhipMetrics`; bars keep
    insertion order. Bars above the dashed 1.0 line amplify demand variance, below it smooth it;
    the hover carries Var(orders) and Var(demand). ``log_y`` puts the y-axis on a log scale (the
    ratio range is wide). Raises :class:`ValueError` if ``metrics`` is empty. A ``nan`` ratio
    (constant demand) is plotted as a gap, not an error.
    """
    if not metrics:
        raise ValueError("metrics is empty: need at least one run to plot")

    labels = list(metrics.keys())
    ratios = [m.bullwhip_ratio for m in metrics.values()]
    customdata = [[m.demand_variance, m.order_variance] for m in metrics.values()]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=ratios,
            name="Bullwhip ratio",
            customdata=customdata,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Bullwhip ratio: %{y:.2f}<br>"
                "Var(orders): %{customdata[1]:.2f}<br>"
                "Var(demand): %{customdata[0]:.2f}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=1.0, line_dash="dash", annotation_text="pass-through (1.0)")
    fig.update_layout(
        title="Bullwhip — order-vs-demand variance amplification",
        xaxis_title="Policy",
        yaxis_title="Bullwhip ratio = Var(orders) / Var(demand)",
    )
    if log_y:
        fig.update_yaxes(type="log")
    return fig
