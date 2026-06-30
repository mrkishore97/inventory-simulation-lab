"""Policy-comparison factories — the side-by-side "race".

Stateless presentation factories for Page 2. They take the
materialized per-policy run outputs and return a Plotly figure / a tidy DataFrame — no
Streamlit, no state, trivially unit-testable. The page layer (a later bullet) runs each policy
on the **same demand stream** — identical ``master_seed`` with a different ``policy`` yields a
byte-identical demand realization (demand/lead-time RNG is seeded independently of the policy),
so the race is apples-to-apples with no extra common-random-numbers plumbing — and renders
these.

Two factories:

- :func:`policy_race_timeseries` overlays one metric (default on-hand) across policies — the
  visual "race" where trajectories diverge.
- :func:`comparison_table` shapes ``{policy: KPIs}`` into a one-row-per-policy DataFrame — the
  comparison table. It returns a frame, not a figure: this module holds pure *presentation*
  factories, tabular as well as graphical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Final

import pandas as pd
import plotly.graph_objects as go

from analytics.kpis import KPIs

_DEFAULT_METRIC: Final[str] = "on_hand"
# Friendly y-axis / title labels for the common race metrics; any other present column is
# allowed too (it just falls back to its raw name).
_METRIC_LABELS: Final[dict[str, str]] = {
    "on_hand": "On hand (units)",
    "inventory_position": "Inventory position (units)",
    "on_order": "On order (units)",
}


def policy_race_timeseries(
    ledgers: Mapping[str, pd.DataFrame],
    *,
    metric: str = _DEFAULT_METRIC,
    shade_windows: Sequence[tuple[int, int, str]] | None = None,
) -> go.Figure:
    """Overlay ``metric`` across policies — one line per policy on a shared ``period`` x-axis.

    ``ledgers`` maps a policy label to its run's Ledger frame (all run on the same demand
    stream). Raises :class:`ValueError` if ``ledgers`` is empty or any frame is missing the
    ``period`` or ``metric`` column.

    ``shade_windows`` optionally shades ``[x0, x1)`` spans — each a ``(x0, x1, label)`` — as
    translucent bands below the traces (the stress-test disruption timeline; the label
    carries the severity, e.g. ``"LT ×2"``). ``None`` leaves the figure band-free (byte-
    identical to the unshaded chart, so the race on Pages 2/8 is unchanged).
    """
    if not ledgers:
        raise ValueError("ledgers is empty: need at least one policy to plot")
    for label, df in ledgers.items():
        missing = {"period", metric} - set(df.columns)
        if missing:
            raise ValueError(f"ledger {label!r} is missing required columns: {sorted(missing)}")

    label_text = _METRIC_LABELS.get(metric, metric)
    fig = go.Figure()
    for label, df in ledgers.items():
        fig.add_trace(go.Scatter(x=df["period"], y=df[metric], mode="lines", name=label))
    fig.update_layout(
        title=f"Policy race — {label_text}",
        xaxis_title="Period",
        yaxis_title=label_text,
        hovermode="x unified",
        legend={"orientation": "h"},
    )
    for x0, x1, label in shade_windows or ():
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="rgba(220,70,70,0.12)",  # faint warning-red; reads on the dark theme
            line_width=0,
            layer="below",  # behind the policy traces
            annotation_text=label,
            annotation_position="top left",
        )
    return fig


def comparison_table(kpis_by_policy: Mapping[str, KPIs]) -> pd.DataFrame:
    """Shape ``{policy label: KPIs}`` into a one-row-per-policy KPI table (raw numbers).

    Index = policy label (insertion order preserved); columns = the 12 ``KPIs`` fields in
    field order. Formatting/rounding is the page's job. Raises :class:`ValueError` if empty.
    """
    if not kpis_by_policy:
        raise ValueError("kpis_by_policy is empty: need at least one policy to tabulate")
    return pd.DataFrame.from_dict(
        {label: asdict(kpis) for label, kpis in kpis_by_policy.items()}, orient="index"
    )
