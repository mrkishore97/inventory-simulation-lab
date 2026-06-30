"""Pareto point-selection → policy params — the click-to-load mapping.

Page 3 renders :func:`visualization.pareto.pareto_scatter` with Streamlit point selection
(``st.plotly_chart(..., on_select="rerun")``). Each plotted point carries its source
``sweep_df`` row index as ``customdata``; these pure helpers turn a selection event back into
the clicked point's swept policy parameters, which the page merges with the held base params
and hands off to Single Run via :data:`ui.components.config_builder.POLICY_HANDOFF_KEY`.

Kept here — an importable component, not the digit-prefixed page script — so the click→params
mapping is unit-testable without the Streamlit runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any, Final

import pandas as pd

from analytics.kpis import KPIs

# KPI-mean columns in a sweep frame; every other column (minus ``on_frontier``) is a swept
# policy parameter (cf. visualization.pareto._KPI_FIELDS — recomputed here to avoid a
# cross-module private import).
_KPI_FIELDS: Final[frozenset[str]] = frozenset(f.name for f in fields(KPIs))
# Policy fields the config-builder widgets collect as ints; every other param is a float.
_INT_FIELDS: Final[frozenset[str]] = frozenset({"order_quantity", "review_period"})


def clicked_index(selection: Mapping[str, Any] | None) -> int | None:
    """Source-frame row index of the first selected point, or ``None`` if none is selected.

    ``selection`` is the ``st.plotly_chart`` selection state (``{"points": [...]}``); each
    point's ``customdata`` is the ``[idx]`` stamped by
    :func:`visualization.pareto.pareto_scatter`.
    """
    points = (selection or {}).get("points") or []
    if not points:
        return None
    return int(points[0]["customdata"][0])


def clicked_params(
    sweep_df: pd.DataFrame, selection: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Swept policy parameters of the clicked point, cast to the widgets' types.

    Returns only the *swept* columns (neither KPI means nor ``on_frontier``); the page merges
    them over the held base params. ``None`` when nothing is selected (or the selected index is
    stale — e.g. a click that predates a re-sweep to a smaller grid).
    """
    idx = clicked_index(selection)
    if idx is None or idx not in sweep_df.index:
        return None
    row = sweep_df.loc[idx]
    param_cols = [c for c in sweep_df.columns if c not in _KPI_FIELDS and c != "on_frontier"]
    return {c: (int(row[c]) if c in _INT_FIELDS else float(row[c])) for c in param_cols}
