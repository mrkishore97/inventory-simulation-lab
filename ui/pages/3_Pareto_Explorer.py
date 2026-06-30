"""Pareto Explorer — sweep a policy grid and read the service-vs-cost frontier.

Pick a policy *kind*, set its base parameters, and sweep them over a symmetric ±range grid; each
grid point is Monte-Carlo evaluated (``analytics.pareto.pareto_sweep`` via ``sweep_cached``) and
the non-dominated points form the efficient frontier (``visualization.pareto.pareto_scatter``).
**Click a point** to select a policy, then **Load into Single Run** — the parameters hand off to
Page 1 via :data:`ui.components.config_builder.POLICY_HANDOFF_KEY` (consumed there to seed the
sidebar widgets), so the frontier doubles as a launcher into the detailed single-run view.

Drift model (mirrors Pages 1–2): **Run sweep** pins the whole spec (world · kind · base params ·
grid · replications · axes); results recompute from the *pinned* spec via the memoized
``sweep_cached``. Any change to the sidebar or the sweep controls since the last Run warns to
re-run. Selecting a point and loading is **live** — it operates on the pinned frontier, not a
re-sweep. Axes are part of the spec (the frontier mask depends on them), so changing an axis is
a re-run, not a presentation toggle.

Uses: ui.components.config_builder (shared world + per-kind editor + handoff key) +
      ui.components.run_cache.sweep_cached + ui.components.pareto_select.clicked_params +
      visualization.pareto.pareto_scatter.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import streamlit as st
from pydantic import ValidationError

from core.config import RunConfig
from ui.components.config_builder import (
    POLICY_HANDOFF_KEY,
    build_config_sidebar,
    build_policy_param_inputs,
)
from ui.components.glossary import tip
from ui.components.pareto_select import clicked_params
from ui.components.run_cache import sweep_cached
from visualization.pareto import pareto_scatter

_KINDS = ["sQ", "sS", "base_stock", "RS"]
_COST_KPIS = [
    "total_cost",
    "total_holding_cost",
    "total_ordering_cost",
    "total_stockout_cost",
    "total_purchase_cost",
]
_SERVICE_KPIS = ["fill_rate", "cycle_service_level", "ready_rate"]
# (field, is_int, minimum) per kind — drives the symmetric ± grid built around each base value.
_SWEEP_FIELDS: dict[str, list[tuple[str, bool, float | None]]] = {
    "sQ": [("reorder_point", False, None), ("order_quantity", True, 1.0)],
    "sS": [("reorder_point", False, None), ("order_up_to", False, 0.01)],
    "base_stock": [("target_level", False, 0.01)],
    "RS": [("review_period", True, 1.0), ("order_up_to", False, 0.01)],
}
# grid points x replications guard (~the wall-time budget); the user must shrink the sweep.
_MAX_TOTAL_RUNS = 6000


def _policy_label(kind: str, p: Mapping[str, Any]) -> str:
    """A self-documenting label for the selected point (``:g`` drops the trailing ``.0``)."""
    if kind == "sQ":
        return f"sQ(s={p['reorder_point']:g}, Q={p['order_quantity']})"
    if kind == "sS":
        return f"sS(s={p['reorder_point']:g}, S={p['order_up_to']:g})"
    if kind == "base_stock":
        return f"BaseStock(S={p['target_level']:g})"
    return f"RS(R={p['review_period']}, S={p['order_up_to']:g})"


def _build_grid(
    kind: str, base_params: Mapping[str, Any], steps: int, range_frac: float
) -> dict[str, list[float | int]]:
    """A symmetric ``±range_frac`` grid (``steps`` points) around each of the kind's base values.

    Int fields are rounded to whole units; values below a field's minimum are dropped; each axis
    is de-duplicated. A field that collapses to nothing is omitted (held at its base).
    """
    grid: dict[str, list[float | int]] = {}
    for field, is_int, minimum in _SWEEP_FIELDS[kind]:
        base = float(base_params[field])
        values = np.linspace(base * (1 - range_frac), base * (1 + range_frac), steps)
        if is_int:
            values = np.round(values)
        if minimum is not None:
            values = values[values >= minimum]
        uniq = sorted({(int(v) if is_int else float(v)) for v in values.tolist()})
        if uniq:
            grid[field] = uniq
    return grid


def _identity(
    world_hash: str,
    base_params: Mapping[str, Any],
    grid: Mapping[str, Any],
    reps: int,
    cost_kpi: str,
    service_kpi: str,
) -> str:
    """A stable identity for the sweep spec — the 'results pin to last Run' drift comparison."""
    return json.dumps(
        {
            "world": world_hash,
            "base": dict(base_params),
            "grid": dict(grid),
            "reps": reps,
            "cost": cost_kpi,
            "service": service_kpi,
        },
        sort_keys=True,
    )


st.set_page_config(page_title="Pareto Explorer · Inventory Twin", page_icon="📐", layout="wide")

st.title("Pareto Explorer")

base = build_config_sidebar(include_policy=False)
st.caption(f"Shared scenario (demand · lead time · costs · seed): `{base.config_hash()[:12]}`")

kind = st.selectbox("Policy kind to sweep", _KINDS, index=0, key="pe_kind")
st.caption(
    "Sweep this policy's parameters over a grid; each point is Monte-Carlo evaluated and the "
    "non-dominated (low-cost, high-service) points form the efficient frontier."
)

st.subheader("Base parameters")
base_params = build_policy_param_inputs(kind, key_prefix="pe_pol")

st.subheader("Sweep")
col_a, col_b, col_c = st.columns(3)
with col_a:
    steps = int(
        st.number_input(
            "Grid steps per parameter", min_value=2, max_value=8, value=5, step=1, key="pe_steps"
        )
    )
with col_b:
    range_pct = int(
        st.number_input("Range (± %)", min_value=5, max_value=90, value=50, step=5, key="pe_range")
    )
with col_c:
    reps = int(
        st.number_input(
            "Replications per point", min_value=1, max_value=500, value=50, step=10, key="pe_reps"
        )
    )

with st.expander("Axes"):
    cost_kpi = st.selectbox("Cost (y-axis)", _COST_KPIS, index=0, key="pe_cost_kpi")
    service_kpi = st.selectbox("Service (x-axis)", _SERVICE_KPIS, index=0, key="pe_service_kpi")

grid = _build_grid(kind, base_params, steps, range_pct / 100.0)
n_points = math.prod(len(v) for v in grid.values()) if grid else 0
total_runs = n_points * reps
st.caption(f"Grid: {n_points} point(s) × {reps} replication(s) = {total_runs} run(s)")

if st.button("Run sweep", type="primary", key="pe_run"):
    if not grid:
        st.error("No parameters to sweep — increase the steps or range.")
    elif total_runs > _MAX_TOTAL_RUNS:
        st.error(
            f"Grid too large: {total_runs} runs (max {_MAX_TOTAL_RUNS}). "
            "Reduce steps, range, or replications."
        )
    else:
        try:
            sweep_config = RunConfig.model_validate({**base.model_dump(), "policy": base_params})
        except ValidationError:
            st.error("Base policy parameters are invalid — fix them before sweeping.")
        else:
            st.session_state["pe_spec_json"] = json.dumps(
                {
                    "config_json": sweep_config.to_json(),
                    "grid": grid,
                    "kind": kind,
                    "base_params": dict(base_params),
                    "reps": reps,
                    "cost_kpi": cost_kpi,
                    "service_kpi": service_kpi,
                    "identity": _identity(
                        base.config_hash(), base_params, grid, reps, cost_kpi, service_kpi
                    ),
                }
            )

if "pe_spec_json" in st.session_state:
    spec = json.loads(st.session_state["pe_spec_json"])

    current = _identity(base.config_hash(), base_params, grid, reps, cost_kpi, service_kpi)
    if current != spec["identity"]:
        st.warning("Sweep settings changed. Click **Run sweep** to refresh results.")

    sweep_df = sweep_cached(
        RunConfig.from_json(spec["config_json"]),
        spec["grid"],
        spec["reps"],
        spec["cost_kpi"],
        spec["service_kpi"],
    )
    st.subheader("Pareto frontier", help=tip("chart_pareto"))
    st.caption(
        f"Swept {spec['kind']} · {len(sweep_df)} grid points · "
        f"{int(sweep_df['on_frontier'].sum())} on the frontier"
    )
    event = st.plotly_chart(
        pareto_scatter(sweep_df, cost_kpi=spec["cost_kpi"], service_kpi=spec["service_kpi"]),
        on_select="rerun",
        selection_mode="points",
        use_container_width=True,
        key="pe_chart",
    )

    clicked = clicked_params(sweep_df, getattr(event, "selection", None))
    if clicked is not None:
        full = {**spec["base_params"], **clicked}
        st.success(f"Selected **{_policy_label(spec['kind'], full)}** — load it to inspect.")
        if st.button("Load into Single Run", type="primary", key="pe_load"):
            st.session_state[POLICY_HANDOFF_KEY] = full
            st.switch_page("pages/1_Single_Run.py")
    else:
        st.info("Click a point on the chart to select a policy, then load it into Single Run.")
else:
    st.info("Configure the sweep, then click **Run sweep**.")
