"""Sensitivity Analysis — one-at-a-time tornado: which input matters most?.

Perturb each selected config parameter low/high (±% around its baseline) and rank the parameters by
the swing they induce in an output KPI — the classic tornado. Structurally the twin of Page 3
(Pareto): a shared-world sidebar + a spec picker + a memoized analytic (``sensitivity_cached`` over
``analytics.sensitivity.sensitivity_oat``) + the ``tornado_chart``. The perturbable catalogue +
bounds are the Streamlit-free ``ui.components.sensitivity_spec`` (unit-tested apart from the page).

Drift model (mirrors Pages 3/4): **Run sensitivity** pins the whole spec (world · parameters · ±% ·
replications · output KPI); results recompute from the pinned spec via the memoized
``sensitivity_cached``. Any change since the last Run warns to re-run. The output KPI is part of the
spec (sensitivity of total_cost vs fill_rate are *different computations*, not a relabel), so
changing it is a re-run, not a live toggle.

Uses: ui.components.config_builder.build_config_sidebar + ui.components.sensitivity_spec
      (numeric_paths / low_high) + ui.components.run_cache.sensitivity_cached +
      visualization.sensitivity.tornado_chart.
"""

from __future__ import annotations

import json

import streamlit as st
from pydantic import ValidationError

from core.config import RunConfig
from ui.components.config_builder import build_config_sidebar
from ui.components.glossary import tip
from ui.components.run_cache import sensitivity_cached
from ui.components.sensitivity_spec import low_high, numeric_paths
from visualization.sensitivity import tornado_chart

_OUTPUT_KPIS = [
    "total_cost",
    "fill_rate",
    "cycle_service_level",
    "ready_rate",
    "total_holding_cost",
    "total_ordering_cost",
    "total_stockout_cost",
    "average_on_hand",
    "inventory_turns",
]
# A sensible default selection (cost drivers + demand variability) — intersected with the live
# catalogue so it degrades gracefully if the demand/cost arm changes.
_PREFERRED = [
    "costs.ordering_fixed",
    "costs.holding_per_unit_per_period",
    "demand.std",
    "demand.mean",
]
_MAX_TOTAL_RUNS = 6000  # (1 + 2·params) × reps guard (~the wall-time budget)


def _identity(
    world_hash: str, specs: dict[str, list[float | int]], output_kpi: str, reps: int
) -> str:
    """A stable identity for the spec — the 'results pin to last Run' drift comparison."""
    return json.dumps(
        {"world": world_hash, "specs": specs, "kpi": output_kpi, "reps": reps}, sort_keys=True
    )


st.set_page_config(page_title="Sensitivity · Inventory Twin", page_icon="🌪️", layout="wide")

st.title("Sensitivity Analysis")

base = build_config_sidebar()  # full world incl. policy — every numeric field is a perturb target
st.caption(f"Baseline scenario: `{base.config_hash()[:12]}`")
st.caption(
    "One-at-a-time: each parameter is moved low/high around its baseline; the bars rank the "
    "parameters by the swing they induce in the output KPI (widest = most influential)."
)

catalogue = numeric_paths(base)
options = sorted(catalogue)
default = [p for p in _PREFERRED if p in catalogue] or options[:3]
selected = st.multiselect("Parameters to test", options, default=default, key="se_params")

col_a, col_b, col_c = st.columns(3)
with col_a:
    range_pct = int(
        st.number_input(
            "Perturbation (± %)", min_value=5, max_value=90, value=25, step=5, key="se_range"
        )
    )
with col_b:
    reps = int(
        st.number_input(
            "Replications", min_value=1, max_value=500, value=50, step=10, key="se_reps"
        )
    )
with col_c:
    output_kpi = st.selectbox("Output KPI", _OUTPUT_KPIS, index=0, key="se_kpi")

frac = range_pct / 100.0
specs = {p: list(low_high(catalogue[p], frac)) for p in selected}
total_runs = (1 + 2 * len(selected)) * reps
st.caption(
    f"Perturbing {len(selected)} parameter(s) at ±{range_pct}% · "
    f"(1 + 2×{len(selected)}) × {reps} = {total_runs} run(s)"
)

if st.button("Run sensitivity", type="primary", key="se_run"):
    if not selected:
        st.error("Select at least one parameter to test.")
    elif total_runs > _MAX_TOTAL_RUNS:
        st.error(
            f"Too many runs: {total_runs} (max {_MAX_TOTAL_RUNS}). "
            "Reduce parameters or replications."
        )
    else:
        st.session_state["se_spec_json"] = json.dumps(
            {
                "config_json": base.to_json(),
                "specs": specs,
                "output_kpi": output_kpi,
                "reps": reps,
                "identity": _identity(base.config_hash(), specs, output_kpi, reps),
            }
        )

if "se_spec_json" in st.session_state:
    spec = json.loads(st.session_state["se_spec_json"])

    if _identity(base.config_hash(), specs, output_kpi, reps) != spec["identity"]:
        st.warning("Settings changed. Click **Run sensitivity** to refresh results.")

    try:
        df = sensitivity_cached(
            RunConfig.from_json(spec["config_json"]),
            spec["specs"],
            spec["reps"],
            spec["output_kpi"],
        )
    except (ValidationError, ValueError) as exc:
        st.error(f"Sensitivity run failed — a perturbed value was invalid:\n\n{exc}")
        st.stop()

    st.subheader("Tornado", help=tip("chart_tornado"))
    st.caption(f"Ranked by |swing| in **{spec['output_kpi']}** · {len(df)} parameter(s)")
    st.plotly_chart(tornado_chart(df, output_kpi=spec["output_kpi"]), use_container_width=True)

    st.subheader("Sensitivity ranking")
    st.dataframe(df.round(2))
else:
    st.info("Select parameters and click **Run sensitivity**.")
