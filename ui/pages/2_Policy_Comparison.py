"""Policy Comparison — the side-by-side "race" (Page 2).

V2: every selected policy is **user-tunable** and raced on ONE shared demand stream — "would
*my* base-stock beat *my* sQ?". The sidebar configures the shared world (demand · lead time ·
costs · seed) with its Policy section suppressed (``include_policy=False``); the main page picks
which policy *kinds* to race and exposes a parameter editor for each.

Drift model (mirrors Page 1, applied to the world): **Run race** pins the sidebar world; the
policy multiselect, the per-policy editors, and the metric selectbox are **live** controls that
race against the *pinned* world via ``run_cached`` (memoized). Only a sidebar change since the
last Run counts as drift (warn → click Run). An invalid policy (the one reachable case: (s,S)
with ``s >= S``) is reported and skipped — the other policies still race.

Layout note: the per-policy editors sit side by side in ``st.columns`` (the "contestants"); if
that feels cramped on a narrow screen, swap the columns block for one ``st.expander`` per kind
(``build_policy_param_inputs`` is layout-agnostic).

Uses: experiments.single_run.run (via run_cached, one per policy on the shared demand stream)
      + analytics.kpis.compute_kpis + the visualization.comparison race/table factories.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from pydantic import ValidationError

from analytics.kpis import compute_kpis
from core.config import RunConfig
from ui.components.config_builder import (
    build_config_sidebar,
    build_policy_param_inputs,
    policy_label,
)
from ui.components.glossary import tip
from ui.components.run_cache import run_cached
from visualization.comparison import comparison_table, policy_race_timeseries

_KINDS = ["sQ", "sS", "base_stock", "RS"]
_METRICS = ["on_hand", "inventory_position", "on_order"]


st.set_page_config(page_title="Policy Comparison · Inventory Twin", page_icon="🏁", layout="wide")

st.title("Policy Comparison")

base = build_config_sidebar(include_policy=False)
st.caption(f"Shared scenario (demand · lead time · costs · seed): `{base.config_hash()[:12]}`")

selected = st.multiselect("Policies to race", _KINDS, default=_KINDS, key="pc_policies")
st.caption(
    "Every policy races on the **same** demand stream. Edit any policy's parameters below — "
    "defaults are sensible baselines."
)

policy_dicts: dict[str, dict[str, Any]] = {}
if selected:
    columns = st.columns(len(selected))
    for column, kind in zip(columns, [k for k in _KINDS if k in selected], strict=True):
        with column:
            st.markdown(f"**{kind}**")
            policy_dicts[kind] = build_policy_param_inputs(kind, key_prefix="pc_pol")

if st.button("Run race", type="primary", key="pc_run"):
    st.session_state["pc_base_json"] = base.to_json()

if "pc_base_json" in st.session_state:
    world = RunConfig.from_json(st.session_state["pc_base_json"])

    if base.config_hash() != world.config_hash():
        st.warning("Scenario changed in the sidebar. Click **Run race** to refresh results.")

    if not selected:
        st.warning("Select at least one policy to race.")
    else:
        ledgers: dict[str, Any] = {}
        for kind in [k for k in _KINDS if k in selected]:
            params = policy_dicts[kind]
            try:
                config = RunConfig.model_validate({**world.model_dump(), "policy": params})
            except ValidationError:
                st.error(f"**{kind}** policy parameters are invalid — fix them to race it.")
                continue
            ledgers[policy_label(kind, params)] = run_cached(config)

        if ledgers:
            st.caption(f"Race on scenario `{world.config_hash()[:12]}` · {len(ledgers)} policies")
            st.subheader("Policy race", help=tip("chart_race"))
            metric = st.selectbox("Race metric", _METRICS, key="pc_metric")
            st.plotly_chart(
                policy_race_timeseries(ledgers, metric=metric), use_container_width=True
            )
            st.subheader("KPI comparison", help=tip("kpi_table"))
            st.dataframe(comparison_table({k: compute_kpis(v) for k, v in ledgers.items()}).T)
else:
    st.info("Configure the scenario and policies, then click **Run race**.")
