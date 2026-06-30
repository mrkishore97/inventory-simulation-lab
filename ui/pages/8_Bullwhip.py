"""Bullwhip Preview — order-vs-demand variance amplification across policies.

Every selected policy is raced on ONE shared demand stream (the sidebar world, built with
``include_policy=False``); each run's Ledger is reduced once by
:func:`analytics.bullwhip.bullwhip_metrics` and the bundles feed BOTH the ``bullwhip_bar``
chart and the per-policy ratio tiles — compute once, reuse (mirrors Page 2's KPI table). Because
all policies face the identical demand, any difference in the ratio is pure policy-induced
amplification: the demo punchline that (s, Q) batching towers over a smoothing base-stock.

Drift model (mirrors Pages 2/7): **Run** pins the sidebar world; the policy multiselect, the
per-policy editors, and the **log-scale checkbox** are live controls that re-render against the
*pinned* world via ``run_cached`` (memoized) — only a sidebar change since the last Run is drift.
An invalid policy (the one reachable case: (s, S) with ``s >= S``) is reported and skipped; the
others still plot. A ``nan`` ratio (constant demand — Var(demand) = 0) is honest: a gap in the
chart and a ``—`` / "constant demand" tile, never a fake number.

Uses: ui.components.config_builder (sidebar · per-policy editor · shared policy_label)
      + ui.components.run_cache.run_cached + analytics.bullwhip.bullwhip_metrics
      + the visualization.bullwhip.bullwhip_bar factory.
"""

from __future__ import annotations

import math
from typing import Any

import streamlit as st
from pydantic import ValidationError

from analytics.bullwhip import BullwhipMetrics, bullwhip_metrics
from core.config import RunConfig
from ui.components.config_builder import (
    build_config_sidebar,
    build_policy_param_inputs,
    policy_label,
)
from ui.components.glossary import tip
from ui.components.run_cache import run_cached
from visualization.bullwhip import bullwhip_bar

_KINDS = ["sQ", "sS", "base_stock", "RS"]


def _band(ratio: float) -> str:
    """The qualitative bullwhip band for a tile descriptor."""
    if math.isnan(ratio):  # Var(demand) == 0 -> ratio undefined (constant demand)
        return "constant demand"
    if ratio > 1.0:
        return "amplifies"
    if ratio < 1.0:
        return "smooths"
    return "pass-through"


st.set_page_config(page_title="Bullwhip Preview · Inventory Twin", page_icon="🌊", layout="wide")

st.title("Bullwhip Preview")
st.caption(
    "**Bullwhip ratio = Var(orders) / Var(demand).** Above 1 → orders vary *more* than demand "
    "(amplification, e.g. (s, Q) batching); below 1 → the policy smooths demand (e.g. base-stock)."
)

base = build_config_sidebar(include_policy=False)
st.caption(f"Shared scenario (demand · lead time · costs · seed): `{base.config_hash()[:12]}`")

selected = st.multiselect("Policies to compare", _KINDS, default=_KINDS, key="bw_policies")
st.caption(
    "Every policy faces the **same** demand stream, so differences in the ratio are pure "
    "policy-induced amplification. Edit any policy's parameters below."
)

policy_dicts: dict[str, dict[str, Any]] = {}
if selected:
    columns = st.columns(len(selected))
    for column, kind in zip(columns, [k for k in _KINDS if k in selected], strict=True):
        with column:
            st.markdown(f"**{kind}**")
            policy_dicts[kind] = build_policy_param_inputs(kind, key_prefix="bw_pol")

if st.button("Run", type="primary", key="bw_run"):
    st.session_state["bw_base_json"] = base.to_json()

if "bw_base_json" in st.session_state:
    world = RunConfig.from_json(st.session_state["bw_base_json"])

    if base.config_hash() != world.config_hash():
        st.warning("Scenario changed in the sidebar. Click **Run** to refresh results.")

    if not selected:
        st.warning("Select at least one policy to compare.")
    else:
        metrics: dict[str, BullwhipMetrics] = {}
        for kind in [k for k in _KINDS if k in selected]:
            params = policy_dicts[kind]
            try:
                config = RunConfig.model_validate({**world.model_dump(), "policy": params})
            except ValidationError:
                st.error(f"**{kind}** policy parameters are invalid — fix them to include it.")
                continue
            metrics[policy_label(kind, params)] = bullwhip_metrics(run_cached(config))

        if metrics:
            st.caption(
                f"Bullwhip on scenario `{world.config_hash()[:12]}` · {len(metrics)} policies"
            )
            st.subheader("Bullwhip", help=tip("chart_bullwhip"))
            log_y = st.checkbox("Log scale (y)", key="bw_log")  # live, not drift
            st.plotly_chart(bullwhip_bar(metrics, log_y=log_y), use_container_width=True)

            tiles = st.columns(len(metrics))
            for tile, (label, m) in zip(tiles, metrics.items(), strict=True):
                value = "—" if math.isnan(m.bullwhip_ratio) else f"{m.bullwhip_ratio:,.2f}×"
                tile.metric(
                    label,
                    value,
                    delta=_band(m.bullwhip_ratio),
                    delta_color="off",  # the word, not a colour, carries the meaning
                    help=tip("bullwhip_ratio"),
                )

            st.caption(
                "The dashed line marks pass-through (ratio = 1). Bullwhip effect — "
                "Lee, Padmanabhan & Whang (1997); Chen et al. (2000)."
            )
else:
    st.info("Configure the scenario and policies, then click **Run**.")
