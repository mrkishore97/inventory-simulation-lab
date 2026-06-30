"""Policy Advisor — a recommended policy for a demand profile, explained.

Pick an M5 archetype (or point at a demand parquet), set the business context (lead time, costs,
service level), and the heuristic advisor (:func:`recommender.advisor.recommend`) classifies the
demand (Syntetos-Boylan), maps it to a policy, derives the policy's parameters from the classical
inventory laws, and explains the reasoning in plain English. **Load into Single Run** hands the
recommended policy off to Page 1 via :data:`ui.components.config_builder.POLICY_HANDOFF_KEY`.

This page completes the capability (the Page 6 stub was blocked on the
``recommender/`` module, now built). The advisor is cheap (classify + closed-form, no
simulation), so the recommendation is **live**: it recomputes on any input change — no pin / drift.

Uses: recommender.advisor.recommend + ui.components.archetypes.load_archetypes +
      ui.components.demand_input.read_demand_column + ui.components.config_builder
      (policy_label + the POLICY_HANDOFF_KEY handoff).
"""

from __future__ import annotations

import streamlit as st

from recommender.advisor import recommend
from ui.components.archetypes import Archetype, load_archetypes
from ui.components.config_builder import POLICY_HANDOFF_KEY, policy_label
from ui.components.demand_input import read_demand_column
from ui.components.glossary import tip

st.set_page_config(page_title="Policy Advisor · Inventory Twin", page_icon="🧭", layout="wide")

st.title("Policy Advisor")
st.markdown(
    "Pick a demand profile and a business context; the advisor classifies the demand "
    "(Syntetos-Boylan), recommends a policy, derives its starting parameters from the classical "
    "inventory laws, and explains the reasoning."
)

# Business context — the four scalars the advisor consumes.
with st.sidebar:
    st.header("Business context")
    lead_time = float(
        st.number_input("Lead time (periods)", min_value=1, value=3, step=1, key="adv_lead_time")
    )
    ordering_cost = float(
        st.number_input("Ordering cost (K)", min_value=0.0, value=20.0, key="adv_ordering_cost")
    )
    holding_cost = float(
        st.number_input(
            "Holding cost (/unit/period)", min_value=0.01, value=0.5, key="adv_holding_cost"
        )
    )
    service_level = float(
        st.slider(
            "Service level",
            min_value=0.50,
            max_value=0.99,
            value=0.95,
            step=0.01,
            key="adv_service_level",
        )
    )

# Demand source — an M5 archetype or any parquet with a 'demand' column.
st.subheader("Demand profile")
source = st.radio("Source", ["M5 archetype", "Custom path"], horizontal=True, key="adv_source")
arch: Archetype | None
if source == "M5 archetype":
    archetypes = load_archetypes()
    arch = archetypes[st.selectbox("Archetype", list(archetypes), key="adv_arch")]
    history_path: str = arch.history_path
    st.caption(f"**{arch.name.title()}** archetype · {arch.item_id} · {arch.n_rows} periods")
else:
    arch = None
    history_path = st.text_input(
        "Demand parquet path", value="data/m5/sample_smooth.parquet", key="adv_path"
    )

try:
    demand = read_demand_column(history_path)
    rec = recommend(
        demand,
        lead_time=lead_time,
        ordering_cost=ordering_cost,
        holding_cost=holding_cost,
        service_level=service_level,
    )
except (ValueError, OSError) as exc:
    st.error(f"Cannot advise on this demand: {exc}")
    st.stop()

# Classification — the Syntetos-Boylan quadrant and the two metrics behind it.
st.subheader("Demand classification")
c = rec.classification
col1, col2, col3 = st.columns(3)
col1.metric("SB class", c.sb_class, help=tip("sb_class"))
col2.metric("ADI", f"{c.adi:.2f}", help=tip("adi"))
col3.metric("CV²", f"{c.cv2:.2f}", help=tip("cv2"))
if arch is not None:
    st.caption(
        f"Business archetype **{arch.name.title()}** → statistical SB class **{c.sb_class}** "
        "(they need not match — *seasonal* is *erratic*, *promotional* is *lumpy*)."
    )

# Recommendation — the policy, the context that drove it, and the plain-English rationale.
st.subheader("Recommended policy")
st.metric("Policy", policy_label(rec.policy.kind, rec.policy.model_dump()), help=tip("policy"))
st.caption(
    f"Context used: lead time = {lead_time:g} · ordering cost = {ordering_cost:g} · "
    f"holding cost = {holding_cost:g} · service level = {service_level:.0%} — these drive the "
    "parameters, so changing them in the sidebar moves the recommendation."
)
st.info(rec.rationale)
if rec.feasibility_warning:
    st.warning(rec.feasibility_warning)

# Hand off the recommended policy to Single Run (policy only — see the caption).
st.caption(
    "This loads **only the recommended policy** into Single Run. Match demand, lead time, and "
    "costs manually there if you want to reproduce this exact recommendation context."
)
if st.button("Load into Single Run", type="primary", key="adv_load"):
    st.session_state[POLICY_HANDOFF_KEY] = rec.policy.model_dump()
    st.switch_page("pages/1_Single_Run.py")
