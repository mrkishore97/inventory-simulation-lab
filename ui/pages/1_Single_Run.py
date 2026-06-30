"""Single Run — one configuration end-to-end (Page 1).

Wires the shared components into the first usable page: ``build_config_sidebar`` (sidebar →
``RunConfig``) + ``run_cached`` (memoized single run) + the inventory-timeseries and
cost-waterfall factories + the KPI summary card + the time-scrubber (a period slider that
moves a marker on the trajectory and shows that moment's state).

Results **pin to the last config that was Run**: the Run button is the explicit trigger, so
changing a sidebar input shows a "configuration changed" warning rather than silently
recomputing (which would make a simulator's numbers a moving target). The include-purchase
checkbox is presentation-only — it re-renders the waterfall live without counting as drift.

Uses: experiments.single_run.run (via run_cached) + analytics.kpis.compute_kpis
      + visualization timeseries/waterfall factories + the KPI card + the time-scrubber.
"""

from __future__ import annotations

import streamlit as st

from analytics.kpis import compute_kpis
from core.config import RunConfig
from ui.components.config_builder import build_config_sidebar
from ui.components.glossary import tip
from ui.components.kpi_card import render_kpi_card
from ui.components.run_cache import run_cached
from ui.components.time_scrubber import render_period_detail, render_time_scrubber
from visualization.timeseries import inventory_timeseries
from visualization.waterfall import cost_waterfall

st.set_page_config(page_title="Single Run · Inventory Twin", page_icon="📈", layout="wide")

st.title("Single Run")

config = build_config_sidebar()
st.caption(f"Current sidebar config: `{config.config_hash()[:12]}`")

if st.button("Run simulation", type="primary", key="sr_run"):
    st.session_state["sr_run_config_json"] = config.to_json()

if "sr_run_config_json" in st.session_state:
    run_config = RunConfig.from_json(st.session_state["sr_run_config_json"])
    ledger = run_cached(run_config)

    if config.config_hash() != run_config.config_hash():
        st.warning(
            "Configuration changed in the sidebar. Click **Run simulation** to refresh results."
        )
    st.caption(f"Showing results for: `{run_config.config_hash()[:12]}`")

    render_kpi_card(compute_kpis(ledger))

    st.subheader("Inventory trajectory", help=tip("chart_inventory"))
    scrub_t = render_time_scrubber(ledger)
    st.plotly_chart(
        inventory_timeseries(ledger, highlight_period=scrub_t), use_container_width=True
    )
    render_period_detail(ledger, scrub_t)

    st.subheader("Cost decomposition", help=tip("chart_waterfall"))
    include_purchase = st.checkbox("Include purchase cost", key="sr_inc_purchase")
    st.plotly_chart(
        cost_waterfall(ledger, include_purchase=include_purchase), use_container_width=True
    )
else:
    st.info("Configure the scenario in the sidebar, then click **Run simulation**.")
