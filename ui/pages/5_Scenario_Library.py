"""Scenario Library — browse / edit / run / export scenarios (Page 5).

A config-management page over the curated ``data/scenarios/*.yaml`` and any saved customs.
Browse a scenario, load it into a YAML editor, validate it live, **run it as-is** (the exact
config — faithful to every scenario, including disruptions, unlike the disruption-fixed Single
Run sidebar), and download or save the canonical YAML. Each config is identified by its
SHA-256 hash, shown throughout.

Running here uses the *loaded* config directly (``run_cached`` over the validated editor text),
so a built-in scenario reproduces its documented results without retyping it into the Single Run
sidebar. There is no config sidebar on this page — it manages files, it does not compose a run.

Uses: ui.components.scenario_io (list/validate/save) + ui.components.run_cache.run_cached +
      analytics.kpis.compute_kpis + the KPI card + the inventory-timeseries factory.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from analytics.kpis import compute_kpis
from core.config import RunConfig
from ui.components.glossary import tip
from ui.components.kpi_card import render_kpi_card
from ui.components.run_cache import run_cached
from ui.components.scenario_io import (
    list_scenarios,
    save_scenario,
    scenario_exists,
    validate_yaml,
)
from visualization.timeseries import inventory_timeseries

_SCENARIOS_DIR = Path("data/scenarios")


def _summary(c: RunConfig) -> str:
    """A one-line readable digest of a parsed config."""
    return (
        f"name: {c.name or '—'} · demand: {c.demand.kind} · pattern: {c.pattern.kind} · "
        f"lead time: {c.lead_time.kind} · policy: {c.policy.kind} · "
        f"horizon: {c.simulation.horizon} · seed: {c.master_seed}"
    )


st.set_page_config(page_title="Scenario Library · Inventory Twin", page_icon="📚", layout="wide")

st.title("Scenario Library")
st.caption(
    "Browse the curated scenarios, edit the YAML, run one as-is, and download or save it. "
    "Each config is identified by its SHA-256 hash."
)

files = list_scenarios(_SCENARIOS_DIR)
if not files:
    st.warning(f"No scenarios found in `{_SCENARIOS_DIR}`.")
    st.stop()

names = [f.name for f in files]
if "sl_editor" not in st.session_state:  # pre-fill the editor on first visit
    st.session_state["sl_editor"] = files[0].read_text()

# --- Browse ---------------------------------------------------------------------------
selected = st.selectbox("Scenario", names, key="sl_file")
sel_config, sel_error = validate_yaml((_SCENARIOS_DIR / selected).read_text())
if sel_error is not None:
    st.caption(f"`{selected}` — ⚠️ does not parse")
else:
    assert sel_config is not None
    st.caption(f"`{selected}` — hash `{sel_config.config_hash()[:12]}`")
if st.button("Load selected into editor", key="sl_load"):  # seeds before the text_area renders
    st.session_state["sl_editor"] = (_SCENARIOS_DIR / selected).read_text()

# --- Edit + validate ------------------------------------------------------------------
st.subheader("Edit")
text = st.text_area("Scenario YAML", key="sl_editor", height=360)
config, error = validate_yaml(text)
if error is not None:
    st.error(f"Invalid scenario:\n\n{error}")
else:
    assert config is not None
    st.success(f"Valid · hash `{config.config_hash()[:12]}`")
    st.caption(_summary(config))

# Run / export are available only for a valid config.
if config is not None:
    # --- Run this scenario (the exact loaded config) ----------------------------------
    st.subheader("Run this scenario")
    if st.button("Run", type="primary", key="sl_run"):
        st.session_state["sl_run_json"] = config.to_json()
    if "sl_run_json" in st.session_state:
        run_config = RunConfig.from_json(st.session_state["sl_run_json"])
        if run_config.config_hash() != config.config_hash():
            st.warning("Edited since the last run. Click **Run** to refresh results.")
        ledger = run_cached(run_config)
        st.caption(f"Results for `{run_config.config_hash()[:12]}`")
        render_kpi_card(compute_kpis(ledger))
        st.subheader("Inventory trajectory", help=tip("chart_inventory"))
        st.plotly_chart(inventory_timeseries(ledger), use_container_width=True)

    # --- Export -----------------------------------------------------------------------
    st.subheader("Export")
    st.download_button(
        "Download canonical YAML",
        config.to_yaml(),
        file_name=selected,
        mime="text/yaml",
        key="sl_download",
    )
    save_name = st.text_input("Save as", value=selected, key="sl_save_name")
    if scenario_exists(_SCENARIOS_DIR, save_name):
        st.warning(f"`{save_name}` already exists — saving overwrites it.")
    if st.button("Save to library", key="sl_save"):
        saved = save_scenario(_SCENARIOS_DIR, save_name, config)
        st.success(f"Saved `{saved.name}` to the library.")
