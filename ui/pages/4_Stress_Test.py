"""Stress Test Composer — lead-time disruptions vs an undisrupted baseline.

Compose a ``ScheduledDisruption`` timeline (one or more ``(start, duration, multiplier)``
windows) and race the **stressed** run against the **undisrupted baseline** of the *same*
scenario. The Single Run sidebar fixes ``disruption: none``, so the baseline comes
straight from ``build_config_sidebar()`` and this page composes the disruption itself; the
disrupted run is the same config with the composed windows, on the **same ``master_seed``**.

Because the disruption channel is deterministic (it draws no RNG), the baseline and disrupted
runs share a byte-identical demand stream — only the realized lead times inside the windows
diverge. So this is a clean A/B (the race premise), reusing the race factory + KPI
table for the overlay, plus ``st.metric`` deltas for the headline impact.

Drift model (mirrors Pages 1–3): **Run stress test** pins the baseline config + the composed
windows; results recompute from the pinned spec via ``run_cached``. A sidebar or window change
since the last Run warns to re-run; the race-metric selectbox is live.

Uses: ui.components.config_builder.build_config_sidebar + ui.components.run_cache.run_cached +
      analytics.kpis.compute_kpis + the visualization.comparison race/table factories +
      ui.components.scenario_io (save/share the composed scenario).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
from pydantic import ValidationError

from analytics.kpis import compute_kpis
from core.config import DeterministicLeadTimeConfig, RunConfig
from leadtime.base import round_to_periods
from ui.components.config_builder import build_config_sidebar
from ui.components.glossary import tip
from ui.components.run_cache import run_cached
from ui.components.scenario_io import save_scenario, scenario_exists
from visualization.comparison import comparison_table, policy_race_timeseries

_METRICS = ["on_hand", "inventory_position", "on_order"]
_MAX_WINDOWS = 5
_SCENARIOS_DIR = Path("data/scenarios")  # shared with Page 5 → the save-and-share loop


def _identity(base_hash: str, windows: list[dict[str, Any]]) -> str:
    """Stable identity for the stress spec — the 'results pin to last Run' drift comparison."""
    return json.dumps({"base": base_hash, "windows": windows}, sort_keys=True)


st.set_page_config(page_title="Stress Test · Inventory Twin", page_icon="⚠️", layout="wide")

st.title("Stress Test Composer")

base = build_config_sidebar()  # include_policy=True; disruption is fixed to none = the baseline
st.caption(f"Baseline scenario (disruption-free): `{base.config_hash()[:12]}`")

st.subheader("Disruption timeline")
st.caption(
    "An order placed in a window's `[start, start + duration)` has its lead time multiplied: "
    "`> 1` delays replenishment (the typical supply shock), `< 1` expedites it."
)
n_windows = int(
    st.number_input(
        "Number of windows",
        min_value=1,
        max_value=_MAX_WINDOWS,
        value=1,
        step=1,
        key="st_n_windows",
    )
)

windows: list[dict[str, Any]] = []
for i in range(n_windows):
    c1, c2, c3 = st.columns(3)
    with c1:
        start = int(
            st.number_input(
                f"Start (window {i + 1})", min_value=0, value=30, step=1, key=f"st_w{i}_start"
            )
        )
    with c2:
        duration = int(
            st.number_input(
                f"Duration (window {i + 1})", min_value=1, value=30, step=1, key=f"st_w{i}_dur"
            )
        )
    with c3:
        multiplier = float(
            st.number_input(
                f"Multiplier (window {i + 1})",
                min_value=0.01,
                value=2.0,
                step=0.5,
                key=f"st_w{i}_mult",
            )
        )
    windows.append({"start": start, "duration": duration, "multiplier": multiplier})

# Effective lead-time caption: the disruption multiplies the base lead time, then
# round_to_periods() rounds to the nearest whole period (>=1). For a deterministic base L this
# collapses bands of multipliers onto one integer — e.g. L=3 with ×1.6 / ×1.7 / ×1.8 all round to
# 5 — so nearby multipliers can produce identical runs and behavior steps only at the thresholds.
lt = base.lead_time
if isinstance(lt, DeterministicLeadTimeConfig):
    parts: list[str] = []
    for i, w in enumerate(windows):
        raw = lt.lead_time * w["multiplier"]
        eff = round_to_periods(raw)
        parts.append(f"w{i + 1}: {lt.lead_time}×{w['multiplier']:g} = {raw:g} → **{eff} periods**")
    st.caption(
        "Effective lead time, base L×multiplier rounded to the nearest whole period (≥1): "
        + "; ".join(parts)
    )
else:
    st.caption(
        f"Effective lead time: each order's sampled `{lt.kind}` lead time is multiplied by the "
        "window factor, then rounded to the nearest whole period (≥1)."
    )

horizon = base.simulation.horizon
if all(w["start"] >= horizon for w in windows):
    st.caption(f"⚠️ Every window starts at or after the horizon ({horizon}) — no effect on the run.")

if st.button("Run stress test", type="primary", key="st_run"):
    st.session_state["st_spec_json"] = json.dumps({"base_json": base.to_json(), "windows": windows})

if "st_spec_json" in st.session_state:
    spec = json.loads(st.session_state["st_spec_json"])
    base_run = RunConfig.from_json(spec["base_json"])
    spec_windows = spec["windows"]

    if _identity(base.config_hash(), windows) != _identity(base_run.config_hash(), spec_windows):
        st.warning("Settings changed. Click **Run stress test** to refresh results.")

    try:
        disrupted = RunConfig.model_validate(
            {**base_run.model_dump(), "disruption": {"kind": "scheduled", "windows": spec_windows}}
        )
    except ValidationError as exc:
        st.error(f"Invalid disruption timeline:\n\n{exc}")
        st.stop()

    baseline_ledger = run_cached(base_run)
    disrupted_ledger = run_cached(disrupted)
    base_kpis = compute_kpis(baseline_ledger)
    disr_kpis = compute_kpis(disrupted_ledger)

    spans = ", ".join(
        f"[{w['start']}, {w['start'] + w['duration']})×{w['multiplier']:g}" for w in spec_windows
    )
    st.caption(
        f"Baseline `{base_run.config_hash()[:12]}` vs disrupted `{disrupted.config_hash()[:12]}` "
        f"· windows: {spans}"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Total cost",
        f"${disr_kpis.total_cost:,.0f}",
        delta=f"{disr_kpis.total_cost - base_kpis.total_cost:+,.0f}",
        delta_color="inverse",  # a cost increase is bad
        help=tip("total_cost"),
    )
    m2.metric(
        "Fill rate",
        f"{disr_kpis.fill_rate:.1%}",
        delta=f"{disr_kpis.fill_rate - base_kpis.fill_rate:+.1%}",
        help=tip("fill_rate"),
    )
    m3.metric(
        "Cycle service",
        f"{disr_kpis.cycle_service_level:.1%}",
        delta=f"{disr_kpis.cycle_service_level - base_kpis.cycle_service_level:+.1%}",
        help=tip("cycle_service"),
    )

    ledgers = {"Baseline": baseline_ledger, "Disrupted": disrupted_ledger}
    run_horizon = base_run.simulation.horizon
    shade = [
        (w["start"], min(w["start"] + w["duration"], run_horizon), f"LT ×{w['multiplier']:g}")
        for w in spec_windows
        if w["start"] < run_horizon  # clip the timeline to the plotted domain (page owns this)
    ]
    st.subheader("Policy race", help=tip("chart_race"))
    metric = st.selectbox("Race metric", _METRICS, key="st_metric")
    st.plotly_chart(
        policy_race_timeseries(ledgers, metric=metric, shade_windows=shade),
        use_container_width=True,
    )

    st.subheader("KPI comparison", help=tip("kpi_table"))
    st.dataframe(comparison_table({"Baseline": base_kpis, "Disrupted": disr_kpis}).T)

    st.subheader("Export composed scenario")
    st.caption(
        "Saves the disrupted config (baseline + this timeline) — reload it on **Scenario "
        "Library** to reproduce or share the stressed run."
    )
    fname = f"stress_{disrupted.config_hash()[:12]}.yaml"  # hash-identified
    st.download_button(
        "Download composed scenario YAML",
        disrupted.to_yaml(),
        file_name=fname,
        mime="text/yaml",
        key="st_download",
    )
    save_name = st.text_input("Save as", value=fname, key="st_save_name")
    if scenario_exists(_SCENARIOS_DIR, save_name):
        st.warning(f"`{save_name}` already exists — saving overwrites it.")
    if st.button("Save to library", key="st_save"):
        saved = save_scenario(_SCENARIOS_DIR, save_name, disrupted)
        st.success(f"Saved `{saved.name}` to the library.")
else:
    st.info("Configure the scenario and disruption timeline, then click **Run stress test**.")
