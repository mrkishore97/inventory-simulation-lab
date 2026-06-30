"""Monte Carlo & Risk — the per-rep KPI distribution + VaR/CVaR tail risk.

Configure a scenario, run many replications **once** (`monte_carlo` via `mc_cached`), then analyze
the resulting KPI distribution **live**: the KPI and confidence controls re-derive
`analytics.risk.risk_metrics` over the *pinned* distribution on every change — a pure reduction, no
re-simulation (the same live-vs-pinned model as the other pages). The histogram shows
the spread a single mean hides, with mean / VaR / CVaR markers.

Drift model: **Run Monte Carlo** pins the scenario + replication count; only a sidebar or
replications change since the last Run warns to re-run. The KPI / confidence controls are live.

The "worst" tail is derived from each KPI's semantics: higher-is-better KPIs (the service trinity +
inventory turns) use the **lower** tail; cost/quantity KPIs — including days-of-supply, where a high
value is overstock risk — use the **upper** tail. The analyzed tail is stated in a caption.

Uses: ui.components.config_builder.build_config_sidebar + ui.components.run_cache.mc_cached +
      analytics.risk.risk_metrics + the visualization.distribution histogram factory.
"""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Literal

import streamlit as st

from analytics.kpis import KPIs
from analytics.risk import risk_metrics
from core.config import RunConfig
from ui.components.config_builder import build_config_sidebar
from ui.components.glossary import tip
from ui.components.run_cache import mc_cached
from visualization.distribution import kpi_distribution_histogram

_KPIS = [f.name for f in fields(KPIs)]
# Higher-is-better KPIs → the bad tail is the LOW one; everything else (cost/quantity, incl.
# days_of_supply where a high value is overstock risk) → the HIGH tail.
_LOWER_TAIL_KPIS = {"cycle_service_level", "fill_rate", "ready_rate", "inventory_turns"}
_RATE_KPIS = {"cycle_service_level", "fill_rate", "ready_rate"}


def _tail_for(kpi: str) -> Literal["upper", "lower"]:
    return "lower" if kpi in _LOWER_TAIL_KPIS else "upper"


def _fmt(kpi: str, value: float) -> str:
    """Format a KPI value: rates as a percentage, everything else thousands-grouped."""
    return f"{value:.1%}" if kpi in _RATE_KPIS else f"{value:,.2f}"


st.set_page_config(page_title="Monte Carlo & Risk · Inventory Twin", page_icon="🎲", layout="wide")

st.title("Monte Carlo & Risk")

config = build_config_sidebar()
st.caption(f"Scenario: `{config.config_hash()[:12]}`")

reps = int(
    st.number_input("Replications", min_value=50, max_value=2000, value=500, step=50, key="mc_reps")
)

if st.button("Run Monte Carlo", type="primary", key="mc_run"):
    st.session_state["mc_spec_json"] = json.dumps({"config_json": config.to_json(), "reps": reps})

if "mc_spec_json" in st.session_state:
    spec = json.loads(st.session_state["mc_spec_json"])
    run_config = RunConfig.from_json(spec["config_json"])
    run_reps = int(spec["reps"])

    if config.config_hash() != run_config.config_hash() or reps != run_reps:
        st.warning(
            "Scenario or replications changed. Click **Run Monte Carlo** to refresh results."
        )

    distribution = mc_cached(run_config, run_reps)
    st.caption(f"{run_reps} replications of `{run_config.config_hash()[:12]}`")

    # Live analysis controls — pure reductions over the pinned distribution, no re-simulation.
    col_kpi, col_conf = st.columns(2)
    with col_kpi:
        kpi = st.selectbox("KPI", _KPIS, index=_KPIS.index("total_cost"), key="mc_kpi")
    with col_conf:
        confidence = float(
            st.slider(
                "Confidence level",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                key="mc_conf",
            )
        )
    tail = _tail_for(kpi)
    risk = risk_metrics(distribution, column=kpi, confidence_level=confidence, tail=tail)
    st.caption(f"Analyzing the **{tail}** tail — the worse outcomes for `{kpi}`.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean", _fmt(kpi, risk.mean), help=tip("mc_mean"))
    m2.metric(f"VaR ({confidence:.0%})", _fmt(kpi, risk.value_at_risk), help=tip("var"))
    m3.metric(
        f"CVaR ({confidence:.0%})", _fmt(kpi, risk.conditional_value_at_risk), help=tip("cvar")
    )
    m4.metric("Tail probability", f"{risk.tail_probability:.0%}", help=tip("tail_probability"))

    st.subheader("Outcome distribution", help=tip("chart_distribution"))
    st.plotly_chart(
        kpi_distribution_histogram(distribution, column=kpi, risk=risk), use_container_width=True
    )

    st.subheader("Distribution summary")
    st.dataframe(distribution.drop(columns=["replication", "seed"]).describe().T)
else:
    st.info("Configure the scenario, set the replication count, then click **Run Monte Carlo**.")
