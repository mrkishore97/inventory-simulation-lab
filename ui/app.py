"""Inventory Twin — Streamlit app entry point (Home / landing page).

Run locally::

    uv run streamlit run ui/app.py

This is the M4 multipage shell. Streamlit auto-discovers the
numbered scripts in ``ui/pages/`` and builds the sidebar navigation from them; this
module is the landing page. The pages are filled in by the later pages — for
now each is an annotated stub naming the M1–M3 function it will consume.

The UI is a thin **presentation layer** over the finished M1–M3 compute surface: every
number it will show comes from an already-tested function (``single_run.run``,
``compute_kpis``, ``monte_carlo``, ``pareto_sweep``, ``sensitivity_oat``,
``risk_metrics``, ``bullwhip_metrics``). No business logic lives in ``ui/``.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Inventory Twin",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📦 Inventory Twin")
st.markdown(
    "An interactive **inventory-policy simulator**. Configure a policy, a demand model, "
    "and a lead-time model; run a digital twin of the replenishment process; and measure "
    "the service-level-vs-cost trade-off with Monte Carlo rigor."
)

st.subheader("Pages")
st.markdown(
    "Use the sidebar to navigate. Each page focuses on a single mental model:\n\n"
    "1. **Single Run** — one configuration: inventory time-series, cost waterfall, and a "
    "KPI card, with a time-scrubber to inspect any period.\n"
    "2. **Policy Comparison** — race several policies on the *same* demand stream.\n"
    "3. **Pareto Explorer** — sweep a policy grid and read the service-vs-cost frontier.\n"
    "4. **Stress Test** — place lead-time disruptions on a timeline and compare to a baseline.\n"
    "5. **Scenario Library** — browse, load, edit, and save scenario configurations.\n"
    "6. **Policy Advisor** — pick a demand archetype and get a recommended policy.\n"
    "7. **Monte Carlo & Risk** — run many replications and read the cost/service "
    "*distribution* with VaR/CVaR tail risk.\n"
    "8. **Bullwhip Preview** — compare how policies amplify or smooth demand variance "
    "(the bullwhip ratio) on a shared demand stream.\n"
    "9. **Sensitivity Analysis** — perturb each input ±% and rank which one moves the KPI "
    "most (a tornado chart).\n"
)

st.subheader("Reproducibility")
st.markdown(
    "Every run is identified by the SHA-256 **config hash** of its `RunConfig`. The same "
    "config reproduces byte-identical results — note the hash to recreate any experiment."
)

st.caption("Phase 1 · a presentation layer over the M1–M3 engine + analytics surface.")
