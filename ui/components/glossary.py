"""Central glossary of formula/explanation tooltips.

every KPI tile and key chart in the app carries a
hover ``help=`` tooltip with the formula and a one-line explanation, so the twin reads as a
*teaching tool*, not just a calculator. This module is the single source of truth for those
strings — :func:`tip` is called from ``kpi_card``, ``time_scrubber``, and Pages 1–8 so the same
KPI never gets two slightly different explanations.

Streamlit-free (pure data + one accessor), so it is unit-testable in isolation even though it
lives under coverage-omitted ``ui/`` — same precedent as :mod:`ui.components.demand_input`. The
prose reference table is complementary: that is documentation, this is the
runtime tooltip source. Every key here has a UI call site (no dead entries).

One value is a ``str.format`` template (``recent_demand`` carries ``{n}``); all others are
ready-to-render markdown.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

TOOLTIPS: Final[Mapping[str, str]] = MappingProxyType(
    {
        # — service trinity / costs / efficiency (kpi_card) —
        "cycle_service": (
            "**Cycle service (Type 1)** — fraction of replenishment cycles that end with no "
            "stockout."
        ),
        "fill_rate": (
            "**Fill rate (Type 2)** — sales ÷ demand: fraction of demand met immediately from "
            "stock."
        ),
        "ready_rate": "**Ready rate (Type 3)** — fraction of periods ending with on-hand > 0.",
        "total_cost": "**Total cost** = holding + ordering + purchase + stockout.",
        "holding_cost": "**Holding** = Σ h · (on-hand) over the run.",
        "ordering_cost": "**Ordering** = Σ K per ordering event (K × orders).",
        "purchase_cost": "**Purchase** = Σ c · (units bought).",
        "stockout_cost": "**Stockout** = Σ penalty on unmet demand (backorder or lost-sale).",
        "inventory_turns": (
            "**Inventory turns** = total demand ÷ average on-hand (over the horizon, not "
            "annualized)."
        ),
        "days_of_supply": (
            "**Days of supply** = average on-hand ÷ mean demand per period (expressed in periods)."
        ),
        "avg_on_hand": "**Avg on-hand** — mean end-of-period physical stock.",
        "order_count": "**Orders** — number of ordering events (order_placed > 0).",
        # — time-scrubber state / flows —
        "inventory_position": (
            "**Inventory position** = on-hand + on-order − backorders (what the policy reacts to)."
        ),
        "on_hand": "**On hand** — physical stock at end of period.",
        "on_order": "**On order** — units ordered but not yet received (the pipeline).",
        "demand": "**Demand** — units demanded in this period.",
        "recent_demand": (
            "**Recent demand** — mean demand over the trailing {n} periods (inclusive)."
        ),
        "order_placed": "**Order placed** — units the policy ordered this period.",
        "order_received": "**Order received** — units that arrived this period (pipeline outflow).",
        # — Monte Carlo / risk (Page 7) —
        "mc_mean": "**Mean** — average of the KPI across all Monte Carlo replications.",
        "var": (
            "**Value-at-Risk (VaR)** — the KPI level the chosen tail breaches with probability "
            "1 − confidence."
        ),
        "cvar": (
            "**Conditional VaR (CVaR)** — mean of the outcomes beyond VaR (expected shortfall)."
        ),
        "tail_probability": "**Tail probability** — fraction of replications in the worse tail.",
        # — Syntetos-Boylan classification (Page 6) —
        "sb_class": (
            "**Syntetos-Boylan class** — demand quadrant from ADI and CV²: smooth, erratic, "
            "intermittent, or lumpy."
        ),
        "adi": (
            "**ADI** (Average Demand Interval) = periods ÷ non-zero periods; > 1.32 ⇒ intermittent."
        ),
        "cv2": (
            "**CV²** — squared coefficient of variation of non-zero demand; > 0.49 ⇒ erratic/lumpy."
        ),
        "policy": (
            "**Recommended policy** and its starting parameters, derived from the classical "
            "inventory laws."
        ),
        # — bullwhip (Page 8) —
        "bullwhip_ratio": (
            "**Bullwhip ratio** = Var(orders) ÷ Var(demand); > 1 means the policy amplifies "
            "demand variability."
        ),
        # — charts (hung on the section subheader's help=) —
        "chart_inventory": (
            "On-hand, inventory position, and orders over time — stockouts appear where on-hand "
            "hits 0."
        ),
        "chart_waterfall": (
            "Cost build-up: holding + ordering + (purchase) + stockout = total cost."
        ),
        "chart_race": (
            "Each policy simulated on the *same* demand stream; compare the selected metric "
            "period by period."
        ),
        "chart_pareto": (
            "Service vs. cost for every swept policy; the frontier is the set you can't beat on "
            "both at once."
        ),
        "chart_distribution": (
            "Histogram of the KPI across replications; markers show VaR and the CVaR tail."
        ),
        "chart_bullwhip": (
            "Var(orders) ÷ Var(demand) per policy; the dashed line marks pass-through (ratio = 1)."
        ),
        "chart_tornado": (
            "Ranks parameters by the swing they cause in the KPI when moved ±% around baseline; "
            "the widest bar is the most influential input."
        ),
        "kpi_table": "Full KPI bundle for each policy, side by side.",
    }
)


def tip(key: str) -> str:
    """Return the tooltip markdown for ``key``; raise :class:`KeyError` (loud) on an unknown key.

    Failing loud on a typo means a missing tooltip surfaces in a test, never as a silently blank
    ``help=`` icon in the app. ``recent_demand`` is a ``str.format`` template — call sites pass
    ``.format(n=...)``.
    """
    return TOOLTIPS[key]
