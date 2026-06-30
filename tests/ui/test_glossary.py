"""Unit tests for the tooltip glossary.

Pure data + accessor — no Streamlit, so this runs as a plain unit test even though ``glossary``
lives under coverage-omitted ``ui/``. The contract: every key a page asks for exists (a missing
key would be a runtime ``KeyError`` in the app), every value is non-empty, and the loud accessor
raises on a typo. Extra keys are allowed (the glossary may grow); the subset check is the gate.
"""

from __future__ import annotations

import pytest

from ui.components.glossary import TOOLTIPS, tip

# Every key referenced by kpi_card / time_scrubber / Pages 1–8. The runtime contract: these MUST
# resolve. (Subset, not equality — extra glossary entries are fine.)
EXPECTED_KEYS = frozenset(
    {
        # kpi_card
        "cycle_service",
        "fill_rate",
        "ready_rate",
        "total_cost",
        "holding_cost",
        "ordering_cost",
        "purchase_cost",
        "stockout_cost",
        "inventory_turns",
        "days_of_supply",
        "avg_on_hand",
        "order_count",
        # time_scrubber
        "inventory_position",
        "on_hand",
        "on_order",
        "demand",
        "recent_demand",
        "order_placed",
        "order_received",
        # Monte Carlo / risk (Page 7)
        "mc_mean",
        "var",
        "cvar",
        "tail_probability",
        # classification (Page 6)
        "sb_class",
        "adi",
        "cv2",
        "policy",
        # bullwhip (Page 8)
        "bullwhip_ratio",
        # charts
        "chart_inventory",
        "chart_waterfall",
        "chart_race",
        "chart_pareto",
        "chart_distribution",
        "chart_bullwhip",
        "chart_tornado",
        "kpi_table",
    }
)


def test_every_required_key_present() -> None:
    """Each key a page needs resolves — a subset check, so extra entries are allowed to grow."""
    assert set(TOOLTIPS) >= EXPECTED_KEYS


def test_all_values_are_nonempty_strings() -> None:
    for key, value in TOOLTIPS.items():
        assert isinstance(value, str), key
        assert value.strip(), key


@pytest.mark.parametrize(
    ("key", "needle"),
    [
        ("bullwhip_ratio", "Var(orders)"),
        ("fill_rate", "÷"),
        ("var", "Value-at-Risk"),
        ("adi", "1.32"),  # the Syntetos-Boylan cutoff — must match recommender/classify.py
        ("cv2", "0.49"),
        ("total_cost", "holding"),
    ],
)
def test_formula_spot_checks(key: str, needle: str) -> None:
    """Spot-check that the tooltips actually carry the formula, not just a label."""
    assert needle in tip(key)


def test_recent_demand_is_a_format_template() -> None:
    """The one ``{n}`` template renders cleanly and leaves no stray brace."""
    rendered = tip("recent_demand").format(n=5)
    assert "5" in rendered
    assert "{" not in rendered


def test_tip_raises_loudly_on_unknown_key() -> None:
    with pytest.raises(KeyError):
        tip("does_not_exist")


def test_tooltips_mapping_is_read_only() -> None:
    """``MappingProxyType`` — a stray write fails instead of silently mutating the source."""
    with pytest.raises(TypeError):
        TOOLTIPS["fill_rate"] = "tampered"  # type: ignore[index]
