"""Integration: the time-scrubber component."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _host_multi() -> None:
    # from_function execs this body as a standalone script — imports live inside.
    import pandas as pd
    import streamlit as st

    from ui.components.time_scrubber import render_time_scrubber

    ledger = pd.DataFrame({"period": list(range(10)), "demand": [10.0] * 10})
    st.session_state["picked"] = render_time_scrubber(ledger)


def _host_single() -> None:
    import pandas as pd
    import streamlit as st

    from ui.components.time_scrubber import render_time_scrubber

    ledger = pd.DataFrame({"period": [0], "demand": [10.0]})
    st.session_state["picked"] = render_time_scrubber(ledger)


def _host_detail() -> None:
    import pandas as pd

    from ui.components.time_scrubber import render_period_detail

    # demand window for t=6 is periods 2..6 = [8, 10, 12, 9, 11] -> mean 10.0.
    # periods 0,1 (demand 5) sit OUTSIDE the trailing 5-window and must be excluded.
    ledger = pd.DataFrame(
        {
            "period": list(range(7)),
            "on_hand": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0],
            "on_order": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0],
            "inventory_position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 42.0],
            "demand": [5.0, 5.0, 8.0, 10.0, 12.0, 9.0, 11.0],
            "order_placed": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "order_received": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0],
        }
    )
    render_period_detail(ledger, 6)


def test_slider_present_and_defaults_to_last_period() -> None:
    at = AppTest.from_function(_host_multi, default_timeout=30).run()
    assert not at.exception
    assert any(s.key == "sr_scrub_t" for s in at.slider)
    assert at.session_state["picked"] == 9  # last period


def test_slider_returns_selected_period() -> None:
    at = AppTest.from_function(_host_multi, default_timeout=30).run()
    next(s for s in at.slider if s.key == "sr_scrub_t").set_value(4).run()
    assert at.session_state["picked"] == 4


def test_single_period_skips_slider() -> None:
    at = AppTest.from_function(_host_single, default_timeout=30).run()
    assert not at.exception
    assert len(at.slider) == 0  # no range to slide over
    assert at.session_state["picked"] == 0
    assert len(at.caption) >= 1  # the single-period note


def test_detail_renders_seven_tiles() -> None:
    at = AppTest.from_function(_host_detail, default_timeout=30).run()
    assert not at.exception
    assert len(at.metric) == 7


def test_detail_recent_demand_is_trailing_five_mean() -> None:
    at = AppTest.from_function(_host_detail, default_timeout=30).run()
    recent = next(m for m in at.metric if m.label == "Recent demand (avg 5)")
    assert recent.value == "10.0"  # mean(periods 2..6) = mean(8,10,12,9,11)
    # The glossary tooltip is a {n} template — assert it rendered the window, not a stray brace.
    assert "5" in recent.help
    assert "{" not in recent.help


def test_detail_state_tiles_read_the_frozen_row() -> None:
    at = AppTest.from_function(_host_detail, default_timeout=30).run()
    pos = next(m for m in at.metric if m.label == "Inventory position")
    assert pos.value == "42"
