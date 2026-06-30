"""Integration: the Single Run page assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

_PAGE = str(Path(__file__).resolve().parents[2] / "ui" / "pages" / "1_Single_Run.py")


def _widget(at: AppTest, widget_type: str, key: str) -> Any:
    for w in getattr(at, widget_type):
        if w.key == key:
            return w
    raise KeyError(f"no {widget_type} widget with key {key!r}")


def test_gate_before_run() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    assert not at.exception
    assert len(at.info) >= 1  # the "configure then Run" gate
    assert len(at.metric) == 0  # no results until Run is clicked


def test_run_renders_kpis_without_warning() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.metric) >= 12
    assert len(at.warning) == 0  # no drift immediately after running
    assert any("Showing results for" in getattr(c, "value", "") for c in at.caption)


def test_sidebar_drift_warns_but_keeps_results() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    at.button[0].click().run()
    _widget(at, "number_input", "cfg_horizon").set_value(120)  # change config AFTER running
    at.run()
    assert not at.exception
    assert len(at.warning) >= 1  # "configuration changed" warning
    assert len(at.metric) >= 12  # results stay, pinned to the last-run config


def test_include_purchase_toggle_is_not_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    at.button[0].click().run()
    _widget(at, "checkbox", "sr_inc_purchase").set_value(True)
    at.run()
    assert not at.exception
    assert len(at.warning) == 0  # the checkbox is presentation-only, not config drift
    assert len(at.metric) >= 12


def test_scrubber_slider_present_after_run() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    at.button[0].click().run()
    assert not at.exception
    assert _widget(at, "slider", "sr_scrub_t") is not None  # the time-scrubber


def test_scrubber_is_not_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    at.button[0].click().run()
    _widget(at, "slider", "sr_scrub_t").set_value(5)  # scrub to an earlier period
    at.run()
    assert not at.exception
    assert len(at.warning) == 0  # the slider is presentation-only, not config drift
    assert len(at.metric) >= 12
