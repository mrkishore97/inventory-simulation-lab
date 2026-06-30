"""Integration: the Bullwhip Preview page assembly — Page 8."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

_PAGE = str(Path(__file__).resolve().parents[2] / "ui" / "pages" / "8_Bullwhip.py")


def _widget(at: AppTest, widget_type: str, key: str) -> Any:
    for w in getattr(at, widget_type):
        if w.key == key:
            return w
    raise KeyError(f"no {widget_type} widget with key {key!r}")


def _has(elements: Any, text: str) -> bool:
    return any(text in getattr(e, "value", "") for e in elements)


def _run(at: AppTest) -> None:
    """Click Run and re-run (applying any pending widget edits too)."""
    _widget(at, "button", "bw_run").click().run()


def test_gate_before_run() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    assert not at.exception
    assert _has(at.info, "Run")  # the "configure then Run" gate
    assert len(at.metric) == 0  # no ratio tiles until Run is clicked


def test_run_renders_chart_and_tiles() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run(at)
    assert not at.exception
    assert len(at.metric) == 4  # one ratio tile per raced policy
    assert _has(at.caption, "Bullwhip on scenario")
    assert len(at.warning) == 0  # no drift immediately after running


def test_parameterized_tile_labels() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run(at)
    assert any(m.label.startswith("sQ(s=20") for m in at.metric)  # shared self-documenting label


def test_log_scale_is_live_not_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run(at)
    _widget(at, "checkbox", "bw_log").set_value(True)
    at.run()
    assert not at.exception
    assert len(at.warning) == 0  # the log toggle is presentation-only, not a re-run
    assert len(at.metric) == 4


def test_multiselect_is_live_not_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run(at)
    _widget(at, "multiselect", "bw_policies").unselect("sS")
    at.run()
    assert not at.exception
    assert len(at.warning) == 0  # changing the contestant set is live, not config drift
    assert len(at.metric) == 3


def test_policy_edit_is_live_not_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run(at)
    _widget(at, "number_input", "bw_pol_sq_q").set_value(60)
    at.run()
    assert not at.exception
    assert len(at.warning) == 0  # tuning a policy re-derives live against the pinned world
    assert len(at.metric) == 4


def test_sidebar_change_is_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run(at)
    _widget(at, "number_input", "cfg_horizon").set_value(120)  # change the shared world
    at.run()
    assert not at.exception
    assert len(at.warning) >= 1  # the world changed -> drift warning
    assert len(at.metric) == 4  # results stay, pinned to the last Run


def test_empty_selection_guarded() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run(at)
    _widget(at, "multiselect", "bw_policies").set_value([])
    at.run()
    assert not at.exception
    assert len(at.warning) >= 1  # "select at least one policy"
    assert len(at.metric) == 0


def test_invalid_policy_skipped() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run(at)
    _widget(at, "number_input", "bw_pol_ss_s").set_value(70.0)  # s=70 >= S=60 -> invalid (s,S)
    at.run()
    assert not at.exception
    assert len(at.error) >= 1  # the invalid policy is reported
    assert len(at.metric) == 3  # sS dropped, the other 3 still plot


def test_constant_demand_nan_does_not_crash() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _widget(at, "number_input", "cfg_dem_n_std").set_value(0.0)  # constant demand -> Var=0 -> nan
    _run(at)
    assert not at.exception
    assert len(at.metric) == 4  # tiles still render
    assert any(m.value == "—" for m in at.metric)  # nan ratio shown honestly, not a fake number
