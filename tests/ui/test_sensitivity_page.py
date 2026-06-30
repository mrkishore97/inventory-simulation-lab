"""Integration: the Sensitivity Analysis page assembly — Page 9.

AppTest smoke over the page shell: the run gate, a real (tiny) OAT render, the drift model, the
empty-selection guard, and the run-count guard. The catalogue/bounds logic is unit-tested apart in
test_sensitivity_spec.py; the tornado geometry in test_sensitivity_viz.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

_PAGE = str(Path(__file__).resolve().parents[2] / "ui" / "pages" / "9_Sensitivity.py")


def _widget(at: AppTest, widget_type: str, key: str) -> Any:
    for w in getattr(at, widget_type):
        if w.key == key:
            return w
    raise KeyError(f"no {widget_type} widget with key {key!r}")


def _has(elements: Any, text: str) -> bool:
    return any(text in getattr(e, "value", "") for e in elements)


def _run_small(at: AppTest) -> None:
    """Pin a tiny run (the default ~4 params x 2 reps = a handful of MC sims) and render it."""
    _widget(at, "number_input", "se_reps").set_value(2)
    _widget(at, "button", "se_run").click().run()


def test_gate_before_run() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    assert not at.exception
    assert _has(at.info, "Select parameters")  # the select-then-run gate
    assert "se_spec_json" not in at.session_state


def test_run_renders_tornado() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run_small(at)
    assert not at.exception
    assert len(at.dataframe) == 1  # the ranking table
    table = at.dataframe[0].value
    assert table.shape[1] == 9 and table.shape[0] >= 2  # sensitivity_oat columns x >=2 params
    assert _has(at.caption, "Ranked by")  # the results caption
    assert _widget(at, "selectbox", "se_kpi") is not None
    assert len(at.warning) == 0  # no drift immediately after running


def test_sidebar_change_is_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run_small(at)
    _widget(at, "number_input", "cfg_horizon").set_value(120)  # change the shared world
    at.run()
    assert not at.exception
    assert len(at.warning) >= 1  # world changed -> drift warning
    assert len(at.dataframe) == 1  # results persist, pinned to the last Run


def test_perturbation_change_is_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run_small(at)
    _widget(at, "number_input", "se_range").set_value(50)  # change the +/- % -> new specs
    at.run()
    assert not at.exception
    assert len(at.warning) >= 1  # the spec changed -> drift


def test_empty_selection_is_guarded() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _widget(at, "multiselect", "se_params").set_value([])
    _widget(at, "button", "se_run").click().run()
    assert not at.exception
    assert len(at.error) >= 1  # "select at least one parameter"
    assert "se_spec_json" not in at.session_state  # nothing pinned
    assert _has(at.info, "Select parameters")  # the gate remains


def test_too_many_runs_is_guarded() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    ms = _widget(at, "multiselect", "se_params")
    ms.set_value(list(ms.options))  # all ~9 params...
    _widget(at, "number_input", "se_reps").set_value(500)  # x500 -> (1 + 2*9)*500 = 9500 > 6000
    _widget(at, "button", "se_run").click().run()
    assert not at.exception
    assert len(at.error) >= 1  # the run-count guard fires
    assert "se_spec_json" not in at.session_state  # nothing pinned
