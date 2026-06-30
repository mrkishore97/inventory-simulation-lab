"""Integration: the Pareto Explorer page assembly — Page 3.

AppTest cannot synthesize a Plotly point-selection, so the click→load path is covered by the
pure ``ui.components.pareto_select`` unit tests (and the handoff landing by the config-builder
tests); these smoke tests exercise the page shell: the run gate, a real (small) sweep render,
the drift model, and the cardinality guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

_PAGE = str(Path(__file__).resolve().parents[2] / "ui" / "pages" / "3_Pareto_Explorer.py")


def _widget(at: AppTest, widget_type: str, key: str) -> Any:
    for w in getattr(at, widget_type):
        if w.key == key:
            return w
    raise KeyError(f"no {widget_type} widget with key {key!r}")


def _run_small(at: AppTest) -> None:
    """Pin a deliberately tiny sweep (2x2 grid x 2 reps = a handful of runs) and render it."""
    _widget(at, "number_input", "pe_steps").set_value(2)
    _widget(at, "number_input", "pe_reps").set_value(2)
    _widget(at, "button", "pe_run").click().run()


def _has(elements: Any, text: str) -> bool:
    return any(text in getattr(e, "value", "") for e in elements)


def test_gate_before_run() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    assert not at.exception
    assert _has(at.info, "Configure the sweep")  # the configure-then-run gate
    assert "pe_spec_json" not in at.session_state


def test_run_renders_frontier() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run_small(at)
    assert not at.exception
    assert _has(at.caption, "Swept sQ")  # the results caption (default kind)
    assert _has(at.info, "Click a point")  # no selection under AppTest -> the prompt shows
    assert len(at.warning) == 0  # no drift immediately after running


def test_sidebar_change_is_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run_small(at)
    _widget(at, "number_input", "cfg_horizon").set_value(120)  # change the shared world
    at.run()
    assert not at.exception
    assert len(at.warning) >= 1  # world changed -> drift warning
    assert _has(at.caption, "Swept sQ")  # results persist, pinned to the last Run


def test_sweep_control_change_is_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run_small(at)
    _widget(at, "number_input", "pe_steps").set_value(3)  # change the grid granularity
    at.run()
    assert not at.exception
    assert len(at.warning) >= 1  # the sweep spec changed -> drift


def test_grid_too_large_is_guarded() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _widget(at, "number_input", "pe_reps").set_value(400)  # 25 pts x 400 reps = 10000 > 6000
    _widget(at, "button", "pe_run").click().run()
    assert not at.exception
    assert len(at.error) >= 1  # the cardinality guard fires
    assert "pe_spec_json" not in at.session_state  # nothing pinned
    assert _has(at.info, "Configure the sweep")  # the gate remains


def test_change_kind_resweeps() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _widget(at, "selectbox", "pe_kind").set_value("base_stock")
    at.run()  # the base-params editor now shows base_stock's single field
    _run_small(at)
    assert not at.exception
    assert _has(at.caption, "Swept base_stock")
