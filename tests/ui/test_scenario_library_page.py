"""Integration: the Scenario Library page assembly — Page 5.

Smoke tests for the page shell: browse/list, the editor + live validation, the load button,
and the inline run. Disk-write (Save) is covered by the ``scenario_io`` unit tests with
``tmp_path``; these tests never click Save, so the real ``data/scenarios/`` is untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

_PAGE = str(Path(__file__).resolve().parents[2] / "ui" / "pages" / "5_Scenario_Library.py")


def _widget(at: AppTest, widget_type: str, key: str) -> Any:
    for w in getattr(at, widget_type):
        if w.key == key:
            return w
    raise KeyError(f"no {widget_type} widget with key {key!r}")


def _has(elements: Any, text: str) -> bool:
    return any(text in getattr(e, "value", "") for e in elements)


def test_page_loads_lists_and_validates() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    assert not at.exception
    assert len(_widget(at, "selectbox", "sl_file").options) >= 1
    assert _widget(at, "text_area", "sl_editor").value  # editor pre-filled on first visit
    assert len(at.success) >= 1  # the default scenario validates
    assert _has(at.success, "hash")


def test_malformed_yaml_shows_error() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _widget(at, "text_area", "sl_editor").set_value("a: b: c")  # not valid YAML
    at.run()
    assert not at.exception
    assert len(at.error) >= 1
    assert len(at.success) == 0


def test_invalid_config_shows_error() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _widget(at, "text_area", "sl_editor").set_value("master_seed: 42\n")  # valid YAML, bad config
    at.run()
    assert not at.exception
    assert len(at.error) >= 1


def test_load_button_swaps_editor() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _widget(at, "selectbox", "sl_file").set_value("example_sS.yaml")
    _widget(at, "button", "sl_load").click().run()
    assert not at.exception
    expected = Path("data/scenarios/example_sS.yaml").read_text()
    assert _widget(at, "text_area", "sl_editor").value == expected


def test_run_renders_kpis() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _widget(at, "button", "sl_run").click().run()  # run the default (valid) scenario as-is
    assert not at.exception
    assert len(at.metric) >= 12  # the KPI card's 12 tiles
    assert _has(at.caption, "Results for")


def test_export_controls_present_when_valid() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    assert _widget(at, "button", "sl_save") is not None
    assert _widget(at, "text_input", "sl_save_name") is not None
