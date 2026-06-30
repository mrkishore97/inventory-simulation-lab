"""Integration: the Policy Comparison page assembly — Page 2 V2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

_PAGE = str(Path(__file__).resolve().parents[2] / "ui" / "pages" / "2_Policy_Comparison.py")


def _widget(at: AppTest, widget_type: str, key: str) -> Any:
    for w in getattr(at, widget_type):
        if w.key == key:
            return w
    raise KeyError(f"no {widget_type} widget with key {key!r}")


def _run(at: AppTest) -> None:
    """Click the Run race button and re-run (applying any pending widget edits too)."""
    _widget(at, "button", "pc_run").click().run()


def test_gate_before_run() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    assert not at.exception
    assert len(at.info) >= 1  # the "configure then Run" gate
    assert len(at.dataframe) == 0  # no results until Run is clicked


def test_run_renders_race() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _run(at)
    assert not at.exception
    assert at.dataframe[0].value.shape == (12, 4)  # 12 KPIs x 4 policies (transposed table)
    assert len(at.warning) == 0  # no drift immediately after running
    assert any("Race on scenario" in getattr(c, "value", "") for c in at.caption)
    assert _widget(at, "selectbox", "pc_metric") is not None


def test_parameterized_labels_in_table() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _run(at)
    cols = list(at.dataframe[0].value.columns)
    assert any(c.startswith("sQ(s=20") for c in cols)  # self-documenting param label


def test_multiselect_is_live_not_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _run(at)
    _widget(at, "multiselect", "pc_policies").unselect("sS")
    at.run()
    assert not at.exception
    assert len(at.warning) == 0  # changing the contestant set is live, not config drift
    assert at.dataframe[0].value.shape == (12, 3)


def test_metric_is_live_not_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _run(at)
    _widget(at, "selectbox", "pc_metric").set_value("inventory_position")
    at.run()
    assert not at.exception
    assert len(at.warning) == 0  # metric is presentation-only
    assert at.dataframe[0].value.shape == (12, 4)


def test_policy_param_edit_is_live_not_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _run(at)
    _widget(at, "number_input", "pc_pol_sq_q").set_value(45)
    at.run()
    assert not at.exception
    assert len(at.warning) == 0  # tuning a policy races live against the pinned world
    assert at.dataframe[0].value.shape == (12, 4)


def test_sidebar_change_is_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _run(at)
    _widget(at, "number_input", "cfg_horizon").set_value(120)  # change the shared world
    at.run()
    assert not at.exception
    assert len(at.warning) >= 1  # the world changed -> drift warning
    assert at.dataframe[0].value.shape == (12, 4)  # results stay, pinned to the last Run


def test_reduced_selection_before_run() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _widget(at, "multiselect", "pc_policies").set_value(["sQ", "base_stock"])
    _run(at)  # the pending selection applies on this run alongside the click
    assert not at.exception
    assert at.dataframe[0].value.shape == (12, 2)


def test_empty_selection_guarded() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _run(at)
    _widget(at, "multiselect", "pc_policies").set_value([])
    at.run()
    assert not at.exception
    assert len(at.warning) >= 1  # "select at least one policy"
    assert len(at.dataframe) == 0


def test_invalid_policy_errors_but_others_race() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _run(at)
    _widget(at, "number_input", "pc_pol_ss_s").set_value(70.0)  # s=70 >= S=60 -> invalid (s,S)
    at.run()
    assert not at.exception
    assert len(at.error) >= 1  # the invalid policy is reported
    assert at.dataframe[0].value.shape == (12, 3)  # sS dropped, the other 3 still race
