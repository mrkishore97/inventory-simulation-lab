"""Integration: the Stress Test Composer page assembly — Page 4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

_PAGE = str(Path(__file__).resolve().parents[2] / "ui" / "pages" / "4_Stress_Test.py")


def _widget(at: AppTest, widget_type: str, key: str) -> Any:
    for w in getattr(at, widget_type):
        if w.key == key:
            return w
    raise KeyError(f"no {widget_type} widget with key {key!r}")


def _has(elements: Any, text: str) -> bool:
    return any(text in getattr(e, "value", "") for e in elements)


def _run(at: AppTest) -> None:
    _widget(at, "button", "st_run").click().run()


def test_gate_before_run() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    assert not at.exception
    assert _has(at.info, "Run stress test")
    assert len(at.dataframe) == 0


def test_run_renders_comparison() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _run(at)
    assert not at.exception
    assert at.dataframe[0].value.shape == (12, 2)  # 12 KPIs x {Baseline, Disrupted} (transposed)
    assert _has(at.caption, "vs disrupted")
    assert len(at.metric) == 3 and all(m.help for m in at.metric)  # tooltips on each tile
    assert _widget(at, "selectbox", "st_metric") is not None
    # export — the composed (disrupted) scenario saves/shares as YAML. AppTest can't query
    # download_button, so assert the save-path widgets (mirrors the Page 5 export test).
    assert _widget(at, "text_input", "st_save_name") is not None
    assert _widget(at, "button", "st_save") is not None
    assert len(at.warning) == 0  # no drift immediately after running


def test_disruption_raises_cost() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _run(at)  # default window: [30, 60) x2.0
    table = at.dataframe[0].value
    assert table.loc["total_cost", "Disrupted"] > table.loc["total_cost", "Baseline"]


def test_noop_multiplier_matches_baseline() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _widget(at, "number_input", "st_w0_mult").set_value(1.0)  # x1.0 round-trips to L=3 = baseline
    _run(at)
    table = at.dataframe[0].value
    assert table.loc["total_cost", "Disrupted"] == table.loc["total_cost", "Baseline"]


def test_second_window_runs() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _widget(at, "number_input", "st_n_windows").set_value(2)
    _run(at)  # the pending window-count applies on this run
    assert not at.exception
    assert at.dataframe[0].value.shape == (12, 2)


def test_sidebar_change_is_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=30).run()
    _run(at)
    _widget(at, "number_input", "cfg_horizon").set_value(120)  # change the baseline scenario
    at.run()
    assert not at.exception
    assert len(at.warning) >= 1  # drift warning
    assert at.dataframe[0].value.shape == (12, 2)  # results persist, pinned to the last Run
