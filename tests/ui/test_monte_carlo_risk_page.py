"""Integration: the Monte Carlo & Risk page assembly — Page 7."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

_PAGE = str(Path(__file__).resolve().parents[2] / "ui" / "pages" / "7_Monte_Carlo_Risk.py")


def _widget(at: AppTest, widget_type: str, key: str) -> Any:
    for w in getattr(at, widget_type):
        if w.key == key:
            return w
    raise KeyError(f"no {widget_type} widget with key {key!r}")


def _has(elements: Any, text: str) -> bool:
    return any(text in getattr(e, "value", "") for e in elements)


def _run(at: AppTest) -> None:
    """Pin a small (fast) Monte Carlo run."""
    _widget(at, "number_input", "mc_reps").set_value(50)
    _widget(at, "button", "mc_run").click().run()


def test_gate_before_run() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    assert not at.exception
    assert _has(at.info, "Run Monte Carlo")
    assert len(at.metric) == 0


def test_run_renders_distribution_and_risk() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run(at)
    assert not at.exception
    assert len(at.metric) >= 4  # Mean, VaR, CVaR, Tail probability
    assert all(m.help for m in at.metric)  #: every risk tile carries a formula tooltip
    assert at.dataframe[0].value.shape == (12, 8)  # 12 KPIs x describe()'s 8 stats
    assert _has(at.caption, "replications of")
    assert _has(at.caption, "upper")  # default KPI total_cost -> upper-tail risk
    assert len(at.warning) == 0


def test_kpi_change_to_service_flips_tail_live() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run(at)
    _widget(at, "selectbox", "mc_kpi").set_value("fill_rate")
    at.run()
    assert not at.exception
    assert len(at.warning) == 0  # KPI is a live reduction, not drift
    assert _has(at.caption, "lower")  # fill_rate -> lower-tail risk


def test_confidence_change_is_live_not_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run(at)
    _widget(at, "slider", "mc_conf").set_value(0.90)
    at.run()
    assert not at.exception
    assert len(at.warning) == 0  # re-deriving risk at a new confidence is not a re-run


def test_sidebar_change_is_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run(at)
    _widget(at, "number_input", "cfg_horizon").set_value(120)
    at.run()
    assert not at.exception
    assert len(at.warning) >= 1  # the scenario changed -> drift warning
    assert at.dataframe[0].value.shape == (12, 8)  # results persist, pinned to the last Run


def test_replications_change_is_drift() -> None:
    at = AppTest.from_file(_PAGE, default_timeout=60).run()
    _run(at)
    _widget(at, "number_input", "mc_reps").set_value(100)
    at.run()
    assert not at.exception
    assert len(at.warning) >= 1  # replications are pinned, so a change is drift
