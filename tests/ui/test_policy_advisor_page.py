"""AppTest coverage for the Policy Advisor page.

The page is fully AppTest-drivable (no plotly click or file upload): the demand source is a
selectbox / text-input path, so every arm — archetype, custom path, error, and the policy
handoff — is exercised here. The handoff is asserted by the *stash* in ``session_state`` (the
``st.switch_page`` navigation is not asserted — it is brittle under AppTest, per the review).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from ui.components.config_builder import POLICY_HANDOFF_KEY

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PAGE = _REPO_ROOT / "ui" / "pages" / "6_Policy_Advisor.py"


def _run() -> AppTest:
    at = AppTest.from_file(str(_PAGE), default_timeout=30).run()
    assert not at.exception, at.exception
    return at


def _metric_values(at: AppTest) -> list[str]:
    return [str(m.value) for m in at.metric]


def test_default_recommends_sQ_for_smooth() -> None:
    # the first archetype ("smooth") classifies smooth -> (s, Q)
    values = _metric_values(_run())
    assert "smooth" in values  # SB-class tile
    assert any(v.startswith("sQ(") for v in values)  # recommended-policy tile


def test_promotional_recommends_base_stock() -> None:
    at = _run()
    at.selectbox(key="adv_arch").select("promotional").run()
    assert not at.exception
    # the promotional SKU is SB-lumpy -> base-stock
    assert any(v.startswith("BaseStock(") for v in _metric_values(at))


def test_service_level_change_is_live() -> None:
    # no Run button: changing the service level re-derives the recommendation in place
    at = _run()
    at.slider(key="adv_service_level").set_value(0.80).run()
    assert not at.exception
    assert any(v.startswith("sQ(") for v in _metric_values(at))


def test_custom_path_renders(tmp_path: Path) -> None:
    p = tmp_path / "smooth.parquet"
    pd.DataFrame({"demand": [10.0, 12.0, 8.0, 11.0, 9.0, 10.0]}).to_parquet(p)
    at = _run()
    at.radio(key="adv_source").set_value("Custom path").run()
    at.text_input(key="adv_path").set_value(str(p)).run()
    assert not at.exception
    assert any(v.startswith("sQ(") for v in _metric_values(at))


def test_bad_custom_path_shows_error(tmp_path: Path) -> None:
    p = tmp_path / "bad.parquet"
    pd.DataFrame({"sales": [1.0, 2.0]}).to_parquet(p)  # no 'demand' column
    at = _run()
    at.radio(key="adv_source").set_value("Custom path").run()
    at.text_input(key="adv_path").set_value(str(p)).run()
    assert at.error  # st.error rendered, then st.stop()


def test_classification_and_policy_tiles_carry_tooltips() -> None:
    #: every KPI tile carries a formula tooltip (sourced from ui.components.glossary).
    by_label = {m.label: m for m in _run().metric}
    for label in ("SB class", "ADI", "CV²", "Policy"):
        assert by_label[label].help


def test_load_into_single_run_stashes_policy() -> None:
    at = _run()
    at.button(key="adv_load").click().run()
    # the stash happens before st.switch_page, so it is set regardless of navigation
    assert POLICY_HANDOFF_KEY in at.session_state
    assert at.session_state[POLICY_HANDOFF_KEY]["kind"] == "sQ"
