"""Headless smoke tests for the Streamlit shell.

Streamlit's own ``AppTest`` runtime executes each script without a browser and records
any exception it raises. These tests are the automated gate for the UI layer (which is
coverage-omitted — there is no business logic to measure here, only that every page
renders). Later page bullets extend this file as they add real widgets.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# Resolve paths from this file (not the pytest CWD) so the tests are location-robust.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_APP = _REPO_ROOT / "ui" / "app.py"
_PAGES = sorted((_REPO_ROOT / "ui" / "pages").glob("[0-9]*_*.py"))

# The page names the Home page advertises (the set + the Monte Carlo & Risk page that
# fills the distributional-analytics gap + the Bullwhip Preview closing + the
# Sensitivity Analysis page completing the analytics catalogue).
_PAGE_NAMES = (
    "Single Run",
    "Policy Comparison",
    "Pareto Explorer",
    "Stress Test",
    "Scenario Library",
    "Policy Advisor",
    "Monte Carlo & Risk",
    "Bullwhip Preview",
    "Sensitivity Analysis",
)


def test_home_app_runs() -> None:
    at = AppTest.from_file(str(_APP), default_timeout=30).run()
    assert not at.exception
    assert "Inventory Twin" in str(at.title[0].value)


def test_home_lists_all_pages() -> None:
    at = AppTest.from_file(str(_APP), default_timeout=30).run()
    blob = " ".join(str(md.value) for md in at.markdown)
    for name in _PAGE_NAMES:
        assert name in blob, f"Home page does not mention {name!r}"


def test_all_pages_discovered() -> None:
    # The full set of nine pages; guard drift.
    assert len(_PAGES) == 9, [p.name for p in _PAGES]


@pytest.mark.parametrize("page", _PAGES, ids=lambda p: p.name)
def test_each_page_runs(page: Path) -> None:
    at = AppTest.from_file(str(page), default_timeout=30).run()
    assert not at.exception
