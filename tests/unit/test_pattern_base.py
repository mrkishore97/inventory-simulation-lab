"""Tests for ``demand.patterns.base.Pattern`` (ABC contract)."""

from __future__ import annotations

import pytest

from demand.patterns.base import Pattern


class TestPatternABCContract:
    """The Pattern ABC enforces a single abstract method via Python's ABC machinery.

    Mirrors ``tests/unit/test_demand_base.TestDemandABCContract`` (if it
    existed — the demand ABC is exercised indirectly through subclass tests;
    the pattern ABC gets explicit coverage here because it is the
    first time it's introduced).
    """

    def test_pattern_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            Pattern()  # type: ignore[abstract]

    def test_pattern_subclass_without_apply_raises(self) -> None:
        class IncompletePattern(Pattern):
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompletePattern()  # type: ignore[abstract]

    def test_pattern_subclass_with_apply_instantiates(self) -> None:
        class MinimalPattern(Pattern):
            def apply(self, base: float, t: int) -> float:
                return base * 2

        p = MinimalPattern()
        assert p.apply(3.0, 0) == 6.0
        assert p.apply(0.0, 100) == 0.0
