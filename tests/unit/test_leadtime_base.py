"""Tests for the ``LeadTime`` ABC."""

from __future__ import annotations

import pytest

from leadtime.base import LeadTime, round_to_periods


class TestLeadTimeABC:
    def test_lead_time_is_abstract(self) -> None:
        # Cannot instantiate the ABC directly; mirrors the Demand ABC contract.
        with pytest.raises(TypeError):
            LeadTime()  # type: ignore[abstract]

    def test_subclass_without_sample_is_still_abstract(self) -> None:
        class PartialLeadTime(LeadTime):
            pass

        with pytest.raises(TypeError):
            PartialLeadTime()  # type: ignore[abstract]

    def test_subclass_with_sample_is_concrete(self) -> None:
        class ConstantLeadTime(LeadTime):
            def sample(self) -> int:
                return 5

        instance = ConstantLeadTime()
        assert isinstance(instance, LeadTime)
        assert instance.sample() == 5


class TestRoundToPeriods:
    """The shared round-then-clip helper used by every stochastic arm."""

    def test_rounds_to_nearest(self) -> None:
        assert round_to_periods(2.4) == 2
        assert round_to_periods(2.6) == 3

    def test_bankers_rounding_at_half(self) -> None:
        # Python's round() is round-half-to-even (banker's rounding). Documented
        # so a future reader isn't surprised; lead times don't care about the
        # tie-break direction, but the behavior is pinned.
        assert round_to_periods(2.5) == 2
        assert round_to_periods(3.5) == 4

    def test_clips_to_one(self) -> None:
        assert round_to_periods(0.3) == 1  # rounds to 0, clipped to 1
        assert round_to_periods(0.0) == 1
        assert round_to_periods(-5.0) == 1  # absorbs a Normal's negative tail

    def test_exact_integer_passes_through(self) -> None:
        assert round_to_periods(3.0) == 3

    def test_returns_int_type(self) -> None:
        assert isinstance(round_to_periods(4.2), int)
