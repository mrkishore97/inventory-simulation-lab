"""Tests for ``leadtime.disruptions.scheduled.ScheduledDisruption``."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from core.config import DisruptionWindow, ScheduledDisruptionConfig
from leadtime.disruptions.scheduled import ScheduledDisruption


def _make(windows: list[tuple[int, int, float]], seed: int = 0) -> ScheduledDisruption:
    cfg = ScheduledDisruptionConfig(
        windows=[DisruptionWindow(start=s, duration=d, multiplier=m) for (s, d, m) in windows]
    )
    return ScheduledDisruption(cfg, np.random.default_rng(seed))


class TestScheduledDisruptionWindow:
    def test_multiplies_inside_window(self) -> None:
        # Window [10, 20): base 3 x 2.0 -> 6 inside.
        d = _make([(10, 10, 2.0)])
        assert d.apply(3, 15) == 6

    def test_identity_outside_window(self) -> None:
        d = _make([(10, 10, 2.0)])
        assert d.apply(3, 5) == 3  # before
        assert d.apply(3, 25) == 3  # after

    def test_half_open_interval_boundaries(self) -> None:
        # Active for start <= t < start + duration: active at start and
        # start+duration-1; inactive at start-1 and start+duration.
        d = _make([(10, 5, 2.0)])  # active for t in [10, 15)
        assert d.apply(3, 9) == 3  # just before -> identity
        assert d.apply(3, 10) == 6  # at start -> active
        assert d.apply(3, 14) == 6  # last active period
        assert d.apply(3, 15) == 3  # at start+duration -> inactive

    def test_multiplier_one_is_noop(self) -> None:
        d = _make([(0, 100, 1.0)])
        for t in (0, 50, 99):
            assert d.apply(3, t) == 3

    def test_overlapping_windows_compose_multiplicatively(self) -> None:
        # Two windows both active at t=15: 2.0 * 1.5 = 3.0 -> base 2 x 3 = 6.
        d = _make([(10, 10, 2.0), (12, 6, 1.5)])  # overlap on [12, 18)
        assert d.apply(2, 15) == 6
        # Only the first active at t=11 (second starts at 12): base 2 x 2 = 4.
        assert d.apply(2, 11) == 4

    def test_window_order_independence(self) -> None:
        a = _make([(10, 10, 2.0), (12, 6, 1.5)])
        b = _make([(12, 6, 1.5), (10, 10, 2.0)])
        for t in range(25):
            assert a.apply(4, t) == b.apply(4, t)

    def test_round_then_clip_applied(self) -> None:
        # base 3 x 1.5 = 4.5 -> banker's round (4 is even) -> 4.
        assert _make([(0, 10, 1.5)]).apply(3, 5) == 4
        # base 5 x 1.5 = 7.5 -> banker's round (8 is even) -> 8.
        assert _make([(0, 10, 1.5)]).apply(5, 5) == 8
        # Expedite below 1: base 1 x 0.1 = 0.1 -> rounds to 0 -> clipped to 1.
        assert _make([(0, 10, 0.1)]).apply(1, 5) == 1

    def test_apply_returns_int_type(self) -> None:
        d = _make([(0, 10, 2.0)])
        assert isinstance(d.apply(3, 5), int)  # active branch
        assert isinstance(d.apply(3, 50), int)  # identity branch


class TestScheduledDisruptionRNGAndDeterminism:
    def test_apply_does_not_consume_rng(self) -> None:
        # The schedule is deterministic — ScheduledDisruption accepts but never
        # draws from the "disruptions" stream. Pinned via bit_generator.state.
        rng = np.random.default_rng(42)
        state_before = rng.bit_generator.state
        cfg = ScheduledDisruptionConfig(
            windows=[DisruptionWindow(start=10, duration=10, multiplier=2.0)]
        )
        disruption = ScheduledDisruption(cfg, rng)
        state_after_construct = rng.bit_generator.state
        for t in range(1000):
            disruption.apply(3, t)
        state_after_apply = rng.bit_generator.state
        assert state_before == state_after_construct == state_after_apply

    def test_determinism_under_same_seed(self) -> None:
        windows = [(10, 10, 2.0), (12, 6, 1.5)]
        a = _make(windows, seed=7)
        b = _make(windows, seed=7)
        for base in (1, 3, 5):
            for t in range(25):
                assert a.apply(base, t) == b.apply(base, t)


class TestScheduledDisruptionProperty:
    @settings(max_examples=50)
    @given(
        base=st.integers(min_value=1, max_value=1000),
        t=st.integers(min_value=0, max_value=10_000),
        start=st.integers(min_value=0, max_value=5000),
        duration=st.integers(min_value=1, max_value=5000),
        multiplier=st.floats(
            min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_apply_always_returns_positive_int(
        self, base: int, t: int, start: int, duration: int, multiplier: float
    ) -> None:
        d = _make([(start, duration, multiplier)])
        result = d.apply(base, t)
        assert isinstance(result, int)
        assert result >= 1
