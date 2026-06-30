"""Tests for ``demand.patterns.lumpy.LumpyPattern``.

Sibling of the Intermittent arm: same Bernoulli mask, but the kept value is amplified by
``burst_multiplier`` (> 1). Second stochastic pattern arm — consumes the seeded ``"pattern"``
stream (one uniform per ``apply()``, drawn before the burst/zero branch).
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from core.config import LumpyPatternConfig
from demand.patterns.lumpy import LumpyPattern


class TestLumpyPatternSemantics:
    """The Bernoulli burst-mask: ``apply(base, t)`` is ``base * m`` w.p. ``p``, else ``0.0``."""

    def test_kept_value_is_base_times_multiplier(self) -> None:
        # p=1 → every period a burst → base * 5.0 exactly.
        cfg = LumpyPatternConfig(occurrence_probability=1.0, burst_multiplier=5.0)
        pattern = LumpyPattern(cfg, np.random.default_rng(0))
        for base in (1.0, 10.0, 13.7, 100.0):
            assert pattern.apply(base, 0) == base * 5.0

    def test_apply_decision_matches_a_twin_rng(self) -> None:
        # The burst/zero decision is exactly `draw < p` for the SAME uniform a twin rng produces,
        # and apply consumes EXACTLY ONE draw per call (the rngs stay in lockstep afterward).
        cfg = LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=5.0)
        rng_a = np.random.default_rng(123)
        rng_b = np.random.default_rng(123)  # twin: predicts each draw apply() will consume
        pattern = LumpyPattern(cfg, rng_a)
        for t in range(100):
            expected = 10.0 * 5.0 if rng_b.random() < 0.3 else 0.0
            assert pattern.apply(10.0, t) == expected
        assert rng_a.random() == rng_b.random()  # one draw/apply → still aligned

    def test_output_is_zero_or_base_times_multiplier(self) -> None:
        cfg = LumpyPatternConfig(occurrence_probability=0.5, burst_multiplier=3.0)
        pattern = LumpyPattern(cfg, np.random.default_rng(7))
        for t in range(500):
            result = pattern.apply(13.7, t)
            assert result == 0.0 or result == 13.7 * 3.0

    def test_zero_base_returns_zero(self) -> None:
        # base=0 → burst yields 0 * m = 0.0, mask yields 0.0; either way 0.0.
        cfg = LumpyPatternConfig(occurrence_probability=0.6, burst_multiplier=5.0)
        pattern = LumpyPattern(cfg, np.random.default_rng(0))
        for t in range(50):
            assert pattern.apply(0.0, t) == 0.0


class TestLumpyPatternInvariants:
    def test_apply_returns_float_type(self) -> None:
        cfg = LumpyPatternConfig(occurrence_probability=0.6, burst_multiplier=5.0)
        pattern = LumpyPattern(cfg, np.random.default_rng(0))
        assert isinstance(pattern.apply(10.0, 0), float)

    def test_apply_returns_non_negative(self) -> None:
        cfg = LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=5.0)
        pattern = LumpyPattern(cfg, np.random.default_rng(1))
        for t in range(1000):
            assert pattern.apply(10.0, t) >= 0.0

    def test_apply_consumes_one_draw_per_call(self) -> None:
        # State-advance via a twin: after N applies the source rng has advanced exactly N draws
        # (the twin drawn N times stays in lockstep). Robust — no bit_generator internals.
        cfg = LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=5.0)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        pattern = LumpyPattern(cfg, rng_a)
        for t in range(1000):
            pattern.apply(float(t), t)
            rng_b.random()
        assert rng_a.random() == rng_b.random()

    def test_determinism_under_same_seed(self) -> None:
        cfg = LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=5.0)
        a = LumpyPattern(cfg, np.random.default_rng(2024))
        b = LumpyPattern(cfg, np.random.default_rng(2024))
        for t in range(200):
            assert a.apply(10.0, t) == b.apply(10.0, t)

    def test_distinct_seeds_diverge(self) -> None:
        # Fixed seeds 1 and 2 → deterministic (not probabilistic); the burst/zero sequences
        # differ over 200 periods. Mirrors the fixed-seed divergence pin.
        cfg = LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=5.0)
        a = LumpyPattern(cfg, np.random.default_rng(1))
        b = LumpyPattern(cfg, np.random.default_rng(2))
        assert [a.apply(10.0, t) for t in range(200)] != [b.apply(10.0, t) for t in range(200)]

    def test_multiplier_scales_magnitude_not_fire_positions(self) -> None:
        # Same seed + same p, different m → SAME fire positions (the mask draw is independent of
        # m), magnitudes scaled by m_large / m_small. Decouples occurrence from burst magnitude.
        p = 0.3
        small = LumpyPattern(
            LumpyPatternConfig(occurrence_probability=p, burst_multiplier=2.0),
            np.random.default_rng(99),
        )
        large = LumpyPattern(
            LumpyPatternConfig(occurrence_probability=p, burst_multiplier=10.0),
            np.random.default_rng(99),
        )
        for t in range(200):
            small_out = small.apply(10.0, t)
            large_out = large.apply(10.0, t)
            if small_out == 0.0:
                assert large_out == 0.0  # same fire positions
            else:
                assert large_out == small_out * 5.0  # m_large / m_small = 10 / 2

    def test_keep_frequency_approximates_probability(self) -> None:
        # Fixed seed → deterministic. Over 20k draws at p=0.3 the burst fraction is within ±0.02.
        cfg = LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=5.0)
        pattern = LumpyPattern(cfg, np.random.default_rng(2024))
        n = 20_000
        kept = sum(1 for t in range(n) if pattern.apply(10.0, t) > 0.0)
        assert abs(kept / n - 0.3) < 0.02


class TestLumpyPatternProperty:
    @given(
        p=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
        m=st.floats(min_value=1.0 + 1e-9, max_value=1e3, allow_nan=False, allow_infinity=False),
        base=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        t=st.integers(min_value=0, max_value=10_000),
    )
    def test_apply_returns_zero_or_burst_nonneg_float(
        self, p: float, m: float, base: float, t: int
    ) -> None:
        # Property: every apply() output is a non-negative float that is either 0.0 (masked) or
        # exactly the amplified base draw (burst).
        cfg = LumpyPatternConfig(occurrence_probability=p, burst_multiplier=m)
        pattern = LumpyPattern(cfg, np.random.default_rng(0))
        result = pattern.apply(base, t)
        assert isinstance(result, float)
        assert result == 0.0 or result == base * m
        assert result >= 0.0
