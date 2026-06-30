"""Tests for ``demand.patterns.intermittent.IntermittentPattern``.

The first STOCHASTIC pattern arm: unlike Stationary / Seasonal / Trending (which accept an rng
for uniform API but never draw), this arm consumes the seeded ``"pattern"`` stream — one uniform
per ``apply()``, drawn before the keep/zero branch. The deterministic arms' "rng untouched" pin
is therefore inverted here into a twin-RNG consumption pin.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from core.config import IntermittentPatternConfig
from demand.patterns.intermittent import IntermittentPattern


class TestIntermittentPatternSemantics:
    """The Bernoulli zero-mask: ``apply(base, t)`` is ``base`` w.p. ``p``, else ``0.0``."""

    def test_probability_one_is_identity(self) -> None:
        # p=1 → rng.random() < 1.0 is always True (random() ∈ [0, 1)) → always keep.
        # The degenerate-to-identity case (parity with amplitude=0 / slope=0).
        cfg = IntermittentPatternConfig(occurrence_probability=1.0)
        pattern = IntermittentPattern(cfg, np.random.default_rng(0))
        for t in range(200):
            assert pattern.apply(10.0, t) == 10.0

    def test_apply_decision_matches_a_twin_rng(self) -> None:
        # The keep/zero decision is exactly `draw < p` for the SAME uniform a twin rng produces,
        # and apply consumes EXACTLY ONE draw per call (the two rngs stay in lockstep afterward).
        cfg = IntermittentPatternConfig(occurrence_probability=0.6)
        rng_a = np.random.default_rng(123)
        rng_b = np.random.default_rng(123)  # twin: predicts each draw apply() will consume
        pattern = IntermittentPattern(cfg, rng_a)
        for t in range(100):
            expected = 10.0 if rng_b.random() < 0.6 else 0.0
            assert pattern.apply(10.0, t) == expected
        # Exactly one draw consumed per apply → the rngs are still aligned.
        assert rng_a.random() == rng_b.random()

    def test_output_is_zero_or_base(self) -> None:
        cfg = IntermittentPatternConfig(occurrence_probability=0.5)
        pattern = IntermittentPattern(cfg, np.random.default_rng(7))
        for t in range(500):
            result = pattern.apply(13.7, t)
            assert result == 0.0 or result == 13.7

    def test_zero_base_returns_zero(self) -> None:
        # base=0 → keep yields 0.0, mask yields 0.0; either way 0.0.
        cfg = IntermittentPatternConfig(occurrence_probability=0.6)
        pattern = IntermittentPattern(cfg, np.random.default_rng(0))
        for t in range(50):
            assert pattern.apply(0.0, t) == 0.0


class TestIntermittentPatternInvariants:
    def test_apply_returns_float_type(self) -> None:
        cfg = IntermittentPatternConfig(occurrence_probability=0.6)
        pattern = IntermittentPattern(cfg, np.random.default_rng(0))
        assert isinstance(pattern.apply(10.0, 0), float)

    def test_apply_returns_non_negative(self) -> None:
        cfg = IntermittentPatternConfig(occurrence_probability=0.3)
        pattern = IntermittentPattern(cfg, np.random.default_rng(1))
        for t in range(1000):
            assert pattern.apply(10.0, t) >= 0.0

    def test_apply_consumes_one_draw_per_call(self) -> None:
        # State-advance check via a twin: after N applies, the source rng has advanced exactly N
        # draws (the twin drawn N times stays in lockstep). Robust — no bit_generator internals.
        cfg = IntermittentPatternConfig(occurrence_probability=0.6)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        pattern = IntermittentPattern(cfg, rng_a)
        for t in range(1000):
            pattern.apply(float(t), t)
            rng_b.random()
        assert rng_a.random() == rng_b.random()

    def test_determinism_under_same_seed(self) -> None:
        cfg = IntermittentPatternConfig(occurrence_probability=0.6)
        a = IntermittentPattern(cfg, np.random.default_rng(2024))
        b = IntermittentPattern(cfg, np.random.default_rng(2024))
        for t in range(200):
            assert a.apply(10.0, t) == b.apply(10.0, t)

    def test_distinct_seeds_diverge(self) -> None:
        cfg = IntermittentPatternConfig(occurrence_probability=0.6)
        a = IntermittentPattern(cfg, np.random.default_rng(1))
        b = IntermittentPattern(cfg, np.random.default_rng(2))
        assert [a.apply(10.0, t) for t in range(200)] != [b.apply(10.0, t) for t in range(200)]

    def test_keep_frequency_approximates_probability(self) -> None:
        # Fixed seed → deterministic. Over 20k draws at p=0.6 the kept fraction is within ±0.02
        # (~5.7σ band; never flakes, yet a swapped 1−p=0.4 is ~57σ away and caught).
        cfg = IntermittentPatternConfig(occurrence_probability=0.6)
        pattern = IntermittentPattern(cfg, np.random.default_rng(2024))
        n = 20_000
        kept = sum(1 for t in range(n) if pattern.apply(10.0, t) > 0.0)
        assert abs(kept / n - 0.6) < 0.02


class TestIntermittentPatternProperty:
    @given(
        p=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
        base=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        t=st.integers(min_value=0, max_value=10_000),
    )
    def test_apply_returns_zero_or_base_nonneg_float(self, p: float, base: float, t: int) -> None:
        # Property: across the inventory-relevant parameter space, every apply() output is a
        # non-negative float that is either 0.0 (masked) or exactly the base draw (kept).
        cfg = IntermittentPatternConfig(occurrence_probability=p)
        pattern = IntermittentPattern(cfg, np.random.default_rng(0))
        result = pattern.apply(base, t)
        assert isinstance(result, float)
        assert result == 0.0 or result == base
        assert result >= 0.0
