"""Tests for ``demand.patterns.trending.TrendingPattern``."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.config import TrendingPatternConfig
from demand.patterns.trending import TrendingPattern


class TestTrendingPatternFormula:
    """Explicit formula verification at known ``(t, expected_factor)`` pairs.

    TrendingPattern is deterministic; the "sequence equivalence" tests of
    the demand-arm suite become "formula verification" tests here. Each
    test pins a specific numerical relationship the linear-multiplicative
    form ``apply(base, t) = max(0.0, base * (1 + slope * t))`` must
    satisfy. Pre-clip factor values land exactly in IEEE 754 (no
    transcendental functions in the formula), so most assertions use
    exact ``==`` equality.
    """

    def test_apply_at_known_t_values_positive_slope(self) -> None:
        # slope=0.02, positive growth:
        #   t=0  → factor = 1.00, demand = 10.0
        #   t=10 → factor = 1.20, demand = 12.0
        #   t=50 → factor = 2.00, demand = 20.0
        #   t=89 → factor = 2.78, demand = 27.8
        cfg = TrendingPatternConfig(slope=0.02)
        pattern = TrendingPattern(cfg, np.random.default_rng(0))
        for t, expected_factor in [(0, 1.0), (10, 1.2), (50, 2.0), (89, 2.78)]:
            assert pattern.apply(10.0, t) == pytest.approx(10.0 * expected_factor)

    def test_apply_at_known_t_values_negative_slope_no_clip_yet(self) -> None:
        # slope=-0.01, before clip activates (clip activates at t >= 100):
        #   t=0  → factor = 1.00
        #   t=25 → factor = 0.75
        #   t=50 → factor = 0.50
        #   t=75 → factor = 0.25
        cfg = TrendingPatternConfig(slope=-0.01)
        pattern = TrendingPattern(cfg, np.random.default_rng(0))
        for t, expected_factor in [(0, 1.0), (25, 0.75), (50, 0.5), (75, 0.25)]:
            assert pattern.apply(10.0, t) == pytest.approx(10.0 * expected_factor)

    def test_apply_clips_at_steep_negative_slope(self) -> None:
        # slope=-0.02: factor crosses 0 EXACTLY at t=50 (1 + -0.02*50 = 0.0
        # exactly in IEEE 754). At t >= 51, factor < 0 and the clip fires.
        cfg = TrendingPatternConfig(slope=-0.02)
        pattern = TrendingPattern(cfg, np.random.default_rng(0))
        # At t=50, factor is exactly 0, so apply = max(0, 10*0) = 0
        assert pattern.apply(10.0, 50) == 0.0
        # At t > 50, factor is negative; clip activates
        for t in (51, 60, 89, 100, 1000):
            assert pattern.apply(10.0, t) == 0.0

    def test_apply_at_t_zero_is_base(self) -> None:
        # At t=0, factor = 1 + slope*0 = 1.0 EXACTLY in IEEE 754, so
        # apply(base, 0) == base for any base >= 0 and any slope.
        # Locks the mean-preservation-at-t=0 invariant.
        for slope in (-1.0, -0.05, 0.0, 0.05, 1.0, 100.0):
            cfg = TrendingPatternConfig(slope=slope)
            pattern = TrendingPattern(cfg, np.random.default_rng(0))
            for base in (0.0, 1.0, 10.0, 100.0, 1000.0):
                assert pattern.apply(base, 0) == base


class TestTrendingPatternInvariants:
    """Invariant guards — third pattern arm, second time-modulated pattern.

    Mirrors the demand-arm invariant test classes with adaptations for a
    deterministic linear-trend pattern: float type, non-negativity (by
    clip, not by bound), zero-base passthrough, time dependence, the
    slope=0 degenerate-to-identity case, monotonicity in both signs, the
    clip activation boundary, the no-reset-at-high-t guarantee, the
    rng-state preservation check, and determinism under same seed.
    """

    def test_apply_returns_float_type(self) -> None:
        cfg = TrendingPatternConfig(slope=0.02)
        pattern = TrendingPattern(cfg, np.random.default_rng(0))
        assert isinstance(pattern.apply(10.0, 0), float)

    def test_apply_returns_non_negative(self) -> None:
        # At slope=-0.05 (worst case for clip), 1000 t-values all yield
        # apply(10.0, t) >= 0.0. Clip activates at t = 20 (factor=0) and
        # holds for the rest. Pins non-negativity-via-clip.
        cfg = TrendingPatternConfig(slope=-0.05)
        pattern = TrendingPattern(cfg, np.random.default_rng(0))
        for t in range(1000):
            assert pattern.apply(10.0, t) >= 0.0

    def test_apply_zero_base_returns_zero(self) -> None:
        # Zero times any factor is zero, and max(0.0, 0.0) is 0.0. Holds
        # for all t and all slope values.
        cfg = TrendingPatternConfig(slope=0.02)
        pattern = TrendingPattern(cfg, np.random.default_rng(0))
        for t in range(20):
            assert pattern.apply(0.0, t) == 0.0

    def test_apply_is_time_dependent_at_nonzero_slope(self) -> None:
        # Counterpart to Stationary's test_apply_is_time_invariant;
        # complementary to Seasonal's test_apply_is_time_dependent.
        cfg = TrendingPatternConfig(slope=0.02)
        pattern = TrendingPattern(cfg, np.random.default_rng(0))
        assert pattern.apply(10.0, 0) != pattern.apply(10.0, 1)

    def test_apply_is_time_invariant_at_zero_slope(self) -> None:
        # At slope=0, the formula reduces to apply(base, t) =
        # max(0.0, base * 1.0) = base. EXACT equality across all t.
        # Locks the degenerate-to-Stationary identity case.
        cfg = TrendingPatternConfig(slope=0.0)
        pattern = TrendingPattern(cfg, np.random.default_rng(0))
        baseline = pattern.apply(10.0, 0)
        for t in (1, 10, 100, 1_000, 1_000_000):
            assert pattern.apply(10.0, t) == baseline

    def test_apply_does_not_consume_rng(self) -> None:
        # TrendingPattern stores rng (uniform API) but is fully
        # deterministic — never draws. Locked via byte-for-byte
        # bit_generator.state comparison at three checkpoints: before
        # construction, after construction, and after 1000 apply() calls.
        cfg = TrendingPatternConfig(slope=0.02)
        rng = np.random.default_rng(42)
        state_before = rng.bit_generator.state
        pattern = TrendingPattern(cfg, rng)
        state_after_construct = rng.bit_generator.state
        for i in range(1000):
            pattern.apply(float(i), i)
        state_after_apply = rng.bit_generator.state
        assert state_before == state_after_construct == state_after_apply

    def test_apply_grows_monotonically_at_positive_slope(self) -> None:
        # At slope > 0, factor = 1 + slope*t is strictly increasing in t;
        # since base is fixed positive, apply is strictly increasing too.
        cfg = TrendingPatternConfig(slope=0.02)
        pattern = TrendingPattern(cfg, np.random.default_rng(0))
        previous = pattern.apply(10.0, 0)
        for t in range(1, 101):
            current = pattern.apply(10.0, t)
            assert current > previous
            previous = current

    def test_apply_decays_monotonically_at_negative_slope_before_clip(self) -> None:
        # At slope=-0.005, clip activates at t=200 (factor=0); over
        # t in [0, 100], factor stays positive and apply is strictly
        # decreasing.
        cfg = TrendingPatternConfig(slope=-0.005)
        pattern = TrendingPattern(cfg, np.random.default_rng(0))
        previous = pattern.apply(10.0, 0)
        for t in range(1, 101):
            current = pattern.apply(10.0, t)
            assert current < previous
            previous = current

    def test_clip_activation_point_for_negative_slope(self) -> None:
        # At slope=-0.02, factor = 1 + -0.02*t crosses 0 EXACTLY at t=50
        # in IEEE 754 (-0.02*50 = -1.0 exactly). At t=51, factor is
        # negative and clip activates. Pins the clip boundary.
        cfg = TrendingPatternConfig(slope=-0.02)
        pattern = TrendingPattern(cfg, np.random.default_rng(0))
        assert pattern.apply(10.0, 49) > 0.0  # pre-clip
        assert pattern.apply(10.0, 50) == 0.0  # factor exactly 0
        assert pattern.apply(10.0, 51) == 0.0  # clip active

    def test_high_t_does_not_reset(self) -> None:
        # At slope=0.001 and t=10_000, factor = 1 + 0.001*10000 = 11.0
        # EXACTLY in IEEE 754. No modular reset — trends are horizon-long.
        cfg = TrendingPatternConfig(slope=0.001)
        pattern = TrendingPattern(cfg, np.random.default_rng(0))
        assert pattern.apply(10.0, 10_000) == 110.0

    def test_determinism_under_same_seed(self) -> None:
        # Trivially true (TrendingPattern has no stochasticity), but
        # locks the symmetric API across the pattern test seam.
        cfg = TrendingPatternConfig(slope=0.02)
        a = TrendingPattern(cfg, np.random.default_rng(42))
        b = TrendingPattern(cfg, np.random.default_rng(42))
        for base in (0.0, 1.0, 5.0, 100.0):
            for t in (0, 1, 10, 100):
                assert a.apply(base, t) == b.apply(base, t)


class TestTrendingPatternProperty:
    @given(
        slope=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
        base=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        t=st.integers(min_value=0, max_value=10_000),
    )
    def test_apply_returns_non_negative_float(self, slope: float, base: float, t: int) -> None:
        # Property: across the inventory-relevant parameter space, every
        # apply() output is a non-negative float. Bounds rationale:
        #   * slope ∈ [-0.5, 0.5] covers realistic per-period rates; at
        #     slope=-0.5 the factor crosses 0 at t=2, so the clip is
        #     heavily exercised across t ∈ [2, 10_000]
        #   * base ∈ [0, 1e6] is the demand non-negativity contract; 1e6
        #     covers high-volume SKUs
        #   * t ∈ [0, 10_000] covers reasonable simulation horizons and
        #     stresses the clip across the full range
        cfg = TrendingPatternConfig(slope=slope)
        pattern = TrendingPattern(cfg, np.random.default_rng(0))
        result = pattern.apply(base, t)
        assert isinstance(result, float)
        assert result >= 0.0
