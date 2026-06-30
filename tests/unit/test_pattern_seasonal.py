"""Tests for ``demand.patterns.seasonal.SeasonalPattern``."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.config import SeasonalPatternConfig
from demand.patterns.seasonal import SeasonalPattern


class TestSeasonalPatternFormula:
    """Explicit formula verification at known ``(t, expected_factor)`` pairs.

    SeasonalPattern is deterministic; the "sequence equivalence" tests of
    the demand-arm suite become "formula verification" tests here. Each
    test pins a specific numerical relationship the multiplicative
    sinusoidal form ``apply(base, t) = base * (1 + amplitude * sin(2π·t/period + phase))``
    must satisfy. ``pytest.approx`` is used where intermediate ``sin``
    values are not exact in IEEE 754 (e.g., ``sin(π)`` ≈ 1.22e-16).
    """

    def test_apply_at_known_t_values(self) -> None:
        # amplitude=0.5, period=4, phase=0:
        #   sin(2π·t/4) at t=0,1,2,3 is 0, 1, 0, -1 (modulo float precision)
        #   factors:                    1.0, 1.5, 1.0, 0.5
        cfg = SeasonalPatternConfig(amplitude=0.5, period=4, phase=0.0)
        pattern = SeasonalPattern(cfg, np.random.default_rng(0))
        for t, expected_factor in enumerate([1.0, 1.5, 1.0, 0.5]):
            assert pattern.apply(10.0, t) == pytest.approx(10.0 * expected_factor)

    def test_apply_at_period_12(self) -> None:
        # The scenario YAML's period. Expected factors at the four
        # cardinal points: mean (t=0), peak (t=3), mean (t=6), trough (t=9).
        cfg = SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)
        pattern = SeasonalPattern(cfg, np.random.default_rng(0))
        for t, expected_factor in [(0, 1.0), (3, 1.5), (6, 1.0), (9, 0.5)]:
            assert pattern.apply(10.0, t) == pytest.approx(10.0 * expected_factor)

    def test_phase_pi_inverts_cycle(self) -> None:
        # sin(x + π) = -sin(x), so two SeasonalPatterns with the same
        # (amplitude, period) but phase=0 vs phase=π produce factors
        # whose AVERAGE is exactly 1.0 — i.e., the two outputs sum to
        # 2 * base for every t. Locks the phase parameter is correctly
        # threaded through apply() into math.sin.
        cfg_a = SeasonalPatternConfig(amplitude=0.5, period=4, phase=0.0)
        cfg_b = SeasonalPatternConfig(amplitude=0.5, period=4, phase=math.pi)
        pa = SeasonalPattern(cfg_a, np.random.default_rng(0))
        pb = SeasonalPattern(cfg_b, np.random.default_rng(0))
        for t in range(20):
            assert pa.apply(10.0, t) + pb.apply(10.0, t) == pytest.approx(20.0)


class TestSeasonalPatternInvariants:
    """Invariant guards — second pattern arm, first time-modulated pattern.

    Mirrors the demand-arm invariant test classes with adaptations for a
    deterministic pattern: float type, non-negativity (by bound, not by
    clip), zero-base passthrough, time dependence, periodicity, the
    amplitude=0 degenerate-to-identity case, the rng-state preservation
    check (reused from Stationary), the mean-preservation over a full
    cycle, the open-at-1 amplitude bound's trough behavior, and
    determinism under same seed.
    """

    def test_apply_returns_float_type(self) -> None:
        cfg = SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)
        pattern = SeasonalPattern(cfg, np.random.default_rng(0))
        assert isinstance(pattern.apply(10.0, 0), float)

    def test_apply_returns_non_negative(self) -> None:
        # At amplitude=0.999 (worst case under the locked < 1 bound), the
        # factor's minimum is just above 0. Verify the multiplicative form
        # stays > 0 across 1000 t-values. Non-negativity is GUARANTEED by
        # the bound, no runtime clip — this test pins that guarantee.
        cfg = SeasonalPatternConfig(amplitude=0.999, period=12, phase=0.0)
        pattern = SeasonalPattern(cfg, np.random.default_rng(0))
        for t in range(1000):
            assert pattern.apply(10.0, t) > 0.0

    def test_apply_zero_base_returns_zero(self) -> None:
        # Zero times any factor is exactly zero. Holds for all t and all
        # parameter combinations.
        cfg = SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)
        pattern = SeasonalPattern(cfg, np.random.default_rng(0))
        for t in range(20):
            assert pattern.apply(0.0, t) == 0.0

    def test_apply_is_time_dependent(self) -> None:
        # Counterpart to Stationary's test_apply_is_time_invariant.
        # At amplitude > 0 and period > 1, the factor varies with t.
        cfg = SeasonalPatternConfig(amplitude=0.5, period=4, phase=0.0)
        pattern = SeasonalPattern(cfg, np.random.default_rng(0))
        assert pattern.apply(10.0, 0) != pattern.apply(10.0, 1)

    def test_apply_is_periodic(self) -> None:
        # apply(base, t) == apply(base, t + period) for any t. The cyclic
        # property holds at all multiples of the period.
        cfg = SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.7)
        pattern = SeasonalPattern(cfg, np.random.default_rng(0))
        for t in range(0, 100, 7):
            assert pattern.apply(10.0, t) == pytest.approx(pattern.apply(10.0, t + 12))

    def test_amplitude_zero_is_identity(self) -> None:
        # At amplitude=0, the formula reduces to base * (1 + 0 * sin(...))
        # = base * 1 = base. EXACT equality (not approx): 1 + 0 * anything
        # = 1.0 exactly in IEEE 754. Locks the degenerate-to-Stationary
        # behavior used for A/B comparisons.
        cfg = SeasonalPatternConfig(amplitude=0.0, period=12, phase=0.0)
        pattern = SeasonalPattern(cfg, np.random.default_rng(0))
        for t in range(20):
            for base in (0.0, 1.0, 10.0, 100.0):
                assert pattern.apply(base, t) == base

    def test_apply_does_not_consume_rng(self) -> None:
        # SeasonalPattern stores rng (uniform API) but is fully
        # deterministic — never draws. Locked via byte-for-byte
        # bit_generator.state comparison at three checkpoints: before
        # construction, after construction, and after 1000 apply() calls.
        cfg = SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)
        rng = np.random.default_rng(42)
        state_before = rng.bit_generator.state
        pattern = SeasonalPattern(cfg, rng)
        state_after_construct = rng.bit_generator.state
        for i in range(1000):
            pattern.apply(float(i), i)
        state_after_apply = rng.bit_generator.state
        assert state_before == state_after_construct == state_after_apply

    def test_mean_over_full_cycle_approx_base_mean(self) -> None:
        # Sin averages to 0 over a full period, so the multiplicative
        # factor averages to 1.0 and the time-averaged demand equals the
        # base value. Verified by summing apply(base, t) across one
        # complete cycle and dividing by the period.
        cfg = SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)
        pattern = SeasonalPattern(cfg, np.random.default_rng(0))
        base = 10.0
        cycle_sum = sum(pattern.apply(base, t) for t in range(12))
        assert cycle_sum / 12 == pytest.approx(base)

    def test_high_amplitude_at_trough_approaches_zero(self) -> None:
        # At amplitude=0.999, period=4, phase=0, the trough (t=3, sin=-1)
        # produces factor ≈ 0.001, strictly above zero. Locks the
        # open-at-1 bound's quantitative effect — confirms the factor
        # never actually reaches 0 under the bound.
        cfg = SeasonalPatternConfig(amplitude=0.999, period=4, phase=0.0)
        pattern = SeasonalPattern(cfg, np.random.default_rng(0))
        trough = pattern.apply(10.0, 3)
        assert 0.0 < trough < 0.02  # ~0.01 expected

    def test_determinism_under_same_seed(self) -> None:
        # Trivially true (SeasonalPattern has no stochasticity), but
        # locks the symmetric API across the pattern test seam.
        cfg = SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.3)
        a = SeasonalPattern(cfg, np.random.default_rng(42))
        b = SeasonalPattern(cfg, np.random.default_rng(42))
        for base in (0.0, 1.0, 5.0, 100.0):
            for t in (0, 1, 10, 100):
                assert a.apply(base, t) == b.apply(base, t)


class TestSeasonalPatternProperty:
    @given(
        amplitude=st.floats(min_value=0.0, max_value=0.999, allow_nan=False, allow_infinity=False),
        period=st.integers(min_value=1, max_value=1000),
        phase=st.floats(
            min_value=-2.0 * math.pi,
            max_value=2.0 * math.pi,
            allow_nan=False,
            allow_infinity=False,
        ),
        base=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        t=st.integers(min_value=0, max_value=10_000),
    )
    def test_apply_returns_non_negative_float(
        self, amplitude: float, period: int, phase: float, base: float, t: int
    ) -> None:
        # Property: across the inventory-relevant parameter space, every
        # apply() output is a non-negative float. Bounds rationale:
        #   * amplitude ∈ [0, 0.999] stays strictly inside the Pydantic
        #     [0, 1) bound; at 0.999 the worst-case trough factor is 0.001
        #   * period ∈ [1, 1000] covers daily-within-3-years and beyond
        #   * phase ∈ [-2π, 2π] covers all phase positions (mod 2π)
        #   * base ∈ [0, 1e6] is the demand non-negative contract; 1e6
        #     covers high-volume SKUs
        #   * t ∈ [0, 10_000] covers reasonable simulation horizons
        # Edge cases: period=1 (degenerate constant factor), amplitude=0
        # (identity), base=0 (zero output), are all valid corners.
        cfg = SeasonalPatternConfig(amplitude=amplitude, period=period, phase=phase)
        pattern = SeasonalPattern(cfg, np.random.default_rng(0))
        result = pattern.apply(base, t)
        assert isinstance(result, float)
        assert result >= 0.0
