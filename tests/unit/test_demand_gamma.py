"""Tests for ``demand.gamma.GammaDemand``."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from core.config import GammaDemandConfig
from demand.gamma import GammaDemand


class TestGammaDemandSequenceEquivalence:
    """Exact bit-for-bit equivalence with ``np.random.Generator.gamma``.

    The GammaDemand class is meant to be a thin wrapper around
    ``rng.gamma(shape, scale)`` with a float upcast at the boundary.
    These tests pin that contract: the class produces the identical
    sequence given the same RNG and (shape, scale). Regression guard
    against accidental changes that would alter the demand stream.
    """

    def test_sequence_matches_numpy_gamma(self) -> None:
        cfg = GammaDemandConfig(shape=5.0, scale=2.0)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)

        demand = GammaDemand(cfg, rng_a)
        actual = [demand.draw() for _ in range(100)]
        expected = [float(rng_b.gamma(5.0, 2.0)) for _ in range(100)]

        assert actual == expected

    def test_sequence_at_low_shape_regime(self) -> None:
        # Low shape exercises the mode-at-zero / heavy-tail regime where
        # numpy's Gamma algorithm path can differ from moderate-shape.
        # Pinning equivalence across regimes catches subtle wrapper bugs.
        cfg = GammaDemandConfig(shape=0.5, scale=1.0)
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)

        demand = GammaDemand(cfg, rng_a)
        actual = [demand.draw() for _ in range(200)]
        expected = [float(rng_b.gamma(0.5, 1.0)) for _ in range(200)]

        assert actual == expected


class TestGammaDemandInvariants:
    """Invariant guards — continuous-support test redesign applied.

    First continuous-support arm to go through the recipe; the
    integer-rejection assertion that anchored Poisson and NegBin
    (``not isinstance(sample, np.integer)``) is dropped here — vacuously
    true for a continuous distribution. Replaced by a positive
    ``test_draw_returns_continuous_values`` (assert at least one draw
    has a non-zero fractional part). Strict positivity (``> 0.0``)
    replaces the integer arms' ``>= 0.0`` to capture Gamma's open-at-zero
    support.
    """

    def test_draw_returns_float_type(self) -> None:
        # numpy 2.4.4's rng.gamma returns Python float directly for
        # scalar calls — the float(...) wrap is a documentation seam,
        # not a runtime cast for this arm. On older numpy versions it
        # returns np.float64 (a Python float subclass); the wrap
        # normalizes to a plain float instance. The
        # `not isinstance(sample, np.integer)` line that anchored
        # Poisson/NegBin tests is intentionally dropped here — vacuously
        # true for a continuous distribution.
        cfg = GammaDemandConfig(shape=5.0, scale=2.0)
        demand = GammaDemand(cfg, np.random.default_rng(0))
        sample = demand.draw()
        assert isinstance(sample, float)

    def test_draw_returns_continuous_values(self) -> None:
        # Positive proof of continuous support: at least one of 100
        # draws has a non-zero fractional part. Direct counterpart to
        # the dropped `not isinstance(np.integer)` assertion; catches
        # accidental wiring to an integer distribution (where every
        # draw would have `.0` fractional part).
        cfg = GammaDemandConfig(shape=5.0, scale=2.0)
        demand = GammaDemand(cfg, np.random.default_rng(0))
        samples = [demand.draw() for _ in range(100)]
        assert any(s != float(int(s)) for s in samples), (
            "expected at least one continuous (non-integer) sample"
        )

    def test_draw_returns_positive(self) -> None:
        # Gamma support is the open interval (0, ∞), NOT [0, ∞) as for
        # Poisson/NegBin. Strict positivity (> 0.0) is the right
        # assertion; the open-at-zero contract holds numerically as well
        # as mathematically (verified via pre-impl sanity check: even
        # gamma(0.01, 1.0) returns 7.44e-12, not exactly 0.0).
        cfg = GammaDemandConfig(shape=5.0, scale=2.0)
        demand = GammaDemand(cfg, np.random.default_rng(0))
        for _ in range(1000):
            assert demand.draw() > 0.0

    def test_small_mean_concentrates_low(self) -> None:
        # At (shape=2, scale=0.1) expected mean = shape * scale = 0.2.
        # Over 1000 draws, at least 95% should be below 1.0. Same
        # categorical-shape-check style as NegBin's
        # test_high_p_concentrates_near_zero, with Gamma-appropriate
        # parameters. Catches accidental wiring to a different
        # distribution shape (e.g., Normal at mean=0.2, std=1 would
        # only have ~80% below 1.0).
        cfg = GammaDemandConfig(shape=2.0, scale=0.1)
        demand = GammaDemand(cfg, np.random.default_rng(0))
        below_one = sum(1 for _ in range(1000) if demand.draw() < 1.0)
        assert below_one > 950, f"expected >95% < 1.0 at (shape=2, scale=0.1), got {below_one}/1000"

    def test_determinism_under_same_seed(self) -> None:
        cfg = GammaDemandConfig(shape=5.0, scale=2.0)
        a = GammaDemand(cfg, np.random.default_rng(42))
        b = GammaDemand(cfg, np.random.default_rng(42))
        for _ in range(100):
            assert a.draw() == b.draw()

    def test_distinct_seeds_produce_different_sequences(self) -> None:
        cfg = GammaDemandConfig(shape=5.0, scale=2.0)
        a = GammaDemand(cfg, np.random.default_rng(42))
        b = GammaDemand(cfg, np.random.default_rng(43))
        seq_a = [a.draw() for _ in range(50)]
        seq_b = [b.draw() for _ in range(50)]
        assert seq_a != seq_b


class TestGammaDemandProperty:
    @given(
        shape=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        scale=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=1_000_000),
    )
    def test_draw_returns_positive_float(self, shape: float, scale: float, seed: int) -> None:
        # Property: across the inventory-relevant (shape, scale) space,
        # every draw is a strictly positive float. Bounds chosen to keep
        # worst-case mean = 10,000 (microsecond-computable) and avoid
        # the very-large-mean rejection-sampling slowdown. Lower bound
        # scale=0.01 keeps very-small-mean cases testable without
        # underflow concerns (sanity-checked: even gamma(0.01, 1.0)
        # returns positive non-zero). Boundary validation for
        # shape <= 0 and scale <= 0 lives in tests/unit/test_config.py.
        cfg = GammaDemandConfig(shape=shape, scale=scale)
        demand = GammaDemand(cfg, np.random.default_rng(seed))
        sample = demand.draw()
        assert isinstance(sample, float)
        assert sample > 0.0
