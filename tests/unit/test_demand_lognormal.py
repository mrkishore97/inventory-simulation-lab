"""Tests for ``demand.lognormal.LognormalDemand``."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from core.config import LognormalDemandConfig
from demand.lognormal import LognormalDemand


class TestLognormalDemandSequenceEquivalence:
    """Exact bit-for-bit equivalence with ``np.random.Generator.lognormal``.

    The LognormalDemand class is a thin wrapper around
    ``rng.lognormal(mu, sigma)`` with a float upcast at the boundary.
    These tests pin that contract: the class produces the identical
    sequence given the same RNG and (mu, sigma). Regression guard
    against accidental changes that would alter the demand stream.

    Note on the parameter mapping: our ``mu, sigma`` names map directly
    onto numpy's ``rng.lognormal(mean, sigma)`` positional arguments —
    the numpy ``mean`` parameter is the underlying Normal's location
    (the well-known numpy foot-gun the config docstring documents).
    """

    def test_sequence_matches_numpy_lognormal(self) -> None:
        cfg = LognormalDemandConfig(mu=2.2114, sigma=0.4271)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)

        demand = LognormalDemand(cfg, rng_a)
        actual = [demand.draw() for _ in range(100)]
        expected = [float(rng_b.lognormal(2.2114, 0.4271)) for _ in range(100)]

        assert actual == expected

    def test_sequence_at_high_sigma_regime(self) -> None:
        # High sigma exercises the heavy-tail regime where Lognormal's
        # right tail extends to extreme values. Pinning equivalence across
        # regimes catches subtle wrapper bugs that only surface at the
        # parameter-space corners (Gamma's analogue is the low-shape
        # mode-at-zero regime; for Lognormal the corner is high sigma).
        cfg = LognormalDemandConfig(mu=0.0, sigma=2.0)
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)

        demand = LognormalDemand(cfg, rng_a)
        actual = [demand.draw() for _ in range(200)]
        expected = [float(rng_b.lognormal(0.0, 2.0)) for _ in range(200)]

        assert actual == expected


class TestLognormalDemandInvariants:
    """Invariant guards — continuous-support seam reused from Gamma.

    Second continuous-support arm through the recipe; the continuous
    test seam locked here (drop integer-rejection, add
    ``test_draw_returns_continuous_values``, strict ``> 0.0``) is
    reused unchanged. The small-mean concentration parameters and the
    sequence-equivalence regime test are Lognormal-specific.
    """

    def test_draw_returns_float_type(self) -> None:
        # numpy 2.4.4's rng.lognormal returns Python float directly for
        # scalar calls — the float(...) wrap is a documentation seam,
        # not a runtime cast for this arm. Continuous-support seam
        #: the ``not isinstance(sample, np.integer)``
        # assertion is intentionally absent — vacuously true for a
        # continuous distribution.
        cfg = LognormalDemandConfig(mu=2.2114, sigma=0.4271)
        demand = LognormalDemand(cfg, np.random.default_rng(0))
        sample = demand.draw()
        assert isinstance(sample, float)

    def test_draw_returns_continuous_values(self) -> None:
        # Positive proof of continuous support (continuous-support seam):
        # at least one of 100 draws has a non-zero fractional part.
        # Counterpart to the dropped ``not isinstance(np.integer)``
        # assertion; catches accidental wiring to an integer distribution.
        cfg = LognormalDemandConfig(mu=2.2114, sigma=0.4271)
        demand = LognormalDemand(cfg, np.random.default_rng(0))
        samples = [demand.draw() for _ in range(100)]
        assert any(s != float(int(s)) for s in samples), (
            "expected at least one continuous (non-integer) sample"
        )

    def test_draw_returns_positive(self) -> None:
        # Lognormal support is the open interval (0, ∞) — same as Gamma,
        # distinct from Poisson/NegBin's [0, ∞). Strict positivity
        # (> 0.0) is the right assertion; the open-at-zero contract
        # holds numerically (pre-impl sanity check: even
        # lognormal(-5.0, 0.01) returns 6.76e-3, not exactly 0.0).
        cfg = LognormalDemandConfig(mu=2.2114, sigma=0.4271)
        demand = LognormalDemand(cfg, np.random.default_rng(0))
        for _ in range(1000):
            assert demand.draw() > 0.0

    def test_small_mean_concentrates_low(self) -> None:
        # At (mu=-2, sigma=0.5) expected mean = exp(-2 + 0.125) ≈ 0.153.
        # Lognormal at small mu and moderate sigma is extremely
        # concentrated below 1.0 — P(X > 1) = P(N(-2, 0.5) > 0) which is
        # essentially zero. Same categorical-shape-check style as Gamma's
        # test_small_mean_concentrates_low, with Lognormal-appropriate
        # parameters. Catches accidental wiring to a wider-shaped
        # distribution at the same nominal location parameter.
        cfg = LognormalDemandConfig(mu=-2.0, sigma=0.5)
        demand = LognormalDemand(cfg, np.random.default_rng(0))
        below_one = sum(1 for _ in range(1000) if demand.draw() < 1.0)
        assert below_one > 950, f"expected >95% < 1.0 at (mu=-2, sigma=0.5), got {below_one}/1000"

    def test_determinism_under_same_seed(self) -> None:
        cfg = LognormalDemandConfig(mu=2.2114, sigma=0.4271)
        a = LognormalDemand(cfg, np.random.default_rng(42))
        b = LognormalDemand(cfg, np.random.default_rng(42))
        for _ in range(100):
            assert a.draw() == b.draw()

    def test_distinct_seeds_produce_different_sequences(self) -> None:
        cfg = LognormalDemandConfig(mu=2.2114, sigma=0.4271)
        a = LognormalDemand(cfg, np.random.default_rng(42))
        b = LognormalDemand(cfg, np.random.default_rng(43))
        seq_a = [a.draw() for _ in range(50)]
        seq_b = [b.draw() for _ in range(50)]
        assert seq_a != seq_b


class TestLognormalDemandProperty:
    @given(
        mu=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        sigma=st.floats(min_value=0.01, max_value=3.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=1_000_000),
    )
    def test_draw_returns_positive_float(self, mu: float, sigma: float, seed: int) -> None:
        # Property: across the inventory-relevant (mu, sigma) space,
        # every draw is a strictly positive float. Bounds rationale:
        #   * mu ∈ [-5, 5] — any real; at sigma=0 gives means in
        #     [exp(-5) ≈ 6.7e-3, exp(5) ≈ 148]. mu can be negative
        #     (slow-moving SKUs with mean < 1).
        #   * sigma ∈ [0.01, 3.0] — lower bound avoids degenerate point
        #     mass at exp(mu); upper bound caps exp(mu + sigma²/2) at
        #     ≈ exp(5 + 4.5) ≈ 13360 worst case (still computable).
        # Boundary validation for sigma <= 0 lives in test_config.py.
        cfg = LognormalDemandConfig(mu=mu, sigma=sigma)
        demand = LognormalDemand(cfg, np.random.default_rng(seed))
        sample = demand.draw()
        assert isinstance(sample, float)
        assert sample > 0.0
