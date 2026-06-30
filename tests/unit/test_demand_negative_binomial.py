"""Tests for ``demand.negative_binomial.NegativeBinomialDemand``."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from core.config import NegativeBinomialDemandConfig
from demand.negative_binomial import NegativeBinomialDemand


class TestNegativeBinomialDemandSequenceEquivalence:
    """Exact bit-for-bit equivalence with ``np.random.Generator.negative_binomial``.

    The NegativeBinomialDemand class is meant to be a thin wrapper around
    ``rng.negative_binomial(n, p)`` with a float upcast at the boundary.
    These tests pin that contract: the class produces the identical
    sequence given the same RNG and (n, p). Regression guard against
    accidental changes that would alter the demand stream.
    """

    def test_sequence_matches_numpy_negative_binomial(self) -> None:
        cfg = NegativeBinomialDemandConfig(n=10.0, p=0.5)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)

        demand = NegativeBinomialDemand(cfg, rng_a)
        actual = [demand.draw() for _ in range(100)]
        expected = [float(rng_b.negative_binomial(10.0, 0.5)) for _ in range(100)]

        assert actual == expected

    def test_sequence_at_skewed_p(self) -> None:
        # Low p with moderate n exercises the heavy-tail regime — numpy's
        # NegBin uses a Gamma-Poisson mixture algorithm that takes a
        # different code path here vs the symmetric p=0.5 case. Pinning
        # equivalence across regimes catches subtle wrapper bugs.
        cfg = NegativeBinomialDemandConfig(n=2.0, p=0.1)
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)

        demand = NegativeBinomialDemand(cfg, rng_a)
        actual = [demand.draw() for _ in range(200)]
        expected = [float(rng_b.negative_binomial(2.0, 0.1)) for _ in range(200)]

        assert actual == expected


class TestNegativeBinomialDemandInvariants:
    def test_draw_returns_float_type(self) -> None:
        # numpy's rng.negative_binomial returns int64 / numpy.int64; the
        # wrapper MUST upcast to Python float so the engine's
        # _period_demand: float contract (and the Ledger float64 column)
        # is preserved.
        cfg = NegativeBinomialDemandConfig(n=10.0, p=0.5)
        demand = NegativeBinomialDemand(cfg, np.random.default_rng(0))
        sample = demand.draw()
        assert isinstance(sample, float)
        assert not isinstance(sample, np.integer)

    def test_draw_returns_non_negative(self) -> None:
        cfg = NegativeBinomialDemandConfig(n=5.0, p=0.5)
        demand = NegativeBinomialDemand(cfg, np.random.default_rng(0))
        for _ in range(1000):
            assert demand.draw() >= 0.0

    def test_high_p_concentrates_near_zero(self) -> None:
        # At (n=1, p=0.99) expected mean = n(1-p)/p ≈ 0.0101. Most draws
        # should be exactly 0. Asserting "at least 95% are zero" is a
        # safe shape check that would fail loudly if someone accidentally
        # wired up the wrong distribution (e.g., Normal-shaped or
        # complementary p semantics).
        cfg = NegativeBinomialDemandConfig(n=1.0, p=0.99)
        demand = NegativeBinomialDemand(cfg, np.random.default_rng(0))
        zeros = sum(1 for _ in range(1000) if demand.draw() == 0.0)
        assert zeros > 950, f"expected >95% zeros at (n=1, p=0.99), got {zeros}/1000"

    def test_determinism_under_same_seed(self) -> None:
        cfg = NegativeBinomialDemandConfig(n=10.0, p=0.5)
        a = NegativeBinomialDemand(cfg, np.random.default_rng(42))
        b = NegativeBinomialDemand(cfg, np.random.default_rng(42))
        for _ in range(100):
            assert a.draw() == b.draw()

    def test_distinct_seeds_produce_different_sequences(self) -> None:
        cfg = NegativeBinomialDemandConfig(n=10.0, p=0.5)
        a = NegativeBinomialDemand(cfg, np.random.default_rng(42))
        b = NegativeBinomialDemand(cfg, np.random.default_rng(43))
        seq_a = [a.draw() for _ in range(50)]
        seq_b = [b.draw() for _ in range(50)]
        assert seq_a != seq_b


class TestNegativeBinomialDemandProperty:
    @given(
        n=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        p=st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=1_000_000),
    )
    def test_draw_returns_non_negative_float(self, n: float, p: float, seed: int) -> None:
        # Property: across the inventory-relevant (n, p) space, every draw
        # is a non-negative float. Tightened bounds vs the math-permitted
        # ranges avoid numerically pathological combinations (e.g., low p
        # with high n → mean ~ 1e5 + slow rejection sampling). Boundary
        # validation for p ∈ {0, 1, <0, >1} lives in
        # tests/unit/test_config.py — separate concerns.
        cfg = NegativeBinomialDemandConfig(n=n, p=p)
        demand = NegativeBinomialDemand(cfg, np.random.default_rng(seed))
        sample = demand.draw()
        assert isinstance(sample, float)
        assert sample >= 0.0
