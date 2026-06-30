"""Tests for ``demand.poisson.PoissonDemand``."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from core.config import PoissonDemandConfig
from demand.poisson import PoissonDemand


class TestPoissonDemandSequenceEquivalence:
    """Exact bit-for-bit equivalence with ``np.random.Generator.poisson``.

    The PoissonDemand class is meant to be a thin wrapper around
    ``rng.poisson(rate)`` with a float upcast at the boundary. These tests
    pin that contract: the class produces the identical sequence given
    the same RNG and rate. Regression guard against accidental changes
    that would alter the demand stream.
    """

    def test_sequence_matches_numpy_poisson(self) -> None:
        cfg = PoissonDemandConfig(rate=10.0)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)

        demand = PoissonDemand(cfg, rng_a)
        actual = [demand.draw() for _ in range(100)]
        expected = [float(rng_b.poisson(10.0)) for _ in range(100)]

        assert actual == expected

    def test_sequence_at_low_rate(self) -> None:
        # Low rate exercises a different branch of numpy's Poisson sampler
        # (the small-lambda direct algorithm vs the large-lambda rejection).
        cfg = PoissonDemandConfig(rate=0.5)
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)

        demand = PoissonDemand(cfg, rng_a)
        actual = [demand.draw() for _ in range(200)]
        expected = [float(rng_b.poisson(0.5)) for _ in range(200)]

        assert actual == expected


class TestPoissonDemandInvariants:
    def test_draw_returns_float_type(self) -> None:
        # numpy's rng.poisson returns int64 / numpy.int64; the wrapper
        # MUST upcast to Python float so the engine's _period_demand: float
        # contract (and the Ledger float64 column) is preserved.
        cfg = PoissonDemandConfig(rate=10.0)
        demand = PoissonDemand(cfg, np.random.default_rng(0))
        sample = demand.draw()
        assert isinstance(sample, float)
        assert not isinstance(sample, np.integer)  # explicit guard against int64 leaking through

    def test_draw_returns_non_negative(self) -> None:
        cfg = PoissonDemandConfig(rate=2.0)
        demand = PoissonDemand(cfg, np.random.default_rng(0))
        for _ in range(1000):
            assert demand.draw() >= 0.0

    def test_low_rate_produces_zeros(self) -> None:
        # At rate=0.1, P(X=0) = e^{-0.1} ≈ 0.905 — most draws are 0.
        # Asserting "at least 50% are zero" is a safe shape check that
        # would fail loudly if someone accidentally drew from Normal or
        # similar (Normal(mean=0.1, std=0.1) would give roughly 50%
        # negatives, NOT 90% zeros).
        cfg = PoissonDemandConfig(rate=0.1)
        demand = PoissonDemand(cfg, np.random.default_rng(0))
        zeros = sum(1 for _ in range(1000) if demand.draw() == 0.0)
        assert zeros > 500, f"expected >50% zeros at rate=0.1, got {zeros}/1000"

    def test_determinism_under_same_seed(self) -> None:
        cfg = PoissonDemandConfig(rate=10.0)
        a = PoissonDemand(cfg, np.random.default_rng(42))
        b = PoissonDemand(cfg, np.random.default_rng(42))
        for _ in range(100):
            assert a.draw() == b.draw()

    def test_distinct_seeds_produce_different_sequences(self) -> None:
        cfg = PoissonDemandConfig(rate=10.0)
        a = PoissonDemand(cfg, np.random.default_rng(42))
        b = PoissonDemand(cfg, np.random.default_rng(43))
        seq_a = [a.draw() for _ in range(50)]
        seq_b = [b.draw() for _ in range(50)]
        assert seq_a != seq_b


class TestPoissonDemandProperty:
    @given(
        rate=st.floats(min_value=0.001, max_value=1e3, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=1_000_000),
    )
    def test_draw_returns_non_negative_float(self, rate: float, seed: int) -> None:
        # Property: across the inventory-relevant rate range, every draw is
        # a non-negative float. Poisson is integer-valued so range tests
        # are trivial; this pins type + sign across the parameter space.
        cfg = PoissonDemandConfig(rate=rate)
        demand = PoissonDemand(cfg, np.random.default_rng(seed))
        sample = demand.draw()
        assert isinstance(sample, float)
        assert sample >= 0.0
