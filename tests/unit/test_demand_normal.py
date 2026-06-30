"""Tests for ``demand.normal.NormalDemand``."""

from __future__ import annotations

import numpy as np

from core.config import NormalDemandConfig
from demand.normal import NormalDemand


class TestNormalDemandSequenceEquivalence:
    """Exact bit-for-bit equivalence with the pre-refactor inline draw.

    Pre-refactor, the engine ran ``max(0.0, float(rng.normal(mean, std)))``
    per period. ``NormalDemand`` must produce the identical sequence given
    the same RNG and (mean, std). This is the regression guard: if anyone
    reorders the call, drops the clip, or upcasts in a different order,
    these tests break loudly.
    """

    def test_sequence_matches_numpy_normal_with_clip(self) -> None:
        cfg = NormalDemandConfig(mean=10.0, std=2.0)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)

        demand = NormalDemand(cfg, rng_a)
        actual = [demand.draw() for _ in range(100)]
        expected = [max(0.0, float(rng_b.normal(10.0, 2.0))) for _ in range(100)]

        assert actual == expected

    def test_sequence_with_low_mean_exercises_clip(self) -> None:
        # Low mean + high std => negative samples occur and the clip fires;
        # this exercises the max() branch (not just the passthrough branch).
        cfg = NormalDemandConfig(mean=1.0, std=3.0)
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)

        demand = NormalDemand(cfg, rng_a)
        actual = [demand.draw() for _ in range(200)]
        expected = [max(0.0, float(rng_b.normal(1.0, 3.0))) for _ in range(200)]

        assert actual == expected
        # Sanity: at least one zero in the sequence — proves the clip ran.
        assert any(v == 0.0 for v in actual)


class TestNormalDemandInvariants:
    def test_draw_returns_non_negative(self) -> None:
        cfg = NormalDemandConfig(mean=1.0, std=3.0)
        demand = NormalDemand(cfg, np.random.default_rng(0))
        for _ in range(1000):
            assert demand.draw() >= 0.0

    def test_determinism_under_same_seed(self) -> None:
        cfg = NormalDemandConfig(mean=10.0, std=2.0)
        a = NormalDemand(cfg, np.random.default_rng(42))
        b = NormalDemand(cfg, np.random.default_rng(42))
        for _ in range(100):
            assert a.draw() == b.draw()

    def test_distinct_seeds_produce_different_sequences(self) -> None:
        cfg = NormalDemandConfig(mean=10.0, std=2.0)
        a = NormalDemand(cfg, np.random.default_rng(42))
        b = NormalDemand(cfg, np.random.default_rng(43))
        seq_a = [a.draw() for _ in range(50)]
        seq_b = [b.draw() for _ in range(50)]
        # At least some pairwise disagreement — unique sequences not the
        # same prefix by accident.
        assert seq_a != seq_b
