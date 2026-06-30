"""Tests for ``leadtime.lognormal.LognormalLeadTime``."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from core.config import LognormalLeadTimeConfig
from leadtime.lognormal import LognormalLeadTime

# Matched-moment parameters: mean = 3, std = 1 (see example_leadtime_lognormal.yaml).
_MU = 1.045931
_SIGMA = 0.324593


class TestLognormalLeadTime:
    def test_sample_returns_int_type(self) -> None:
        cfg = LognormalLeadTimeConfig(mu=_MU, sigma=_SIGMA)
        lt = LognormalLeadTime(cfg, np.random.default_rng(0))
        assert isinstance(lt.sample(), int)

    def test_sample_always_at_least_one(self) -> None:
        cfg = LognormalLeadTimeConfig(mu=_MU, sigma=_SIGMA)
        lt = LognormalLeadTime(cfg, np.random.default_rng(0))
        assert all(lt.sample() >= 1 for _ in range(1000))

    def test_clip_fires_for_negative_mu(self) -> None:
        # mu=-2 ⇒ resulting mean = exp(-2 + 0.045) ≈ 0.14: most draws round to 0
        # and are clipped to 1. The clip absorbs the sub-0.5 mass.
        cfg = LognormalLeadTimeConfig(mu=-2.0, sigma=0.3)
        lt = LognormalLeadTime(cfg, np.random.default_rng(0))
        samples = [lt.sample() for _ in range(1000)]
        assert min(samples) >= 1
        assert samples.count(1) > 900

    def test_sample_advances_rng_state(self) -> None:
        cfg = LognormalLeadTimeConfig(mu=1.5, sigma=0.5)
        rng = np.random.default_rng(42)
        lt = LognormalLeadTime(cfg, rng)
        state_before = rng.bit_generator.state
        for _ in range(100):
            lt.sample()
        assert rng.bit_generator.state != state_before

    def test_distribution_matches_mean(self) -> None:
        # Matched-moment params give a resulting mean of 3; the clip almost never
        # fires at this mean, so the sample mean tracks 3.
        cfg = LognormalLeadTimeConfig(mu=_MU, sigma=_SIGMA)
        lt = LognormalLeadTime(cfg, np.random.default_rng(123))
        samples = [lt.sample() for _ in range(10_000)]
        assert abs(sum(samples) / len(samples) - 3.0) < 0.15

    def test_determinism_under_same_seed(self) -> None:
        cfg = LognormalLeadTimeConfig(mu=1.2, sigma=0.4)
        a = LognormalLeadTime(cfg, np.random.default_rng(7))
        b = LognormalLeadTime(cfg, np.random.default_rng(7))
        assert [a.sample() for _ in range(100)] == [b.sample() for _ in range(100)]

    @settings(max_examples=50)
    @given(
        mu=st.floats(min_value=-5.0, max_value=5.0),
        sigma=st.floats(min_value=0.01, max_value=3.0),
    )
    def test_property_returns_int_at_least_one(self, mu: float, sigma: float) -> None:
        cfg = LognormalLeadTimeConfig(mu=mu, sigma=sigma)
        lt = LognormalLeadTime(cfg, np.random.default_rng(0))
        result = lt.sample()
        assert isinstance(result, int)
        assert result >= 1
