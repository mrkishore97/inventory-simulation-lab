"""Tests for ``leadtime.gamma.GammaLeadTime``."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from core.config import GammaLeadTimeConfig
from leadtime.gamma import GammaLeadTime


class TestGammaLeadTime:
    def test_sample_returns_int_type(self) -> None:
        cfg = GammaLeadTimeConfig(shape=9.0, scale=1.0 / 3.0)
        lt = GammaLeadTime(cfg, np.random.default_rng(0))
        assert isinstance(lt.sample(), int)

    def test_sample_always_at_least_one(self) -> None:
        cfg = GammaLeadTimeConfig(shape=9.0, scale=1.0 / 3.0)
        lt = GammaLeadTime(cfg, np.random.default_rng(0))
        assert all(lt.sample() >= 1 for _ in range(1000))

    def test_clip_fires_for_small_mean(self) -> None:
        # shape*scale = 0.25 (well below the 0.5 round-up boundary): most draws
        # round to 0 and are clipped to 1.
        cfg = GammaLeadTimeConfig(shape=0.5, scale=0.5)
        lt = GammaLeadTime(cfg, np.random.default_rng(0))
        samples = [lt.sample() for _ in range(1000)]
        assert min(samples) >= 1
        assert samples.count(1) > 500

    def test_sample_advances_rng_state(self) -> None:
        cfg = GammaLeadTimeConfig(shape=4.0, scale=1.0)
        rng = np.random.default_rng(42)
        lt = GammaLeadTime(cfg, rng)
        state_before = rng.bit_generator.state
        for _ in range(100):
            lt.sample()
        assert rng.bit_generator.state != state_before

    def test_distribution_matches_mean(self) -> None:
        # Gamma(9, 1/3): mean = shape*scale = 3, std = sqrt(shape)*scale = 1.
        # The clip almost never fires at this mean, so the sample mean tracks 3.
        cfg = GammaLeadTimeConfig(shape=9.0, scale=1.0 / 3.0)
        lt = GammaLeadTime(cfg, np.random.default_rng(123))
        samples = [lt.sample() for _ in range(10_000)]
        assert abs(sum(samples) / len(samples) - 3.0) < 0.15

    def test_determinism_under_same_seed(self) -> None:
        cfg = GammaLeadTimeConfig(shape=5.0, scale=0.8)
        a = GammaLeadTime(cfg, np.random.default_rng(7))
        b = GammaLeadTime(cfg, np.random.default_rng(7))
        assert [a.sample() for _ in range(100)] == [b.sample() for _ in range(100)]

    @settings(max_examples=50)
    @given(
        shape=st.floats(min_value=0.1, max_value=100.0),
        scale=st.floats(min_value=0.01, max_value=50.0),
    )
    def test_property_returns_int_at_least_one(self, shape: float, scale: float) -> None:
        cfg = GammaLeadTimeConfig(shape=shape, scale=scale)
        lt = GammaLeadTime(cfg, np.random.default_rng(0))
        result = lt.sample()
        assert isinstance(result, int)
        assert result >= 1
