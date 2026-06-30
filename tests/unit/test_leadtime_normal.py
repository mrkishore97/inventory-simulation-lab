"""Tests for ``leadtime.normal.NormalLeadTime``."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from core.config import NormalLeadTimeConfig
from leadtime.normal import NormalLeadTime


class TestNormalLeadTime:
    def test_sample_returns_int_type(self) -> None:
        cfg = NormalLeadTimeConfig(mean=3.0, std=1.0)
        lt = NormalLeadTime(cfg, np.random.default_rng(0))
        assert isinstance(lt.sample(), int)

    def test_sample_always_at_least_one(self) -> None:
        cfg = NormalLeadTimeConfig(mean=3.0, std=1.0)
        lt = NormalLeadTime(cfg, np.random.default_rng(0))
        assert all(lt.sample() >= 1 for _ in range(1000))

    def test_clip_fires_for_small_mean(self) -> None:
        # Mean below the clip boundary: most draws round to 0 (clipped to 1) or
        # round to 1, so the realized lead time is dominated by 1.
        cfg = NormalLeadTimeConfig(mean=0.6, std=0.3)
        lt = NormalLeadTime(cfg, np.random.default_rng(0))
        samples = [lt.sample() for _ in range(1000)]
        assert min(samples) >= 1
        assert samples.count(1) > 500

    def test_zero_std_is_constant_round_mean(self) -> None:
        # std=0 degenerates to round(mean) every draw (a near-deterministic arm).
        cfg = NormalLeadTimeConfig(mean=3.0, std=0.0)
        lt = NormalLeadTime(cfg, np.random.default_rng(0))
        assert all(lt.sample() == 3 for _ in range(50))

    def test_sample_advances_rng_state(self) -> None:
        # First RNG-consuming lead-time arm: state MUST advance (the counterpart
        # to DeterministicLeadTime.test_sample_does_not_consume_rng).
        cfg = NormalLeadTimeConfig(mean=5.0, std=2.0)
        rng = np.random.default_rng(42)
        lt = NormalLeadTime(cfg, rng)
        state_before = rng.bit_generator.state
        for _ in range(100):
            lt.sample()
        assert rng.bit_generator.state != state_before

    def test_distribution_matches_mean(self) -> None:
        # mean=10, std=2: clip never fires and round is ~unbiased, so the sample
        # mean tracks the configured mean. Loose ±5% bound to avoid flakiness.
        cfg = NormalLeadTimeConfig(mean=10.0, std=2.0)
        lt = NormalLeadTime(cfg, np.random.default_rng(123))
        samples = [lt.sample() for _ in range(10_000)]
        assert abs(sum(samples) / len(samples) - 10.0) < 0.5

    def test_determinism_under_same_seed(self) -> None:
        cfg = NormalLeadTimeConfig(mean=4.0, std=1.5)
        a = NormalLeadTime(cfg, np.random.default_rng(7))
        b = NormalLeadTime(cfg, np.random.default_rng(7))
        assert [a.sample() for _ in range(100)] == [b.sample() for _ in range(100)]

    @settings(max_examples=50)
    @given(
        mean=st.floats(min_value=0.1, max_value=100.0),
        std=st.floats(min_value=0.0, max_value=50.0),
    )
    def test_property_returns_int_at_least_one(self, mean: float, std: float) -> None:
        cfg = NormalLeadTimeConfig(mean=mean, std=std)
        lt = NormalLeadTime(cfg, np.random.default_rng(0))
        result = lt.sample()
        assert isinstance(result, int)
        assert result >= 1
