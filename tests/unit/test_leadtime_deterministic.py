"""Tests for ``leadtime.deterministic.DeterministicLeadTime``."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import DeterministicLeadTimeConfig
from leadtime.deterministic import DeterministicLeadTime


class TestDeterministicLeadTime:
    def test_sample_returns_configured_value(self) -> None:
        cfg = DeterministicLeadTimeConfig(lead_time=3)
        lt = DeterministicLeadTime(cfg, np.random.default_rng(0))
        assert lt.sample() == 3

    def test_sample_returns_int_type(self) -> None:
        cfg = DeterministicLeadTimeConfig(lead_time=5)
        lt = DeterministicLeadTime(cfg, np.random.default_rng(0))
        assert isinstance(lt.sample(), int)

    @pytest.mark.parametrize("lead_time", [1, 3, 100])
    def test_sample_is_constant_across_calls(self, lead_time: int) -> None:
        cfg = DeterministicLeadTimeConfig(lead_time=lead_time)
        lt = DeterministicLeadTime(cfg, np.random.default_rng(0))
        assert [lt.sample() for _ in range(50)] == [lead_time] * 50

    def test_sample_does_not_consume_rng(self) -> None:
        # Byte-equivalence lock: the deterministic arm holds the RNG for a
        # uniform factory API but must NEVER draw from it, so the "lead_time"
        # stream stays untouched (mirrors StationaryPattern). If this fires,
        # the chassis extraction is no longer byte-equivalent with M1.
        cfg = DeterministicLeadTimeConfig(lead_time=4)
        rng = np.random.default_rng(42)
        lt = DeterministicLeadTime(cfg, rng)
        state_before = rng.bit_generator.state
        for _ in range(1000):
            lt.sample()
        assert rng.bit_generator.state == state_before
