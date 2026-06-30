"""Tests for the disruption factory in ``leadtime.disruptions.registry``.

The factory has two arms — ``NoDisruption`` (identity, the
default) and ``ScheduledDisruption`` (the deterministic windows). The defensive
raise is exercised by passing a non-disruption config type.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import (
    DisruptionWindow,
    NoDisruptionConfig,
    ScheduledDisruptionConfig,
    SimulationConfig,
)
from leadtime.disruptions.base import Disruption
from leadtime.disruptions.none import NoDisruption
from leadtime.disruptions.registry import make_disruption
from leadtime.disruptions.scheduled import ScheduledDisruption


class TestMakeDisruption:
    def test_dispatches_none_config_to_NoDisruption_instance(self) -> None:
        d = make_disruption(NoDisruptionConfig(), np.random.default_rng(0))
        assert isinstance(d, NoDisruption)

    def test_dispatches_scheduled_config_to_ScheduledDisruption_instance(self) -> None:
        cfg = ScheduledDisruptionConfig(
            windows=[DisruptionWindow(start=10, duration=10, multiplier=2.0)]
        )
        d = make_disruption(cfg, np.random.default_rng(0))
        assert isinstance(d, ScheduledDisruption)

    def test_returns_disruption_abc_instance(self) -> None:
        d = make_disruption(NoDisruptionConfig(), np.random.default_rng(0))
        assert isinstance(d, Disruption)

    def test_factory_passthrough_none_matches_direct_construction(self) -> None:
        cfg = NoDisruptionConfig()
        for base, t in [(1, 0), (3, 1), (7, 10), (100, 100)]:
            factory = make_disruption(cfg, np.random.default_rng(123))
            direct = NoDisruption(cfg, np.random.default_rng(123))
            assert factory.apply(base, t) == direct.apply(base, t)

    def test_factory_passthrough_scheduled_matches_direct_construction(self) -> None:
        cfg = ScheduledDisruptionConfig(
            windows=[
                DisruptionWindow(start=10, duration=10, multiplier=2.0),
                DisruptionWindow(start=12, duration=6, multiplier=1.5),
            ]
        )
        factory = make_disruption(cfg, np.random.default_rng(123))
        direct = ScheduledDisruption(cfg, np.random.default_rng(123))
        for base in (1, 3, 5):
            for t in range(25):
                assert factory.apply(base, t) == direct.apply(base, t)

    def test_factory_accepts_rng_without_consuming(self) -> None:
        cfg = ScheduledDisruptionConfig(
            windows=[DisruptionWindow(start=10, duration=10, multiplier=2.0)]
        )
        rng = np.random.default_rng(42)
        state_before = rng.bit_generator.state
        d = make_disruption(cfg, rng)
        state_after_construct = rng.bit_generator.state
        for t in range(1000):
            d.apply(3, t)
        state_after_apply = rng.bit_generator.state
        assert state_before == state_after_construct == state_after_apply

    def test_unregistered_config_raises(self) -> None:
        # Defensive guard: a non-DisruptionConfig object raises ValueError — the
        # runtime safety net for a callsite that bypassed Pydantic discriminator
        # validation. Bypasses static typing via ``# type: ignore`` to exercise
        # the raise.
        fake = SimulationConfig(horizon=10, initial_on_hand=0)
        with pytest.raises(ValueError, match="Unknown disruption config"):
            make_disruption(fake, np.random.default_rng(0))  # type: ignore[arg-type]
