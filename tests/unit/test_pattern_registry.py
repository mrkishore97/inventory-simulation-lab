"""Tests for the pattern factory in ``demand.patterns.registry``.

The factory starts with a single Stationary arm, then widens to 2 arms
by registering Seasonal, then to 3 arms
by registering Trending. The defensive raise is exercised by passing a
non-pattern config type.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.config import (
    IntermittentPatternConfig,
    LumpyPatternConfig,
    SeasonalPatternConfig,
    SimulationConfig,
    StationaryPatternConfig,
    TrendingPatternConfig,
)
from demand.patterns.base import Pattern
from demand.patterns.intermittent import IntermittentPattern
from demand.patterns.lumpy import LumpyPattern
from demand.patterns.registry import make_pattern
from demand.patterns.seasonal import SeasonalPattern
from demand.patterns.stationary import StationaryPattern
from demand.patterns.trending import TrendingPattern


class TestMakePattern:
    def test_dispatches_stationary_config_to_StationaryPattern_instance(self) -> None:
        cfg = StationaryPatternConfig()
        pattern = make_pattern(cfg, np.random.default_rng(0))
        assert isinstance(pattern, StationaryPattern)

    def test_returns_pattern_abc_instance(self) -> None:
        cfg = StationaryPatternConfig()
        pattern = make_pattern(cfg, np.random.default_rng(0))
        assert isinstance(pattern, Pattern)

    def test_factory_passthrough_stationary_matches_direct_construction(self) -> None:
        """Pin: factory adds zero behavior beyond direct construction.

        If a future bullet ever wraps the factory output (decorator,
        adapter, telemetry shim), this test fails immediately and forces
        an explicit decision instead of silent behavior change.
        """
        cfg = StationaryPatternConfig()
        for base, t in [(0.0, 0), (1.0, 1), (3.7, 10), (100.0, 100), (1e-9, 0), (1e9, 1)]:
            factory_pattern = make_pattern(cfg, np.random.default_rng(123))
            direct_pattern = StationaryPattern(cfg, np.random.default_rng(123))
            assert factory_pattern.apply(base, t) == direct_pattern.apply(base, t), (
                f"factory diverged from direct construction at base={base}, t={t}"
            )

    def test_factory_accepts_rng_without_consuming(self) -> None:
        """The factory accepts an rng argument; Stationary's apply() does not
        advance the rng's state.

        Pinned via ``bit_generator.state`` snapshot — byte-for-byte identical
        before construction, after construction, and after 1000 apply() calls.
        """
        cfg = StationaryPatternConfig()
        rng = np.random.default_rng(42)
        state_before = rng.bit_generator.state
        pattern = make_pattern(cfg, rng)
        state_after_construct = rng.bit_generator.state
        for i in range(1000):
            pattern.apply(float(i), i)
        state_after_apply = rng.bit_generator.state
        assert state_before == state_after_construct == state_after_apply

    def test_unregistered_config_raises(self) -> None:
        """Defensive guard: a non-PatternConfig object raises ValueError.

        Post-Bullet-11, the union has 2 arms; the defensive raise remains
        the runtime safety net for any future config that bypasses
        Pydantic discriminator validation (or for a Bullet-12+ regression
        where a new arm is added but the registry isn't updated).
        Bypasses static typing via ``# type: ignore`` to exercise the raise.
        """
        fake = SimulationConfig(horizon=10, initial_on_hand=0)
        with pytest.raises(ValueError, match="Unknown pattern config"):
            make_pattern(fake, np.random.default_rng(0))  # type: ignore[arg-type]

    def test_dispatches_seasonal_config_to_SeasonalPattern_instance(self) -> None:
        cfg = SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)
        pattern = make_pattern(cfg, np.random.default_rng(0))
        assert isinstance(pattern, SeasonalPattern)

    def test_factory_passthrough_seasonal_matches_direct_construction(self) -> None:
        """Same passthrough property as Stationary, exercised across a
        grid of (amplitude, period, phase) combinations to cover the
        parameter space corners: amplitude=0 (degenerate-to-identity),
        moderate amplitude with phase=0 (scenario default), moderate
        amplitude with phase=π/2 (peak-at-zero), high amplitude with
        negative phase. The factory output's apply() must produce
        bit-identical results to direct construction across all combinations.
        """
        for amp, period, phase in [
            (0.0, 4, 0.0),
            (0.5, 12, 0.0),
            (0.5, 12, math.pi / 2),
            (0.9, 30, -math.pi / 4),
        ]:
            cfg = SeasonalPatternConfig(amplitude=amp, period=period, phase=phase)
            factory_pattern = make_pattern(cfg, np.random.default_rng(123))
            direct_pattern = SeasonalPattern(cfg, np.random.default_rng(123))
            for base, t in [(0.0, 0), (1.0, 1), (10.0, 5), (100.0, 100), (1e-9, 0)]:
                assert factory_pattern.apply(base, t) == direct_pattern.apply(base, t), (
                    f"factory diverged from direct construction at "
                    f"amp={amp}, period={period}, phase={phase}, base={base}, t={t}"
                )

    def test_dispatches_trending_config_to_TrendingPattern_instance(self) -> None:
        cfg = TrendingPatternConfig(slope=0.02)
        pattern = make_pattern(cfg, np.random.default_rng(0))
        assert isinstance(pattern, TrendingPattern)

    def test_factory_passthrough_trending_matches_direct_construction(self) -> None:
        """Same passthrough property as Stationary and Seasonal, exercised
        across a grid of slope values that span the parameter corners:
        slope=0 (degenerate-to-identity), small positive (mild growth),
        small negative (mild decline), moderate positive (scenario default),
        large positive, and steep negative (exercises clip at high t).
        """
        for slope in (-0.05, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1):
            cfg = TrendingPatternConfig(slope=slope)
            factory_pattern = make_pattern(cfg, np.random.default_rng(123))
            direct_pattern = TrendingPattern(cfg, np.random.default_rng(123))
            for base, t in [(0.0, 0), (1.0, 1), (10.0, 5), (100.0, 100), (1e-9, 0), (10.0, 1000)]:
                assert factory_pattern.apply(base, t) == direct_pattern.apply(base, t), (
                    f"factory diverged from direct construction at "
                    f"slope={slope}, base={base}, t={t}"
                )

    def test_dispatches_intermittent_config_to_IntermittentPattern_instance(self) -> None:
        cfg = IntermittentPatternConfig(occurrence_probability=0.6)
        pattern = make_pattern(cfg, np.random.default_rng(0))
        assert isinstance(pattern, IntermittentPattern)

    def test_factory_passthrough_intermittent_matches_direct_construction(self) -> None:
        """Same passthrough property as the deterministic arms, for the first STOCHASTIC arm:
        two instances seeded identically draw in lockstep, so the factory output's apply()
        sequence is bit-identical to direct construction's across the probability corners
        (small, mid, scenario default, and the p=1 degenerate-identity).
        """
        for p in (0.1, 0.5, 0.6, 1.0):
            cfg = IntermittentPatternConfig(occurrence_probability=p)
            factory_pattern = make_pattern(cfg, np.random.default_rng(123))
            direct_pattern = IntermittentPattern(cfg, np.random.default_rng(123))
            for t in range(50):
                assert factory_pattern.apply(10.0, t) == direct_pattern.apply(10.0, t), (
                    f"factory diverged from direct construction at p={p}, t={t}"
                )

    def test_dispatches_lumpy_config_to_LumpyPattern_instance(self) -> None:
        cfg = LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=5.0)
        pattern = make_pattern(cfg, np.random.default_rng(0))
        assert isinstance(pattern, LumpyPattern)

    def test_factory_passthrough_lumpy_matches_direct_construction(self) -> None:
        # Stochastic arm: identically-seeded instances draw in lockstep, so the factory output's
        # apply() sequence is bit-identical to direct construction's across the param corners.
        for p, m in [(0.1, 2.0), (0.3, 5.0), (0.6, 1.5), (1.0, 10.0)]:
            cfg = LumpyPatternConfig(occurrence_probability=p, burst_multiplier=m)
            factory_pattern = make_pattern(cfg, np.random.default_rng(123))
            direct_pattern = LumpyPattern(cfg, np.random.default_rng(123))
            for t in range(50):
                assert factory_pattern.apply(10.0, t) == direct_pattern.apply(10.0, t), (
                    f"factory diverged from direct construction at p={p}, m={m}, t={t}"
                )
