"""Tests for ``leadtime.registry.make_lead_time``."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import BaseModel

from core.config import (
    DeterministicLeadTimeConfig,
    GammaLeadTimeConfig,
    LognormalLeadTimeConfig,
    NormalLeadTimeConfig,
)
from leadtime.deterministic import DeterministicLeadTime
from leadtime.gamma import GammaLeadTime
from leadtime.lognormal import LognormalLeadTime
from leadtime.normal import NormalLeadTime
from leadtime.registry import make_lead_time


class TestMakeLeadTime:
    def test_dispatches_deterministic_config_to_DeterministicLeadTime(self) -> None:
        cfg = DeterministicLeadTimeConfig(lead_time=3)
        lt = make_lead_time(cfg, np.random.default_rng(0))
        assert isinstance(lt, DeterministicLeadTime)

    def test_dispatches_normal_config_to_NormalLeadTime(self) -> None:
        cfg = NormalLeadTimeConfig(mean=3.0, std=1.0)
        lt = make_lead_time(cfg, np.random.default_rng(0))
        assert isinstance(lt, NormalLeadTime)

    def test_factory_passthrough_matches_direct_construction(self) -> None:
        # The factory must stay a passthrough: a future bullet that wraps the
        # output (adapter, telemetry) breaks this loudly. Mirrors make_demand /
        # make_policy passthrough tests.
        cfg = DeterministicLeadTimeConfig(lead_time=7)
        factory = make_lead_time(cfg, np.random.default_rng(0))
        direct = DeterministicLeadTime(cfg, np.random.default_rng(0))
        assert factory.sample() == direct.sample()

    def test_factory_passthrough_normal_matches_direct(self) -> None:
        # Normal consumes the RNG, so compare draw sequences under a fixed seed.
        cfg = NormalLeadTimeConfig(mean=5.0, std=2.0)
        factory = make_lead_time(cfg, np.random.default_rng(0))
        direct = NormalLeadTime(cfg, np.random.default_rng(0))
        assert [factory.sample() for _ in range(50)] == [direct.sample() for _ in range(50)]

    def test_dispatches_gamma_config_to_GammaLeadTime(self) -> None:
        cfg = GammaLeadTimeConfig(shape=9.0, scale=1.0 / 3.0)
        lt = make_lead_time(cfg, np.random.default_rng(0))
        assert isinstance(lt, GammaLeadTime)

    def test_factory_passthrough_gamma_matches_direct(self) -> None:
        cfg = GammaLeadTimeConfig(shape=4.0, scale=0.75)
        factory = make_lead_time(cfg, np.random.default_rng(0))
        direct = GammaLeadTime(cfg, np.random.default_rng(0))
        assert [factory.sample() for _ in range(50)] == [direct.sample() for _ in range(50)]

    def test_dispatches_lognormal_config_to_LognormalLeadTime(self) -> None:
        cfg = LognormalLeadTimeConfig(mu=1.045931, sigma=0.324593)
        lt = make_lead_time(cfg, np.random.default_rng(0))
        assert isinstance(lt, LognormalLeadTime)

    def test_factory_passthrough_lognormal_matches_direct(self) -> None:
        cfg = LognormalLeadTimeConfig(mu=1.2, sigma=0.4)
        factory = make_lead_time(cfg, np.random.default_rng(0))
        direct = LognormalLeadTime(cfg, np.random.default_rng(0))
        assert [factory.sample() for _ in range(50)] == [direct.sample() for _ in range(50)]

    def test_unregistered_config_raises(self) -> None:
        # The defensive raise became reachable when LeadTimeConfig widened to a
        # 2-arm union; pragma retired. A config type with no arm
        # raises ValueError carrying the offending type name.
        class _FakeLeadTimeConfig(BaseModel):
            pass

        with pytest.raises(ValueError, match="_FakeLeadTimeConfig"):
            make_lead_time(_FakeLeadTimeConfig(), np.random.default_rng(0))  # type: ignore[arg-type]
