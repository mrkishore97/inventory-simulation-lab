"""Tests for the demand factory in ``demand.registry``.

The factory starts with a single Normal arm, then widens with Poisson
— the defensive raise is now reachable and tested. NegBin, Gamma, and
Lognormal follow, closing the demand quartet. The Empirical arm is
last (first non-parametric arm; 5→6 widening). Each arm adds
one dispatch test plus an entry in the factory-passthrough property
suite below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.config import (
    EmpiricalDemandConfig,
    GammaDemandConfig,
    LognormalDemandConfig,
    NegativeBinomialDemandConfig,
    NormalDemandConfig,
    PoissonDemandConfig,
    SimulationConfig,
)
from demand.base import Demand
from demand.empirical import EmpiricalDemand
from demand.gamma import GammaDemand
from demand.lognormal import LognormalDemand
from demand.negative_binomial import NegativeBinomialDemand
from demand.normal import NormalDemand
from demand.poisson import PoissonDemand
from demand.registry import make_demand


class TestMakeDemand:
    def test_dispatches_normal_config_to_NormalDemand_instance(self) -> None:
        cfg = NormalDemandConfig(mean=10.0, std=2.0)
        demand = make_demand(cfg, np.random.default_rng(0))
        assert isinstance(demand, NormalDemand)

    def test_dispatches_poisson_config_to_PoissonDemand_instance(self) -> None:
        cfg = PoissonDemandConfig(rate=10.0)
        demand = make_demand(cfg, np.random.default_rng(0))
        assert isinstance(demand, PoissonDemand)

    def test_dispatches_negative_binomial_config_to_NegativeBinomialDemand_instance(
        self,
    ) -> None:
        cfg = NegativeBinomialDemandConfig(n=10.0, p=0.5)
        demand = make_demand(cfg, np.random.default_rng(0))
        assert isinstance(demand, NegativeBinomialDemand)

    def test_dispatches_gamma_config_to_GammaDemand_instance(self) -> None:
        cfg = GammaDemandConfig(shape=5.0, scale=2.0)
        demand = make_demand(cfg, np.random.default_rng(0))
        assert isinstance(demand, GammaDemand)

    def test_dispatches_lognormal_config_to_LognormalDemand_instance(self) -> None:
        cfg = LognormalDemandConfig(mu=2.2114, sigma=0.4271)
        demand = make_demand(cfg, np.random.default_rng(0))
        assert isinstance(demand, LognormalDemand)

    def test_dispatches_empirical_config_to_EmpiricalDemand_instance(self, tmp_path: Path) -> None:
        path = tmp_path / "h.parquet"
        pd.DataFrame({"demand": np.arange(1.0, 100.0)}).to_parquet(path, index=False)
        cfg = EmpiricalDemandConfig(history_path=path)
        demand = make_demand(cfg, np.random.default_rng(0))
        assert isinstance(demand, EmpiricalDemand)

    def test_returns_demand_abc_instance(self) -> None:
        cfg = NormalDemandConfig(mean=10.0, std=2.0)
        demand = make_demand(cfg, np.random.default_rng(0))
        assert isinstance(demand, Demand)

    def test_factory_passthrough_normal_matches_direct_construction(self) -> None:
        """Pin: factory adds zero behavior beyond direct construction.

        If a future bullet ever wraps the factory output (decorator,
        adapter, telemetry shim), this test fails immediately and forces
        an explicit decision instead of silent behavior change.
        """
        for mean, std in [(0.5, 0.1), (10.0, 2.0), (100.0, 25.0)]:
            cfg = NormalDemandConfig(mean=mean, std=std)
            factory_demand = make_demand(cfg, np.random.default_rng(123))
            direct_demand = NormalDemand(cfg, np.random.default_rng(123))
            for i in range(50):
                assert factory_demand.draw() == direct_demand.draw(), (
                    f"factory diverged from direct construction at mean={mean}, "
                    f"std={std}, draw #{i}"
                )

    def test_factory_passthrough_poisson_matches_direct_construction(self) -> None:
        """Same passthrough property, Poisson variant."""
        for rate in [0.1, 1.0, 10.0, 100.0]:
            cfg = PoissonDemandConfig(rate=rate)
            factory_demand = make_demand(cfg, np.random.default_rng(123))
            direct_demand = PoissonDemand(cfg, np.random.default_rng(123))
            for i in range(50):
                assert factory_demand.draw() == direct_demand.draw(), (
                    f"factory diverged from direct construction at rate={rate}, draw #{i}"
                )

    def test_factory_passthrough_negative_binomial_matches_direct_construction(
        self,
    ) -> None:
        """Same passthrough property, NegBin variant — grid of (n, p) pairs."""
        for n, p in [(1.0, 0.1), (5.0, 0.5), (10.0, 0.9), (50.0, 0.3)]:
            cfg = NegativeBinomialDemandConfig(n=n, p=p)
            factory_demand = make_demand(cfg, np.random.default_rng(123))
            direct_demand = NegativeBinomialDemand(cfg, np.random.default_rng(123))
            for i in range(50):
                assert factory_demand.draw() == direct_demand.draw(), (
                    f"factory diverged from direct construction at n={n}, p={p}, draw #{i}"
                )

    def test_factory_passthrough_gamma_matches_direct_construction(self) -> None:
        """Same passthrough property, Gamma variant — grid of (shape, scale) pairs.

        Exercises the heavy-tail mode-at-zero regime (shape < 1), the
        exponential special case (shape = 1), and moderate-shape /
        high-shape regimes — the recipe must pass through identically
        across Gamma's parameter space.
        """
        for shape, scale in [(0.5, 1.0), (1.0, 1.0), (5.0, 2.0), (10.0, 0.5)]:
            cfg = GammaDemandConfig(shape=shape, scale=scale)
            factory_demand = make_demand(cfg, np.random.default_rng(123))
            direct_demand = GammaDemand(cfg, np.random.default_rng(123))
            for i in range(50):
                assert factory_demand.draw() == direct_demand.draw(), (
                    f"factory diverged from direct construction at "
                    f"shape={shape}, scale={scale}, draw #{i}"
                )

    def test_factory_passthrough_lognormal_matches_direct_construction(self) -> None:
        """Same passthrough property, Lognormal variant — grid of (mu, sigma) pairs.

        Exercises the negative-mu slow-moving regime, the unit-mu /
        unit-sigma reference case, the moment-matched scenario
        parameters (mu=2.2114, sigma=0.4271), and a low-mu small-sigma
        concentrated case — the recipe must pass through identically
        across Lognormal's parameter space, including the negative-mu
        corner (Lognormal is the only demand arm where the first field
        accepts negative values).
        """
        for mu, sigma in [(0.0, 0.5), (1.0, 1.0), (2.2114, 0.4271), (-1.0, 0.3)]:
            cfg = LognormalDemandConfig(mu=mu, sigma=sigma)
            factory_demand = make_demand(cfg, np.random.default_rng(123))
            direct_demand = LognormalDemand(cfg, np.random.default_rng(123))
            for i in range(50):
                assert factory_demand.draw() == direct_demand.draw(), (
                    f"factory diverged from direct construction at "
                    f"mu={mu}, sigma={sigma}, draw #{i}"
                )

    def test_factory_passthrough_empirical_matches_direct_construction(
        self, tmp_path: Path
    ) -> None:
        """Same passthrough property, Empirical variant.

        Empirical is the first non-parametric arm; the test asserts the
        factory output produces the same sequence as direct construction
        across a range of history shapes (small / medium / large).
        """
        for size in [10, 100, 1000]:
            history = np.random.default_rng(size).uniform(0.0, 20.0, size=size)
            path = tmp_path / f"h_{size}.parquet"
            pd.DataFrame({"demand": history}).to_parquet(path, index=False)
            cfg = EmpiricalDemandConfig(history_path=path)
            factory_demand = make_demand(cfg, np.random.default_rng(123))
            direct_demand = EmpiricalDemand(cfg, np.random.default_rng(123))
            for i in range(50):
                assert factory_demand.draw() == direct_demand.draw(), (
                    f"factory diverged from direct construction at size={size}, draw #{i}"
                )

    def test_unregistered_config_raises(self) -> None:
        """Defensive guard: a non-DemandConfig object raises ValueError.

        Now reachable post-Bullet-6 (the union has ≥2 arms; constructing
        a non-demand BaseModel and passing it through the factory hits
        the runtime safety net). Bypasses static typing via
        ``# type: ignore`` to exercise the raise.
        """
        fake = SimulationConfig(horizon=10, initial_on_hand=0)
        with pytest.raises(ValueError, match="Unknown demand config"):
            make_demand(fake, np.random.default_rng(0))  # type: ignore[arg-type]
