"""Demand factory: dispatch from ``DemandConfig`` to a concrete ``Demand``.

Mirrors :mod:`policies.registry`. The factory is the single construction
site for demand generators; the engine's ``InventoryEngine.__init__``
calls it instead of dispatching on ``config.demand.kind`` inline. Keeping
demand instantiation behind one function means the engine call site is
stable as M2 bullets add distribution variants — each new arm lands as
one additional ``isinstance`` arm here.

To add a new distribution in a future bullet:
    1. Add the ``*DemandConfig`` class to ``core.config`` and widen the
       ``DemandConfig`` union (with ``Field(discriminator="kind")``).
    2. Add the matching ``Demand`` subclass under ``demand/``.
    3. Add an ``isinstance`` arm here mapping the config to the class.
    4. Add a dispatch test in ``tests/unit/test_demand_registry.py``.
"""

from __future__ import annotations

import numpy as np

from core.config import (
    DemandConfig,
    EmpiricalDemandConfig,
    GammaDemandConfig,
    LognormalDemandConfig,
    NegativeBinomialDemandConfig,
    NormalDemandConfig,
    PoissonDemandConfig,
)
from demand.base import Demand
from demand.empirical import EmpiricalDemand
from demand.gamma import GammaDemand
from demand.lognormal import LognormalDemand
from demand.negative_binomial import NegativeBinomialDemand
from demand.normal import NormalDemand
from demand.poisson import PoissonDemand


def make_demand(config: DemandConfig, rng: np.random.Generator) -> Demand:
    """Return the ``Demand`` concrete instance for ``config`` bound to ``rng``.

    The caller is responsible for sourcing ``rng`` from the run's
    :class:`~core.seeding.SeedManager` (``seeds.rng("demand")``); the
    factory does not spawn or reseed. Raises ``ValueError`` if no arm is
    registered for the given config type — defensive guard against
    runtime misuse (e.g. a callsite that bypassed Pydantic discriminator
    validation).
    """
    if isinstance(config, NormalDemandConfig):
        return NormalDemand(config, rng)
    if isinstance(config, PoissonDemandConfig):
        return PoissonDemand(config, rng)
    if isinstance(config, NegativeBinomialDemandConfig):
        return NegativeBinomialDemand(config, rng)
    if isinstance(config, GammaDemandConfig):
        return GammaDemand(config, rng)
    if isinstance(config, LognormalDemandConfig):
        return LognormalDemand(config, rng)
    if isinstance(config, EmpiricalDemandConfig):
        return EmpiricalDemand(config, rng)
    raise ValueError(
        f"Unknown demand config type: {type(config).__name__}. "
        f"Register it in demand.registry.make_demand."
    )
