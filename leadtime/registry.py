"""Lead-time factory: dispatch from ``LeadTimeConfig`` to a concrete ``LeadTime``.

Mirrors :mod:`demand.registry` and :mod:`policies.registry`. The factory is the
single construction site for lead-time generators; the engine's
``InventoryEngine.__init__`` calls it instead of reading ``config.lead_time``
inline. Keeping lead-time instantiation behind one function means the engine
call site is stable as future bullets add stochastic / disruption variants —
each new arm lands as one additional ``isinstance`` arm here.

To add a new lead-time arm in a future bullet:
    1. Add the ``*LeadTimeConfig`` class to ``core.config`` and widen the
       ``LeadTimeConfig`` alias to a discriminated union
       (``Field(discriminator="kind")``).
    2. Add the matching ``LeadTime`` subclass under ``leadtime/``.
    3. Add an ``isinstance`` arm here mapping the config to the class.
    4. Add a dispatch test in ``tests/unit/test_leadtime_registry.py``.
"""

from __future__ import annotations

import numpy as np

from core.config import (
    DeterministicLeadTimeConfig,
    GammaLeadTimeConfig,
    LeadTimeConfig,
    LognormalLeadTimeConfig,
    NormalLeadTimeConfig,
)
from leadtime.base import LeadTime
from leadtime.deterministic import DeterministicLeadTime
from leadtime.gamma import GammaLeadTime
from leadtime.lognormal import LognormalLeadTime
from leadtime.normal import NormalLeadTime


def make_lead_time(config: LeadTimeConfig, rng: np.random.Generator) -> LeadTime:
    """Return the ``LeadTime`` concrete instance for ``config`` bound to ``rng``.

    The caller sources ``rng`` from the run's
    :class:`~core.seeding.SeedManager` (``seeds.rng("lead_time")``); the
    factory does not spawn or reseed.
    """
    if isinstance(config, DeterministicLeadTimeConfig):
        return DeterministicLeadTime(config, rng)
    if isinstance(config, NormalLeadTimeConfig):
        return NormalLeadTime(config, rng)
    if isinstance(config, GammaLeadTimeConfig):
        return GammaLeadTime(config, rng)
    if isinstance(config, LognormalLeadTimeConfig):
        return LognormalLeadTime(config, rng)
    raise ValueError(
        f"Unknown lead-time config type: {type(config).__name__}. "
        f"Register it in leadtime.registry.make_lead_time."
    )
