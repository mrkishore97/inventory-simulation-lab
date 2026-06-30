"""Disruption factory: dispatch from ``DisruptionConfig`` to a concrete ``Disruption``.

Mirrors :mod:`demand.patterns.registry`. The factory is the single construction
site for disruption overlays; the engine's ``InventoryEngine.__init__`` calls it
instead of dispatching on ``config.disruption.kind`` inline. Keeping disruption
instantiation behind one function means the engine call site is stable as future
bullets add disruption variants (e.g. a stochastic disruption generator) — each
new arm lands as one additional ``isinstance`` arm here.

To add a new disruption arm in a future bullet:
    1. Add the ``*DisruptionConfig`` class to ``core.config`` and widen the
       ``DisruptionConfig`` union (with ``Field(discriminator="kind")``).
    2. Add the matching ``Disruption`` subclass under ``leadtime/disruptions/``.
    3. Add an ``isinstance`` arm here mapping the config to the class.
    4. Add a dispatch test in ``tests/unit/test_disruption_registry.py``.
"""

from __future__ import annotations

import numpy as np

from core.config import (
    DisruptionConfig,
    NoDisruptionConfig,
    ScheduledDisruptionConfig,
)
from leadtime.disruptions.base import Disruption
from leadtime.disruptions.none import NoDisruption
from leadtime.disruptions.scheduled import ScheduledDisruption


def make_disruption(config: DisruptionConfig, rng: np.random.Generator) -> Disruption:
    """Return the ``Disruption`` concrete instance for ``config`` bound to ``rng``.

    The caller sources ``rng`` from the run's
    :class:`~core.seeding.SeedManager` (``seeds.rng("disruptions")``); the
    factory does not spawn or reseed. Raises ``ValueError`` if no arm is
    registered for the given config type — defensive guard against runtime
    misuse (e.g. a callsite that bypassed Pydantic discriminator validation).
    """
    if isinstance(config, NoDisruptionConfig):
        return NoDisruption(config, rng)
    if isinstance(config, ScheduledDisruptionConfig):
        return ScheduledDisruption(config, rng)
    raise ValueError(
        f"Unknown disruption config type: {type(config).__name__}. "
        f"Register it in leadtime.disruptions.registry.make_disruption."
    )
