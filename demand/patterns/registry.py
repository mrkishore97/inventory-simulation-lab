"""Pattern factory: dispatch from ``PatternConfig`` to a concrete ``Pattern``.

Mirrors :mod:`demand.registry`. The factory is the single construction
site for pattern overlays; the engine's ``InventoryEngine.__init__`` calls
it instead of dispatching on ``config.pattern.kind`` inline. Keeping
pattern instantiation behind one function means the engine call site is
stable as M2 bullets add pattern variants — each new arm lands as one
additional ``isinstance`` arm here.

To add a new pattern in a future bullet:
    1. Add the ``*PatternConfig`` class to ``core.config`` and widen the
       ``PatternConfig`` union (with ``Field(discriminator="kind")``).
    2. Add the matching ``Pattern`` subclass under ``demand/patterns/``.
    3. Add an ``isinstance`` arm here mapping the config to the class.
    4. Add a dispatch test in ``tests/unit/test_pattern_registry.py``.
"""

from __future__ import annotations

import numpy as np

from core.config import (
    IntermittentPatternConfig,
    LumpyPatternConfig,
    PatternConfig,
    SeasonalPatternConfig,
    StationaryPatternConfig,
    TrendingPatternConfig,
)
from demand.patterns.base import Pattern
from demand.patterns.intermittent import IntermittentPattern
from demand.patterns.lumpy import LumpyPattern
from demand.patterns.seasonal import SeasonalPattern
from demand.patterns.stationary import StationaryPattern
from demand.patterns.trending import TrendingPattern


def make_pattern(config: PatternConfig, rng: np.random.Generator) -> Pattern:
    """Return the ``Pattern`` concrete instance for ``config`` bound to ``rng``.

    The caller is responsible for sourcing ``rng`` from the run's
    :class:`~core.seeding.SeedManager` (``seeds.rng("pattern")``); the
    factory does not spawn or reseed. Raises ``ValueError`` if no arm is
    registered for the given config type — defensive guard against runtime
    misuse (e.g. a callsite that bypassed Pydantic discriminator
    validation).
    """
    if isinstance(config, StationaryPatternConfig):
        return StationaryPattern(config, rng)
    if isinstance(config, SeasonalPatternConfig):
        return SeasonalPattern(config, rng)
    if isinstance(config, TrendingPatternConfig):
        return TrendingPattern(config, rng)
    if isinstance(config, IntermittentPatternConfig):
        return IntermittentPattern(config, rng)
    if isinstance(config, LumpyPatternConfig):
        return LumpyPattern(config, rng)
    raise ValueError(
        f"Unknown pattern config type: {type(config).__name__}. "
        f"Register it in demand.patterns.registry.make_pattern."
    )
