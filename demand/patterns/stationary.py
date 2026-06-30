"""Stationary (identity) pattern — the chassis arm of the pattern engine."""

from __future__ import annotations

import numpy as np

from core.config import StationaryPatternConfig
from demand.patterns.base import Pattern


class StationaryPattern(Pattern):
    """Identity arm of the pattern engine — pass-through, no time modulation.

    ``apply(base, t)`` returns ``base`` unchanged for every ``(base, t)``.
    Constructed with an ``np.random.Generator`` for uniform API with future
    stochastic patterns (Intermittent / Lumpy in Bullets 13-14, which DO
    consume RNG), but ``StationaryPattern`` is fully stateless and never
    calls into the RNG.

    The chassis arm of the pattern engine and the Pydantic-level default for
    ``RunConfig.pattern``. Combined with the locked default-factory on
    ``RunConfig``, every existing scenario YAML continues to produce
    byte-for-byte identical demand streams without any schema change.

    Identity is **unconditional**: ``apply(base, t) == base`` for ALL base
    (including negative). Stationary does not police input; non-negativity
    is the BASE distribution's responsibility (which all 5 demand arms
    unconditionally satisfy, so identity never observes a negative value
    at runtime).
    """

    def __init__(self, config: StationaryPatternConfig, rng: np.random.Generator) -> None:
        # Accept rng for uniform API with future stochastic patterns; never
        # actually draws from it. The test class pins this — the rng state
        # is byte-identical before and after any number of apply() calls.
        self._rng: np.random.Generator = rng

    def apply(self, base: float, t: int) -> float:
        return base
