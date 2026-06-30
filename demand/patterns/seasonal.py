"""Seasonal (sinusoidal multiplicative) pattern — first concrete arm of ``PatternConfig``."""

from __future__ import annotations

import math

import numpy as np

from core.config import SeasonalPatternConfig
from demand.patterns.base import Pattern


class SeasonalPattern(Pattern):
    """Sinusoidal multiplicative seasonal demand modulation.

    ``apply(base, t)`` returns ``base * (1 + amplitude * sin(2π·t/period + phase))``.
    The textbook seasonal decomposition (Box-Jenkins, Hyndman-Athanasopoulos):
    seasonality scales with demand level (multiplicative), mean is preserved
    over a full cycle (sin averages to 0), and the factor range under the
    locked ``amplitude ∈ [0, 1)`` bound is strictly ``(1 - amplitude, 1 +
    amplitude) ⊂ (0, 2)`` — non-negativity is guaranteed by the parameter
    constraint, **no runtime clip in apply()**.

    Uses ``math.sin`` and ``math.pi`` (Python stdlib) rather than
    ``numpy.sin`` / ``numpy.pi``: ``math.sin`` returns Python ``float``
    natively for scalar inputs, so no ``float(...)`` cast is needed at the
    boundary. Reserve numpy for vectorized or RNG-coupled operations.

    Constructor accepts an ``np.random.Generator`` for uniform API with
    future stochastic patterns (Intermittent / Lumpy,
    which DO consume RNG), but ``SeasonalPattern`` is fully deterministic
    and never draws from the rng. The test class pins this — the rng state
    is byte-identical before and after any number of ``apply()`` calls.
    """

    def __init__(self, config: SeasonalPatternConfig, rng: np.random.Generator) -> None:
        self._amplitude: float = float(config.amplitude)
        self._period: int = int(config.period)
        self._phase: float = float(config.phase)
        # Accept rng for uniform API with future stochastic patterns; never
        # actually draws from it.
        self._rng: np.random.Generator = rng

    def apply(self, base: float, t: int) -> float:
        angle = 2.0 * math.pi * t / self._period + self._phase
        return base * (1.0 + self._amplitude * math.sin(angle))
