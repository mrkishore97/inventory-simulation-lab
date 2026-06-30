"""Intermittent (Bernoulli zero-mask) pattern — third concrete arm, first stochastic arm."""

from __future__ import annotations

import numpy as np

from core.config import IntermittentPatternConfig
from demand.patterns.base import Pattern


class IntermittentPattern(Pattern):
    """Bernoulli zero-mask intermittent demand modulation.

    ``apply(base, t)`` returns ``base`` with probability ``occurrence_probability``
    and ``0.0`` otherwise — masking the base draw to zero on a random subset of
    periods to model intermittent demand (slow-moving SKUs, spare parts,
    low-velocity items with many no-demand periods).

    **First stochastic pattern arm.** Stationary / Seasonal / Trending accept an
    rng for uniform API but never draw; this arm consumes the seeded ``"pattern"``
    stream the chassis reserved. ``apply()`` draws **exactly one**
    uniform (``self._rng.random()``) *before* the keep/zero branch, so the stream
    advances deterministically by one per period regardless of outcome — the
    keep/zero decision never changes how much randomness is consumed.

    The ``"pattern"`` and ``"demand"`` streams are independent children of the
    master :class:`~core.seeding.SeedManager`, so for a fixed ``master_seed`` the
    base demand sequence is identical to a Stationary run's; intermittent demand
    is exactly that sequence with some periods zeroed (``apply(base, t) ∈
    {0.0, base}``).

    Non-negativity is unconditional: the output is ``base`` (non-negative by the
    ``Demand`` contract) or ``0.0`` — no clip needed (the Stationary
    unconditional-passthrough path, not the Trending clip-at-source path).
    """

    def __init__(self, config: IntermittentPatternConfig, rng: np.random.Generator) -> None:
        self._occurrence_probability: float = float(config.occurrence_probability)
        self._rng: np.random.Generator = rng

    def apply(self, base: float, t: int) -> float:
        # One uniform draw per period, BEFORE the branch, so the "pattern" stream
        # advances by exactly one regardless of the keep/zero outcome.
        if self._rng.random() < self._occurrence_probability:
            return base
        return 0.0
