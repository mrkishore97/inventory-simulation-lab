"""Lumpy (Bernoulli burst-mask) pattern — fourth concrete arm; closes the M2 pattern set."""

from __future__ import annotations

import numpy as np

from core.config import LumpyPatternConfig
from demand.patterns.base import Pattern


class LumpyPattern(Pattern):
    """Bernoulli burst-mask lumpy demand modulation.

    ``apply(base, t)`` returns ``base * burst_multiplier`` with probability
    ``occurrence_probability`` and ``0.0`` otherwise — the sibling of
    :class:`~demand.patterns.intermittent.IntermittentPattern`, but the kept
    value is *amplified* by ``burst_multiplier`` (> 1). Models the colloquial
    lumpy-demand shape: long runs of no-demand periods punctuated by occasional
    large bursts (spare parts, capital-goods orders, promotional spikes).

    **SB caveat:** a *constant* ``burst_multiplier`` scales the magnitude of the
    nonzero periods but does NOT by itself raise the **CV² of the nonzero demand
    sizes** (nonzero size = ``base * m``, so its CV² equals the base's). So the
    realized Syntetos-Boylan quadrant depends on the base distribution's CV² —
    a low-variance base (e.g. Normal(10, 2)) yields sparse *amplified* bursts
    that may still classify as *intermittent* rather than *lumpy*. This arm is
    the colloquial lumpy shape, not a guaranteed SB-lumpy generator.

    Second stochastic pattern arm (after Intermittent). ``apply()`` draws
    **exactly one** uniform (``self._rng.random()``) *before* the burst/zero
    branch, so the seeded ``"pattern"`` stream advances by one per period
    regardless of outcome. The ``"pattern"`` and ``"demand"`` streams are
    independent SeedManager children, so for a fixed ``master_seed`` a Lumpy run
    fires on exactly the same periods as an Intermittent run with the same
    ``occurrence_probability`` — with the kept demand scaled by
    ``burst_multiplier`` (``lumpy.demand == intermittent.demand * m``).

    Non-negativity is unconditional: the output is ``base * m`` (non-negative by
    the ``Demand`` contract, ``m > 1``) or ``0.0`` — no clip needed.
    """

    def __init__(self, config: LumpyPatternConfig, rng: np.random.Generator) -> None:
        self._occurrence_probability: float = float(config.occurrence_probability)
        self._burst_multiplier: float = float(config.burst_multiplier)
        self._rng: np.random.Generator = rng

    def apply(self, base: float, t: int) -> float:
        # One uniform draw per period, BEFORE the branch, so the "pattern" stream
        # advances by exactly one regardless of the burst/zero outcome.
        if self._rng.random() < self._occurrence_probability:
            return base * self._burst_multiplier
        return 0.0
