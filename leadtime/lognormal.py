"""Lognormal-distributed lead time, rounded to an integer number of periods."""

from __future__ import annotations

import numpy as np

from core.config import LognormalLeadTimeConfig
from leadtime.base import LeadTime, round_to_periods


class LognormalLeadTime(LeadTime):
    """Lognormal lead time: ``round_to_periods(Lognormal(mu, sigma))``.

    The third and final stochastic lead-time arm — closes the trio (Normal,
    Gamma, Lognormal), mirroring the demand quartet's continuous arms. Heavy
    right tail: delivery delays that are usually short but occasionally very
    long. ``sample()`` draws a continuous ``X ~ Lognormal(mu, sigma)`` on
    ``(0, ∞)`` and converts it to an integer ``>= 1`` via the shared
    :func:`round_to_periods` helper. Lognormal never produces a negative value,
    so the clip fires only when a sub-0.5 draw rounds to 0 (common when ``mu``
    is small or negative).

    Parameter-naming foot-gun (same as ``LognormalDemand``): ``mu, sigma`` are
    the parameters of the *underlying* Normal, not the resulting Lognormal's
    moments. The numpy call maps literally — ``rng.lognormal(self._mu,
    self._sigma)`` — but the config surface uses the unambiguous textbook names.
    """

    def __init__(self, config: LognormalLeadTimeConfig, rng: np.random.Generator) -> None:
        self._mu: float = float(config.mu)
        self._sigma: float = float(config.sigma)
        self._rng: np.random.Generator = rng

    def sample(self) -> int:
        return round_to_periods(self._rng.lognormal(self._mu, self._sigma))
