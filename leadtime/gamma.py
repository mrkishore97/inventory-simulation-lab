"""Gamma-distributed lead time, rounded to an integer number of periods."""

from __future__ import annotations

import numpy as np

from core.config import GammaLeadTimeConfig
from leadtime.base import LeadTime, round_to_periods


class GammaLeadTime(LeadTime):
    """Gamma lead time: each order's delay is ``round_to_periods(Gamma(shape, scale))``.

    The second stochastic lead-time arm — the textbook model for delivery
    delays (positive support, right-skewed). ``sample()`` draws a continuous
    ``X ~ Gamma(shape, scale)`` on ``(0, ∞)`` and converts it to an integer
    ``>= 1`` via the shared :func:`round_to_periods` helper. Gamma never
    produces a negative value, so the clip fires only when a sub-0.5 draw rounds
    to 0 (common for small ``shape × scale``). Parameter naming ``shape, scale``
    mirrors numpy's API and ``GammaDemandConfig`` (locked over textbook
    ``(k, θ)`` / rate ``(α, β)``).
    """

    def __init__(self, config: GammaLeadTimeConfig, rng: np.random.Generator) -> None:
        self._shape: float = float(config.shape)
        self._scale: float = float(config.scale)
        self._rng: np.random.Generator = rng

    def sample(self) -> int:
        return round_to_periods(self._rng.gamma(self._shape, self._scale))
