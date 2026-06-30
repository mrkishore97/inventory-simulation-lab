"""Normal-distributed lead time, rounded to an integer number of periods."""

from __future__ import annotations

import numpy as np

from core.config import NormalLeadTimeConfig
from leadtime.base import LeadTime, round_to_periods


class NormalLeadTime(LeadTime):
    """Normal lead time: each order's delay is ``round_to_periods(N(mean, std))``.

    The first RNG-consuming lead-time arm (``DeterministicLeadTime`` holds but
    never draws). ``sample()`` draws a continuous ``X ~ Normal(mean, std)`` and
    converts it to an integer ``>= 1`` via the shared :func:`round_to_periods`
    helper (round-to-nearest, clip-≥1). Round-to-nearest keeps the realized lead
    time centred on the configured ``mean`` (away from the clip boundary); the
    clip also absorbs the negative tail of the Normal.
    """

    def __init__(self, config: NormalLeadTimeConfig, rng: np.random.Generator) -> None:
        self._mean: float = float(config.mean)
        self._std: float = float(config.std)
        self._rng: np.random.Generator = rng

    def sample(self) -> int:
        return round_to_periods(self._rng.normal(self._mean, self._std))
