"""Deterministic (fixed) lead time."""

from __future__ import annotations

import numpy as np

from core.config import DeterministicLeadTimeConfig
from leadtime.base import LeadTime


class DeterministicLeadTime(LeadTime):
    """Fixed lead time: every order arrives a constant number of periods later.

    ``sample()`` returns the configured integer ``lead_time`` on every call
    and does NOT draw from the RNG. The ``rng`` is accepted for a uniform
    factory API with future stochastic arms (which DO consume it) but is never
    used here — this is what makes the chassis extraction
    byte-equivalent to the pre-extraction engine: the ``"lead_time"`` stream
    stays untouched, exactly as before. Mirrors ``StationaryPattern``'s
    accept-but-never-draw contract.
    """

    def __init__(self, config: DeterministicLeadTimeConfig, rng: np.random.Generator) -> None:
        self._lead_time: int = config.lead_time
        self._rng: np.random.Generator = rng

    def sample(self) -> int:
        return self._lead_time
