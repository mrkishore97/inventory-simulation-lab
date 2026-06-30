"""No-op (identity) disruption — the chassis arm of the disruption channel."""

from __future__ import annotations

import numpy as np

from core.config import NoDisruptionConfig
from leadtime.disruptions.base import Disruption


class NoDisruption(Disruption):
    """Identity arm of the disruption overlay — no lead-time modulation.

    ``apply(base_lead_time, t)`` returns ``base_lead_time`` unchanged for every
    ``(base_lead_time, t)``. Constructed with an ``np.random.Generator`` for
    uniform API with a future stochastic disruption generator, but
    ``NoDisruption`` is fully stateless and never calls into the RNG — the
    ``"disruptions"`` stream is byte-identical before and after any number of
    ``apply`` calls.

    The chassis arm of the disruption channel and the Pydantic-level default for
    ``RunConfig.disruption``. Combined with the locked ``default_factory`` on
    ``RunConfig``, every existing scenario YAML that omits the ``disruption:``
    block produces byte-for-byte identical Ledgers without any schema change.
    Mirrors ``StationaryPattern``.
    """

    def __init__(self, config: NoDisruptionConfig, rng: np.random.Generator) -> None:
        # Accept rng for uniform API with a future stochastic disruption arm;
        # never actually draws from it (there is no schedule to evaluate). The
        # test class pins this — the rng state is byte-identical before and
        # after any number of apply() calls.
        self._rng: np.random.Generator = rng

    def apply(self, base_lead_time: int, t: int) -> int:
        return base_lead_time
