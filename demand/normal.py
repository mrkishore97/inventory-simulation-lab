"""Normal demand with non-negativity clip."""

from __future__ import annotations

import numpy as np

from core.config import NormalDemandConfig
from demand.base import Demand


class NormalDemand(Demand):
    """Normal-distributed demand clipped at zero.

    ``draw()`` samples ``X ~ Normal(mean, std)`` and returns
    ``max(0.0, X)``. The clip is required because Normal can produce
    negative values; demand must be non-negative for the engine's
    physical-validity invariants (``order_placed >= 0``,
    ``sales <= demand``). On the typical M2 fixture (mean=10, std=2)
    the clip almost never fires (z = -5), but it is the contract, not
    a heuristic.
    """

    def __init__(self, config: NormalDemandConfig, rng: np.random.Generator) -> None:
        self._mean: float = float(config.mean)
        self._std: float = float(config.std)
        self._rng: np.random.Generator = rng

    def draw(self) -> float:
        # Preserve the exact form from the pre-extraction engine code
        # (``max(0.0, float(rng.normal(...)))``) so the refactor is
        # byte-equivalent. Functionally identical to ``x if x > 0 else 0``
        # for finite floats, but ``max`` propagates NaN where the ternary
        # would silently zero it — keep behaviour identical to M1.
        return max(0.0, float(self._rng.normal(self._mean, self._std)))
