"""Poisson demand."""

from __future__ import annotations

import numpy as np

from core.config import PoissonDemandConfig
from demand.base import Demand


class PoissonDemand(Demand):
    """Poisson-distributed demand at constant rate ``λ``.

    ``draw()`` samples ``X ~ Poisson(rate)`` and upcasts to ``float`` to
    satisfy the ``Demand.draw() -> float`` contract. **No clip** — Poisson
    support is `[0, ∞)` integers, so the cargo-cult ``max(0.0, …)`` from
    ``NormalDemand`` would be dead code (forbidden by the *Demand Support
    and Clipping Rule*).

    The float upcast is mandatory: ``Generator.poisson(lam)`` returns
    ``int64`` (or ``numpy.int64`` for scalar calls); the engine's
    ``_period_demand: float`` and the Ledger's ``demand`` column dtype
    (``float64``) require a Python ``float`` at the boundary.
    """

    def __init__(self, config: PoissonDemandConfig, rng: np.random.Generator) -> None:
        self._rate: float = float(config.rate)
        self._rng: np.random.Generator = rng

    def draw(self) -> float:
        return float(self._rng.poisson(self._rate))
