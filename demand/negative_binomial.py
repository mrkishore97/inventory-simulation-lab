"""Negative Binomial demand."""

from __future__ import annotations

import numpy as np

from core.config import NegativeBinomialDemandConfig
from demand.base import Demand


class NegativeBinomialDemand(Demand):
    """Negative-Binomial-distributed demand on `[0, ∞)` integers.

    ``draw()`` samples ``X ~ NegBin(n, p)`` and upcasts to ``float`` to
    satisfy the ``Demand.draw() -> float`` contract. **No clip** — NegBin
    support is `[0, ∞)` integers natively, so the cargo-cult ``max(0.0, …)``
    from ``NormalDemand`` would be dead code (forbidden by the *Demand
    Support and Clipping Rule*).

    The float upcast is mandatory: ``Generator.negative_binomial(n, p)``
    returns ``int64`` (or ``numpy.int64`` for scalar calls); the engine's
    ``_period_demand: float`` and the Ledger's ``demand`` column dtype
    (``float64``) require a Python ``float`` at the boundary. Even on
    numpy versions where scalar calls happen to return a Python ``int``,
    the explicit ``float(...)`` wrap is the locked contract — robust
    across numpy versions and explicit at the seam.
    """

    def __init__(self, config: NegativeBinomialDemandConfig, rng: np.random.Generator) -> None:
        self._n: float = float(config.n)
        self._p: float = float(config.p)
        self._rng: np.random.Generator = rng

    def draw(self) -> float:
        return float(self._rng.negative_binomial(self._n, self._p))
