"""Gamma demand."""

from __future__ import annotations

import numpy as np

from core.config import GammaDemandConfig
from demand.base import Demand


class GammaDemand(Demand):
    """Gamma-distributed demand on ``(0, ∞)`` continuous reals.

    ``draw()`` samples ``X ~ Gamma(shape, scale)`` and upcasts to
    ``float`` to satisfy the ``Demand.draw() -> float`` contract.
    **No clip** — Gamma support is ``(0, ∞)`` continuous reals natively,
    so the cargo-cult ``max(0.0, …)`` from ``NormalDemand`` would be
    dead code (forbidden by the *Demand Support and Clipping Rule*; the
    ABC docstring also explicitly locks this
    anticipatorily for continuous arms).

    The ``float(...)`` wrap is a **documentation seam**, not a runtime
    cast for this arm: numpy 2.4.4's ``Generator.gamma(shape, scale)``
    returns a Python ``float`` directly for scalar calls. The explicit
    wrap is kept for three reasons: (a) older numpy versions return
    ``np.float64`` (a Python ``float`` subclass) and the wrap normalizes
    to a plain ``float`` instance; (b) it documents the contract at the
    seam — every ``Demand.draw()`` implementation visibly produces a
    ``float`` at the boundary; (c) it keeps the recipe visually uniform
    with the integer-valued arms (Poisson, NegBin) where the wrap is a
    genuine runtime cast from ``int64``.

    Strict positivity: Gamma support is mathematically open at zero
    ``(0, ∞)``, distinct from Poisson/NegBin's ``[0, ∞)``. The
    open-at-zero contract holds numerically too — even ``gamma(0.01, 1.0)``
    returns ``7.44e-12``, not exactly ``0.0``.
    """

    def __init__(self, config: GammaDemandConfig, rng: np.random.Generator) -> None:
        self._shape: float = float(config.shape)
        self._scale: float = float(config.scale)
        self._rng: np.random.Generator = rng

    def draw(self) -> float:
        return float(self._rng.gamma(self._shape, self._scale))
