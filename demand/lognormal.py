"""Lognormal demand."""

from __future__ import annotations

import numpy as np

from core.config import LognormalDemandConfig
from demand.base import Demand


class LognormalDemand(Demand):
    """Lognormal-distributed demand on ``(0, ∞)`` continuous reals.

    ``draw()`` samples ``X ~ Lognormal(mu, sigma)`` and upcasts to
    ``float`` to satisfy the ``Demand.draw() -> float`` contract.
    **No clip** — Lognormal support is ``(0, ∞)`` continuous reals
    natively (the ABC docstring explicitly names Lognormal in the
    no-clip lock; the *Demand Support and Clipping Rule*
    applies).

    Parameter-naming foot-gun (well-known numpy pitfall): numpy's
    ``Generator.lognormal(mean, sigma)`` uses ``mean`` for the **underlying
    Normal's location parameter**, NOT the mean of the resulting Lognormal.
    A YAML reader seeing ``mean: 2.21`` would reasonably assume the
    resulting demand averages 2.21 — but the actual mean is
    ``exp(2.21 + sigma²/2)``. We therefore deviate from Gamma's
    numpy-native naming convention and use ``mu, sigma`` (textbook
    standard: parameters of the underlying Normal). The mapping into the
    numpy call is literal — ``rng.lognormal(self._mu, self._sigma)`` —
    but the YAML/config surface is unambiguous.

    Resulting moments (NOT the parameters):
        ``E[X]   = exp(mu + sigma²/2)``
        ``Var[X] = (exp(sigma²) − 1) × exp(2*mu + sigma²)``

    The ``float(...)`` wrap is a **documentation seam**, not a runtime
    cast for this arm: numpy 2.4.4's ``Generator.lognormal(mean, sigma)``
    returns a Python ``float`` directly for scalar calls. The explicit
    wrap is kept for the same reasons as Gamma's wrap — older numpy
    returns ``np.float64``, the wrap documents intent at the seam, and
    visual recipe uniformity across arms.

    Strict positivity: Lognormal support is mathematically open at zero
    ``(0, ∞)`` — same as Gamma, distinct from Poisson/NegBin's ``[0, ∞)``.
    The open-at-zero contract holds numerically (verified via pre-impl
    sanity check: even ``lognormal(-5.0, 0.01)`` returns ``6.76e-3``,
    not exactly ``0.0``).
    """

    def __init__(self, config: LognormalDemandConfig, rng: np.random.Generator) -> None:
        self._mu: float = float(config.mu)
        self._sigma: float = float(config.sigma)
        self._rng: np.random.Generator = rng

    def draw(self) -> float:
        return float(self._rng.lognormal(self._mu, self._sigma))
