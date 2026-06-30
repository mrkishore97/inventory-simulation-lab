"""Stochastic demand generators.

Public surface intentionally narrow: consumers configure demand via
``core.config.DemandConfig`` (a Pydantic discriminated union — the M2
demand quartet is complete: Normal (inline pre-M2 baseline) plus
Poisson, NegativeBinomial, Gamma, and Lognormal added across M2 Bullets
6–9) and call ``make_demand(config, rng)`` to obtain a ``Demand``.
Concrete classes (``NormalDemand``, etc.) are package-level re-exports
for tests and one-off scripts; they are NOT pulled up to
``inventory_twin/__init__.py`` because demand classes are plumbing, not
nameable business artifacts the way classical policies are.

Pattern overlays — the second axis of demand variability — live under
the :mod:`demand.patterns` subpackage; the chassis landed in M2 Bullet
10 with ``StationaryPattern`` identity. The pattern surface mirrors the
demand surface: ``Pattern`` ABC + ``make_pattern`` factory at the
umbrella facade; concrete pattern classes (``StationaryPattern``, and
future ``SeasonalPattern`` etc.) are subpackage-level
only.
"""

from demand.base import Demand
from demand.gamma import GammaDemand
from demand.lognormal import LognormalDemand
from demand.negative_binomial import NegativeBinomialDemand
from demand.normal import NormalDemand
from demand.poisson import PoissonDemand
from demand.registry import make_demand

__all__ = [
    "Demand",
    "GammaDemand",
    "LognormalDemand",
    "NegativeBinomialDemand",
    "NormalDemand",
    "PoissonDemand",
    "make_demand",
]
