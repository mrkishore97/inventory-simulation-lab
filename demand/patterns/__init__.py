"""Demand-pattern overlay subpackage.

Pattern engine infrastructure ships the
``StationaryPattern`` identity arm (the Pydantic-level default for
``RunConfig.pattern`` — equivalent to "no pattern overlay"). ``SeasonalPattern``
(first concrete non-identity arm; sinusoidal multiplicative modulation)
follows. ``TrendingPattern`` (second concrete non-identity
arm; linear-multiplicative drift with clip-at-source) follows. The
remaining concrete patterns (Intermittent / Lumpy) are additive arms
of the ``PatternConfig`` discriminated
union and the ``make_pattern`` factory.

Public surface intentionally mirrors the demand convention: ``Pattern``
(the ABC) and ``make_pattern`` (the factory) are re-exported at the
``inventory_twin`` umbrella facade; concrete pattern classes
(``StationaryPattern``, ``SeasonalPattern``, ``TrendingPattern``, and
future Intermittent/Lumpy) are package-level re-exports here for tests
and one-off scripts but NOT pulled up to ``inventory_twin/__init__.py``.
"""

from demand.patterns.base import Pattern
from demand.patterns.registry import make_pattern
from demand.patterns.seasonal import SeasonalPattern
from demand.patterns.stationary import StationaryPattern
from demand.patterns.trending import TrendingPattern

__all__ = [
    "Pattern",
    "SeasonalPattern",
    "StationaryPattern",
    "TrendingPattern",
    "make_pattern",
]
