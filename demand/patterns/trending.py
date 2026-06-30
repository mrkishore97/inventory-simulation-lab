"""Trending (linear-multiplicative) pattern — second concrete arm of ``PatternConfig``."""

from __future__ import annotations

import numpy as np

from core.config import TrendingPatternConfig
from demand.patterns.base import Pattern


class TrendingPattern(Pattern):
    """Linear-multiplicative trending demand modulation.

    ``apply(base, t)`` returns ``max(0.0, base * (1 + slope * t))``. The
    textbook linear-trend model: monotone drift parameterized by a single
    per-period fractional change. ``slope=0.02`` grows demand 2%/period;
    ``slope=-0.01`` declines demand 1%/period. ``slope=0`` is the degenerate
    identity case (factor is constantly ``1.0`` in IEEE 754; equivalent to
    :class:`StationaryPattern`).

    Multiplicative chassis matches :class:`SeasonalPattern`: both forms have
    ``apply = base * factor`` shape, so future M3 Seasonal+Trending
    composition is derivable as ``base * seasonal_factor * trending_factor``
    without re-thinking the abstraction.

    Non-negativity strategy: **clip-at-source**. At sufficiently steep
    negative slope, the factor ``1 + slope * t`` crosses zero at
    ``t = -1/slope`` and goes negative beyond; the ``max(0.0, ...)`` clip
    handles this. Mirrors :class:`NormalDemand`'s ``max(0.0, val)`` rule.
    This is the **third non-negativity path** in the codebase, alongside
    :class:`StationaryPattern`'s unconditional passthrough and
    :class:`SeasonalPattern`'s bounds-based guarantee. Picked here because
    bounds-based for slope would require cross-field validation with
    ``simulation.horizon``, which is brittle (every horizon change would
    re-validate every Trending config). The clip rarely fires for typical
    SKU slopes (``|slope| <= 0.05`` over horizon 90 keeps factor in
    ``[-3.5, 3.5]`` worst-case, clip activates only at very steep negative
    slope) — it is a safety net, not a routine code path.

    Pure arithmetic: no ``math`` import, no transcendental functions. The
    pattern math conventions lock (prefer stdlib ``math``
    over numpy for deterministic scalar pattern math) degenerates cleanly
    when no math is needed — ``max``, ``+``, ``*`` are Python built-ins.

    Constructor accepts an ``np.random.Generator`` for uniform API with
    future stochastic patterns (Intermittent / Lumpy,
    which DO consume RNG), but ``TrendingPattern`` is fully deterministic
    and never draws from the rng. The test class pins this — the rng state
    is byte-identical before and after any number of ``apply()`` calls.

    NaN/+inf/-inf rejection happens at config-load time via the
    :class:`TrendingPatternConfig` ``slope: FiniteFloat`` gate — the
    pattern's ``apply()`` assumes a finite ``slope`` and trusts that
    contract.
    """

    def __init__(self, config: TrendingPatternConfig, rng: np.random.Generator) -> None:
        self._slope: float = float(config.slope)
        # Accept rng for uniform API with future stochastic patterns; never
        # actually draws from it.
        self._rng: np.random.Generator = rng

    def apply(self, base: float, t: int) -> float:
        return max(0.0, base * (1.0 + self._slope * t))
