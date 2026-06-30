"""Pattern abstract base.

A ``Pattern`` modulates per-period demand: each call to ``apply(base, t)``
takes one period's pre-pattern demand value from ``Demand.draw()`` and
the integer period index ``t`` from the engine, and returns the
post-pattern demand value the engine consumes.

The contract is stateful (the internal RNG may mutate on every call,
for stochastic patterns like the Intermittent / Lumpy arms landing in
later additions). Storing the RNG on the instance mirrors the
``Demand`` ABC and keeps the engine's call site clean
(``self._pattern.apply(self._demand.draw(), self._t)``).

Concrete arms are constructed via ``demand.patterns.registry.make_pattern``
from a ``PatternConfig`` plus a spawned ``np.random.Generator``. The
chassis arm is ``StationaryPattern`` (identity
passthrough — equivalent to "no pattern overlay"). Subsequent bullets
add Seasonal, Trending, Intermittent, Lumpy.

Sample-domain note: ``apply(base, t) -> float`` is the locked return type.
Any pattern that *modifies* its input MUST return a non-negative float —
the engine writes the result directly to ``self._period_demand`` and the
Ledger's ``demand`` column, which assume non-negative. The identity
``StationaryPattern`` is the exception that proves the rule: it never
modifies the input, so it trivially preserves whatever non-negativity the
base ``Demand`` produced. Future patterns whose math could mathematically
dip below zero (e.g., a trending pattern with steep negative slope) MUST
clip at the source — the rule mirrors ``NormalDemand``'s ``max(0.0, …)``
clip.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Pattern(ABC):
    """Abstract base for demand-pattern overlays."""

    @abstractmethod
    def apply(self, base: float, t: int) -> float:
        """Modulate one period's pre-pattern ``base`` demand at integer time ``t``.

        ``base`` is the value returned by ``Demand.draw()`` for the current
        period; ``t`` is the engine's integer period index (0-indexed,
        advanced each ``InventoryEngine.step()``). The return value is
        written directly to ``self._period_demand`` in
        ``InventoryEngine._draw_demand`` and ultimately to the Ledger's
        ``demand`` column.

        Implementations must return a finite, non-negative ``float`` for any
        ``base >= 0``. The identity case (``StationaryPattern``) preserves
        ``base`` unchanged; non-identity patterns own clipping to enforce
        non-negativity if their math could otherwise produce negatives.
        """
        ...
