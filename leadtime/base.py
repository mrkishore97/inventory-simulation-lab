"""Lead-time abstract base.

A ``LeadTime`` is a per-order stochastic source: each call to ``sample()``
returns one order's lead time as an integer number of periods, advancing an
internal RNG (for stochastic arms). The contract is **stateful** — mirroring
``Demand``, distinct from the stateless ``Policy``. Storing the RNG on the
instance keeps the engine's call site clean (``self._lead_time.sample()``)
and the seed-cascade plumbing (``SeedManager.rng("lead_time")``) entirely
inside engine ``__init__``.

Concrete arms are constructed via ``leadtime.registry.make_lead_time`` from a
``LeadTimeConfig`` plus a spawned ``np.random.Generator``. Subclasses are
internal — consumers configure lead time through YAML / config, not by
instantiating lead-time classes directly.

Return-type note: ``sample() -> int`` is the locked return type. Lead time is
an integer number of periods because the engine schedules arrivals in a
``dict[int, float]`` pending-arrivals queue keyed by integer arrival period
(``pending[t + L]``). The deterministic arm returns its fixed configured
integer and does NOT draw from the RNG. Stochastic arms (Normal / Gamma /
Lognormal) draw a continuous value and convert it to an integer ``>= 1`` via
the shared :func:`round_to_periods` helper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


def round_to_periods(draw: float) -> int:
    """Convert a continuous lead-time draw to an integer number of periods.

    Rounds to the nearest period (Python's ``round`` is banker's rounding —
    round-half-to-even) and clips to ``>= 1``: the periodic model requires a
    strictly positive lead time, and the clip also absorbs any non-positive
    draw (the negative tail of a Normal, or a sub-0.5 Gamma/Lognormal draw that
    rounds to 0). Shared by every stochastic lead-time arm so the round-then-
    clip policy lives in exactly one place.
    """
    return max(1, round(float(draw)))


class LeadTime(ABC):
    """Abstract base for per-order lead-time generators."""

    @abstractmethod
    def sample(self) -> int:
        """Return one order's lead time as a positive integer number of periods.

        The internal RNG advances on every call for stochastic arms; the
        deterministic arm returns a fixed value without drawing. The engine
        schedules the order's arrival at period ``self._t + sample()``.
        """
        ...
