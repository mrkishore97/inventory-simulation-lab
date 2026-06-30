"""Lead-time disruption overlay subpackage.

Disruption infrastructure ships as a combined chassis + first
concrete arm: the ``Disruption`` ABC, the ``make_disruption`` factory, the
``NoDisruption`` identity arm (the Pydantic-level default for
``RunConfig.disruption`` — equivalent to "no disruption"), and the
``ScheduledDisruption`` arm (deterministic ``(start, duration, multiplier)``
windows that multiply the lead time during active periods). A future stochastic
disruption generator would land as an additive arm of the ``DisruptionConfig``
discriminated union and the ``make_disruption`` factory, drawing from the seeded
``"disruptions"`` stream.

This overlay is the lead-time twin of ``demand.patterns``: ``Disruption.apply``
modulates lead time by period exactly as ``Pattern.apply`` modulates demand.

Public surface mirrors the pattern convention: ``Disruption`` (the ABC) and
``make_disruption`` (the factory) are re-exported at the ``inventory_twin``
umbrella facade; concrete disruption classes (``NoDisruption``,
``ScheduledDisruption``) are package-level re-exports here for tests and one-off
scripts but NOT pulled up to ``inventory_twin/__init__.py``.
"""

from leadtime.disruptions.base import Disruption
from leadtime.disruptions.none import NoDisruption
from leadtime.disruptions.registry import make_disruption
from leadtime.disruptions.scheduled import ScheduledDisruption

__all__ = [
    "Disruption",
    "NoDisruption",
    "ScheduledDisruption",
    "make_disruption",
]
