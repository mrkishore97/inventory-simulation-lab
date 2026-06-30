"""Disruption abstract base.

A ``Disruption`` overlays the lead-time generator: each call to
``apply(base_lead_time, t)`` takes one order's pre-disruption lead time (the
integer returned by ``LeadTime.sample()``) and the integer period index ``t``
at which the order is placed, and returns the effective integer lead time the
engine schedules the arrival with (``arrival = t + apply(...)``).

This is the lead-time twin of the demand ``Pattern`` overlay: ``Pattern.apply``
modulates demand by period; ``Disruption.apply`` modulates lead time by period.
The default arm is ``NoDisruption`` (identity passthrough — equivalent to "no
disruption"), the Pydantic-level default for ``RunConfig.disruption``; the
concrete ``ScheduledDisruption`` arm multiplies the lead time during configured
``(start, duration, multiplier)`` windows.

The contract is stateful (the internal RNG may mutate on every call for a
future stochastic disruption generator). Storing the RNG on the instance
mirrors the ``LeadTime`` / ``Pattern`` ABCs and keeps the engine's call site
clean (``self._disruption.apply(self._lead_time.sample(), self._t)``).

Concrete arms are constructed via
``leadtime.disruptions.registry.make_disruption`` from a ``DisruptionConfig``
plus a spawned ``np.random.Generator`` sourced from
``SeedManager.rng("disruptions")``.

Return-type note: ``apply(base_lead_time, t) -> int`` is the locked return type,
and the result must be ``>= 1`` (the periodic model requires a strictly positive
lead time). The identity ``NoDisruption`` returns its already-valid
``base_lead_time`` unchanged; arms that scale the lead time clip to ``>= 1`` via
the shared :func:`leadtime.base.round_to_periods` helper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Disruption(ABC):
    """Abstract base for lead-time disruption overlays."""

    @abstractmethod
    def apply(self, base_lead_time: int, t: int) -> int:
        """Return the effective lead time for an order placed at period ``t``.

        ``base_lead_time`` is the pre-disruption integer lead time from
        ``LeadTime.sample()``; ``t`` is the engine's integer period index at
        order placement. The engine schedules the arrival at
        ``t + apply(base_lead_time, t)``.

        Implementations must return an ``int >= 1``. The identity case
        (``NoDisruption``) returns ``base_lead_time`` unchanged; scaling arms own
        the round-then-clip via ``round_to_periods``.
        """
        ...
