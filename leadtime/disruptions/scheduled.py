"""Scheduled lead-time disruption — multiply L during configured windows."""

from __future__ import annotations

import numpy as np

from core.config import ScheduledDisruptionConfig
from leadtime.base import round_to_periods
from leadtime.disruptions.base import Disruption


class ScheduledDisruption(Disruption):
    """Deterministic ``(start, duration, multiplier)`` lead-time disruptions.

    ``apply(base_lead_time, t)`` multiplies ``base_lead_time`` by the product of
    the multipliers of every window active at period ``t`` (a window with start
    ``s`` and duration ``d`` is active for ``s <= t < s + d`` — the half-open
    interval), then rounds to the nearest period and clips to ``>= 1`` via
    :func:`leadtime.base.round_to_periods`. Overlapping windows compose
    multiplicatively; with no active window the factor is ``1.0`` and the
    (already integer, already ``>= 1``) ``base_lead_time`` is returned unchanged.

    The schedule is fully deterministic. The constructor accepts an
    ``np.random.Generator`` for uniform API with a future stochastic disruption
    generator but never draws from it — the ``"disruptions"`` stream is reserved
    for that arm. Mirrors ``SeasonalPattern`` as the first concrete arm of its
    overlay union.
    """

    def __init__(self, config: ScheduledDisruptionConfig, rng: np.random.Generator) -> None:
        # Precompute (start, end, multiplier) with the half-open end = start +
        # duration so apply() is a tight membership test. Stored as a tuple so
        # the instance is effectively immutable.
        self._windows: tuple[tuple[int, int, float], ...] = tuple(
            (w.start, w.start + w.duration, float(w.multiplier)) for w in config.windows
        )
        # Accept rng for uniform API; the deterministic schedule never draws.
        self._rng: np.random.Generator = rng

    def apply(self, base_lead_time: int, t: int) -> int:
        factor = 1.0
        for start, end, multiplier in self._windows:
            if start <= t < end:
                factor *= multiplier
        return round_to_periods(base_lead_time * factor)
