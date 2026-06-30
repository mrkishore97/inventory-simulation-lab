"""Syntetos-Boylan demand classifier — ADI / CV² → smooth/intermittent/erratic/lumpy.

The single source of truth for Syntetos-Boylan-Croston (2005) demand classification, lifted out of
``scripts/extract_m5_archetypes.py`` so the curation script, the Policy Advisor, and the tests
all share one implementation. Pure ``numpy``; no I/O, no Streamlit,
no ``core`` — a 100%-coverage-gated compute module
like ``analytics/``.

The classification partitions a demand history on two axes against fixed cutoffs:

- **ADI** (average demand interval = #periods / #non-zero periods): how *intermittent* demand is.
  ``ADI >= 1.32`` → intermittent axis high.
- **CV²** (squared coefficient of variation of the *non-zero* demand sizes): how *erratic* the
  size of a demand, when it occurs, is. ``CV² >= 0.49`` → erratic axis high.

The four quadrants: smooth (low/low), erratic (low ADI / high CV²), intermittent (high ADI / low
CV²), lumpy (high/high). Leading (pre-introduction) zeros are trimmed before metrics, matching the
curation pipeline, so the same series classifies identically whether or not it was pre-trimmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

# Syntetos-Boylan-Croston (2005) classification cutoffs (inclusive on the high side).
SB_ADI_CUTOFF: Final = 1.32
SB_CV2_CUTOFF: Final = 0.49

SBClass = Literal["smooth", "intermittent", "erratic", "lumpy"]


@dataclass(frozen=True)
class SBClassification:
    """An SB class plus the metrics behind it, so the Advisor needs no recomputation.

    - ``sb_class``: the quadrant — ``smooth`` / ``intermittent`` / ``erratic`` / ``lumpy``.
    - ``adi`` / ``cv2``: the two classification metrics (over the leading-zero-trimmed history).
    - ``n_periods`` / ``n_nonzero``: the trimmed (active) length and how many periods had demand.
    """

    sb_class: SBClass
    adi: float
    cv2: float
    n_periods: int
    n_nonzero: int


def trim_leading_zeros(series: np.ndarray) -> np.ndarray:
    """Drop the pre-introduction zero run; return demand from the first sale on.

    Returns an empty array when the series is all zeros (demand never occurs).
    """
    nonzero = np.flatnonzero(series > 0.0)
    if nonzero.shape[0] == 0:
        return series[:0]
    return series[int(nonzero[0]) :]


def adi(series: np.ndarray) -> float:
    """Average demand interval = #periods / #non-zero periods (``inf`` if all zero)."""
    nonzero = int(np.count_nonzero(series))
    if nonzero == 0:
        return float("inf")
    return float(series.shape[0]) / float(nonzero)


def cv_squared(series: np.ndarray) -> float:
    """Squared coefficient of variation of the NON-ZERO demand sizes (``0.0`` if fewer than 2).

    Uses population std (``ddof=0``) for determinism. ``mean`` is always positive here — the sizes
    are filtered strictly ``> 0`` — so no zero-mean guard is needed.
    """
    sizes = series[series > 0.0]
    if sizes.shape[0] < 2:
        return 0.0
    return (float(np.std(sizes)) / float(np.mean(sizes))) ** 2


def classify_metrics(adi: float, cv2: float) -> SBClass:
    """Map an ``(ADI, CV²)`` pair to its SB quadrant; the cutoffs are inclusive on the high side.

    Useful directly when the metrics are already known (e.g. the bundled manifest records them).
    """
    high_adi = adi >= SB_ADI_CUTOFF
    high_cv2 = cv2 >= SB_CV2_CUTOFF
    if not high_adi and not high_cv2:
        return "smooth"
    if not high_adi and high_cv2:
        return "erratic"
    if high_adi and not high_cv2:
        return "intermittent"
    return "lumpy"


def classify(history: npt.ArrayLike) -> SBClassification:
    """Classify a 1-D demand history into an :class:`SBClassification`.

    Leading (pre-introduction) zeros are trimmed first, so a raw and a pre-trimmed history of the
    same SKU classify identically. Raises :class:`ValueError` on an empty history or one that is all
    zeros — demand that never occurs is not meaningfully classifiable.
    """
    arr = np.asarray(history, dtype=float)
    if arr.size == 0:
        raise ValueError("history is empty: nothing to classify")
    active = trim_leading_zeros(arr)
    if active.size == 0:
        raise ValueError("history is all zeros: demand never occurs, cannot classify")
    a = adi(active)
    c = cv_squared(active)
    return SBClassification(
        sb_class=classify_metrics(a, c),
        adi=a,
        cv2=c,
        n_periods=int(active.size),
        n_nonzero=int(np.count_nonzero(active)),
    )
