"""Heuristic Policy Advisor — demand class → policy kind + classical-formula parameters.

The differentiator: given a demand history and a small business context (lead time, costs,
service target), recommend an inventory policy *and explain why* in plain English. The engine is a
three-step pipeline — classify, summarize, parameterize — not a lookup table:

1. **classify** the history with :func:`recommender.classify.classify` (Syntetos-Boylan quadrant).
2. **summarize** it into the two demand moments the classical laws consume (mean / std).
3. **map** the quadrant to a distinct, theory-justified policy *and derive its parameters* from
   :mod:`analytics.classical` (EOQ, reorder point, order-up-to over the right protection interval).

One policy per quadrant, each with a coherent protection-interval story:

============  ====================  =========================  ==================
SB class      policy                protection interval        parameters
============  ====================  =========================  ==================
smooth        ``(s, Q)``            ``L`` (continuous)         ``Q`` = EOQ, ``s`` = ROP
erratic       ``(s, S)``            ``L`` (continuous)         ``s`` = ROP, ``S`` = s + EOQ
intermittent  ``(R, S)``            ``R + L`` (periodic)       ``R`` ≈ ADI, ``S`` = up-to(R+L)
lumpy         base-stock            ``L + 1`` (every period)   ``S`` = up-to(L+1)
============  ====================  =========================  ==================

The recommendation carries the *validated* policy config (drops straight into a ``RunConfig``), the
classification bundle, the demand stats, and a number-rich rationale, so the page renders
everything without recomputation. These are **starting parameters**, not guaranteed optima — the
lumpy recommendation in particular says so explicitly: high-CV² demand can drive a large target
level that should be tuned against simulated cost.

The smooth ``(s, Q)`` recommendation also carries a **feasibility warning** when the EOQ batch is
smaller than one period's mean demand. A once-per-period ``(s, Q)`` cannot reorder more than once a
period, so a sub-period economic cycle (``Q < μ``) means the policy may order every period and still
fall behind — surfacing the warning stops the Advisor confidently presenting an infeasible policy
(the full fix — simulation-validated ranking and an ``(s, nQ)`` candidate — is later work).

Pure compute (no I/O, no Streamlit); 100%-gated like ``analytics/`` and ``recommender.classify``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from analytics import classical
from core.config import (
    BaseStockPolicyConfig,
    PeriodicReviewPolicyConfig,
    PolicyConfig,
    SQPolicyConfig,
    SSPolicyConfig,
)
from recommender.classify import (
    SB_ADI_CUTOFF,
    SB_CV2_CUTOFF,
    SBClassification,
    classify,
    trim_leading_zeros,
)

# Default cycle service level for the safety-stock / order-up-to buffers (z ≈ 1.645).
DEFAULT_SERVICE_LEVEL: Final = 0.95

# Feasibility guardrail for the smooth (s, Q) recommendation: when the EOQ batch is below one
# period's mean demand, a once-per-period (s, Q) cannot keep up (its reorder cycle would be
# sub-period). We warn rather than silently recommend an infeasible policy.
FEASIBILITY_WARNING_SQ: Final = (
    "EOQ is smaller than one average review-period's demand. In a once-per-period (s, Q) model "
    "this policy may order every period and still fail to keep up. Consider increasing Q, using "
    "(s, S), or an (s, nQ) policy."
)


@dataclass(frozen=True)
class DemandStats:
    """The two demand moments the classical laws consume, over the active (trimmed) history.

    - ``mean`` (``μ``): demand per period — feeds EOQ's ``D`` and the cycle-stock term ``μ·L``.
    - ``std`` (``σ``): population standard deviation (``ddof=0``, matching ``cv_squared``) — feeds
      the safety term ``z·σ·√(protection interval)``.
    """

    mean: float
    std: float


@dataclass(frozen=True)
class Recommendation:
    """A recommendation bundle: the policy plus all that is needed to render and justify it.

    - ``policy``: a *validated* :data:`core.config.PolicyConfig` — drops into a ``RunConfig``.
    - ``classification``: the Syntetos-Boylan bundle (class + ADI/CV² + counts).
    - ``stats``: the demand moments the parameters were derived from.
    - ``rationale``: plain-English, number-rich reasoning (class, metrics, derived parameters).
    - ``feasibility_warning``: a non-fatal caveat when the recommended policy may be infeasible
      (currently the smooth ``(s, Q)`` EOQ-``< μ`` case); ``None`` when no concern applies.
    """

    policy: PolicyConfig
    classification: SBClassification
    stats: DemandStats
    rationale: str
    feasibility_warning: str | None = None


def summarize(history: npt.ArrayLike) -> DemandStats:
    """Compute :class:`DemandStats` over the leading-zero-trimmed history.

    Leading (pre-introduction) zeros are dropped first — the same active domain
    :func:`recommender.classify.classify` uses — so the moments describe demand from the first sale
    on. Internal zeros are kept (they are genuine no-demand periods and belong in the per-period
    rate). Raises :class:`ValueError` on an empty or all-zero history, mirroring ``classify``.
    """
    arr = np.asarray(history, dtype=float)
    if arr.size == 0:
        raise ValueError("history is empty: nothing to summarize")
    active = trim_leading_zeros(arr)
    if active.size == 0:
        raise ValueError("history is all zeros: demand never occurs, cannot summarize")
    return DemandStats(mean=float(np.mean(active)), std=float(np.std(active)))


def recommend(
    history: npt.ArrayLike,
    *,
    lead_time: float,
    ordering_cost: float,
    holding_cost: float,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> Recommendation:
    """Recommend an inventory policy for ``history`` given the business context.

    The context is uniform across quadrants (``ordering_cost`` / ``holding_cost`` feed EOQ only for
    the smooth/erratic recommendations, but are validated regardless for a predictable contract):

    - ``lead_time`` (``L``): mean replenishment lead time in periods; must be ``> 0``.
    - ``ordering_cost`` (``K``): fixed cost per order; must be ``>= 0``.
    - ``holding_cost`` (``h``): holding cost per unit per period; must be ``> 0`` (EOQ is undefined
      otherwise). EOQ uses ``D = μ per period`` and ``h per period`` — unit-consistent.
    - ``service_level``: target cycle service level in ``(0, 1)`` → the safety factor ``z``.

    Raises :class:`ValueError` for an out-of-domain context, and propagates ``classify``'s
    :class:`ValueError` for an empty or all-zero history.
    """
    if lead_time <= 0:
        raise ValueError(f"lead_time must be > 0, got {lead_time}")
    if ordering_cost < 0:
        raise ValueError(f"ordering_cost must be >= 0, got {ordering_cost}")
    if holding_cost <= 0:
        raise ValueError(f"holding_cost must be > 0, got {holding_cost}")
    z = classical.z_for_service_level(service_level)  # raises if service_level not in (0, 1)

    arr = np.asarray(history, dtype=float)
    classification = classify(arr)  # raises on empty / all-zero history
    stats = summarize(arr)
    policy, rationale, feasibility_warning = _build_policy(
        classification,
        stats,
        z=z,
        lead_time=float(lead_time),
        ordering_cost=ordering_cost,
        holding_cost=holding_cost,
        service_level=service_level,
    )
    return Recommendation(
        policy=policy,
        classification=classification,
        stats=stats,
        rationale=rationale,
        feasibility_warning=feasibility_warning,
    )


def _classification_clause(c: SBClassification) -> str:
    """The shared opening sentence: the class and the two metrics against their cutoffs."""
    adi_cmp = "≥" if c.adi >= SB_ADI_CUTOFF else "<"
    cv2_cmp = "≥" if c.cv2 >= SB_CV2_CUTOFF else "<"
    return (
        f"Demand is **{c.sb_class}** "
        f"(ADI {c.adi:.2f} {adi_cmp} {SB_ADI_CUTOFF}, CV² {c.cv2:.2f} {cv2_cmp} {SB_CV2_CUTOFF})."
    )


def _sq_feasibility_warning(order_quantity: int, mean_demand: float) -> str | None:
    """Guard the smooth ``(s, Q)`` recommendation: an EOQ batch below one period's mean demand
    implies a sub-period reorder cycle the once-per-period model cannot deliver. Returns the
    warning text when ``Q < μ``, else ``None``."""
    return FEASIBILITY_WARNING_SQ if order_quantity < mean_demand else None


def _build_policy(
    classification: SBClassification,
    stats: DemandStats,
    *,
    z: float,
    lead_time: float,
    ordering_cost: float,
    holding_cost: float,
    service_level: float,
) -> tuple[PolicyConfig, str, str | None]:
    """Map an SB classification + demand moments to a policy config, its rationale, and an
    optional feasibility warning (``None`` unless a recommendation may be infeasible).

    Each level parameter is rounded to whole units (demand is discrete) and floored so the
    Pydantic constraints always hold: ``Q >= 1``, ``s < S`` for (s,S), and ``S``/``target_level
    >= 1`` for the order-up-to policies (matters for low-volume intermittent/lumpy SKUs).
    """
    mu = stats.mean
    sigma = stats.std
    lt = lead_time
    clause = _classification_clause(classification)
    sb = classification.sb_class

    if sb == "smooth":
        q = max(1, round(classical.eoq(mu, ordering_cost, holding_cost)))
        s = float(round(classical.reorder_point(mu, sigma, lt, z)))
        cycle = mu * lt
        safety = z * sigma * math.sqrt(lt)
        rationale = (
            f"{clause} Frequent, stable demand suits a continuous-review (s, Q) policy with a "
            f"fixed economic batch. EOQ gives Q≈{q} from D={mu:.2f} per period and "
            f"h={holding_cost:g} per unit per period (K={ordering_cost:g}); reorder point "
            f"s≈{s:g} = μ·L ({cycle:.1f}) + z·σ·√L ({safety:.1f}) at {service_level:.0%} service."
        )
        warning = _sq_feasibility_warning(q, mu)
        return SQPolicyConfig(reorder_point=s, order_quantity=q), rationale, warning

    if sb == "erratic":
        s = float(round(classical.reorder_point(mu, sigma, lt, z)))
        batch = max(1, round(classical.eoq(mu, ordering_cost, holding_cost)))
        big_s = max(s + batch, 1.0)
        rationale = (
            f"{clause} Demand arrives often but in variable sizes, so a fixed order quantity "
            f"misfits; an (s, S) order-up-to absorbs the size variability. Reorder point s≈{s:g} = "
            f"μ·L + z·σ·√L at {service_level:.0%} service; order-up-to S≈{big_s:g} = s + EOQ batch "
            f"({batch}) from D={mu:.2f} per period, h={holding_cost:g} per unit per period."
        )
        return SSPolicyConfig(reorder_point=s, order_up_to=big_s), rationale, None

    if sb == "intermittent":
        r = max(1, round(classification.adi))
        big_s = max(float(round(mu * (r + lt) + z * sigma * math.sqrt(r + lt))), 1.0)
        rationale = (
            f"{clause} Demand is sporadic but consistent in size, so continuous review is "
            f"wasteful; a periodic-review (R, S) reviews every R={r} periods (≈ the average "
            f"demand interval) and orders up to S≈{big_s:g} = μ·(R+L) + z·σ·√(R+L) at "
            f"{service_level:.0%} service."
        )
        return PeriodicReviewPolicyConfig(review_period=r, order_up_to=big_s), rationale, None

    # lumpy — sporadic AND erratic (the hardest quadrant); a conservative high-service start.
    target = max(float(round(mu * (lt + 1) + z * sigma * math.sqrt(lt + 1))), 1.0)
    rationale = (
        f"{clause} Demand is both sporadic and erratic in size — the hardest pattern. A base-stock "
        f"policy reviewed every period responds the period after a spike, ordering up to "
        f"target≈{target:g} = μ·(L+1) + z·σ·√(L+1) at {service_level:.0%} service. This is a "
        f"**conservative starting point, not a guaranteed optimum**: with high CV² the target can "
        f"run large — tune it against simulated cost."
    )
    return BaseStockPolicyConfig(target_level=target), rationale, None
