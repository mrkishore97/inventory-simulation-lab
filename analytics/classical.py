"""Closed-form classical inventory laws — analytical benchmarks for validation.

This module collects the textbook closed-form results that the simulator's
policies are expected to reproduce. They are pure functions of their scalar
arguments — no RNG, no config, no state — consumed by the ``tests/validation``
suite to assert that the digital twin matches analytical theory. EOQ
lands first; safety stock, reorder point,
and the newsvendor critical ratio follow.

Deliberately *not* an ABC + factory like the policy / demand / lead-time
modules: those dispatch runtime strategies chosen by config, whereas these are
static formulas with no polymorphism to select.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def eoq(demand_rate: float, ordering_cost: float, holding_cost: float) -> float:
    """Economic Order Quantity — the cost-minimizing fixed order quantity.

    Returns the classic Wilson EOQ ``sqrt(2 * D * K / h)``, where

    - ``demand_rate`` (``D``) is demand per period (constant and known),
    - ``ordering_cost`` (``K``) is the fixed cost per order
      (maps to ``CostConfig.ordering_fixed``), and
    - ``holding_cost`` (``h``) is the holding cost per unit per period
      (maps to ``CostConfig.holding_per_unit_per_period``).

    EOQ minimizes the sum of per-period ordering cost ``K * D / Q`` and holding
    cost ``h * Q / 2``. It assumes constant known demand, a fixed (or zero)
    lead time, no stockouts, and a unit cost independent of ``Q`` — so the only
    ``Q``-dependent costs are ordering and holding. The result is
    **lead-time-independent**: lead time shifts the reorder point, not the
    optimal quantity.

    ``holding_cost`` must be strictly positive (a zero holding cost makes total
    cost monotonically decreasing in ``Q`` — order everything in one batch);
    ``demand_rate`` and ``ordering_cost`` must be non-negative.
    """
    if holding_cost <= 0:
        raise ValueError(f"holding_cost must be > 0, got {holding_cost}")
    if demand_rate < 0:
        raise ValueError(f"demand_rate must be >= 0, got {demand_rate}")
    if ordering_cost < 0:
        raise ValueError(f"ordering_cost must be >= 0, got {ordering_cost}")
    return math.sqrt(2 * demand_rate * ordering_cost / holding_cost)


def safety_stock(z: float, demand_std: float, lead_time: float) -> float:
    """Safety stock — the buffer that absorbs demand variability over the lead time.

    Returns ``z * demand_std * sqrt(lead_time)``, where

    - ``z`` is the standard-normal safety factor for the target service level
      (use :func:`z_for_service_level` to convert a service level to ``z``),
    - ``demand_std`` (``σ``) is the per-period demand standard deviation
      (maps to ``NormalDemandConfig.std``), and
    - ``lead_time`` (``L``) is the replenishment lead time in periods
      (maps to ``DeterministicLeadTimeConfig.lead_time``).

    The ``sqrt(L)`` is the heart of the result: under IID per-period demand,
    variance adds over the ``L`` periods of a replenishment cycle, so the
    standard deviation of *lead-time demand* is ``σ·sqrt(L)`` — and safety stock
    is the ``z``-quantile offset of that lead-time-demand distribution. It
    composes with cycle stock ``μ·L`` to give the reorder point.

    ``demand_std`` and ``lead_time`` must be non-negative. ``z`` is unrestricted:
    a negative ``z`` (service level below 50%) yields a negative safety stock —
    holding less than mean lead-time demand — a valid, if unusual, posture.
    """
    if demand_std < 0:
        raise ValueError(f"demand_std must be >= 0, got {demand_std}")
    if lead_time < 0:
        raise ValueError(f"lead_time must be >= 0, got {lead_time}")
    return z * demand_std * math.sqrt(lead_time)


def z_for_service_level(service_level: float) -> float:
    """Standard-normal safety factor ``z`` for a target cycle service level.

    Returns ``Φ⁻¹(service_level)`` — the inverse standard-normal CDF (quantile
    function) at ``service_level``, via ``scipy.stats.norm.ppf``. This is the
    ``z`` consumed by :func:`safety_stock` and the reorder point:
    a 95% service level gives ``z ≈ 1.645``, 97.5% gives ``≈ 1.960``, and 50%
    gives ``0`` (no safety stock).

    ``service_level`` must lie strictly in ``(0, 1)``: ``Φ⁻¹(0) = -inf`` and
    ``Φ⁻¹(1) = +inf`` carry no finite safety factor.
    """
    if not 0.0 < service_level < 1.0:
        raise ValueError(f"service_level must be in (0, 1), got {service_level}")
    return float(norm.ppf(service_level))


def reorder_point(demand_rate: float, demand_std: float, lead_time: float, z: float) -> float:
    """Reorder point — the ``(s, Q)`` inventory level that triggers a replenishment.

    Returns ``demand_rate * lead_time + safety_stock(z, demand_std, lead_time)`` —
    the two classic components:

    - **cycle stock** ``μ·L`` (``demand_rate`` × ``lead_time``): the demand
      expected to arrive during the replenishment lead time, and
    - **safety stock** ``z·σ·√L`` (:func:`safety_stock`): the buffer against
      demand variability over that lead time.

    This is the ``s`` of an ``(s, Q)`` policy (maps to ``SQPolicyConfig.reorder_point``):
    place an order when the inventory position falls to it. ``demand_rate`` (``μ``)
    maps to ``NormalDemandConfig.mean``, ``demand_std`` (``σ``) to ``.std``, and
    ``lead_time`` (``L``) to ``DeterministicLeadTimeConfig.lead_time``.

    With ``z = 0`` (or ``σ = 0``) the safety term vanishes and the reorder point is
    pure cycle stock ``μ·L`` — reorder exactly when the remaining stock will be
    consumed as the order arrives. ``demand_rate`` must be non-negative;
    ``demand_std`` / ``lead_time`` are validated by the delegated
    :func:`safety_stock` call.
    """
    if demand_rate < 0:
        raise ValueError(f"demand_rate must be >= 0, got {demand_rate}")
    return demand_rate * lead_time + safety_stock(z, demand_std, lead_time)


def newsvendor_level(
    demand_rate: float, demand_std: float, underage_cost: float, overage_cost: float
) -> float:
    """Newsvendor order-up-to level — the cost-optimal single-period base stock.

    Returns ``demand_rate + z_for_service_level(critical_ratio) * demand_std``,
    the order-up-to level ``S*`` that minimizes expected single-period cost when
    demand is Normal. The **critical ratio**

        ``critical_ratio = underage_cost / (underage_cost + overage_cost)``

    is the cost-optimal in-stock probability, where

    - ``underage_cost`` (``Cu``) is the cost per unit of *unmet* demand
      (stockout / lost margin), and
    - ``overage_cost`` (``Co``) is the cost per unit of *leftover* inventory.

    So ``S* = μ + Φ⁻¹(Cu/(Cu+Co))·σ``. It sets ``BaseStockPolicyConfig.target_level``
    (the base-stock policy is optimal in the no-fixed-ordering-cost newsvendor
    setting). ``demand_rate`` (``μ``) maps to ``NormalDemandConfig.mean`` and
    ``demand_std`` (``σ``) to ``.std``.

    Balanced costs ``Cu = Co`` give ``CR = 0.5`` ⇒ ``S* = μ`` (no bias). A larger
    ``Cu`` (stockouts hurt more than leftovers) raises ``CR`` and ``S*`` — stock
    more; a larger ``Co`` lowers them — stock less.

    ``demand_rate`` and ``demand_std`` must be non-negative; ``underage_cost`` and
    ``overage_cost`` must be **strictly positive** so the critical ratio lies in
    ``(0, 1)`` and ``S*`` is finite (``Cu = 0`` ⇒ never stock; ``Co = 0`` ⇒ stock
    unboundedly).
    """
    if demand_rate < 0:
        raise ValueError(f"demand_rate must be >= 0, got {demand_rate}")
    if demand_std < 0:
        raise ValueError(f"demand_std must be >= 0, got {demand_std}")
    if underage_cost <= 0:
        raise ValueError(f"underage_cost must be > 0, got {underage_cost}")
    if overage_cost <= 0:
        raise ValueError(f"overage_cost must be > 0, got {overage_cost}")
    critical_ratio = underage_cost / (underage_cost + overage_cost)
    return demand_rate + z_for_service_level(critical_ratio) * demand_std
