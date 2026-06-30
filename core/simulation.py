"""Periodic-time inventory engine for one simulation run.

================================================================================
TIMING CONVENTION — READ THIS BEFORE USING ANY ENGINE STATE.
================================================================================

Two distinct observation points exist within every period. They mean different
things and must never be confused.

    Locked period-t sequence:
      1. Receive arrivals scheduled for period t      -> on_hand += order_received
      2. Observe demand                                -> demand[t]
      3. Fulfill demand                                -> sales / lost_sales
                                                         + backorders /
                                                         + backorders_cleared
      --- Observation Point A (policy decision) ---
      4. Apply policy's action                         -> order_placed[t],
                                                         schedule arrival
      5. Accrue costs                                  -> holding / ordering
                                                         + purchase / stockout
      --- Observation Point B (Ledger row t) ---

    * Observation Point A: returned by initial_observation() and step(). Used
      by the policy. Post-fulfillment, pre-order. (s,Q) and (s,S) triggers
      READ THIS.
    * Observation Point B: written to ledger.record(t, ...). End-of-period,
      post-cost-accrual. Analytics, audit, and KPI computation READ THIS.

Same period, two snapshots. Wrong choice silently corrupts policy logic or
analytics — neither will throw, both will produce plausible-looking nonsense.

Field-by-field semantics:

    Field              | Point A (EngineState)        | Point B (Ledger row t)
    -------------------|------------------------------|-----------------------
    t                  | period the *next* decision   | period just closed
                       |   applies to                 |
    on_hand            | post-step-3                  | post-step-3 (steps 4-5
                       |                              |   don't touch it)
    on_order           | post-step-1, pre-step-4      | post-step-4
                       |   (received this period      |   (= prior on_order
                       |    but no new order yet)     |   - order_received[t]
                       |                              |   + order_placed[t])
    backorders         | post-step-3                  | post-step-3 (steps 4-5
                       |                              |   don't touch it)
    inventory_position | computed from above (A)      | computed from above (B)

Audit recipe — from a Ledger row, recover what the policy saw at period t:

    inventory_position_policy[t] = inventory_position_Ledger[t] - order_placed[t]
    on_order_policy[t]           = on_order_Ledger[t]           - order_placed[t]

================================================================================

Implementation notes
--------------------
This is a deterministic forward simulation, not a continuous-time DES. Sim time
is the integer period index ``t``. The engine is a plain state machine driven
externally: ``initial_observation()`` runs steps 1-3 of period 0 and returns
the Point-A snapshot; each ``step(action)`` runs steps 4-5 of the current
period, advances ``t``, and (if not done) runs steps 1-3 of the new period.

Arrivals are tracked as a ``dict[int, float]`` keyed by arrival period. An
order placed in period ``t`` with effective lead time ``L`` (the ``LeadTime``
arm's draw after the disruption overlay; ``L`` equals the raw draw
when no disruption is active) increments ``_pending_arrivals[t + L]`` by the
order quantity. Step 1 of period ``t``
pops ``_pending_arrivals[t]`` (defaulting to 0). Multiple orders that land in
the same period sum into one entry — exactly the same semantics as a DES
event-heap, with no event-ordering concerns to reason about.

Engine constraints:

    * ``SimulationConfig.initial_on_order`` must be 0 (raised by the engine
      ``__init__``). Non-zero requires a cohort arrival convention deferred to
      M2 (when stochastic lead times need cohort tracking anyway).
    * Lead time must be >= 1 period. ``DeterministicLeadTimeConfig.lead_time``
      is a ``PositiveInt`` (enforced at config validation,
      not engine ``__init__``); stochastic arms clip to >= 1 at ``sample()``.
      ``lead_time=0`` is ambiguous in the periodic discrete-time model — by the
      time an order is placed in step 4 of period t, step 1 of period t has
      already run, so the order cannot be received in the same period.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.config import RunConfig
from core.ledger import Ledger
from core.seeding import SeedManager
from demand.base import Demand
from demand.patterns.base import Pattern
from demand.patterns.registry import make_pattern
from demand.registry import make_demand
from leadtime.base import LeadTime
from leadtime.disruptions.base import Disruption
from leadtime.disruptions.registry import make_disruption
from leadtime.registry import make_lead_time


@dataclass(frozen=True)
class EngineState:
    """Observation Point A — post-fulfillment, pre-order, period t.

    See module docstring for Point A vs Point B semantics.
    """

    t: int
    on_hand: float
    on_order: float
    backorders: float
    inventory_position: float


class InventoryEngine:
    """Periodic-time engine for the locked 5-step within-period sequence.

    Driven externally via ``initial_observation()`` (called once) and
    ``step(action)`` (called per period until ``done=True``). Internally a
    plain state machine: ``initial_observation`` runs steps 1-3 of period 0;
    each ``step(action)`` runs steps 4-5 of the current period, advances the
    period counter, and runs steps 1-3 of the next period (or terminates).
    """

    def __init__(self, config: RunConfig) -> None:
        if config.simulation.initial_on_order != 0:
            raise NotImplementedError(
                "M1 requires SimulationConfig.initial_on_order == 0; "
                "non-zero initial pipeline needs a cohort arrival convention, "
                "deferred to M2."
            )

        self._config = config
        self._seeds = SeedManager(config.master_seed)
        self._ledger = Ledger(config.simulation.horizon)
        self._demand: Demand = make_demand(config.demand, self._seeds.rng("demand"))
        self._pattern: Pattern = make_pattern(config.pattern, self._seeds.rng("pattern"))
        self._lead_time: LeadTime = make_lead_time(config.lead_time, self._seeds.rng("lead_time"))
        self._disruption: Disruption = make_disruption(
            config.disruption, self._seeds.rng("disruptions")
        )

        # Mutable inventory state.
        self._on_hand: float = float(config.simulation.initial_on_hand)
        self._on_order: float = 0.0
        self._backorders: float = 0.0
        self._t: int = 0

        # Per-period flow accumulators (set during steps 1-3, read by accrue+record).
        self._period_demand: float = 0.0
        self._period_sales: float = 0.0
        self._period_lost_sales: float = 0.0
        self._period_backorders_cleared: float = 0.0
        self._period_received: float = 0.0

        # Pending arrivals: {arrival_period: aggregated_qty}. Drained at step 1.
        self._pending_arrivals: dict[int, float] = {}

        # Lifecycle flags.
        self._initialized: bool = False
        self._finished: bool = False

    # ---- Public API ----

    def initial_observation(self) -> EngineState:
        if self._initialized:
            raise RuntimeError("initial_observation() already called")
        self._initialized = True
        self._t = 0
        self._begin_period()
        return self._snapshot_a()

    def step(self, order_placed: float) -> tuple[EngineState, bool]:
        if not self._initialized:
            raise RuntimeError("call initial_observation() before step()")
        if self._finished:
            raise RuntimeError("simulation already finished; cannot step()")
        if order_placed < 0:
            raise ValueError(f"order_placed must be >= 0, got {order_placed}")

        self._end_period(order_placed)

        if self._t + 1 >= self._config.simulation.horizon:
            self._finished = True
            return self._final_snapshot(), True

        self._t += 1
        self._begin_period()
        return self._snapshot_a(), False

    @property
    def ledger(self) -> Ledger:
        return self._ledger

    # ---- Within-period halves (steps 1-3 and 4-5) ----

    def _begin_period(self) -> None:
        # Step 1: receive arrivals scheduled for period t.
        received = self._pending_arrivals.pop(self._t, 0.0)
        self._on_hand += received
        self._on_order -= received
        self._period_received = received

        # Step 2: observe demand.
        self._period_demand = self._draw_demand()

        # Step 3: fulfill (Convention C — sales[t] <= demand[t] always).
        if self._config.simulation.backorder_policy == "backorder":
            self._fulfill_backorder_mode(self._period_demand)
        else:
            self._fulfill_lost_sales_mode(self._period_demand)

    def _end_period(self, action: float) -> None:
        # Step 4: place order. The base lead time from the LeadTime arm is passed
        # through the disruption overlay, which multiplies it
        # during active ``(start, duration, multiplier)`` windows; the default
        # NoDisruption returns it unchanged. Keyed on the order-placement period.
        if action > 0:
            base_lead_time = self._lead_time.sample()
            arrival_t = self._t + self._disruption.apply(base_lead_time, self._t)
            self._pending_arrivals[arrival_t] = self._pending_arrivals.get(arrival_t, 0.0) + action
            self._on_order += action

        # Step 5: accrue costs and write Ledger row (Observation Point B).
        self._accrue_and_record(action)

    # ---- Demand fulfillment (Convention C) ----

    def _fulfill_backorder_mode(self, demand: float) -> None:
        # Backorders carried from prior periods are satisfied first.
        bo_cleared = min(self._on_hand, self._backorders)
        self._on_hand -= bo_cleared
        self._backorders -= bo_cleared

        # Then current-period demand consumes remaining on-hand.
        new_sales = min(self._on_hand, demand)
        self._on_hand -= new_sales
        new_unfilled = demand - new_sales
        self._backorders += new_unfilled

        self._period_sales = new_sales
        self._period_lost_sales = 0.0
        self._period_backorders_cleared = bo_cleared

    def _fulfill_lost_sales_mode(self, demand: float) -> None:
        new_sales = min(self._on_hand, demand)
        self._on_hand -= new_sales
        self._period_sales = new_sales
        self._period_lost_sales = demand - new_sales
        self._period_backorders_cleared = 0.0

    # ---- Cost accrual + Ledger row (Observation Point B) ----

    def _accrue_and_record(self, action: float) -> None:
        costs = self._config.costs
        holding = costs.holding_per_unit_per_period * self._on_hand
        ordering = costs.ordering_fixed if action > 0 else 0.0
        purchase = costs.unit_cost * action
        if self._config.simulation.backorder_policy == "backorder":
            stockout = costs.backorder_per_unit_per_period * self._backorders
        else:
            stockout = costs.lost_sale_per_unit * self._period_lost_sales

        self._ledger.record(
            self._t,
            on_hand=self._on_hand,
            on_order=self._on_order,
            inventory_position=self._inventory_position,
            demand=self._period_demand,
            sales=self._period_sales,
            backorders=self._backorders,
            lost_sales=self._period_lost_sales,
            backorders_cleared=self._period_backorders_cleared,
            order_placed=action,
            order_received=self._period_received,
            holding_cost=holding,
            ordering_cost=ordering,
            purchase_cost=purchase,
            stockout_cost=stockout,
        )

    # ---- Observation snapshots ----

    @property
    def _inventory_position(self) -> float:
        return self._on_hand + self._on_order - self._backorders

    def _snapshot_a(self) -> EngineState:
        return EngineState(
            t=self._t,
            on_hand=self._on_hand,
            on_order=self._on_order,
            backorders=self._backorders,
            inventory_position=self._inventory_position,
        )

    def _final_snapshot(self) -> EngineState:
        # Post-step-5 of the last period. The caller should ignore the value
        # and look at done=True; included for API uniformity.
        return EngineState(
            t=self._t,
            on_hand=self._on_hand,
            on_order=self._on_order,
            backorders=self._backorders,
            inventory_position=self._inventory_position,
        )

    # ---- Demand draw ----

    def _draw_demand(self) -> float:
        # Composes the base demand distribution with the pattern overlay.
        # Both ``_demand`` and ``_pattern`` are constructed
        # once in ``__init__``. The pattern receives the base draw and the
        # integer period index ``self._t``; the default ``StationaryPattern``
        # returns ``base`` unchanged, so existing scenarios that omit the
        # ``pattern:`` YAML block produce byte-for-byte identical demand
        # streams. Concrete non-identity patterns (Seasonal / Trending /
        # Intermittent / Lumpy) plug in here purely
        # additively — no engine-loop change required.
        return self._pattern.apply(self._demand.draw(), self._t)
