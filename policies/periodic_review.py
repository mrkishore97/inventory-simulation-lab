"""(R,S) periodic-review order-up-to policy.

Categorically distinct from the IP-driven policies in this package: the
firing schedule is gated by the clock (``state.t % R == 0``), not the
inventory level. Between reviews the policy is silent regardless of how
low IP drops. ``Policy.decide`` stays stateless — the clock lives on
``EngineState.t``, not in the policy — so determinism, reseed, and reset
all work without policy-internal counters.

Period 0 IS a review period under ``state.t % R == 0`` (0 % R == 0 for
all positive R). If ``initial_on_hand >= order_up_to`` the t=0 review
returns 0 anyway; otherwise a startup order fires. Both behaviors are
intentional under the locked rule.
"""

from __future__ import annotations

from core.config import PeriodicReviewPolicyConfig
from core.simulation import EngineState
from policies.base import Policy


class PeriodicReviewPolicy(Policy):
    """Order ``S − inventory_position`` when ``state.t % R == 0``, else ``0``.

    The ``gap > 0.0`` clip protects against a review period landing on an
    IP > S window (e.g. right after a large arrival): returning a negative
    ``S − IP`` would violate the engine's ``order_placed >= 0`` invariant.
    Reads only ``state.t`` and ``state.inventory_position``; other fields
    are ignored.
    """

    def __init__(self, config: PeriodicReviewPolicyConfig) -> None:
        self._R: int = int(config.review_period)
        self._S: float = float(config.order_up_to)

    def decide(self, state: EngineState) -> float:
        if state.t % self._R != 0:
            return 0.0
        gap = self._S - state.inventory_position
        return gap if gap > 0.0 else 0.0

    def __repr__(self) -> str:
        return f"PeriodicReviewPolicy(R={self._R}, S={self._S})"
