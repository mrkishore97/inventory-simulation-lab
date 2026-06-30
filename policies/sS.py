"""(s, S) min-max reorder policy."""

from __future__ import annotations

from core.config import SSPolicyConfig
from core.simulation import EngineState
from policies.base import Policy


class SSPolicy(Policy):
    """Order ``S − inventory_position`` whenever inventory position falls to or below ``s``.

    The classical min-max rule. Triggered the same way as (s,Q) — boundary-
    inclusive ``IP <= s`` — but the order quantity tops the inventory position
    back up to ``S`` rather than firing a fixed ``Q``. ``SSPolicyConfig`` enforces
    ``s < S`` via a ``model_validator``, so the order quantity is always strictly
    positive at the trigger; no ``max(0, …)`` clip needed here. Reads only
    ``state.inventory_position`` from the snapshot; other fields are ignored.
    """

    def __init__(self, config: SSPolicyConfig) -> None:
        self._s: float = float(config.reorder_point)
        self._S: float = float(config.order_up_to)

    def decide(self, state: EngineState) -> float:
        if state.inventory_position <= self._s:
            return self._S - state.inventory_position
        return 0.0

    def __repr__(self) -> str:
        return f"SSPolicy(s={self._s}, S={self._S})"
