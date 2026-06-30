"""Base-stock (order-up-to-S) periodic-review policy."""

from __future__ import annotations

from core.config import BaseStockPolicyConfig
from core.simulation import EngineState
from policies.base import Policy


class BaseStockPolicy(Policy):
    """Order ``max(S − inventory_position, 0.0)`` every period.

    The classical order-up-to rule. Distinct from (s,S): there is no reorder
    threshold, so under positive demand the policy fires every period (no
    dead zone). Optimal under the K=0 special case of the periodic-review
    newsvendor setting (no fixed ordering cost). Reads only
    ``state.inventory_position`` from the snapshot; other fields are ignored.
    """

    def __init__(self, config: BaseStockPolicyConfig) -> None:
        self._S: float = float(config.target_level)

    def decide(self, state: EngineState) -> float:
        gap = self._S - state.inventory_position
        return gap if gap > 0.0 else 0.0

    def __repr__(self) -> str:
        return f"BaseStockPolicy(S={self._S})"
