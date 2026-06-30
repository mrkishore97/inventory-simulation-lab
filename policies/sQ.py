"""(s, Q) reorder-point policy."""

from __future__ import annotations

from core.config import SQPolicyConfig
from core.simulation import EngineState
from policies.base import Policy


class SQPolicy(Policy):
    """Order ``Q`` whenever inventory position falls to or below ``s``.

    The classical reorder-point rule. Fires a single ``Q``-sized order per
    period when ``IP <= s`` (boundary inclusive); multi-Q "catch-up"
    ordering ((s, nQ)) is a different policy and is not implemented here.
    Reads only ``state.inventory_position`` from the snapshot; other fields
    are ignored.
    """

    def __init__(self, config: SQPolicyConfig) -> None:
        self._s: float = float(config.reorder_point)
        self._Q: float = float(config.order_quantity)

    def decide(self, state: EngineState) -> float:
        return self._Q if state.inventory_position <= self._s else 0.0

    def __repr__(self) -> str:
        return f"SQPolicy(s={self._s}, Q={self._Q})"
