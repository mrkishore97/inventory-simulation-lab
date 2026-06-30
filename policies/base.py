"""Classical policy abstract base.

A Policy maps the engine's Point-A snapshot (post-fulfillment, pre-order)
to a non-negative order quantity. Classical policies (subclass Policy) plug
into ``InventoryEngine`` directly via the CLI runner; RL agents bypass this
ABC and talk to ``InventoryEnv``.

Why ``decide`` takes ``EngineState`` and not the env's ndarray observation:
classical policies are formulas over named fields (``inventory_position``,
``on_hand``); forcing them to index ``obs[3]`` would silently break if
``InventoryEnv._encode`` ever changes field order, and reads worse. RL
agents speak ndarray natively to ``InventoryEnv``; classical policies speak
``EngineState`` natively to the engine. Two contracts, one per audience.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.simulation import EngineState


class Policy(ABC):
    """Abstract base for classical reorder policies."""

    @abstractmethod
    def decide(self, state: EngineState) -> float:
        """Return the non-negative order quantity for this period.

        ``state`` is the post-fulfillment, pre-order snapshot (Observation
        Point A; see :mod:`core.simulation`). Implementations must return
        a finite, non-negative ``float``; the engine validates
        ``order_placed >= 0`` and ``InventoryEnv`` validates finiteness.
        """
        ...
