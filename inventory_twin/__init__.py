"""Public API for the inventory-twin simulator.

Internal modules live at the repo root (``core/``, ``policies/``); this package
re-exports the stable public surface so consumers can write::

    from inventory_twin import RunConfig, InventoryEnv, SQPolicy

without depending on internal layout. Internal code continues to use
``from core.X`` / ``from policies.X`` directly.
"""

from __future__ import annotations

from core.config import RunConfig
from core.environment import InventoryEnv
from core.simulation import EngineState, InventoryEngine
from demand.base import Demand
from demand.patterns.base import Pattern
from demand.patterns.registry import make_pattern
from demand.registry import make_demand
from leadtime.base import LeadTime
from leadtime.disruptions.base import Disruption
from leadtime.disruptions.registry import make_disruption
from leadtime.registry import make_lead_time
from policies.base import Policy
from policies.base_stock import BaseStockPolicy
from policies.periodic_review import PeriodicReviewPolicy
from policies.registry import make_policy
from policies.sQ import SQPolicy
from policies.sS import SSPolicy

__version__ = "0.1.0"

__all__ = [
    "BaseStockPolicy",
    "Demand",
    "Disruption",
    "EngineState",
    "InventoryEngine",
    "InventoryEnv",
    "LeadTime",
    "Pattern",
    "PeriodicReviewPolicy",
    "Policy",
    "RunConfig",
    "SQPolicy",
    "SSPolicy",
    "__version__",
    "make_demand",
    "make_disruption",
    "make_lead_time",
    "make_pattern",
    "make_policy",
]
