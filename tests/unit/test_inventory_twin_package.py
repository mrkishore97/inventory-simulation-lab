"""Smoke tests for the ``inventory_twin`` public re-export package.

Pins the public surface so consumers depending on
``from inventory_twin import RunConfig`` are not silently broken by an internal
layout refactor.
"""

from __future__ import annotations

import inventory_twin
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


class TestPublicReExports:
    def test_re_exports_resolve_to_internal_classes(self) -> None:
        assert inventory_twin.RunConfig is RunConfig
        assert inventory_twin.InventoryEngine is InventoryEngine
        assert inventory_twin.InventoryEnv is InventoryEnv
        assert inventory_twin.EngineState is EngineState
        assert inventory_twin.Policy is Policy
        assert inventory_twin.SQPolicy is SQPolicy
        assert inventory_twin.SSPolicy is SSPolicy
        assert inventory_twin.BaseStockPolicy is BaseStockPolicy
        assert inventory_twin.PeriodicReviewPolicy is PeriodicReviewPolicy
        assert inventory_twin.Demand is Demand
        assert inventory_twin.Pattern is Pattern
        assert inventory_twin.LeadTime is LeadTime
        assert inventory_twin.Disruption is Disruption
        assert inventory_twin.make_demand is make_demand
        assert inventory_twin.make_pattern is make_pattern
        assert inventory_twin.make_lead_time is make_lead_time
        assert inventory_twin.make_disruption is make_disruption
        assert inventory_twin.make_policy is make_policy

    def test_dunder_all_lists_full_surface(self) -> None:
        assert set(inventory_twin.__all__) == {
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
        }

    def test_concrete_demand_pattern_and_leadtime_classes_not_in_public_facade(self) -> None:
        # Concrete demand, pattern, and lead-time classes are deliberately NOT
        # pulled up to the inventory_twin facade. Consumers configure them via
        # YAML; the implementation classes are plumbing behind ``make_demand`` /
        # ``make_pattern`` / ``make_lead_time``. If a future change exports any
        # of them, this test fires and forces a memory / documentation update
        # first. The lead-time chassis was introduced with
        # ``DeterministicLeadTime``; concrete lead-time classes follow the same
        # internal-only convention as concrete demand and pattern classes —
        # the Demand Public API Convention now spans all three axes.
        for name in (
            "NormalDemand",
            "PoissonDemand",
            "NegativeBinomialDemand",
            "GammaDemand",
            "LognormalDemand",
            "EmpiricalDemand",
            "StationaryPattern",
            "SeasonalPattern",
            "TrendingPattern",
            "DeterministicLeadTime",
            "NoDisruption",
            "ScheduledDisruption",
        ):
            assert not hasattr(inventory_twin, name), (
                f"{name} unexpectedly exported from inventory_twin facade"
            )
            assert name not in inventory_twin.__all__, (
                f"{name} unexpectedly in inventory_twin.__all__"
            )

    def test_version_is_nonempty_string(self) -> None:
        assert isinstance(inventory_twin.__version__, str)
        assert inventory_twin.__version__
