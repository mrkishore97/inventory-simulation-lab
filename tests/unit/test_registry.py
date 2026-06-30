"""Tests for the policy factory in ``policies.registry``.

The factory starts with the (s,Q) arm, then widens the union with
(s,S), base-stock, and (R,S) periodic-review. Each policy
adds one dispatch test plus an entry in the factory-passthrough property
suite below.
"""

from __future__ import annotations

import pytest

from core.config import (
    BaseStockPolicyConfig,
    PeriodicReviewPolicyConfig,
    SimulationConfig,
    SQPolicyConfig,
    SSPolicyConfig,
)
from core.simulation import EngineState
from policies.base import Policy
from policies.base_stock import BaseStockPolicy
from policies.periodic_review import PeriodicReviewPolicy
from policies.registry import make_policy
from policies.sQ import SQPolicy
from policies.sS import SSPolicy


def _state(ip: float) -> EngineState:
    """Build an EngineState whose only meaningful field is inventory_position.

    Both SQPolicy and SSPolicy read only ``state.inventory_position``; the
    other fields are filler. EngineState is a frozen dataclass with no
    IP-identity enforcement, so this is safe for unit-testing the policy's
    decision surface.
    """
    return EngineState(t=0, on_hand=0.0, on_order=0.0, backorders=0.0, inventory_position=ip)


class TestMakePolicy:
    def test_dispatches_sQ_config_to_SQPolicy_instance(self) -> None:
        cfg = SQPolicyConfig(reorder_point=20.0, order_quantity=30)
        policy = make_policy(cfg)
        assert isinstance(policy, SQPolicy)

    def test_dispatches_sS_config_to_SSPolicy_instance(self) -> None:
        cfg = SSPolicyConfig(reorder_point=20.0, order_up_to=60.0)
        policy = make_policy(cfg)
        assert isinstance(policy, SSPolicy)

    def test_dispatches_base_stock_config_to_BaseStockPolicy_instance(self) -> None:
        cfg = BaseStockPolicyConfig(target_level=60.0)
        policy = make_policy(cfg)
        assert isinstance(policy, BaseStockPolicy)

    def test_dispatches_RS_config_to_PeriodicReviewPolicy_instance(self) -> None:
        cfg = PeriodicReviewPolicyConfig(review_period=7, order_up_to=60.0)
        policy = make_policy(cfg)
        assert isinstance(policy, PeriodicReviewPolicy)

    def test_returns_policy_abc_instance(self) -> None:
        cfg = SQPolicyConfig(reorder_point=20.0, order_quantity=30)
        policy = make_policy(cfg)
        assert isinstance(policy, Policy)

    def test_passes_config_params_through(self) -> None:
        cfg = SQPolicyConfig(reorder_point=15.5, order_quantity=42)
        policy = make_policy(cfg)
        # IP at the inclusive boundary triggers an order of Q.
        assert policy.decide(_state(15.5)) == 42.0
        # IP just above the boundary does not.
        assert policy.decide(_state(15.5 + 1e-9)) == 0.0

    def test_factory_passthrough_sQ_matches_direct_construction(self) -> None:
        """Pin: factory adds zero behavior beyond direct construction.

        If a future bullet ever wraps the factory output (adapter, decorator,
        post-processing), this test fails immediately and forces an explicit
        decision instead of silent behavior change.
        """
        cfg = SQPolicyConfig(reorder_point=20.0, order_quantity=30)
        factory_policy = make_policy(cfg)
        direct_policy = SQPolicy(cfg)
        for ip in [-1e6, -100.0, -1.0, 0.0, 19.999, 20.0, 20.001, 100.0, 1e6]:
            assert factory_policy.decide(_state(ip)) == direct_policy.decide(_state(ip)), (
                f"factory diverged from direct construction at IP={ip}"
            )

    def test_factory_passthrough_sS_matches_direct_construction(self) -> None:
        """Same passthrough property, (s,S) variant."""
        cfg = SSPolicyConfig(reorder_point=20.0, order_up_to=60.0)
        factory_policy = make_policy(cfg)
        direct_policy = SSPolicy(cfg)
        for ip in [-1e6, -100.0, -1.0, 0.0, 19.999, 20.0, 20.001, 100.0, 1e6]:
            assert factory_policy.decide(_state(ip)) == direct_policy.decide(_state(ip)), (
                f"factory diverged from direct construction at IP={ip}"
            )

    def test_factory_passthrough_base_stock_matches_direct_construction(self) -> None:
        """Same passthrough property, base-stock variant."""
        cfg = BaseStockPolicyConfig(target_level=60.0)
        factory_policy = make_policy(cfg)
        direct_policy = BaseStockPolicy(cfg)
        for ip in [-1e6, -100.0, -1.0, 0.0, 59.999, 60.0, 60.001, 100.0, 1e6]:
            assert factory_policy.decide(_state(ip)) == direct_policy.decide(_state(ip)), (
                f"factory diverged from direct construction at IP={ip}"
            )

    def test_factory_passthrough_RS_matches_direct_construction(self) -> None:
        """Same passthrough property, (R,S) variant.

        (R,S) reads both ``state.t`` and ``state.inventory_position``, so
        the battery iterates over (t, IP) pairs covering review (t=0, 7,
        14) and non-review (t=1, 6, 8) periods plus a range of IPs.
        """
        cfg = PeriodicReviewPolicyConfig(review_period=7, order_up_to=60.0)
        factory_policy = make_policy(cfg)
        direct_policy = PeriodicReviewPolicy(cfg)
        for t in [0, 1, 6, 7, 8, 14]:
            for ip in [-1e6, -100.0, 0.0, 59.999, 60.0, 60.001, 1e6]:
                state = EngineState(
                    t=t, on_hand=0.0, on_order=0.0, backorders=0.0, inventory_position=ip
                )
                assert factory_policy.decide(state) == direct_policy.decide(state), (
                    f"factory diverged from direct construction at t={t}, IP={ip}"
                )

    def test_unregistered_config_raises(self) -> None:
        """Defensive guard: a non-PolicyConfig object raises ValueError.

        Bypasses static typing via ``# type: ignore`` — the test exercises the
        runtime safety net when an unmapped config slips through (e.g. via a
        callsite that bypassed Pydantic discriminator validation).
        """
        fake = SimulationConfig(horizon=10, initial_on_hand=0)
        with pytest.raises(ValueError, match="Unknown policy config"):
            make_policy(fake)  # type: ignore[arg-type]
