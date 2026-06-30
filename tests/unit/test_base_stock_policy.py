"""Tests for the base-stock (order-up-to-S) policy."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.config import BaseStockPolicyConfig
from core.simulation import EngineState
from policies.base_stock import BaseStockPolicy


def _make_policy(S: float = 60.0) -> BaseStockPolicy:
    return BaseStockPolicy(BaseStockPolicyConfig(target_level=S))


def _state(
    ip: float,
    *,
    on_hand: float = 0.0,
    on_order: float = 0.0,
    backorders: float = 0.0,
) -> EngineState:
    return EngineState(
        t=0,
        on_hand=on_hand,
        on_order=on_order,
        backorders=backorders,
        inventory_position=ip,
    )


class TestBaseStockPolicyDecision:
    def test_orders_S_minus_IP_when_ip_below_S(self) -> None:
        # IP=20, S=60 → order 60 − 20 = 40.
        assert _make_policy(S=60.0).decide(_state(20.0)) == 40.0

    def test_orders_zero_when_ip_equals_S(self) -> None:
        # Boundary at IP == S: gap is exactly 0, no order. The classical
        # rule fires only when there is a strict positive gap to fill.
        assert _make_policy(S=60.0).decide(_state(60.0)) == 0.0

    def test_orders_zero_when_ip_above_S(self) -> None:
        # Overstocked relative to target — no order, no clip needed
        # because the policy returns 0.0 directly (never negative).
        assert _make_policy(S=60.0).decide(_state(80.0)) == 0.0

    def test_orders_above_S_when_ip_negative(self) -> None:
        # Backorder regime: IP can go negative; base-stock tops up to S, so
        # the actual order quantity is S − IP > S.
        assert _make_policy(S=60.0).decide(_state(-15.0)) == 75.0

    def test_no_dead_zone_just_below_S(self) -> None:
        """Sharp distinction from (s,S): base-stock fires for ANY gap > 0.

        (s,S) with reorder_point < S has a [s, S] dead zone where it does
        not fire even though IP < S. Base-stock fires whenever IP < S, no
        matter how small the gap. This test pins the no-dead-zone property.
        """
        policy = _make_policy(S=60.0)
        assert policy.decide(_state(60.0 - 1e-9)) == pytest.approx(1e-9)
        assert policy.decide(_state(59.999)) == pytest.approx(0.001)

    @pytest.mark.parametrize(
        "ip,expected",
        [(1e9, 0.0), (-1e9, 60.0 + 1e9)],
    )
    def test_extreme_inventory_positions(self, ip: float, expected: float) -> None:
        # Stable at extreme magnitudes (relevant for M5 disruptions).
        assert _make_policy(S=60.0).decide(_state(ip)) == expected

    def test_only_ip_drives_decision(self) -> None:
        # Vary on_hand / on_order / backorders while keeping IP constant;
        # decision must be invariant. Proves BaseStockPolicy reads only IP.
        policy = _make_policy(S=60.0)
        action_a = policy.decide(_state(20.0, on_hand=20.0, on_order=0.0, backorders=0.0))
        action_b = policy.decide(_state(20.0, on_hand=100.0, on_order=20.0, backorders=100.0))
        assert action_a == action_b == 40.0  # 60 − 20

    def test_determinism(self) -> None:
        policy = _make_policy()
        state = _state(10.0)
        assert policy.decide(state) == policy.decide(state) == policy.decide(state)


class TestBaseStockPolicyConstruction:
    def test_construction_from_config(self) -> None:
        cfg = BaseStockPolicyConfig(target_level=50.0)
        policy = BaseStockPolicy(cfg)
        # IP = 30 → order 50 − 30 = 20.
        assert policy.decide(_state(30.0)) == 20.0
        # IP = 50 → boundary, no order.
        assert policy.decide(_state(50.0)) == 0.0


class TestBaseStockPolicyRepr:
    def test_repr_format(self) -> None:
        # CLI runner and experiment tracking depend on this being human-
        # readable and including the single parameter.
        policy = _make_policy(S=60.0)
        rep = repr(policy)
        assert "BaseStockPolicy" in rep
        assert "60" in rep


class TestBaseStockPolicyProperty:
    @given(
        S=st.floats(min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False),
        ip=st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
    )
    def test_decision_rule_invariant(self, S: float, ip: float) -> None:
        # Property: action == max(S − IP, 0.0) exactly. No tolerance — the
        # policy is closed-form arithmetic on a single subtraction.
        policy = BaseStockPolicy(BaseStockPolicyConfig(target_level=S))
        action = policy.decide(_state(ip))
        if ip < S:
            assert action == S - ip
            assert action > 0
        else:
            # ip >= S: no order, exact zero (not just non-negative).
            assert action == 0.0
