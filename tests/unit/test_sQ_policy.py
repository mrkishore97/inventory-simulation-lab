"""Tests for the (s, Q) reorder-point policy."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.config import SQPolicyConfig
from core.simulation import EngineState
from policies.sQ import SQPolicy


def _make_policy(s: float = 20.0, Q: int = 50) -> SQPolicy:
    return SQPolicy(SQPolicyConfig(reorder_point=s, order_quantity=Q))


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


class TestSQPolicyDecision:
    def test_orders_Q_when_ip_below_s(self) -> None:
        assert _make_policy(s=20.0, Q=50).decide(_state(10.0)) == 50.0

    def test_orders_zero_when_ip_above_s(self) -> None:
        assert _make_policy(s=20.0, Q=50).decide(_state(30.0)) == 0.0

    def test_orders_Q_when_ip_equals_s(self) -> None:
        # Boundary inclusive: textbook (s,Q) fires at IP == s.
        assert _make_policy(s=20.0, Q=50).decide(_state(20.0)) == 50.0

    def test_orders_Q_when_ip_negative(self) -> None:
        # Single Q, not catch-up nQ. Even at IP=-100 the classical rule
        # places one Q-sized order; subsequent periods catch up.
        assert _make_policy(s=20.0, Q=50).decide(_state(-100.0)) == 50.0

    @pytest.mark.parametrize("ip,expected", [(1e9, 0.0), (-1e9, 50.0)])
    def test_extreme_inventory_positions(self, ip: float, expected: float) -> None:
        # Stable at extreme magnitudes — relevant for M5 disruption
        # scenarios where backorders may explode. Hypothesis covers the
        # full range; this case acts as documented invariant.
        assert _make_policy(s=20.0, Q=50).decide(_state(ip)) == expected

    def test_only_ip_drives_decision(self) -> None:
        # Vary on_hand / on_order / backorders while keeping IP constant;
        # decision must be invariant. Proves SQPolicy reads only IP.
        policy = _make_policy(s=20.0, Q=50)
        action_a = policy.decide(_state(15.0, on_hand=15.0, on_order=0.0, backorders=0.0))
        action_b = policy.decide(_state(15.0, on_hand=100.0, on_order=20.0, backorders=105.0))
        assert action_a == action_b == 50.0

    def test_determinism(self) -> None:
        policy = _make_policy()
        state = _state(10.0)
        assert policy.decide(state) == policy.decide(state) == policy.decide(state)


class TestSQPolicyConstruction:
    def test_construction_from_config(self) -> None:
        cfg = SQPolicyConfig(reorder_point=42.5, order_quantity=17)
        policy = SQPolicy(cfg)
        # Spot-check via behavior: IP just below s -> Q; just above -> 0.
        assert policy.decide(_state(42.0)) == 17.0
        assert policy.decide(_state(43.0)) == 0.0


class TestSQPolicyRepr:
    def test_repr_format(self) -> None:
        # CLI runner (bullet 8) and experiment tracking depend on this
        # being human-readable and including both parameters.
        policy = _make_policy(s=20.0, Q=50)
        rep = repr(policy)
        assert "SQPolicy" in rep
        assert "20" in rep
        assert "50" in rep


class TestSQPolicyProperty:
    @given(
        ip=st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
        s=st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
        Q=st.integers(min_value=1, max_value=1_000_000),
    )
    def test_action_is_zero_or_Q(self, ip: float, s: float, Q: int) -> None:
        policy = SQPolicy(SQPolicyConfig(reorder_point=s, order_quantity=Q))
        action = policy.decide(_state(ip))
        # Action is exactly 0.0 or exactly Q (no clipping, no scaling).
        assert action == 0.0 or action == float(Q)
        # Rule: order iff IP <= s.
        if ip <= s:
            assert action == float(Q)
        else:
            assert action == 0.0
