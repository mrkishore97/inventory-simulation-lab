"""Tests for the (s, S) min-max policy."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.config import SSPolicyConfig
from core.simulation import EngineState
from policies.sS import SSPolicy


def _make_policy(s: float = 20.0, S: float = 60.0) -> SSPolicy:
    return SSPolicy(SSPolicyConfig(reorder_point=s, order_up_to=S))


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


class TestSSPolicyDecision:
    def test_orders_S_minus_IP_when_ip_below_s(self) -> None:
        # IP=15, s=20, S=60 → order 60 − 15 = 45.
        assert _make_policy(s=20.0, S=60.0).decide(_state(15.0)) == 45.0

    def test_orders_zero_when_ip_above_s(self) -> None:
        assert _make_policy(s=20.0, S=60.0).decide(_state(30.0)) == 0.0

    def test_orders_when_ip_equals_s(self) -> None:
        # Boundary inclusive: textbook (s,S) fires at IP == s, order = S − s.
        assert _make_policy(s=20.0, S=60.0).decide(_state(20.0)) == 40.0

    def test_just_above_s_no_order(self) -> None:
        # The boundary is inclusive on the *low* side only.
        assert _make_policy(s=20.0, S=60.0).decide(_state(20.0 + 1e-9)) == 0.0

    def test_orders_more_than_S_when_ip_negative(self) -> None:
        # Backorder regime: IP can go negative; (s,S) tops up to S, so the
        # actual order quantity is S − IP > S.
        assert _make_policy(s=20.0, S=60.0).decide(_state(-100.0)) == 160.0

    def test_negative_s_with_positive_S(self) -> None:
        """Negative reorder_point is allowed (backorder mode); rule unchanged."""
        policy = SSPolicy(SSPolicyConfig(reorder_point=-50.0, order_up_to=10.0))
        # IP = -100 ≤ -50 → order 10 − (-100) = 110.
        assert policy.decide(_state(-100.0)) == 110.0
        # IP = -50 (boundary) → order 10 − (-50) = 60.
        assert policy.decide(_state(-50.0)) == 60.0
        # IP = 0 (above s=-50) → no order.
        assert policy.decide(_state(0.0)) == 0.0

    @pytest.mark.parametrize(
        "ip,expected",
        [(1e9, 0.0), (-1e9, 60.0 + 1e9)],
    )
    def test_extreme_inventory_positions(self, ip: float, expected: float) -> None:
        # Stable at extreme magnitudes (relevant for M5 disruptions). The
        # property test below covers the full range; this case acts as
        # documented invariant.
        assert _make_policy(s=20.0, S=60.0).decide(_state(ip)) == expected

    def test_only_ip_drives_decision(self) -> None:
        # Vary on_hand / on_order / backorders while keeping IP constant;
        # decision must be invariant. Proves SSPolicy reads only IP.
        policy = _make_policy(s=20.0, S=60.0)
        action_a = policy.decide(_state(15.0, on_hand=15.0, on_order=0.0, backorders=0.0))
        action_b = policy.decide(_state(15.0, on_hand=100.0, on_order=20.0, backorders=105.0))
        assert action_a == action_b == 45.0  # 60 − 15

    def test_determinism(self) -> None:
        policy = _make_policy()
        state = _state(10.0)
        assert policy.decide(state) == policy.decide(state) == policy.decide(state)


class TestSSPolicyConstruction:
    def test_construction_from_config(self) -> None:
        cfg = SSPolicyConfig(reorder_point=42.5, order_up_to=70.0)
        policy = SSPolicy(cfg)
        # IP = 42.0 ≤ s = 42.5 → order S − IP = 70.0 − 42.0 = 28.0.
        assert policy.decide(_state(42.0)) == 28.0
        # IP = 43.0 > s = 42.5 → no order.
        assert policy.decide(_state(43.0)) == 0.0


class TestSSPolicyRepr:
    def test_repr_format(self) -> None:
        # CLI runner and experiment tracking depend on this being human-
        # readable and including both parameters.
        policy = _make_policy(s=20.0, S=60.0)
        rep = repr(policy)
        assert "SSPolicy" in rep
        assert "20" in rep
        assert "60" in rep


class TestSSPolicyProperty:
    @given(
        s=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        S_offset=st.floats(min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False),
        ip=st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
    )
    def test_decision_rule_invariant(self, s: float, S_offset: float, ip: float) -> None:
        # Property: action is exactly 0 or exactly (S − IP), with the trigger
        # rule IP ≤ s. With s ≥ 0 and S = s + S_offset > 0, PositiveFloat
        # holds for S; the s < S validator holds because S_offset > 0.
        S = s + S_offset
        policy = SSPolicy(SSPolicyConfig(reorder_point=s, order_up_to=S))
        action = policy.decide(_state(ip))
        if ip <= s:
            assert action == S - ip
            # s < S enforced by validator and ip ≤ s, so action = S − ip > 0.
            assert action > 0
        else:
            assert action == 0.0
