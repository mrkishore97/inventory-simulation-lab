"""Tests for the (R,S) periodic-review order-up-to policy."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.config import PeriodicReviewPolicyConfig
from core.simulation import EngineState
from policies.periodic_review import PeriodicReviewPolicy


def _make_policy(R: int = 7, S: float = 60.0) -> PeriodicReviewPolicy:
    return PeriodicReviewPolicy(PeriodicReviewPolicyConfig(review_period=R, order_up_to=S))


def _state(
    t: int,
    ip: float,
    *,
    on_hand: float = 0.0,
    on_order: float = 0.0,
    backorders: float = 0.0,
) -> EngineState:
    return EngineState(
        t=t,
        on_hand=on_hand,
        on_order=on_order,
        backorders=backorders,
        inventory_position=ip,
    )


class TestPeriodicReviewPolicyDecision:
    def test_fires_at_t_zero_when_below_S(self) -> None:
        # Period 0 IS a review (0 % 7 == 0). With IP=20 < S=60, order 40.
        assert _make_policy(R=7, S=60.0).decide(_state(0, 20.0)) == 40.0

    @pytest.mark.parametrize("t", [1, 2, 3, 4, 5, 6])
    def test_silent_between_reviews(self, t: int) -> None:
        # The clock, not the level, gates firing. Even with IP deeply below
        # S, a non-review period returns 0.
        assert _make_policy(R=7, S=60.0).decide(_state(t, 10.0)) == 0.0

    @pytest.mark.parametrize("t", [0, 7, 14, 21, 70])
    def test_fires_at_review_periods(self, t: int) -> None:
        # Every multiple-of-R period is a review; with IP=20, S=60, order 40.
        assert _make_policy(R=7, S=60.0).decide(_state(t, 20.0)) == 40.0

    def test_zero_when_review_period_and_IP_above_S(self) -> None:
        # The gap > 0 clip — at a review with IP > S, no order fires.
        assert _make_policy(R=7, S=60.0).decide(_state(7, 80.0)) == 0.0

    def test_zero_when_review_period_and_IP_equals_S(self) -> None:
        # Boundary: gap is exactly 0, no order.
        assert _make_policy(R=7, S=60.0).decide(_state(7, 60.0)) == 0.0

    @pytest.mark.parametrize("t", [0, 1, 2, 3, 4, 5, 6, 7, 100])
    def test_R_equals_1_fires_every_period(self, t: int) -> None:
        # Degenerate corner: with R=1, every period is a review. Functionally
        # equivalent to base-stock. Lock the corner case.
        assert _make_policy(R=1, S=60.0).decide(_state(t, 20.0)) == 40.0

    def test_orders_above_S_when_ip_negative_at_review(self) -> None:
        # Backorder regime: IP can go negative; the policy tops up to S, so
        # the actual order quantity is S − IP > S.
        assert _make_policy(R=7, S=60.0).decide(_state(7, -10.0)) == 70.0

    def test_only_t_and_ip_drive_decision(self) -> None:
        # Vary on_hand / on_order / backorders while keeping (t, IP) fixed;
        # decision must be invariant. Proves the policy reads only these two
        # fields.
        policy = _make_policy(R=7, S=60.0)
        a = policy.decide(_state(7, 20.0, on_hand=20.0, on_order=0.0, backorders=0.0))
        b = policy.decide(_state(7, 20.0, on_hand=200.0, on_order=80.0, backorders=300.0))
        assert a == b == 40.0

    def test_determinism(self) -> None:
        policy = _make_policy()
        state = _state(7, 20.0)
        assert policy.decide(state) == policy.decide(state) == policy.decide(state)

    def test_extreme_R_value(self) -> None:
        # Large R: only t=0 and t=1000 are reviews within reasonable horizons.
        policy = _make_policy(R=1000, S=60.0)
        assert policy.decide(_state(0, 20.0)) == 40.0
        assert policy.decide(_state(999, 20.0)) == 0.0
        assert policy.decide(_state(1000, 20.0)) == 40.0


class TestPeriodicReviewPolicyConstruction:
    def test_construction_from_config(self) -> None:
        cfg = PeriodicReviewPolicyConfig(review_period=14, order_up_to=70.0)
        policy = PeriodicReviewPolicy(cfg)
        assert policy.decide(_state(0, 30.0)) == 40.0
        assert policy.decide(_state(7, 30.0)) == 0.0
        assert policy.decide(_state(14, 30.0)) == 40.0


class TestPeriodicReviewPolicyRepr:
    def test_repr_format(self) -> None:
        policy = _make_policy(R=7, S=60.0)
        rep = repr(policy)
        assert "PeriodicReviewPolicy" in rep
        assert "7" in rep
        assert "60" in rep


class TestPeriodicReviewPolicyProperty:
    @given(
        R=st.integers(min_value=1, max_value=100),
        S=st.floats(min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False),
        t=st.integers(min_value=0, max_value=1000),
        ip=st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
    )
    def test_decision_rule_invariant(self, R: int, S: float, t: int, ip: float) -> None:
        # Property: action == (S − IP) exactly when (t % R == 0 AND IP < S);
        # else action == 0. No tolerance — closed-form arithmetic.
        policy = PeriodicReviewPolicy(PeriodicReviewPolicyConfig(review_period=R, order_up_to=S))
        action = policy.decide(_state(t, ip))
        is_review = (t % R) == 0
        if is_review and ip < S:
            assert action == S - ip
            assert action > 0
        else:
            assert action == 0.0
