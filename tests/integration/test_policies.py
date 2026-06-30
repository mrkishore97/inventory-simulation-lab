"""Integration tests for policy + engine end-to-end runs.

These tests prove that classical policies compose with the engine to
produce a valid trajectory. Pure-policy logic lives in
``tests/unit/test_sQ_policy.py``; pure-engine physics lives in
``tests/integration/test_engine.py``. This file owns the seam between
them.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from core.config import (
    BaseStockPolicyConfig,
    DeterministicLeadTimeConfig,
    PeriodicReviewPolicyConfig,
    RunConfig,
    SQPolicyConfig,
    SSPolicyConfig,
)
from core.simulation import InventoryEngine
from policies.base_stock import BaseStockPolicy
from policies.periodic_review import PeriodicReviewPolicy
from policies.sQ import SQPolicy
from policies.sS import SSPolicy


def _run_policy(config: RunConfig) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    assert isinstance(config.policy, SQPolicyConfig)
    policy = SQPolicy(config.policy)
    engine = InventoryEngine(config)
    state = engine.initial_observation()
    done = False
    while not done:
        state, done = engine.step(policy.decide(state))
    return cast(
        "np.ndarray[tuple[int, int], np.dtype[np.float64]]",
        engine.ledger.to_dataframe().to_numpy(),
    )


class TestSQPolicyDriven:
    def test_sQ_policy_full_run(self, basic_config: RunConfig) -> None:
        # Drive the engine to horizon end with (s,Q). Verify ledger
        # invariants, that the policy actually fires, and that costs
        # accumulate to a finite positive total.
        assert isinstance(basic_config.policy, SQPolicyConfig)
        policy = SQPolicy(basic_config.policy)
        engine = InventoryEngine(basic_config)
        state = engine.initial_observation()
        done = False
        while not done:
            state, done = engine.step(policy.decide(state))

        df = engine.ledger.to_dataframe()
        # Loop completed for the full horizon.
        assert len(df) == basic_config.simulation.horizon
        # Policy actually fires at least once across the run — otherwise
        # the test would be vacuously a no-op rollout.
        assert (df["order_placed"] > 0).any()
        # Convention C: sales never exceed demand.
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        # Physical: on_hand never negative under the engine contract.
        assert (df["on_hand"] >= 0.0).all()
        # No NaN/inf anywhere — would mean trajectory poisoning.
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()
        # Total cost is finite and positive.
        total_cost = (
            df["holding_cost"] + df["ordering_cost"] + df["purchase_cost"] + df["stockout_cost"]
        ).sum()
        assert np.isfinite(total_cost) and total_cost > 0

    def test_determinism_under_policy(self, basic_config: RunConfig) -> None:
        # Two parallel runs with the same config + same policy must
        # produce bit-identical ledgers.
        np.testing.assert_array_equal(_run_policy(basic_config), _run_policy(basic_config))


class TestSSPolicyDriven:
    """End-to-end tests for the (s,S) policy driving the engine."""

    @staticmethod
    def _ss_config(basic_config: RunConfig, *, s: float = 20.0, S: float = 60.0) -> RunConfig:
        return basic_config.model_copy(
            update={"policy": SSPolicyConfig(reorder_point=s, order_up_to=S)}
        )

    def test_sS_policy_full_run(self, basic_config: RunConfig) -> None:
        cfg = self._ss_config(basic_config)
        assert isinstance(cfg.policy, SSPolicyConfig)
        policy = SSPolicy(cfg.policy)
        engine = InventoryEngine(cfg)
        state = engine.initial_observation()
        done = False
        while not done:
            state, done = engine.step(policy.decide(state))

        df = engine.ledger.to_dataframe()
        assert len(df) == cfg.simulation.horizon
        # Policy fires at least once across the run.
        assert (df["order_placed"] > 0).any()
        # Convention C: sales never exceed demand.
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        # Physical: on_hand never negative under the engine contract.
        assert (df["on_hand"] >= 0.0).all()
        # No NaN/inf anywhere — would mean trajectory poisoning.
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()
        total_cost = (
            df["holding_cost"] + df["ordering_cost"] + df["purchase_cost"] + df["stockout_cost"]
        ).sum()
        assert np.isfinite(total_cost) and total_cost > 0

    def test_post_order_inventory_position_equals_S(self, basic_config: RunConfig) -> None:
        """(s,S) property: when the policy fires, post-order IP equals S exactly.

        The Ledger row at period t (Point B) records ``on_order`` *after* step 4
        and ``inventory_position`` accordingly. So on the periods where
        ``order_placed > 0``, the (s,S) rule guarantees Point-B
        ``inventory_position == order_up_to`` (a.k.a. ``S``). on_hand and
        backorders are unchanged between Point A and Point B, so the post-
        order IP delta is exactly the order quantity.
        """
        S_target = 60.0
        cfg = self._ss_config(basic_config, s=20.0, S=S_target)
        assert isinstance(cfg.policy, SSPolicyConfig)
        policy = SSPolicy(cfg.policy)
        engine = InventoryEngine(cfg)
        state = engine.initial_observation()
        done = False
        while not done:
            state, done = engine.step(policy.decide(state))

        df = engine.ledger.to_dataframe()
        ordering_periods = df[df["order_placed"] > 0]
        # Sanity: the policy fired at least once.
        assert len(ordering_periods) > 0
        # Point-B IP at every ordering period equals S exactly.
        np.testing.assert_allclose(
            ordering_periods["inventory_position"].to_numpy(),
            S_target,
            rtol=1e-10,
        )

    def test_determinism_under_policy(self, basic_config: RunConfig) -> None:
        cfg = self._ss_config(basic_config)
        assert isinstance(cfg.policy, SSPolicyConfig)

        runs = []
        for _ in range(2):
            policy = SSPolicy(cfg.policy)
            engine = InventoryEngine(cfg)
            state = engine.initial_observation()
            done = False
            while not done:
                state, done = engine.step(policy.decide(state))
            runs.append(engine.ledger.to_dataframe().to_numpy())

        np.testing.assert_array_equal(runs[0], runs[1])


class TestBaseStockPolicyDriven:
    """End-to-end tests for the base-stock (order-up-to-S) policy."""

    @staticmethod
    def _bs_config(basic_config: RunConfig, *, S: float = 60.0) -> RunConfig:
        return basic_config.model_copy(update={"policy": BaseStockPolicyConfig(target_level=S)})

    def test_base_stock_full_run(self, basic_config: RunConfig) -> None:
        cfg = self._bs_config(basic_config)
        assert isinstance(cfg.policy, BaseStockPolicyConfig)
        policy = BaseStockPolicy(cfg.policy)
        engine = InventoryEngine(cfg)
        state = engine.initial_observation()
        done = False
        while not done:
            state, done = engine.step(policy.decide(state))

        df = engine.ledger.to_dataframe()
        assert len(df) == cfg.simulation.horizon
        # Policy fires at least once across the run.
        assert (df["order_placed"] > 0).any()
        # Convention C: sales never exceed demand.
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        # Physical: on_hand never negative under the engine contract.
        assert (df["on_hand"] >= 0.0).all()
        # No NaN/inf anywhere — would mean trajectory poisoning.
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()
        total_cost = (
            df["holding_cost"] + df["ordering_cost"] + df["purchase_cost"] + df["stockout_cost"]
        ).sum()
        assert np.isfinite(total_cost) and total_cost > 0

    def test_post_order_inventory_position_equals_S(self, basic_config: RunConfig) -> None:
        """Base-stock property: every period that fires raises Point-B IP to S exactly.

        Same shape as the (s,S) post-order property — both target S — but
        for base-stock the firing rule is "any IP < S" rather than "IP ≤ s".
        """
        S_target = 60.0
        cfg = self._bs_config(basic_config, S=S_target)
        assert isinstance(cfg.policy, BaseStockPolicyConfig)
        policy = BaseStockPolicy(cfg.policy)
        engine = InventoryEngine(cfg)
        state = engine.initial_observation()
        done = False
        while not done:
            state, done = engine.step(policy.decide(state))

        df = engine.ledger.to_dataframe()
        ordering_periods = df[df["order_placed"] > 0]
        # Sanity: the policy fired at least once.
        assert len(ordering_periods) > 0
        # Point-B IP at every ordering period equals S exactly.
        np.testing.assert_allclose(
            ordering_periods["inventory_position"].to_numpy(),
            S_target,
            rtol=1e-10,
        )

    def test_high_firing_rate_distinguishes_from_sS(self, basic_config: RunConfig) -> None:
        """Base-stock has no dead zone, so it fires far more often than (s,S).

        Concrete contrast: same target S=60 and same fixture; base-stock fires
        every period after the first arrival lands (no dead zone), while
        (s,S) with reorder_point=20 has a [20, 60] dead zone where it sits
        idle. After warm-up (allow lead_time periods for the first arrival
        to start moving IP), base-stock should fire on > 90% of periods.
        """
        cfg = self._bs_config(basic_config, S=60.0)
        assert isinstance(cfg.policy, BaseStockPolicyConfig)
        policy = BaseStockPolicy(cfg.policy)
        engine = InventoryEngine(cfg)
        state = engine.initial_observation()
        done = False
        while not done:
            state, done = engine.step(policy.decide(state))

        df = engine.ledger.to_dataframe()
        # basic_config uses a deterministic lead time; narrow the union before
        # reading .lead_time (LeadTimeConfig became a discriminated union).
        assert isinstance(cfg.lead_time, DeterministicLeadTimeConfig)
        warmup = cfg.lead_time.lead_time + 1
        post_warmup = df.iloc[warmup:]
        firing_rate = (post_warmup["order_placed"] > 0).mean()
        # > 90% — under positive Normal(10, 2) demand the IP drops below
        # S every period, so base-stock fires effectively always after
        # warm-up. (s,S) with the same params would sit in its dead zone
        # for many of these periods.
        assert firing_rate > 0.9, (
            f"base-stock firing rate {firing_rate:.2f} below 0.9 — dead-zone "
            f"behavior would imply this is not actually base-stock"
        )

    def test_determinism_under_policy(self, basic_config: RunConfig) -> None:
        cfg = self._bs_config(basic_config)
        assert isinstance(cfg.policy, BaseStockPolicyConfig)

        runs = []
        for _ in range(2):
            policy = BaseStockPolicy(cfg.policy)
            engine = InventoryEngine(cfg)
            state = engine.initial_observation()
            done = False
            while not done:
                state, done = engine.step(policy.decide(state))
            runs.append(engine.ledger.to_dataframe().to_numpy())

        np.testing.assert_array_equal(runs[0], runs[1])


class TestPeriodicReviewPolicyDriven:
    """End-to-end tests for the (R,S) periodic-review policy."""

    @staticmethod
    def _rs_config(basic_config: RunConfig, *, R: int = 7, S: float = 60.0) -> RunConfig:
        return basic_config.model_copy(
            update={"policy": PeriodicReviewPolicyConfig(review_period=R, order_up_to=S)}
        )

    def test_RS_policy_full_run(self, basic_config: RunConfig) -> None:
        cfg = self._rs_config(basic_config)
        assert isinstance(cfg.policy, PeriodicReviewPolicyConfig)
        policy = PeriodicReviewPolicy(cfg.policy)
        engine = InventoryEngine(cfg)
        state = engine.initial_observation()
        done = False
        while not done:
            state, done = engine.step(policy.decide(state))

        df = engine.ledger.to_dataframe()
        assert len(df) == cfg.simulation.horizon
        # Policy fires at least once across the run.
        assert (df["order_placed"] > 0).any()
        # Convention C: sales never exceed demand.
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        # Physical: on_hand never negative under the engine contract.
        assert (df["on_hand"] >= 0.0).all()
        # No NaN/inf anywhere.
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()
        total_cost = (
            df["holding_cost"] + df["ordering_cost"] + df["purchase_cost"] + df["stockout_cost"]
        ).sum()
        assert np.isfinite(total_cost) and total_cost > 0

    def test_orders_only_at_review_periods(self, basic_config: RunConfig) -> None:
        """The (R,S) signature: every firing lands on a review period.

        Filter ledger rows where ``order_placed > 0`` and assert all of
        them have ``period % R == 0``. No other policy in the suite passes
        this test — (s,Q), (s,S), and base-stock are all IP-driven and
        will fire on arbitrary periods.

        The converse direction (every review period fires) does NOT hold:
        if a review lands on an IP >= S window, the gap clip suppresses
        the order. Asserting only "every firing is at a review" is the
        correct strict invariant.
        """
        R = 7
        cfg = self._rs_config(basic_config, R=R, S=60.0)
        assert isinstance(cfg.policy, PeriodicReviewPolicyConfig)
        policy = PeriodicReviewPolicy(cfg.policy)
        engine = InventoryEngine(cfg)
        state = engine.initial_observation()
        done = False
        while not done:
            state, done = engine.step(policy.decide(state))

        df = engine.ledger.to_dataframe()
        firing_periods = df.loc[df["order_placed"] > 0, "period"].astype(int).to_numpy()
        # Sanity: at least one firing — the test is otherwise vacuous.
        assert firing_periods.size > 0
        # Every firing landed on a review period.
        assert (firing_periods % R == 0).all(), (
            f"non-review firings detected: {firing_periods[firing_periods % R != 0]}"
        )

    def test_post_order_inventory_position_equals_S_at_firing_reviews(
        self, basic_config: RunConfig
    ) -> None:
        """Same Point-B invariant as (s,S) and base-stock — the gap clip
        means we filter to firing periods first, then assert IP == S."""
        S_target = 60.0
        cfg = self._rs_config(basic_config, R=7, S=S_target)
        assert isinstance(cfg.policy, PeriodicReviewPolicyConfig)
        policy = PeriodicReviewPolicy(cfg.policy)
        engine = InventoryEngine(cfg)
        state = engine.initial_observation()
        done = False
        while not done:
            state, done = engine.step(policy.decide(state))

        df = engine.ledger.to_dataframe()
        ordering_periods = df[df["order_placed"] > 0]
        assert len(ordering_periods) > 0
        np.testing.assert_allclose(
            ordering_periods["inventory_position"].to_numpy(),
            S_target,
            rtol=1e-10,
        )

    def test_R_equals_horizon_fires_only_at_t_zero(self, basic_config: RunConfig) -> None:
        """Degenerate: when R == horizon, the only review at-or-before
        ``horizon - 1`` is t=0. ``(horizon - 1) % horizon != 0`` for
        horizon > 1, so the loop never re-enters a review.

        Also requires the t=0 review to have IP < S so the gap clip
        doesn't suppress the sole expected firing — basic_config has
        ``initial_on_hand = 100`` so we lift S above that to force a
        startup order.
        """
        H = basic_config.simulation.horizon
        S_above_initial = float(basic_config.simulation.initial_on_hand) + 50.0
        cfg = self._rs_config(basic_config, R=H, S=S_above_initial)
        assert isinstance(cfg.policy, PeriodicReviewPolicyConfig)
        policy = PeriodicReviewPolicy(cfg.policy)
        engine = InventoryEngine(cfg)
        state = engine.initial_observation()
        done = False
        while not done:
            state, done = engine.step(policy.decide(state))

        df = engine.ledger.to_dataframe()
        firing_periods = df.loc[df["order_placed"] > 0, "period"].astype(int).to_numpy()
        # Exactly one firing, at t=0.
        assert firing_periods.tolist() == [0]

    def test_determinism_under_policy(self, basic_config: RunConfig) -> None:
        cfg = self._rs_config(basic_config)
        assert isinstance(cfg.policy, PeriodicReviewPolicyConfig)

        runs = []
        for _ in range(2):
            policy = PeriodicReviewPolicy(cfg.policy)
            engine = InventoryEngine(cfg)
            state = engine.initial_observation()
            done = False
            while not done:
                state, done = engine.step(policy.decide(state))
            runs.append(engine.ledger.to_dataframe().to_numpy())

        np.testing.assert_array_equal(runs[0], runs[1])
