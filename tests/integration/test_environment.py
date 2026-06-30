"""Integration tests for InventoryEnv (Gymnasium-compatible wrapper).

End-to-end checks of the wrapper's contract: spaces, lifecycle, observation
encoding, reward emission, action validation, determinism inheritance, M1
constraint propagation, and the Phase-2 readiness gate.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import pytest
from pydantic import ValidationError

from core.config import RunConfig
from core.environment import InventoryEnv
from core.simulation import InventoryEngine


class TestSpaces:
    def test_observation_space_bounds(self, basic_config: RunConfig) -> None:
        env = InventoryEnv(basic_config)
        space = env.observation_space
        assert isinstance(space, gym.spaces.Box)
        assert space.shape == (4,)
        np.testing.assert_array_equal(space.low, [0.0, 0.0, 0.0, -np.inf])
        np.testing.assert_array_equal(space.high, [np.inf, np.inf, np.inf, np.inf])
        assert space.dtype == np.float64

    def test_action_space_bounds(self, basic_config: RunConfig) -> None:
        env = InventoryEnv(basic_config)
        space = env.action_space
        assert isinstance(space, gym.spaces.Box)
        assert space.shape == (1,)
        assert space.low[0] == 0.0
        assert space.high[0] == 1e6
        assert space.dtype == np.float64

    def test_obs_in_observation_space_each_period(self, basic_config: RunConfig) -> None:
        env = InventoryEnv(basic_config)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        for _ in range(20):
            obs, _, _, _, _ = env.step(np.array([0.0]))
            assert env.observation_space.contains(obs)


class TestLifecycle:
    def test_reset_returns_obs_and_info(self, basic_config: RunConfig) -> None:
        env = InventoryEnv(basic_config)
        obs, info = env.reset()
        assert obs.shape == (4,)
        assert obs.dtype == np.float64
        assert info == {"period": 0}

    def test_step_before_reset_raises(self, basic_config: RunConfig) -> None:
        env = InventoryEnv(basic_config)
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.array([0.0]))

    def test_ledger_before_reset_raises(self, basic_config: RunConfig) -> None:
        env = InventoryEnv(basic_config)
        with pytest.raises(RuntimeError, match="reset"):
            _ = env.ledger

    def test_step_returns_5tuple(self, basic_config: RunConfig) -> None:
        env = InventoryEnv(basic_config)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.array([5.0]))
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        # info["period"] is the index of the next decision period.
        assert info == {"period": 1}

    def test_horizon_end_truncates_not_terminates(self, basic_config: RunConfig) -> None:
        env = InventoryEnv(basic_config)
        env.reset()
        horizon = basic_config.simulation.horizon
        truncated_at = -1
        for t in range(horizon):
            _, _, terminated, truncated, _ = env.step(np.array([0.0]))
            assert terminated is False, f"terminated set at period {t}; should always be False"
            if truncated:
                truncated_at = t
                break
        assert truncated_at == horizon - 1


class TestObservationEncoding:
    def test_obs_matches_engine_state_field_by_field(self, basic_config: RunConfig) -> None:
        # Two parallel runs (env vs raw engine, same config) must produce the
        # same Point-A snapshot at every period — proves the wrapper passes
        # the engine state through faithfully.
        env = InventoryEnv(basic_config)
        engine = InventoryEngine(basic_config)
        obs, _ = env.reset()
        state = engine.initial_observation()
        np.testing.assert_allclose(
            obs,
            [state.on_hand, state.on_order, state.backorders, state.inventory_position],
        )
        for t in range(basic_config.simulation.horizon - 1):
            qty = 30.0 if t % 5 == 0 else 0.0
            obs, _, _, _, _ = env.step(np.array([qty]))
            state, _ = engine.step(qty)
            np.testing.assert_allclose(
                obs,
                [state.on_hand, state.on_order, state.backorders, state.inventory_position],
            )


class TestReward:
    def test_reward_equals_negative_total_cost(self, basic_config: RunConfig) -> None:
        env = InventoryEnv(basic_config)
        env.reset()
        rewards = []
        for t in range(basic_config.simulation.horizon):
            qty = 30.0 if t % 5 == 0 else 0.0
            _, reward, _, _, _ = env.step(np.array([qty]))
            rewards.append(reward)
        df = env.ledger.to_dataframe()
        expected = -(
            df["holding_cost"] + df["ordering_cost"] + df["purchase_cost"] + df["stockout_cost"]
        )
        np.testing.assert_allclose(rewards, expected.to_numpy())


class TestActionValidation:
    def test_wrong_shape_raises(self, basic_config: RunConfig) -> None:
        env = InventoryEnv(basic_config)
        env.reset()
        with pytest.raises(ValueError, match="shape"):
            env.step(np.array([10.0, 20.0]))  # shape (2,)
        with pytest.raises(ValueError, match="shape"):
            env.step(np.array(5.0))  # shape ()

    def test_negative_value_propagates_engine_error(self, basic_config: RunConfig) -> None:
        env = InventoryEnv(basic_config)
        env.reset()
        with pytest.raises(ValueError, match="must be >= 0"):
            env.step(np.array([-1.0]))

    def test_nan_action_raises(self, basic_config: RunConfig) -> None:
        # NaN bypasses `< 0` (nan comparisons are always False) and would
        # silently poison the trajectory — the env must catch it explicitly.
        env = InventoryEnv(basic_config)
        env.reset()
        with pytest.raises(ValueError, match="finite"):
            env.step(np.array([np.nan]))

    def test_inf_action_raises(self, basic_config: RunConfig) -> None:
        env = InventoryEnv(basic_config)
        env.reset()
        with pytest.raises(ValueError, match="finite"):
            env.step(np.array([np.inf]))


class TestDeterminism:
    def test_same_config_same_trajectory(self, basic_config: RunConfig) -> None:
        env_a = InventoryEnv(basic_config)
        env_b = InventoryEnv(basic_config)
        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()
        np.testing.assert_array_equal(obs_a, obs_b)
        for t in range(basic_config.simulation.horizon - 1):
            action = np.array([30.0 if t % 5 == 0 else 0.0])
            obs_a, r_a, _, _, _ = env_a.step(action)
            obs_b, r_b, _, _, _ = env_b.step(action)
            np.testing.assert_array_equal(obs_a, obs_b)
            assert r_a == r_b

    def test_seed_argument_overrides_config(self, basic_config: RunConfig) -> None:
        # config.master_seed = 42 (per fixture). reset(seed=999) should produce
        # a different demand stream and therefore different rewards.
        env = InventoryEnv(basic_config)
        env.reset(seed=42)
        rewards_42 = [env.step(np.array([0.0]))[1] for _ in range(20)]
        env.reset(seed=999)
        rewards_999 = [env.step(np.array([0.0]))[1] for _ in range(20)]
        assert rewards_42 != rewards_999


class TestM1ConstraintInheritance:
    def test_initial_on_order_propagates_at_reset(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        cfg = make_config(initial_on_order=5)
        env = InventoryEnv(cfg)  # construction succeeds; engine is built lazily.
        with pytest.raises(NotImplementedError, match="initial_on_order"):
            env.reset()

    def test_lead_time_zero_rejected_at_config_load(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        # The L>=1 rule was relocated from engine __init__ to a config
        # PositiveInt, so lead_time=0 now fails when make_config builds the
        # RunConfig — before InventoryEnv is even constructed. This is no longer
        # an engine constraint the env inherits lazily; it is a config invariant.
        with pytest.raises(ValidationError):
            make_config(lead_time=0)


class TestPhase2ReadinessGate:
    """The Phase-2 readiness gate.

    Two complementary 100-step rollouts. ``test_random_policy_rollout_no_errors``
    drives a bounded (<=50) hand-rolled policy — a realistic small-order smoke
    test, also the companion. ``test_gym_random_policy_rollout`` drives a
    canonical Gymnasium random policy sampled from the declared ``action_space``
    — the literal ``RandomPolicy`` gate that makes the Definition-of-Done
    checkbox unambiguous.
    """

    def test_random_policy_rollout_no_errors(self, basic_config: RunConfig) -> None:
        # a 100-step random-policy rollout must complete
        # without errors. This is the gate that says the wrapper can be
        # handed to an RL training loop.
        env = InventoryEnv(basic_config)
        env.reset(seed=0)
        rng = np.random.default_rng(42)
        for _ in range(100):
            action = np.array([float(rng.uniform(0, 50))])
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()

    def test_gym_random_policy_rollout(self, basic_config: RunConfig) -> None:
        # THE Phase-2 readiness gate: a canonical Gymnasium
        # random policy (env.action_space.sample()) drives the env for 100 steps
        # without an invalid observation, a non-finite reward, or a runtime error.
        # Distinct from the bounded <=50 rollout above — the sampled actions may
        # include very large order quantities, matching the declared RL action
        # space, so this exercises the env the way an untrained agent calling
        # action_space.sample() does on step 1. (basic_config horizon is 50, so
        # the 100-step loop crosses an episode boundary and exercises reset.)
        env = InventoryEnv(basic_config)
        env.action_space.seed(0)  # reproducible random policy
        try:
            obs, _ = env.reset(seed=0)
            assert env.observation_space.contains(obs)
            steps = 0
            for _ in range(100):
                action = env.action_space.sample()  # the Gymnasium random policy
                obs, reward, terminated, truncated, _ = env.step(action)
                assert env.observation_space.contains(obs)
                assert np.isfinite(reward)
                steps += 1
                if terminated or truncated:
                    obs, _ = env.reset()
                    assert env.observation_space.contains(obs)
            assert steps == 100
        finally:
            env.close()

    def test_full_pipeline_under_100ms(self, make_config: Callable[..., RunConfig]) -> None:
        # Single-run target: 365-period run through the
        # full Gym wrapper stack in < 100 ms.
        cfg = make_config(horizon=365)
        env = InventoryEnv(cfg)
        start = time.perf_counter()
        env.reset()
        for t in range(365):
            qty = 30.0 if t % 7 == 0 else 0.0
            env.step(np.array([qty]))
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"full pipeline took {elapsed:.3f}s, expected < 0.1s"
