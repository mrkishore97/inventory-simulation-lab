"""Integration tests for core.simulation.InventoryEngine.

These tests drive the engine with fixed action sequences and assert that the
materialized Ledger satisfies the locked invariants.
The (s,Q) policy is *not* implemented yet (bullet 7); these tests verify
engine mechanics, not policy behavior.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from core.config import RunConfig
from core.simulation import EngineState, InventoryEngine


def _drive(engine: InventoryEngine, actions: list[float]) -> list[EngineState]:
    """Run an engine to completion with a fixed action sequence.

    Returns the list of per-period EngineStates the policy would have seen
    (Observation Point A): one per period, in order.
    """
    obs_history = [engine.initial_observation()]
    for action in actions[:-1]:
        obs, done = engine.step(action)
        assert not done, "engine finished before final action"
        obs_history.append(obs)
    _, done = engine.step(actions[-1])
    assert done, "engine did not finish on final action"
    return obs_history


class TestEngineLifecycle:
    def test_runs_to_completion(self, basic_config: RunConfig) -> None:
        engine = InventoryEngine(basic_config)
        actions = [0.0] * basic_config.simulation.horizon
        _drive(engine, actions)
        df = engine.ledger.to_dataframe()
        assert len(df) == basic_config.simulation.horizon

    def test_initial_observation_called_twice_raises(self, basic_config: RunConfig) -> None:
        engine = InventoryEngine(basic_config)
        engine.initial_observation()
        with pytest.raises(RuntimeError, match="already called"):
            engine.initial_observation()

    def test_step_before_init_raises(self, basic_config: RunConfig) -> None:
        engine = InventoryEngine(basic_config)
        with pytest.raises(RuntimeError, match="initial_observation"):
            engine.step(0.0)

    def test_step_past_horizon_raises(self, basic_config: RunConfig) -> None:
        engine = InventoryEngine(basic_config)
        actions = [0.0] * basic_config.simulation.horizon
        _drive(engine, actions)
        with pytest.raises(RuntimeError, match="finished"):
            engine.step(0.0)


class TestActionValidation:
    def test_negative_action_raises(self, basic_config: RunConfig) -> None:
        # Without this guard, action=-10 would silently skip ordering (the
        # `if action > 0` branch) but still record `purchase_cost = c × -10`,
        # paying a negative reward back to the caller — free money for an RL
        # agent that hallucinated a negative quantity.
        engine = InventoryEngine(basic_config)
        engine.initial_observation()
        with pytest.raises(ValueError, match="must be >= 0"):
            engine.step(-1.0)


class TestM1Constraints:
    def test_initial_on_order_nonzero_raises(self, make_config: Callable[..., RunConfig]) -> None:
        cfg = make_config(initial_on_order=5)
        with pytest.raises(NotImplementedError, match="initial_on_order"):
            InventoryEngine(cfg)

    def test_lead_time_zero_rejected_at_config_load(self) -> None:
        # The L>=1 rule was relocated from engine __init__ to a config
        # PositiveInt, so lead_time=0 now fails when the config is built (here,
        # inside the make_config fixture) — a ValidationError, not the old
        # engine-init NotImplementedError.
        from core.config import DeterministicLeadTimeConfig

        with pytest.raises(ValidationError):
            DeterministicLeadTimeConfig(lead_time=0)


class TestLeadTime:
    """Normal stochastic lead time at the engine layer."""

    def test_engine_runs_with_normal_lead_time(self, make_config: Callable[..., RunConfig]) -> None:
        # Standard invariant battery; cost is observed, never
        # asserted. NormalLeadTime consumes the "lead_time" stream, so order
        # arrival offsets vary per order.
        from core.config import NormalLeadTimeConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(update={"lead_time": NormalLeadTimeConfig(mean=3.0, std=1.0)})
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * cfg.simulation.horizon)

        df = engine.ledger.to_dataframe()
        assert len(df) == cfg.simulation.horizon
        assert (df["demand"] >= 0.0).all()
        assert (df["on_hand"] >= 0.0).all()
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()

    def test_normal_lead_time_determinism(self, make_config: Callable[..., RunConfig]) -> None:
        from core.config import NormalLeadTimeConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(update={"lead_time": NormalLeadTimeConfig(mean=3.0, std=1.0)})
        engine_a = InventoryEngine(cfg)
        engine_b = InventoryEngine(cfg)
        _drive(engine_a, [0.0] * cfg.simulation.horizon)
        _drive(engine_b, [0.0] * cfg.simulation.horizon)
        pd.testing.assert_frame_equal(
            engine_a.ledger.to_dataframe(), engine_b.ledger.to_dataframe()
        )

    def test_engine_runs_with_gamma_lead_time(self, make_config: Callable[..., RunConfig]) -> None:
        # Second stochastic lead-time arm. Standard invariant
        # battery; cost observed, never asserted.
        from core.config import GammaLeadTimeConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(update={"lead_time": GammaLeadTimeConfig(shape=9.0, scale=1.0 / 3.0)})
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * cfg.simulation.horizon)

        df = engine.ledger.to_dataframe()
        assert len(df) == cfg.simulation.horizon
        assert (df["demand"] >= 0.0).all()
        assert (df["on_hand"] >= 0.0).all()
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()

    def test_gamma_lead_time_determinism(self, make_config: Callable[..., RunConfig]) -> None:
        from core.config import GammaLeadTimeConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(update={"lead_time": GammaLeadTimeConfig(shape=9.0, scale=1.0 / 3.0)})
        engine_a = InventoryEngine(cfg)
        engine_b = InventoryEngine(cfg)
        _drive(engine_a, [0.0] * cfg.simulation.horizon)
        _drive(engine_b, [0.0] * cfg.simulation.horizon)
        pd.testing.assert_frame_equal(
            engine_a.ledger.to_dataframe(), engine_b.ledger.to_dataframe()
        )

    def test_engine_runs_with_lognormal_lead_time(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        # Third/final stochastic lead-time arm. Standard invariant
        # battery; cost observed, never asserted.
        from core.config import LognormalLeadTimeConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(
            update={"lead_time": LognormalLeadTimeConfig(mu=1.045931, sigma=0.324593)}
        )
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * cfg.simulation.horizon)

        df = engine.ledger.to_dataframe()
        assert len(df) == cfg.simulation.horizon
        assert (df["demand"] >= 0.0).all()
        assert (df["on_hand"] >= 0.0).all()
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()

    def test_lognormal_lead_time_determinism(self, make_config: Callable[..., RunConfig]) -> None:
        from core.config import LognormalLeadTimeConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(
            update={"lead_time": LognormalLeadTimeConfig(mu=1.045931, sigma=0.324593)}
        )
        engine_a = InventoryEngine(cfg)
        engine_b = InventoryEngine(cfg)
        _drive(engine_a, [0.0] * cfg.simulation.horizon)
        _drive(engine_b, [0.0] * cfg.simulation.horizon)
        pd.testing.assert_frame_equal(
            engine_a.ledger.to_dataframe(), engine_b.ledger.to_dataframe()
        )


class TestDisruption:
    """lead-time disruption overlay at the engine layer.

    Periodic orders (30 units every 5th period) place replenishments at periods
    30/35/.../55 inside the [30, 60) disruption window, so a scheduled x2
    disruption measurably changes arrival timing and the Ledger.
    """

    @staticmethod
    def _periodic_actions(horizon: int) -> list[float]:
        return [30.0 if t % 5 == 0 else 0.0 for t in range(horizon)]

    def test_engine_runs_with_scheduled_disruption(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        # Standard invariant battery on a disrupted run; cost observed,
        # never asserted.
        from core.config import DisruptionWindow, ScheduledDisruptionConfig

        cfg = make_config(horizon=90, master_seed=42).model_copy(
            update={
                "disruption": ScheduledDisruptionConfig(
                    windows=[DisruptionWindow(start=30, duration=30, multiplier=2.0)]
                )
            }
        )
        engine = InventoryEngine(cfg)
        _drive(engine, self._periodic_actions(cfg.simulation.horizon))

        df = engine.ledger.to_dataframe()
        assert len(df) == cfg.simulation.horizon
        assert (df["demand"] >= 0.0).all()
        assert (df["on_hand"] >= 0.0).all()
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()

    def test_scheduled_disruption_determinism(self, make_config: Callable[..., RunConfig]) -> None:
        from core.config import DisruptionWindow, ScheduledDisruptionConfig

        cfg = make_config(horizon=90, master_seed=42).model_copy(
            update={
                "disruption": ScheduledDisruptionConfig(
                    windows=[DisruptionWindow(start=30, duration=30, multiplier=2.0)]
                )
            }
        )
        engine_a = InventoryEngine(cfg)
        engine_b = InventoryEngine(cfg)
        actions = self._periodic_actions(cfg.simulation.horizon)
        _drive(engine_a, actions)
        _drive(engine_b, actions)
        pd.testing.assert_frame_equal(
            engine_a.ledger.to_dataframe(), engine_b.ledger.to_dataframe()
        )

    def test_no_disruption_default_matches_baseline(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        # The NoDisruption default must produce a byte-identical Ledger to the
        # same run with the disruption block explicitly set to NoDisruption —
        # the backward-compat thesis at the engine layer.
        from core.config import NoDisruptionConfig

        base = make_config(horizon=90, master_seed=42)
        explicit = base.model_copy(update={"disruption": NoDisruptionConfig()})
        actions = self._periodic_actions(base.simulation.horizon)

        engine_base = InventoryEngine(base)
        engine_explicit = InventoryEngine(explicit)
        _drive(engine_base, actions)
        _drive(engine_explicit, actions)
        pd.testing.assert_frame_equal(
            engine_base.ledger.to_dataframe(), engine_explicit.ledger.to_dataframe()
        )

    def test_scheduled_disruption_changes_ledger(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        # A scheduled x2 disruption over [30, 60) delays replenishments placed
        # inside the window, so the Ledger differs from the no-disruption
        # baseline (same seed, same actions). A non-cost behavioral assertion
        # that the disruption actually acts — cost magnitude is observed,
        # never asserted.
        from core.config import DisruptionWindow, ScheduledDisruptionConfig

        base = make_config(horizon=90, master_seed=42)
        disrupted = base.model_copy(
            update={
                "disruption": ScheduledDisruptionConfig(
                    windows=[DisruptionWindow(start=30, duration=30, multiplier=2.0)]
                )
            }
        )
        actions = self._periodic_actions(base.simulation.horizon)

        engine_base = InventoryEngine(base)
        engine_disrupted = InventoryEngine(disrupted)
        _drive(engine_base, actions)
        _drive(engine_disrupted, actions)
        df_base = engine_base.ledger.to_dataframe()
        df_disrupted = engine_disrupted.ledger.to_dataframe()

        # Deliveries shift later under the x2 disruption: the order_received
        # timeline must differ from the baseline.
        assert not df_base["order_received"].equals(df_disrupted["order_received"])


class TestDeterminism:
    def test_same_seed_byte_identical_ledgers(self, basic_config: RunConfig) -> None:
        actions = [0.0 if t % 5 else 30.0 for t in range(basic_config.simulation.horizon)]
        engine_a = InventoryEngine(basic_config)
        engine_b = InventoryEngine(basic_config)
        _drive(engine_a, actions)
        _drive(engine_b, actions)
        df_a = engine_a.ledger.to_dataframe()
        df_b = engine_b.ledger.to_dataframe()
        assert df_a.equals(df_b)

    def test_different_seed_differs(self, make_config: Callable[..., RunConfig]) -> None:
        cfg_a = make_config(master_seed=42, horizon=100)
        cfg_b = make_config(master_seed=43, horizon=100)
        actions = [0.0] * 100
        engine_a = InventoryEngine(cfg_a)
        engine_b = InventoryEngine(cfg_b)
        _drive(engine_a, actions)
        _drive(engine_b, actions)
        df_a = engine_a.ledger.to_dataframe()
        df_b = engine_b.ledger.to_dataframe()
        assert not df_a["demand"].equals(df_b["demand"])


class TestFlowConservation:
    """Engine-level invariants."""

    def _run_with_periodic_orders(
        self, config: RunConfig, *, q: float = 30.0, every: int = 5
    ) -> pd.DataFrame:
        actions = [q if t % every == 0 else 0.0 for t in range(config.simulation.horizon)]
        engine = InventoryEngine(config)
        _drive(engine, actions)
        return engine.ledger.to_dataframe()

    def test_inventory_position_identity(self, basic_config: RunConfig) -> None:
        df = self._run_with_periodic_orders(basic_config)
        ip = df["on_hand"] + df["on_order"] - df["backorders"]
        assert np.allclose(df["inventory_position"], ip)

    def test_on_hand_recurrence_invariant(self, basic_config: RunConfig) -> None:
        df = self._run_with_periodic_orders(basic_config)
        prev_oh = np.concatenate(
            ([float(basic_config.simulation.initial_on_hand)], df["on_hand"].to_numpy()[:-1])
        )
        rhs = (
            prev_oh
            + df["order_received"].to_numpy()
            - df["sales"].to_numpy()
            - df["backorders_cleared"].to_numpy()
        )
        assert np.allclose(df["on_hand"], rhs)

    def test_on_order_recurrence_invariant(self, basic_config: RunConfig) -> None:
        df = self._run_with_periodic_orders(basic_config)
        prev_oo = np.concatenate(
            ([0.0], df["on_order"].to_numpy()[:-1])
        )  # initial_on_order required to be 0 in M1
        rhs = prev_oo - df["order_received"].to_numpy() + df["order_placed"].to_numpy()
        assert np.allclose(df["on_order"], rhs)

    def test_demand_fulfillment_invariant(self, basic_config: RunConfig) -> None:
        df = self._run_with_periodic_orders(basic_config)
        prev_bo = np.concatenate(([0.0], df["backorders"].to_numpy()[:-1]))
        delta_bo = df["backorders"].to_numpy() - prev_bo
        rhs = (
            df["sales"].to_numpy()
            + df["lost_sales"].to_numpy()
            + delta_bo
            + df["backorders_cleared"].to_numpy()
        )
        assert np.allclose(df["demand"], rhs)


class TestConventionC:
    def test_sales_never_exceeds_demand(self, basic_config: RunConfig) -> None:
        # The whole reason Convention C exists: separating bo_cleared from
        # sales lets sales[t] <= demand[t] always.
        engine = InventoryEngine(basic_config)
        actions = [0.0] * basic_config.simulation.horizon
        _drive(engine, actions)
        df = engine.ledger.to_dataframe()
        assert (df["sales"] <= df["demand"] + 1e-9).all()

    def test_backorders_cleared_drains_existing_backorders(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        # Scenario: start with no inventory, accumulate backorders for several
        # periods, then trigger a large arrival. backorders_cleared > 0 in the
        # arrival period, and total backorders should drop.
        cfg = make_config(
            horizon=15,
            initial_on_hand=0,
            demand_mean=10.0,
            demand_std=0.0,  # deterministic demand
            lead_time=3,
        )
        engine = InventoryEngine(cfg)
        actions = [0.0] * 15
        actions[0] = 200.0  # one big order at t=0, arrives at t=3
        _drive(engine, actions)
        df = engine.ledger.to_dataframe()

        # Periods 0-2: backorders accumulate (no inventory, no arrivals yet).
        assert df.loc[0, "backorders_cleared"] == 0.0
        assert df.loc[1, "backorders_cleared"] == 0.0
        assert df.loc[2, "backorders_cleared"] == 0.0
        bo_at_t2 = float(df.loc[2, "backorders"])
        assert bo_at_t2 > 0

        # Period 3: arrival lands; backorders_cleared > 0 reduces bo balance.
        bo_cleared_t3 = float(df.loc[3, "backorders_cleared"])
        assert bo_cleared_t3 > 0
        # And on_hand[t-1] + received - sales - bo_cleared = on_hand[t] (per
        # the recurrence; covered by test_on_hand_recurrence too).


class TestModeBehavior:
    def test_backorder_mode_no_lost_sales(self, basic_config: RunConfig) -> None:
        engine = InventoryEngine(basic_config)
        actions = [0.0] * basic_config.simulation.horizon
        _drive(engine, actions)
        df = engine.ledger.to_dataframe()
        assert (df["lost_sales"] == 0.0).all()

    def test_lost_sales_mode_no_backorders_or_clearance(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        cfg = make_config(backorder_policy="lost_sales", initial_on_hand=20)
        engine = InventoryEngine(cfg)
        actions = [0.0] * cfg.simulation.horizon
        _drive(engine, actions)
        df = engine.ledger.to_dataframe()
        assert (df["backorders"] == 0.0).all()
        assert (df["backorders_cleared"] == 0.0).all()


class TestCostAccounting:
    def test_costs_match_locked_conventions(self, make_config: Callable[..., RunConfig]) -> None:
        # Deterministic demand (std=0), simple sequence: verify each cost
        # column equals the formula column-by-column.
        cfg = make_config(
            horizon=10,
            initial_on_hand=50,
            demand_mean=8.0,
            demand_std=0.0,
            lead_time=2,
        )
        engine = InventoryEngine(cfg)
        actions = [0.0, 30.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        _drive(engine, actions)
        df = engine.ledger.to_dataframe()

        h = cfg.costs.holding_per_unit_per_period
        K = cfg.costs.ordering_fixed
        c = cfg.costs.unit_cost
        b = cfg.costs.backorder_per_unit_per_period

        assert np.allclose(df["holding_cost"], h * df["on_hand"])
        expected_ordering = np.where(df["order_placed"] > 0, K, 0.0)
        assert np.allclose(df["ordering_cost"], expected_ordering)
        assert np.allclose(df["purchase_cost"], c * df["order_placed"])
        assert np.allclose(df["stockout_cost"], b * df["backorders"])


class TestPolicyAuditRecipe:
    def test_engine_state_matches_audit_recipe(self, basic_config: RunConfig) -> None:
        # IP_policy[t] == IP_Ledger[t] - order_placed[t]
        engine = InventoryEngine(basic_config)
        actions = [30.0 if t % 5 == 0 else 0.0 for t in range(basic_config.simulation.horizon)]
        obs_history = _drive(engine, actions)
        df = engine.ledger.to_dataframe()
        for t, obs in enumerate(obs_history):
            expected_ip = df.loc[t, "inventory_position"] - df.loc[t, "order_placed"]
            assert obs.inventory_position == pytest.approx(expected_ip)
            expected_oo = df.loc[t, "on_order"] - df.loc[t, "order_placed"]
            assert obs.on_order == pytest.approx(expected_oo)


class TestDemand:
    def test_negative_demand_clamped_to_zero(self, make_config: Callable[..., RunConfig]) -> None:
        # Tiny mean, large std => some draws will be negative.
        cfg = make_config(horizon=200, demand_mean=0.5, demand_std=5.0)
        engine = InventoryEngine(cfg)
        actions = [0.0] * 200
        _drive(engine, actions)
        df = engine.ledger.to_dataframe()
        assert (df["demand"] >= 0.0).all()

    def test_demand_factory_preserves_normal_stream(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        """Regression guard: factory wires the right RNG and preserves the clip.

        The inline ``max(0.0, rng.normal(...))`` was extracted from the
        engine into ``demand.normal.NormalDemand`` constructed via
        ``make_demand``. The pre-extraction behaviour was: per period, draw
        from the ``SeedManager``-spawned demand RNG, clip negatives to zero.
        This test reconstructs that expected stream independently from a
        fresh ``SeedManager(master_seed)`` and asserts equality with the
        engine's recorded ``demand`` column. If anyone reorders the call,
        drops the clip, or stops caching the spawned generator, this fires.
        """
        from core.seeding import SeedManager

        master_seed = 12345
        horizon = 75
        mean = 8.0
        std = 1.5

        cfg = make_config(
            horizon=horizon,
            demand_mean=mean,
            demand_std=std,
            master_seed=master_seed,
        )
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * horizon)
        actual = engine.ledger.to_dataframe()["demand"].to_numpy()

        # Independently rebuild the expected stream. SeedManager caches the
        # spawned Generator on construction (verified by inspection of
        # core/seeding.py); ``rng("demand")`` returns the same instance on
        # repeat calls.
        rng = SeedManager(master_seed).rng("demand")
        expected = np.array([max(0.0, float(rng.normal(mean, std))) for _ in range(horizon)])

        np.testing.assert_array_equal(actual, expected)

    def test_engine_runs_with_poisson_demand(self, make_config: Callable[..., RunConfig]) -> None:
        """Poisson demand drives the engine to completion under standard invariants.

        DemandConfig is a 2-arm union; this test pins the
        new arm at the integration layer. Standard battery: horizon
        length, non-negative demand / on_hand, Convention C (sales <=
        demand), no NaN/inf, finite total cost.
        """
        from core.config import PoissonDemandConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(update={"demand": PoissonDemandConfig(rate=10.0)})
        engine = InventoryEngine(cfg)
        # No-op actions — demand stream alone exercises the integration.
        _drive(engine, [0.0] * cfg.simulation.horizon)

        df = engine.ledger.to_dataframe()
        assert len(df) == cfg.simulation.horizon
        assert (df["demand"] >= 0.0).all()
        assert (df["on_hand"] >= 0.0).all()
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()

    def test_poisson_demand_stream_matches_factory_output(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        """Same regression-guard shape as the Normal stream-preservation test.

        Independently rebuild the expected Poisson stream from a fresh
        ``SeedManager(master_seed).rng("demand")`` and assert element-wise
        equality with ``ledger["demand"]``. Locks the seed cascade for
        Poisson — engine → SeedManager → factory → PoissonDemand → ledger.
        """
        from core.config import PoissonDemandConfig
        from core.seeding import SeedManager

        master_seed = 9876
        horizon = 75
        rate = 7.0

        base = make_config(horizon=horizon, master_seed=master_seed)
        cfg = base.model_copy(update={"demand": PoissonDemandConfig(rate=rate)})
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * horizon)
        actual = engine.ledger.to_dataframe()["demand"].to_numpy()

        rng = SeedManager(master_seed).rng("demand")
        expected = np.array([float(rng.poisson(rate)) for _ in range(horizon)])

        np.testing.assert_array_equal(actual, expected)

    def test_engine_runs_with_negative_binomial_demand(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        """Negative Binomial demand drives the engine end-to-end.

        DemandConfig widens to 3 arms; this test pins the new arm
        at the integration layer. Standard battery: horizon length,
        non-negative demand / on_hand, Convention C (sales <= demand),
        no NaN/inf, finite total cost. Cost ranking vs Normal / Poisson
        is NOT asserted here — it is a pedagogical observation.
        """
        from core.config import NegativeBinomialDemandConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(update={"demand": NegativeBinomialDemandConfig(n=10.0, p=0.5)})
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * cfg.simulation.horizon)

        df = engine.ledger.to_dataframe()
        assert len(df) == cfg.simulation.horizon
        assert (df["demand"] >= 0.0).all()
        assert (df["on_hand"] >= 0.0).all()
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()

    def test_negative_binomial_demand_stream_matches_factory_output(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        """Seed-cascade lock for NegBin — mirror of the Poisson test.

        Independently rebuild the expected NegBin stream from a fresh
        ``SeedManager(master_seed).rng("demand")`` and assert element-wise
        equality with ``ledger["demand"]``. Locks the seed cascade:
        engine → SeedManager → factory → NegativeBinomialDemand → ledger.
        """
        from core.config import NegativeBinomialDemandConfig
        from core.seeding import SeedManager

        master_seed = 9876
        horizon = 75
        n_param = 8.0
        p_param = 0.4

        base = make_config(horizon=horizon, master_seed=master_seed)
        cfg = base.model_copy(update={"demand": NegativeBinomialDemandConfig(n=n_param, p=p_param)})
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * horizon)
        actual = engine.ledger.to_dataframe()["demand"].to_numpy()

        rng = SeedManager(master_seed).rng("demand")
        expected = np.array(
            [float(rng.negative_binomial(n_param, p_param)) for _ in range(horizon)]
        )

        np.testing.assert_array_equal(actual, expected)

    def test_engine_runs_with_gamma_demand(self, make_config: Callable[..., RunConfig]) -> None:
        """Gamma demand drives the engine end-to-end.

        DemandConfig widens to 4 arms; this test pins the new
        continuous-support arm at the integration layer. Standard
        battery: horizon length, **strict positive** demand (Gamma's
        open-at-zero support, distinct from Poisson/NegBin's >= 0),
        non-negative on_hand, Convention C (sales <= demand), no
        NaN/inf, finite total cost. Cost ranking vs other arms is NOT
        asserted here — pedagogical observation.
        """
        from core.config import GammaDemandConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(update={"demand": GammaDemandConfig(shape=5.0, scale=2.0)})
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * cfg.simulation.horizon)

        df = engine.ledger.to_dataframe()
        assert len(df) == cfg.simulation.horizon
        # Strict positivity — distinct from integer arms' >= 0. Gamma's
        # support is the open interval (0, ∞).
        assert (df["demand"] > 0.0).all()
        assert (df["on_hand"] >= 0.0).all()
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()

    def test_gamma_demand_stream_matches_factory_output(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        """Seed-cascade lock for Gamma — mirror of the NegBin test.

        Independently rebuild the expected Gamma stream from a fresh
        ``SeedManager(master_seed).rng("demand")`` and assert
        element-wise equality with ``ledger["demand"]``. Locks the seed
        cascade: engine → SeedManager → factory → GammaDemand → ledger
        for the fourth distribution.
        """
        from core.config import GammaDemandConfig
        from core.seeding import SeedManager

        master_seed = 9876
        horizon = 75
        shape_param = 3.0
        scale_param = 2.5

        base = make_config(horizon=horizon, master_seed=master_seed)
        cfg = base.model_copy(
            update={"demand": GammaDemandConfig(shape=shape_param, scale=scale_param)}
        )
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * horizon)
        actual = engine.ledger.to_dataframe()["demand"].to_numpy()

        rng = SeedManager(master_seed).rng("demand")
        expected = np.array([float(rng.gamma(shape_param, scale_param)) for _ in range(horizon)])

        np.testing.assert_array_equal(actual, expected)

    def test_engine_runs_with_lognormal_demand(self, make_config: Callable[..., RunConfig]) -> None:
        """Lognormal demand drives the engine end-to-end.

        DemandConfig widens to 5 arms (closes the demand
        quartet); this test pins the new continuous-support, heavy-tail
        arm at the integration layer. Standard battery: horizon length,
        **strict positive** demand (Lognormal's open-at-zero support,
        same as Gamma, distinct from Poisson/NegBin's >= 0), non-negative
        on_hand, Convention C (sales <= demand), no NaN/inf, finite
        total cost. Cost ranking vs other arms is NOT asserted here —
        pedagogical observation.
        """
        from core.config import LognormalDemandConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(update={"demand": LognormalDemandConfig(mu=2.2114, sigma=0.4271)})
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * cfg.simulation.horizon)

        df = engine.ledger.to_dataframe()
        assert len(df) == cfg.simulation.horizon
        # Strict positivity — same as Gamma; Lognormal support is the
        # open interval (0, ∞).
        assert (df["demand"] > 0.0).all()
        assert (df["on_hand"] >= 0.0).all()
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()

    def test_lognormal_demand_stream_matches_factory_output(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        """Seed-cascade lock for Lognormal — closes the M2 quartet chain.

        Independently rebuild the expected Lognormal stream from a fresh
        ``SeedManager(master_seed).rng("demand")`` and assert
        element-wise equality with ``ledger["demand"]``. Locks the seed
        cascade: engine → SeedManager → factory → LognormalDemand →
        ledger for the fifth (and final M2) demand distribution.
        """
        from core.config import LognormalDemandConfig
        from core.seeding import SeedManager

        master_seed = 9876
        horizon = 75
        mu_param = 1.5
        sigma_param = 0.6

        base = make_config(horizon=horizon, master_seed=master_seed)
        cfg = base.model_copy(
            update={"demand": LognormalDemandConfig(mu=mu_param, sigma=sigma_param)}
        )
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * horizon)
        actual = engine.ledger.to_dataframe()["demand"].to_numpy()

        rng = SeedManager(master_seed).rng("demand")
        expected = np.array([float(rng.lognormal(mu_param, sigma_param)) for _ in range(horizon)])

        np.testing.assert_array_equal(actual, expected)

    def test_engine_runs_with_empirical_demand(self, make_config: Callable[..., RunConfig]) -> None:
        """Empirical demand drives the engine end-to-end.

        First non-parametric demand arm. The bundled
        ``data/m5/sample_smooth.parquet`` (synthetic Normal(10, 2)-shaped)
        plays back via IID resampling. Standard battery: horizon length,
        **non-negative** demand (locked at EmpiricalDemand.__init__ via
        history validation), non-negative on_hand, Convention C, no
        NaN/inf, finite total cost.
        """
        from pathlib import Path

        from core.config import EmpiricalDemandConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(
            update={
                "demand": EmpiricalDemandConfig(history_path=Path("data/m5/sample_smooth.parquet"))
            }
        )
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * cfg.simulation.horizon)

        df = engine.ledger.to_dataframe()
        assert len(df) == cfg.simulation.horizon
        assert (df["demand"] >= 0.0).all()
        assert (df["on_hand"] >= 0.0).all()
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()

    def test_engine_empirical_demand_close_to_normal_baseline(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        """Chassis correctness gate.

        Because the bundled history IS ``Normal(10, 2)`` clipped at 0,
        the empirical resampling should reproduce a policy outcome close
        to the Normal(10, 2) baseline ($5,562.05). Loose ±20% bound —
        the cost depends on the specific RNG sequence through
        ``rng.choice``, which diverges from ``rng.normal`` even at the
        same marginal distribution. A wildly-off cost (outside ±20%)
        signals a chassis bug.
        """
        from pathlib import Path

        from core.config import EmpiricalDemandConfig, NormalDemandConfig

        base = make_config(horizon=90, master_seed=42)
        normal_cfg = base.model_copy(update={"demand": NormalDemandConfig(mean=10.0, std=2.0)})
        empirical_cfg = base.model_copy(
            update={
                "demand": EmpiricalDemandConfig(history_path=Path("data/m5/sample_smooth.parquet"))
            }
        )

        normal_engine = InventoryEngine(normal_cfg)
        _drive(normal_engine, [0.0] * normal_cfg.simulation.horizon)
        normal_costs = (
            normal_engine.ledger.to_dataframe()[
                ["holding_cost", "ordering_cost", "purchase_cost", "stockout_cost"]
            ]
            .sum()
            .sum()
        )

        empirical_engine = InventoryEngine(empirical_cfg)
        _drive(empirical_engine, [0.0] * empirical_cfg.simulation.horizon)
        empirical_costs = (
            empirical_engine.ledger.to_dataframe()[
                ["holding_cost", "ordering_cost", "purchase_cost", "stockout_cost"]
            ]
            .sum()
            .sum()
        )

        rel_diff = abs(empirical_costs - normal_costs) / normal_costs
        assert rel_diff < 0.20, (
            f"empirical cost {empirical_costs:.2f} differs from Normal baseline "
            f"{normal_costs:.2f} by more than ±20% ({rel_diff:.2%}); "
            f"chassis correctness gate failed"
        )

    def test_engine_empirical_demand_stream_matches_factory_output(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        """Seed-cascade lock for Empirical.

        Independently rebuild the expected empirical demand stream from a
        fresh ``SeedManager(master_seed).rng("demand")`` + a fresh load
        of the bundled parquet, then assert element-wise equality with
        ``ledger["demand"]``. Locks the seed cascade: engine →
        SeedManager → factory → EmpiricalDemand (with parquet load +
        validation) → Pattern (Stationary identity) → ledger.
        """
        from pathlib import Path

        import pandas as pd

        from core.config import EmpiricalDemandConfig
        from core.seeding import SeedManager

        master_seed = 9876
        horizon = 75
        history_path = Path("data/m5/sample_smooth.parquet")

        base = make_config(horizon=horizon, master_seed=master_seed)
        cfg = base.model_copy(update={"demand": EmpiricalDemandConfig(history_path=history_path)})
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * horizon)
        actual = engine.ledger.to_dataframe()["demand"].to_numpy()

        history = pd.read_parquet(history_path)["demand"].to_numpy(dtype=np.float64)
        rng = SeedManager(master_seed).rng("demand")
        expected = np.array([float(rng.choice(history)) for _ in range(horizon)])

        np.testing.assert_array_equal(actual, expected)

    # ---- real M5 archetype curation (data-backed) ---------- #
    _ARCHETYPES = ("smooth", "intermittent", "seasonal", "promotional")

    @pytest.mark.parametrize("archetype", _ARCHETYPES)
    def test_engine_runs_with_archetype(
        self, archetype: str, make_config: Callable[..., RunConfig]
    ) -> None:
        """Each real M5 archetype drives the engine end-to-end.

        Standard invariant battery only — NO cost-baseline assertion (real
        data has no known baseline and demand scale varies per archetype).
        Cost ranking across archetypes is observed, never
        asserted.
        """
        from pathlib import Path

        from core.config import EmpiricalDemandConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(
            update={
                "demand": EmpiricalDemandConfig(
                    history_path=Path(f"data/m5/archetype_{archetype}.parquet")
                )
            }
        )
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * cfg.simulation.horizon)

        df = engine.ledger.to_dataframe()
        assert len(df) == cfg.simulation.horizon
        assert (df["demand"] >= 0.0).all()
        assert (df["on_hand"] >= 0.0).all()
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()

    def test_archetypes_are_distinct_marginals(self) -> None:
        """The 4 curated archetypes have genuinely different marginals.

        By construction: smooth is fast-moving (high mean, low zero-fraction);
        intermittent is slow-moving (low mean, high zero-fraction). Locks that
        the curation produced real variety, not four lookalikes — and
        reconfirms every committed parquet satisfies the EmpiricalDemand
        contract (non-empty, finite, non-negative).
        """
        from pathlib import Path

        histories: dict[str, np.ndarray] = {}
        for archetype in self._ARCHETYPES:
            hist = pd.read_parquet(Path(f"data/m5/archetype_{archetype}.parquet"))[
                "demand"
            ].to_numpy(dtype=np.float64)
            assert hist.shape[0] > 0
            assert np.isfinite(hist).all()
            assert (hist >= 0.0).all()
            histories[archetype] = hist

        zero_fracs = {k: float((v == 0.0).mean()) for k, v in histories.items()}
        means = {k: float(v.mean()) for k, v in histories.items()}
        assert zero_fracs["intermittent"] > zero_fracs["smooth"]
        assert means["smooth"] > means["intermittent"]

    def test_archetype_parquets_match_manifest(self) -> None:
        """Committed parquets match their manifest provenance."""
        import json
        from pathlib import Path

        manifest = json.loads(Path("data/m5/archetypes_manifest.json").read_text())["archetypes"]
        for archetype in self._ARCHETYPES:
            entry = manifest[archetype]
            hist = pd.read_parquet(Path(f"data/m5/archetype_{archetype}.parquet"))[
                "demand"
            ].to_numpy(dtype=np.float64)
            assert hist.shape[0] == entry["n_rows"]
            assert float(hist.mean()) == pytest.approx(entry["mean"], abs=1e-4)
            assert float((hist == 0.0).mean()) == pytest.approx(entry["zero_fraction"], abs=1e-4)

    def test_archetype_determinism(self, make_config: Callable[..., RunConfig]) -> None:
        """Two runs of an archetype scenario at the same seed match."""
        from pathlib import Path

        from core.config import EmpiricalDemandConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(
            update={
                "demand": EmpiricalDemandConfig(
                    history_path=Path("data/m5/archetype_smooth.parquet")
                )
            }
        )
        engine_a = InventoryEngine(cfg)
        engine_b = InventoryEngine(cfg)
        _drive(engine_a, [0.0] * cfg.simulation.horizon)
        _drive(engine_b, [0.0] * cfg.simulation.horizon)
        df_a = engine_a.ledger.to_dataframe()["demand"].to_numpy()
        df_b = engine_b.ledger.to_dataframe()["demand"].to_numpy()
        np.testing.assert_array_equal(df_a, df_b)


class TestPatternEngineIntegration:
    """pattern engine chassis integration at the engine layer.

    Pins the four locks the chassis must hold:
      (1) Every existing scenario YAML continues to parse via the
          ``RunConfig.pattern`` default-factory and runs to completion.
      (2) Implicit (default-factory) Stationary and explicit Stationary
          produce byte-for-byte identical Ledgers.
      (3) The engine's ``_pattern`` attribute is constructed via the
          factory and is a ``StationaryPattern`` instance.
      (4) The engine passes ``self._t`` to ``pattern.apply()`` correctly:
          ``t = 0, 1, 2, …, horizon-1`` across the run. This is the
          canonical lock for the Seasonal pattern (which
          consumes ``t`` for sinusoidal modulation).
    """

    @pytest.mark.parametrize(
        "scenario_filename",
        [
            "example.yaml",
            "example_poisson.yaml",
            "example_negative_binomial.yaml",
            "example_gamma.yaml",
            "example_lognormal.yaml",
        ],
    )
    def test_existing_scenarios_run_unchanged_after_pattern_engine(
        self, scenario_filename: str
    ) -> None:
        # Each of the 5 pre-Bullet-10 scenario YAMLs (which omit any
        # ``pattern:`` block) must continue to parse via the default-factory
        # and run end-to-end. The chassis must not perturb any pre-existing
        # behavior. Per-distribution byte-for-byte stream equality is
        # already locked individually by the demand-stream tests above; this
        # parametrized test additionally proves the YAML-omits-pattern path
        # (the default-factory branch) works for every existing scenario.
        import pathlib

        from core.config import StationaryPatternConfig

        repo_root = pathlib.Path(__file__).resolve().parents[2]
        yaml_text = (repo_root / "data" / "scenarios" / scenario_filename).read_text()
        cfg = RunConfig.from_yaml(yaml_text)
        # The default-factory branch fires: cfg.pattern is StationaryPatternConfig.
        assert isinstance(cfg.pattern, StationaryPatternConfig)

        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * cfg.simulation.horizon)
        df = engine.ledger.to_dataframe()
        # Same standard battery as the per-arm engine smokes; if the chassis
        # perturbed any invariant, one of these fires.
        assert len(df) == cfg.simulation.horizon
        assert (df["demand"] >= 0.0).all()
        assert (df["on_hand"] >= 0.0).all()
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()

    def test_engine_with_implicit_stationary_matches_explicit_stationary(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        # Two RunConfigs differ ONLY in whether ``pattern`` is explicit or
        # implicit-via-default-factory. The resulting Ledgers must be
        # byte-for-byte identical across all 15 columns. Locks the
        # default-factory equivalence at the engine level. This is the
        # cornerstone test of the backward-compat thesis at the
        # integration layer.
        from core.config import StationaryPatternConfig

        cfg_implicit = make_config(horizon=50, master_seed=12345)
        cfg_explicit = cfg_implicit.model_copy(update={"pattern": StationaryPatternConfig()})

        engine_a = InventoryEngine(cfg_implicit)
        engine_b = InventoryEngine(cfg_explicit)
        actions = [30.0 if t % 5 == 0 else 0.0 for t in range(50)]
        _drive(engine_a, actions)
        _drive(engine_b, actions)
        df_a = engine_a.ledger.to_dataframe()
        df_b = engine_b.ledger.to_dataframe()
        assert df_a.equals(df_b), (
            "implicit Stationary (default-factory) and explicit Stationary "
            "produced different Ledgers — backward-compat broken"
        )

    def test_engine_constructs_pattern_from_factory(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        # After ``InventoryEngine.__init__``, ``engine._pattern`` must be a
        # ``StationaryPattern`` (the factory-dispatched concrete arm).
        # Accesses the name-mangled private attribute by convention — if a
        # future refactor renames or restructures it, this test forces an
        # explicit re-pinning. The factory contract is also locked by
        # ``test_pattern_registry.py``; this test pins the *engine's*
        # consumption of that factory.
        from demand.patterns.stationary import StationaryPattern

        cfg = make_config(horizon=10)
        engine = InventoryEngine(cfg)
        assert isinstance(engine._pattern, StationaryPattern)  # noqa: SLF001

    def test_engine_t_advances_correctly_across_pattern_apply_calls(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        # The engine MUST pass ``self._t`` to ``pattern.apply(base, t)`` on
        # every period, with t advancing 0, 1, 2, …, horizon-1. This is the
        # canonical lock for the Seasonal pattern, which consumes
        # ``t`` for sinusoidal modulation; if the chassis wires the shim
        # incorrectly (e.g., passes ``self._t + 1``, or passes the
        # _previous_ ``_t`` because of a step-ordering bug), Seasonal will
        # silently produce wrong demand on every period.
        from demand.patterns.base import Pattern

        class _RecordingPattern(Pattern):
            """Test-only pattern that records every (base, t) it receives.

            Not registered with the factory; substituted into the engine
            after construction via the name-mangled private attribute.
            Returns base unchanged so the recorder behaves like Stationary
            from the engine's perspective.
            """

            def __init__(self) -> None:
                self.records: list[tuple[float, int]] = []

            def apply(self, base: float, t: int) -> float:
                self.records.append((base, t))
                return base

        cfg = make_config(horizon=20)
        engine = InventoryEngine(cfg)
        recorder = _RecordingPattern()
        engine._pattern = recorder  # noqa: SLF001
        _drive(engine, [0.0] * cfg.simulation.horizon)

        t_values = [t for _, t in recorder.records]
        assert t_values == list(range(cfg.simulation.horizon)), (
            f"engine passed wrong t-sequence to pattern.apply: {t_values}"
        )
        # Also: every recorded base was the value returned by Demand.draw()
        # for the corresponding period; the ledger preserves it (identity
        # passthrough).
        df = engine.ledger.to_dataframe()
        recorded_bases = [base for base, _ in recorder.records]
        assert list(df["demand"].to_numpy()) == recorded_bases

    def test_engine_runs_with_seasonal_pattern(self, make_config: Callable[..., RunConfig]) -> None:
        # first concrete pattern arm at the integration
        # layer. Standard battery: horizon length, non-negative demand
        # (locked by amplitude < 1 bound), non-negative on_hand, Convention
        # C (sales <= demand), no NaN/inf, finite total cost. The cost
        # comparison vs Stationary baseline is observed only
        # (pedagogical observation, not a test gate).
        from core.config import SeasonalPatternConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(
            update={"pattern": SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)}
        )
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * cfg.simulation.horizon)

        df = engine.ledger.to_dataframe()
        assert len(df) == cfg.simulation.horizon
        assert (df["demand"] >= 0.0).all()
        assert (df["on_hand"] >= 0.0).all()
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()

    def test_engine_seasonal_demand_visibly_oscillates(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        # At amplitude=0.5, period=12, the seasonal factor swings 0.5 to
        # 1.5, multiplying Normal(10, 2) draws. The resulting demand column
        # spans a much wider range than the unmodulated Normal baseline
        # (where ~99% of draws fall within mean ± 3*std = [4, 16], span
        # ≈ 12). Pin a loose lower bound (span > 8.0) to confirm visible
        # oscillation without over-specifying. Smoke test, not precision.
        from core.config import SeasonalPatternConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(
            update={"pattern": SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)}
        )
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * cfg.simulation.horizon)
        df = engine.ledger.to_dataframe()
        demand_span = float(df["demand"].max()) - float(df["demand"].min())
        assert demand_span > 8.0, (
            f"seasonal demand column span={demand_span:.2f} is too narrow; "
            f"expected visible oscillation from amplitude=0.5 modulation"
        )

    def test_engine_seasonal_demand_stream_matches_formula(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        # Locks the full composition: SeedManager → NormalDemand →
        # SeasonalPattern → ledger. Independently rebuild the expected
        # seasonal-modulated stream from a fresh SeedManager rng and assert
        # element-wise equality with the engine's recorded demand column.
        # NormalDemand's max(0.0, ...) clip survives, then the seasonal
        # factor is applied (matches the engine's `_pattern.apply(_demand.draw(),
        # _t)` composition order).
        import math as _math

        from core.config import SeasonalPatternConfig
        from core.seeding import SeedManager

        master_seed = 9876
        horizon = 75
        mean = 8.0
        std = 1.5
        amplitude = 0.4
        period = 10
        phase = 0.3

        base = make_config(
            horizon=horizon,
            demand_mean=mean,
            demand_std=std,
            master_seed=master_seed,
        )
        cfg = base.model_copy(
            update={
                "pattern": SeasonalPatternConfig(amplitude=amplitude, period=period, phase=phase)
            }
        )
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * horizon)
        actual = engine.ledger.to_dataframe()["demand"].to_numpy()

        rng = SeedManager(master_seed).rng("demand")
        expected = np.array(
            [
                max(0.0, float(rng.normal(mean, std)))
                * (1.0 + amplitude * _math.sin(2.0 * _math.pi * t / period + phase))
                for t in range(horizon)
            ]
        )

        np.testing.assert_array_equal(actual, expected)

    def test_engine_runs_with_trending_pattern(self, make_config: Callable[..., RunConfig]) -> None:
        # second concrete pattern arm at the integration
        # layer. Standard battery: horizon length, non-negative demand
        # (locked by clip-at-source in TrendingPattern.apply), non-negative
        # on_hand, Convention C (sales <= demand), no NaN/inf, finite total
        # cost. The cost comparison vs Stationary and Seasonal baselines is
        # observed only (pedagogical observation, not a test
        # gate).
        from core.config import TrendingPatternConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(update={"pattern": TrendingPatternConfig(slope=0.02)})
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * cfg.simulation.horizon)

        df = engine.ledger.to_dataframe()
        assert len(df) == cfg.simulation.horizon
        assert (df["demand"] >= 0.0).all()
        assert (df["on_hand"] >= 0.0).all()
        assert (df["sales"] <= df["demand"] + 1e-9).all()
        assert np.isfinite(df.drop(columns="period").to_numpy()).all()

    def test_engine_trending_demand_visibly_grows(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        # At slope=0.02 over 90 periods, the factor at t=89 is 2.78. The
        # late-horizon demand mean should be notably higher than the
        # early-horizon mean. Pin a loose lower bound (late mean > 1.5 ×
        # early mean) to confirm visible monotone growth without over-
        # specifying. Smoke test, not precision.
        from core.config import TrendingPatternConfig

        base = make_config(horizon=90, master_seed=42)
        cfg = base.model_copy(update={"pattern": TrendingPatternConfig(slope=0.02)})
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * cfg.simulation.horizon)
        df = engine.ledger.to_dataframe()
        early_mean = float(df["demand"].iloc[:10].mean())
        late_mean = float(df["demand"].iloc[-10:].mean())
        assert late_mean > early_mean * 1.5, (
            f"trending demand: early mean={early_mean:.2f}, late mean={late_mean:.2f}; "
            f"expected late mean > 1.5 × early mean (factor at t=89 is 2.78)"
        )

    def test_engine_trending_demand_stream_matches_formula(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        # Locks the full composition: SeedManager → NormalDemand →
        # TrendingPattern → ledger. Independently rebuild the expected
        # trending-modulated stream from a fresh SeedManager rng and assert
        # element-wise equality with the engine's recorded demand column.
        # Two nested clips: NormalDemand's inner max(0.0, ...) and
        # TrendingPattern's outer max(0.0, ...). The composition order
        # matches the engine's `_pattern.apply(_demand.draw(), _t)`.
        from core.config import TrendingPatternConfig
        from core.seeding import SeedManager

        master_seed = 9876
        horizon = 75
        mean = 8.0
        std = 1.5
        slope = 0.02

        base = make_config(
            horizon=horizon,
            demand_mean=mean,
            demand_std=std,
            master_seed=master_seed,
        )
        cfg = base.model_copy(update={"pattern": TrendingPatternConfig(slope=slope)})
        engine = InventoryEngine(cfg)
        _drive(engine, [0.0] * horizon)
        actual = engine.ledger.to_dataframe()["demand"].to_numpy()

        rng = SeedManager(master_seed).rng("demand")
        expected = np.array(
            [
                max(0.0, max(0.0, float(rng.normal(mean, std))) * (1.0 + slope * t))
                for t in range(horizon)
            ]
        )

        np.testing.assert_array_equal(actual, expected)


class TestPerformance:
    def test_365_period_run_under_threshold(self, make_config: Callable[..., RunConfig]) -> None:
        cfg = make_config(horizon=365)
        engine = InventoryEngine(cfg)
        actions = [30.0 if t % 7 == 0 else 0.0 for t in range(cfg.simulation.horizon)]
        start = time.perf_counter()
        _drive(engine, actions)
        engine.ledger.to_dataframe()
        elapsed = time.perf_counter() - start
        # Engine alone budget; full pipeline target is 100 ms after the
        # Gym wrapper lands (bullet 6).
        assert elapsed < 0.25, f"engine took {elapsed:.3f}s, expected < 0.25s"
