"""Integration: the experiments rollout primitive."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from core.config import RunConfig
from core.simulation import InventoryEngine
from experiments import single_run
from experiments.common import rollout
from policies.registry import make_policy


def test_single_run_returns_full_horizon_frame(make_config: Callable[..., RunConfig]) -> None:
    config = make_config()
    frame = single_run.run(config)
    assert len(frame) == config.simulation.horizon
    for column in ("on_hand", "demand", "sales", "holding_cost"):
        assert column in frame.columns


def test_single_run_is_deterministic(make_config: Callable[..., RunConfig]) -> None:
    config = make_config()
    pd.testing.assert_frame_equal(single_run.run(config), single_run.run(config))


def test_single_run_matches_inline_rollout(make_config: Callable[..., RunConfig]) -> None:
    config = make_config()

    # Hand-inlined rollout (the pre-extraction form) must equal single_run.run —
    # any future wrapping of the rollout breaks this loudly.
    policy = make_policy(config.policy)
    engine = InventoryEngine(config)
    state = engine.initial_observation()
    done = False
    while not done:
        state, done = engine.step(policy.decide(state))
    expected = engine.ledger.to_dataframe()

    pd.testing.assert_frame_equal(single_run.run(config), expected)


def test_rollout_drives_engine_to_completion(make_config: Callable[..., RunConfig]) -> None:
    config = make_config()
    policy = make_policy(config.policy)
    engine = InventoryEngine(config)

    rollout(engine, policy)

    frame = engine.ledger.to_dataframe()
    assert len(frame) == config.simulation.horizon
    pd.testing.assert_frame_equal(frame, single_run.run(config))
